# 肺野分割モデル（U-Net）の学習と配置

[English](lung-unet_EN.md) | 日本語

roi-service は CheXmask を教師にした U-Net で肺野・心臓を分割し、胸部野
（肺野＋縦隔＋横隔膜）を作る。この文書は、そのモデルを学習して配置する手順である。

**U-Net は必須である。** 胸部野には心臓の mask が要り、規則ベースの profile
（`nihcc-classical-v1`）は心臓を出さない。`ROI_PROFILE` が規則ベースだと roi-service は
起動時に止まる。`nihcc-classical-v1` は Dice の比較の基準（§5）にだけ使う。

手元（約 5,000 枚）でも EC2（約 11 万枚）でも同じ手順で動く。違いは帯域と GPU の
有無だけで、コマンドの引数で吸収する。

## profile とモデル

**入力解像度が違うモデルは互換ではない**ので、profile を分けてある。

| profile | 肺野の取り方 | 入力 | 用途 |
|---|---|---|---|
| `nihcc-unet-local-v1` | U-Net（CPU で学習できる縮小版） | 160px | **既定**（`.env.example` / `docker-compose.yml`） |
| `nihcc-unet-v1` | U-Net（GPU で学習） | 384px | 高解像度版 |
| `nihcc-classical-v1` | 規則ベース | - | Dice の比較の基準のみ |

**胸部野の分類器は profile に結び付いている。** 分類器（[chest-classifier.md](chest-classifier.md)）は
`nihcc-unet-local-v1` の胸部野で学習してあり、ingest-worker が起動時に突き合わせる。
profile を変えるなら分類器も学習し直し、全量を取り込み直す。

既定の 160px のモデルでの実測は、無作為 1,000 枚で Dice 中央値 0.968、破綻 0.5%。

## 使う Python

この文書の `python` は、いずれも**プロジェクトの開発環境**（`.venv-dev`）である。
素の `python` には cv2 も numpy も入っていないので `ModuleNotFoundError` になる。

```powershell
# PowerShell
.\.venv-dev\Scripts\Activate.ps1
```

```bash
# bash / EC2
source .venv-dev/bin/activate      # EC2 は環境に合わせる
```

GPU での学習（§3）だけはコンテナの中で動かすので、この開発環境とは無関係である。

## 全体像

| | 段 | 何をするか | 手元 | EC2 |
|---|---|---|---|---|
| 1 | [参照 mask の取り込み](#1-参照-mask-の取り込み) | CheXmask から正解 mask を取る | 8 分 | 10〜40 分 |
| 2 | [学習データの作成](#2-学習データの作成) | 画像と mask を配列へ | 5 分 | 20 分 |
| 3 | [学習](#3-学習) | U-Net を学習して ONNX を書き出す | 50 分（160px） | 30〜60 分（384px） |
| 4 | [モデルの配置](#4-モデルの配置) | `infra/lung-unet/` へ置く | 1 分 | 1 分 |
| 5 | [精度の確認](#5-精度の確認) | Dice で測る | 10 分 | 10 分 |
| 6 | [roi-service への反映](#6-roi-service-への反映) | 起動して版を確かめる | 5 分 | 5 分 |
| 7 | [一括 mask 作成](#7-一括-mask-作成任意) | 取り込みとは別に mask を出す（任意） | 20 分 | 3 時間 |

推論（§4 以降）はどちらも onnxruntime の CPU で足りる。

---

## 1. 参照 mask の取り込み

出典は **CheXmask Database v1.0.0**（PhysioNet, CC BY 4.0, 認証不要）。ChestX-ray8
すなわちこのシステムが使う画像そのものに、左肺・右肺・心臓の mask が付いている。

### 手元（約 5,000 枚）

配布元の CSV は 2.2 GB あるが、画像 ID が近いものが大きな塊で並んでいる。手元の
corpus（`images_001`、`00000001`〜`00001335`）は狭い帯なので、該当する塊だけを取れば足りる。

```bash
python scripts/fetch_reference_masks.py --from-mb 1480 --to-mb 1600
```

corpus の ID 帯が違うときは、まず位置を調べる。

```bash
python scripts/fetch_reference_masks.py --probe
```

### EC2（約 11 万枚）

```bash
python scripts/fetch_reference_masks.py --all
```

どちらも `data/reference-masks/chestxray8.csv` へ書く。手元にある画像の分しか
残さないので、11 万枚ぶんで 2 GB ほどになる。

---

## 2. 学習データの作成

```bash
# 既定の 160px（nihcc-unet-local-v1）
python scripts/build_lung_dataset.py --max 1200 --size 160 --out data/lung-dataset-160

# 384px（nihcc-unet-v1）
python scripts/build_lung_dataset.py --max 8000 --size 384
```

384px の出力先は既定の `data/lung-dataset/`。要点は 3 つある。

- **前処理は推論時と同じ `normalize` を通す。** ずれると学習時と推論時で入力の分布が変わる
- **分割は患者単位。** 同じ患者の画像が学習側と検証側に跨がると成績が実力より良く出る
- **参照の品質で絞る。** 既定は Dice RCA 0.8 以上（評価時の 0.7 より厳しくする）

`--max` は枚数の上限である。肺野分割は易しい課題なので 8,000 枚あれば足りる。
11 万枚すべてを使っても Dice はほとんど変わらず、データの作成と学習が 10 倍以上かかる。

---

## 3. 学習

### 160px（CPU で約 50 分）

開発環境に torch (CPU) が入っているので、そのまま動く。

```bash
python scripts/train_lung_unet.py --dataset data/lung-dataset-160 --epochs 15 --batch 8 --base 16 --workers 0
```

CPU での実測（8 スレッド、base=16）:

| 解像度 | 1 枚あたり | 1,000 枚で 1 epoch |
|---|---|---|
| 128px | 0.14 秒 | 2.3 分 |
| 160px | 0.21 秒 | 3.5 分 |
| 256px | 0.55 秒 | 9.2 分 |

`base` を 16 から 24 へ上げると 1.6 倍遅くなる。256px 以上を CPU で学習するのは現実的でない。

### 384px（GPU で 30〜60 分）

**学習専用の image を作る。** 土台は embedding-service で、そこに `onnx` / `onnxscript` /
`opencv-python-headless` を足す。EC2 では `TORCH_INDEX_URL` が cu121 なので、土台の
torch は CUDA 版になる。torch を入れ直さないので、学習と推論で版がずれない。

```bash
docker compose build embedding-service        # 土台（まだなら）
docker build -f infra/train/Dockerfile.lung-unet -t local/nihcc-lung-unet-train:1.0.0 .
```

```bash
mkdir -p models/lung-unet
docker run --rm --gpus all --shm-size=2g \
  -v "$PWD:/work" -w /work \
  --user "$(id -u):$(id -g)" \
  local/nihcc-lung-unet-train:1.0.0 \
  python -u scripts/train_lung_unet.py --epochs 40 --batch 16
```

- **`--shm-size` が要る。** DataLoader の worker は共有メモリでテンソルを渡すが、Docker の
  `/dev/shm` は既定 64 MB しかなく、384px・batch 16 では足りない
- **`docker compose run` は使えない。** 運用のコンテナは `read_only: true` で `/tmp` 以外に
  書けず、実行ユーザーも uid 10001 固定である。`docker run` に `--user` で host の uid を
  渡せば、出力が host の所有になる
- GPU が使えるかは `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` で確かめる

配管だけ確かめたいときは `--smoke` を付ける（2 epoch・16 枚、数十秒）。

### 出力

`models/lung-unet/` に出る（`--out` で変えられる）。

| ファイル | 中身 |
|---|---|
| `lung-unet.onnx` | 推論に使う。**重みを含む 1 ファイル** |
| `best.pt` | 検証 Dice が最良だった重み。追加学習用 |
| `metrics.json` | epoch ごとの検証 Dice（右肺・左肺・心臓） |

**見るべき数字は `val_lung_dice`**（右肺と左肺の平均）。0.95 前後まで上がるのが普通で、
0.90 を下回るなら学習データか前処理を疑う。

---

## 4. モデルの配置

compose は `${LUNG_MODEL_DIR:-./infra/lung-unet}` を `/models/lung-unet` へ read-only で
mount する。モデルは git に入れない。

```bash
# bash / EC2
cp models/lung-unet/lung-unet.onnx infra/lung-unet/
sha256sum infra/lung-unet/lung-unet.onnx
```

```powershell
# PowerShell
Copy-Item models\lung-unet\lung-unet.onnx infra\lung-unet\
Get-FileHash infra\lung-unet\lung-unet.onnx -Algorithm SHA256
```

別の環境で学習したモデルを持ち込むときは、ハッシュを控えて持ち込んだ先で突き合わせる。
所在と SHA-256 を DB へ記録するには `register_model_artifact.py` を使う（[OPERATIONS.md](../OPERATIONS.md) §6.5）。

---

## 5. 精度の確認

**配置する前に測る。** 参照 mask は §1 で取り込んだものを使う。

ホストで動かす。profile が持つ `model_path` はコンテナ内のパス（`/models/lung-unet/...`）
なので、**`LUNG_MODEL_PATH` で置き場所を教える**。忘れると `分割モデルが無い` で止まる。

```bash
# bash / EC2
python scripts/eval_roi_dice.py --limit 500 --profile nihcc-classical-v1 \
    --json tmp/roi-dice/classical.json

LUNG_MODEL_PATH="$PWD/models/lung-unet/lung-unet.onnx" \
python scripts/eval_roi_dice.py --limit 500 --profile nihcc-unet-local-v1 \
    --json tmp/roi-dice/unet.json --inspect 9
```

```powershell
# PowerShell
$env:LUNG_MODEL_PATH = "$PWD\models\lung-unet\lung-unet.onnx"
python scripts/eval_roi_dice.py --limit 500 --profile nihcc-unet-local-v1 --json tmp/roi-dice/unet.json --inspect 9
```

profile はモデルの解像度に合わせる（160px は `nihcc-unet-local-v1`、384px は `nihcc-unet-v1`）。

`--inspect` で成績の悪い順に外周を描く（黄=このシステム、水色=参照）。無作為抽出で測り、
そのうえで悪いものを目で確かめる。

規則ベースの値（無作為 1,000 枚）:

```
Dice 中央値 0.828   四分位 0.725-0.879
実効的な破綻（Dice < 0.7 または抽出失敗）: 23.8%
```

> **参照は模型が作ったもので、専門家の手描きではない。** U-Net も同じ CheXmask で
> 学習するので、Dice は「参照にどれだけ似たか」を測っており、**高く出る方向に偏る。**
> 規則ベースとの差と、目視での破綻の減り方を見る。

結果を DB へ残す手順は [OPERATIONS.md](../OPERATIONS.md) §6.3。

---

## 6. roi-service への反映

`.env` の `ROI_PROFILE` がモデルの解像度と合っていることを確かめ、作り直す。

```bash
grep '^ROI_PROFILE=' .env                       # 既定 nihcc-unet-local-v1
docker compose up -d --force-recreate roi-service
docker compose ps roi-service                   # healthy になるまで 10 秒ほど
```

roi-service はポートを公開していないので、コンテナの中から確かめる。

```bash
docker compose exec roi-service python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/ready').read().decode())"
```

```
{"status":"ok","profile":"nihcc-unet-local-v1",
 "roi_model_version":"unet-lung-spine-local-v2.0.0+256051fe3a84", ...}
```

`roi_model_version` は profile ごとに違う。モデルを差し替えたら、検索の対象を揃えるため
取り込み直す（`python scripts/load_initial_images.py`）。

---

## 7. 一括 mask 作成（任意）

取り込みとは別に、mask だけを画像として出したいときに使う。品質の確認や図の作成用である。
取り込みは roi-service 経由で mask を作るので、通常の運用では要らない。

```bash
# 手元
python scripts/segment_lungs.py --images NIHCC/images --out data/lung-masks --profile nihcc-unet-local-v1

# EC2（約 11 万枚）。GPU があれば provider を指定する
python scripts/segment_lungs.py \
    --images /imports/nihcc/images --out /data/lung-masks \
    --batch 16 --providers CUDAExecutionProvider CPUExecutionProvider
```

出力は 1 枚 1 PNG で、**bit を詰めた単チャネル**（1=右肺 2=左肺 4=心臓）。
途中で止めても既にある分は飛ばして再開する。

`summary.json` に肺野の割合の分布と、`suspicious`（割合が 10〜40% を外れる画像）が
出る。目視で当たりを付けるのに使う。

**解像度は profile ではなくモデルから読む。** `--profile` が違っていても動き、
食い違いを表示する。閾値と前処理だけが profile から来る。

GPU で推論するには `onnxruntime-gpu` が要る。無ければ CPU へ自動で落ちる。
11 万枚を CPU で回すと 3 時間ほどかかる。

---

## つまずきやすいところ

**`onnxruntime が無い`**
: roi-service の image を作り直す。`docker compose build roi-service`

**`Got invalid dimensions for input: image ... Got: 384 Expected: 160`**
: profile とモデルの解像度が食い違っている。roi-service では `SegmenterError` として
  起動時に止まる。`.env` の `ROI_PROFILE` を合わせる。

**`unknown extraction profile: ...`**
: profile は image へ焼き込まれる。`docker compose build roi-service`

**`分割モデルが無い: /models/lung-unet/lung-unet.onnx`**
: `infra/lung-unet/` に置いたか、`LUNG_MODEL_DIR` が合っているかを見る。
  `docker compose exec roi-service ls -l /models/lung-unet`

**`No module named 'onnxscript'` / `No module named 'cv2'`（学習）**
: 学習用 image を作っていない。
  `docker build -f infra/train/Dockerfile.lung-unet -t local/nihcc-lung-unet-train:1.0.0 .`

**`Read-only file system: '/srv/.local'` / `Permission denied: '/work/models'`**
: `docker compose run` で学習しようとしている。§3 の `docker run` を使う。

**`Unexpected bus error encountered in worker ... insufficient shared memory (shm)`**
: `--shm-size=2g` が抜けている。暫定なら `--workers 0` でも回るが、GPU が入力待ちになる。

**ONNX が 2 ファイルになる（`lung-unet.onnx` と `lung-unet.onnx.data`）**
: `train_lung_unet.py` が 1 ファイルへ詰め直すので、通常は起きない。手で書き出した
  ときは `onnx.save_model(..., save_as_external_data=False)` を通す。

**学習の Dice が 0 のまま上がらない**
: ラベルの左右が入れ替わっていないかを疑う。`build_lung_dataset.py` は重心で確かめているが、
  `--size` を変えたときは数枚を描いて確かめる。

**`UnicodeEncodeError: 'cp932'`**
: Windows のコンソールで torch が絵文字を出したとき。`PYTHONIOENCODING=utf-8` を付ける。

**`ModuleNotFoundError: No module named 'cv2'`（ホスト）**
: 素の `python` を使っている。「[使う Python](#使う-python)」を見る。

**PowerShell で `sha256sum` / `cp` が無い**

| Unix | PowerShell |
|---|---|
| `sha256sum F` | `Get-FileHash F -Algorithm SHA256` |
| `cp A B` | `Copy-Item A B` |
| `ls -l` | `Get-ChildItem` |
| `export VAR=x; cmd` | `$env:VAR = "x"; cmd` |
