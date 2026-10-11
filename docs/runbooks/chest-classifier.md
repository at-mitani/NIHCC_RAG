# 胸部野の分類器の作成（教師データ・学習・評価・書き出し）

[English](chest-classifier_EN.md) | 日本語

所見プロファイル（`EMBEDDING_FINE_BACKEND=chest-profile`）が使う分類器
`infra/chest/head-pool-chest.npz` を作る手順。全 112,120 枚から胸部野
（肺野＋縦隔＋横隔膜）の特徴を作り、画像単位のラベル（14 所見）で分類器を学習し、
本番の成果物として書き出す。

作った分類器を本番へ入れる手順は [chest-profile-setup.md](chest-profile-setup.md)。

- 実行場所: EC2（g5.2xlarge、GPU の embedding-service へ特徴を投げる）
- 所要: 半日（全量の特徴の生成が 3〜5 時間）
- DB は使わない

---

## 0. 何が作られるか

ChestX-ray14 のラベルは画像単位である。検索の単位を画像 1 枚の胸部野に揃えているので、
そのラベルがそのまま教師データになる。

`scripts/run_chest_eval.sh` が次を通しで行う（中身は `scripts/chest_image_eval.py` の
サブコマンド）。

| 段 | サブコマンド | 出力（`tmp/chest/`） |
|---|---|---|
| 1 | `coverage` | `coverage.json`（正解 box が肺野・胸部野へ入る割合） |
| 2 | `sheet` | `sheet.png`（無作為 64 枚に輪郭を重ねた一覧） |
| 3 | `vectors` | `vec_<表現>_<profile>_<枚数>.npz`、`shards/`（特徴。再開可能） |
| 4 | `train` | `train.json`（AUC・Test B）、`head_<表現>_<profile>.npz`（分類器）、`ec2-full/teacher_<profile>_<枚数>.npz`（教師データと分割） |

比べる表現は次の 5 つ。本番に使うのは `pool-chest` である。

| 名前 | 中身 | 役割 |
|---|---|---|
| `pool-chest` | 胸部野 mask で RAD-DINO の patch token を pooling | **本番** |
| `pool-lung` | 肺野 mask だけで pooling | 心臓を外した対照 |
| `pool-whole` | 画像全体で pooling | 枠を絞らない対照 |
| `crop-chest` | 胸部野の外を黒塗りした crop の RAD-DINO | 枠外を物理的に消した対照 |
| `deterministic` | 胸部野 crop の画像統計量 | **帰無**（医学的意味なし） |

学習・評価は患者単位で分割する（train / val / test）。`ec2-full/teacher_*.npz` は
検索品質の測定（[chest-evaluation.md](chest-evaluation.md) §1）がクエリを test の患者から
選ぶのに使うので、git で追跡している。

---

## 1. 前提

### 1.1 評価用の仮想環境

スクリプトはホストで動かす（ROI 抽出に cv2 と U-Net を使い、埋め込みだけを HTTP で
embedding-service へ投げる）。

```bash
cd ~/NIHCC_RAG
python3 -m venv ~/.venv-eval && source ~/.venv-eval/bin/activate
pip install numpy httpx opencv-python-headless pyyaml pillow onnxruntime
```

`run_chest_eval.sh` は最初にこれらが import できるかを確かめ、足りなければ止まる。
**nohup で起動するプロセスは起動したシェルの環境を引き継ぐ**ので、同じシェルで venv を
有効にしてから実行する。

### 1.2 U-Net と正解 box

```bash
ls infra/lung-unet/lung-unet.onnx              # 無ければ lung-unet.md
ls tmp/gold/BBox_List_2017.csv                 # git で追跡している
```

profile の既定は `nihcc-unet-local-v1`（160px）。384px のモデルを使うときは、以降の
コマンドに `CHEST_PROFILE=nihcc-unet-v1` を付ける。**分類器は学習に使った profile と
組になる**ので、本番の `ROI_PROFILE` も同じにする。

正解 box は検証専用で、学習には使わない。消してしまったときの取得元:

```bash
mkdir -p tmp/gold && curl -sfL -o tmp/gold/BBox_List_2017.csv \
  https://raw.githubusercontent.com/TRKuan/cxr8/master/dataset/BBox_List_2017.csv
```

### 1.3 ディスク

特徴のキャッシュと shard で約 3 GB 使う（空きが 8 GB 未満なら止まる）。

---

## 2. embedding-service を特徴の生成用に起動する

2 段目に RAD-DINO を立て、ホストへ公開する。**この `.env` では embedding-service 以外を
起動しない。** `EMBEDDING_FINE_DIMENSION=768` は DB の列幅（15）と合わないので、
ingest-worker は起動時に止まり、`db-migrate` を流すと列幅が 768 で作られてしまう。

```bash
cp .env tmp/env.before-chest-classifier          # §7 で戻す

setenv() { if grep -q "^$1=" .env; then sed -i "s#^$1=.*#$1=$2#" .env; else echo "$1=$2" >> .env; fi; }
setenv COMPOSE_FILE docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.gpu.yml:infra/compose/docker-compose.eval.yml
setenv COMPOSE_PATH_SEPARATOR :
setenv EMBEDDING_FINE_BACKEND rad-dino
setenv EMBEDDING_FINE_DIMENSION 768
setenv EMBEDDING_FINE_MODEL_REVISION <RAD-DINO の revision（40 桁）>   # §2.1

docker compose build embedding-service
docker compose up -d --no-deps --force-recreate embedding-service
```

pooling が使えることを確かめる。

```bash
curl -s http://127.0.0.1:8002/api/v1/model | python3 -m json.tool \
  | grep -E 'region_pool|fine_embedding_model_revision'
# "fine_embedding_model_revision": "<40 桁>",
# "region_pool_backend": "rad-dino",
# "region_pool_patch_grid": 37,
```

### 2.1 RAD-DINO の版を固定する

**分類器は RAD-DINO の特徴で学習するので、RAD-DINO の重みが変われば分類器は黙って壊れる。**
特徴を作る前に revision を固定し、同じ値を書き出し（§6）と本番の
`EMBEDDING_FINE_MODEL_REVISION` に使う。`export_chest_head.py` は revision が無ければ
書き出さず、本番の embedding-service は成果物の revision と構成が違えば起動を止める。

使う revision が決まっていなければ、一度 revision を空にして起動し（モデルが
キャッシュへ取得される）、キャッシュから読む。キャッシュは Docker ボリューム
`nihcc-rag_model-cache`（コンテナ内では `/models/hf-cache`）にある。

```bash
docker run --rm --entrypoint sh -v nihcc-rag_model-cache:/m:ro \
  local/nihcc-embedding-service:1.0.0 -c '
for repo in models--microsoft--rad-dino models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224; do
  echo "== $repo"
  echo "refs/main: $(cat /m/hub/$repo/refs/main 2>/dev/null || echo なし)"
  ls -l /m/hub/$repo/snapshots/
  for f in /m/hub/$repo/snapshots/*/*.safetensors /m/hub/$repo/snapshots/*/*.bin; do
    [ -e "$f" ] && sha256sum "$(readlink -f "$f")" | sed "s#  .*#  ${f#/m/hub/}#"
  done
done' | tee tmp/chest/model-revisions.txt
```

- `refs/main` の 40 桁の値を `EMBEDDING_FINE_MODEL_REVISION` に書き、embedding-service を作り直す
- 重みファイルの SHA-256 も残るので、版の値だけでなく中身でも照合できる
- BiomedCLIP も同じく `EMBEDDING_MODEL_REVISION` に固定する。このリポジトリの分類器と
  検索の実測は `9f341de24bfb00180f1b847274256e9b65a3a32e` で取った

---

## 3. 較正（先に 1,000 枚）

```bash
mkdir -p tmp/chest
LIMIT=1000 bash scripts/run_chest_eval.sh 2>&1 | tee tmp/chest/calib.log
```

確かめること:

- 事前確認がすべて通る（`胸部野の pooling に対応` と出る）
- `分割の失敗` が数 % 以内
- 速度（`枚/秒`）。全量の時間は 112,120 ÷ この値
- `tmp/chest/sheet.png` を目で見る。**青 = 肺野、橙 = 胸部野**。橙が縦隔・心陰影・
  横隔膜ドームを含み、頸部や腹部の奥まで伸びていないこと

1,000 枚の AUC は所見ごとの陽性が少ないので、読まない。

---

## 4. 全量

```bash
nohup bash scripts/run_chest_eval.sh > tmp/chest/run.log 2>&1 &
tail -f tmp/chest/run.log
```

- **途中から再開できる。** 特徴は 2,000 枚ごとに `tmp/chest/shards/` へ書き出す。落ちたら
  原因を直して同じコマンドを投げ直せば、続きから処理する
- embedding-service が切断しても、再ロード（1〜2 分）を待って再送する。繰り返すなら
  `EMBEDDING_POOL_BATCH_SIZE` を下げる（原因は `docker inspect` の OOMKilled と `dmesg` で確かめる）
- 速度は EC2（g5.2xlarge）で 9.2〜9.6 枚/秒。律速は U-Net と前処理（CPU）

---

## 5. 結果を確かめる

`tmp/chest/train.json` に表現ごとの AUC（所見別・平均・95% 区間）と Test B が入る。
このリポジトリの分類器を作った実行の値を目安として示す。

| 項目 | 値 |
|---|---|
| 分割に成功した画像 | 112,066 / 112,120 枚（失敗 54 枚はすべて `LUNG_TOO_SMALL`） |
| 患者単位の分割 | train 77,648 / val 11,435 / test 22,983 枚 |
| 胸部野に 50% 以上入る正解 box | 99.7%（肺野だけなら 58.7%） |

test の画像単位 AUC（平均と 95% 区間）:

| `pool-lung` | **`pool-chest`** | `pool-whole` | `crop-chest` | `deterministic`（帰無） |
|---:|---:|---:|---:|---:|
| 0.829 [0.821, 0.835] | **0.833 [0.824, 0.839]** | 0.826 [0.816, 0.832] | 0.815 [0.805, 0.821] | 0.696 [0.686, 0.705] |

`pool-chest` の所見別 AUC は 0.70（Infiltration）〜0.905（Cardiomegaly）。Test B
（上位 20 件の lift）は 14 所見すべてで 1.33 以上になる。

読むときの注意:

- **帰無が高い。** 画像の統計量だけでも AUC 0.696 に届く。撮影条件（ポータブル撮影、
  AP 像など）が所見と相関しているためで、帰無との差が画像の中身から得た分である
- 帰無が本命に近づいたら、特徴の生成（U-Net、RAD-DINO の版、pooling）を疑う

---

## 6. 書き出す

`pool-chest` の分類器に**特徴の作り方**（backbone の revision、pooling の解像度、胸部野
mask の作り方、ROI profile、分割の乱数種）を添えて書き出す。embedding-service は起動時に
これと自分の構成を照合し、ずれていれば起動を止める。成果物は git に入れない。

```bash
python scripts/export_chest_head.py \
  --head tmp/chest/head_pool-chest_nihcc-unet-local-v1.npz \
  --train-json tmp/chest/train.json \
  --backbone-revision <§2.1 の RAD-DINO の revision> \
  --output infra/chest/head-pool-chest.npz
# 「test 平均 AUC 0.8328」と出ること（このリポジトリの分類器の場合。train.json の値と同じになる）
```

384px の profile で学習したときは `--roi-profile nihcc-unet-v1` を付ける。

本番の取り込み経路が同じ特徴を作るかの検査（[chest-profile-setup.md](chest-profile-setup.md) §4）は、
ここで作った `tmp/chest/vec_pool-chest_*.npz` を使う。消さずに残しておく。

---

## 7. 戻す

```bash
cp tmp/env.before-chest-classifier .env
docker compose up -d --force-recreate embedding-service
```

続けて所見プロファイルを有効にするなら、[chest-profile-setup.md](chest-profile-setup.md) へ進む。
評価用の上書き（`docker-compose.eval.yml`）を外したことを確かめる（内部サービスは
Docker ネットワーク内に閉じる方針）。
