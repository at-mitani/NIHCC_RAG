# 手元の PC への環境構築手順

[English](SETUP_LOCAL_EN.md) | 日本語

Windows（Docker Desktop）または Linux の作業機に一式を構築する。

GPU 付き EC2 への構築は [SETUP_EC2.md](SETUP_EC2.md)、日々の運用は
[docs/OPERATIONS.md](docs/OPERATIONS.md) を参照。

---

## 1. 必要なもの

| | 最低 | 推奨 |
|---|---|---|
| メモリ | 16 GB（Docker へ 8 GB 以上） | 32 GB |
| ディスク | 60 GB | 200 GB |
| CPU | 4 コア | 8 コア |

**GPU は要らない。** すべて CPU で動く。ただし取り込みは 5〜10 倍遅い。

> **手元の Intel 内蔵 GPU は使えない。** CUDA は NVIDIA 専用で、Docker Desktop の
> WSL2 からは `/dev/dri` が見えない。モデルの学習が必要になったら EC2 で行い、
> 成果物だけを持ち込む。

### 立ち上がるコンテナの数

**約 16 個**ある。うち 6 個がベクトル DB（Coordinator 1 台 + Worker 5 台）である。
Docker Desktop のメモリ割り当ては 8 GB 以上を推奨する。

---

## 2. 前提ソフト

### Windows

1. **WSL2** を有効にする（`wsl --install`）
2. **Docker Desktop** を入れ、WSL2 バックエンドを使う
3. 設定 → Resources でメモリを **8 GB 以上**、ディスクを 60 GB 以上にする
4. **Git for Windows**（Git Bash が付く）

> このリポジトリの手順は bash 前提のものが多い。**Git Bash から実行する**のが確実。
> PowerShell で動かす場合は、`sha256sum` → `Get-FileHash`、`cp` → `Copy-Item` の
> ように読み替えが要る。

### Linux

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git python3-venv
sudo usermod -aG docker $USER      # 入り直しが要る
```

---

## 3. データセットを置く

NIH ChestX-ray14 を入手し、次の形に置く。

```
NIHCC/
  images/                       *.png
  Data_Entry_2017_v2020.csv
```

**全 112,120 枚でなくてよい。** 数千枚でも動く。ディスクと時間に合わせて絞る。

置き場所は `.env` の `NIHCC_DATA_DIR` で変えられる（既定は `./NIHCC`）。

---

## 4. 構築

```bash
git clone git@github.com:at-mitani/NIHCC_RAG.git
cd NIHCC_RAG
```

### 4.1 secret と `.env` を作る

```bash
bash scripts/bootstrap.sh
```

**初期ユーザーのパスワードが表示される。控えること。** 再表示はできない。

`secrets/` に DB や JWT の鍵が作られる。**git に入らない**（`.gitignore` 済）。

### 4.2 イメージを作る

```bash
docker compose build db-migrate     # 先に。引数なしの build に入らない
docker compose build
```

DB イメージは pgvector をソースからビルドするので、初回は 15〜30 分かかる。

> **`db-migrate` を別に指定するのを忘れないこと。** `migrate` プロファイルに属する
> ので、引数なしの `build` では作られない。migration ファイルはイメージに焼き込まれる
> ため、作り忘れると新しいスキーマが入らない。

### 4.3 起動

```bash
docker compose up -d
docker compose ps        # 全部が healthy になるまで数分
```

初回は BiomedCLIP をインターネットから取得する。閉域で動かす場合は先に
`bash scripts/fetch_model.sh` で取得しておく。

### 4.4 スキーマを入れる

**全コンテナが healthy になってから**実行する。

```bash
docker compose run --rm db-migrate
```

metadata とベクトル用クラスタにスキーマを作り、Citus の Worker 5 台を登録する。

> `embedding_fine` の列の幅は、このとき `.env` の `EMBEDDING_FINE_DIMENSION` で
> **固定される**。後から `.env` だけ変えても列は変わらず、取り込みが全件失敗する。
> 実行前に値を確認すること（既定 15）。

### 4.5 利用者を作る

```bash
python scripts/load_initial_users.py
```

`bootstrap.sh` が作った初期ユーザーを登録する。

### 4.6 確認

```bash
python tests/integration/smoke_test.py
```

ブラウザで `http://localhost:8080` を開く。

---

## 5. 画像を取り込む

画面の「データ取り込み」タブから行うのが分かりやすい。まず **dry-run** で CSV と
画像の突合を確認し、問題がなければ実行する。

スクリプトでもよい。

```bash
python scripts/load_initial_images.py
```

CPU だと 1 枚あたり 1〜2 秒かかる。まず 100 枚ほどで試すこと。

---

## 6. 開発環境（任意）

スクリプトを直接動かすなら Python の仮想環境が要る。

```bash
python -m venv .venv-dev
.venv-dev/Scripts/activate      # Windows
source .venv-dev/bin/activate   # Linux

pip install -e packages/common -e packages/image-pipeline
pip install pytest httpx
```

> **素の `python` では動かない。** `cv2` も `numpy` も入っていないので
> `ModuleNotFoundError` になる。スクリプトを動かすときは仮想環境を有効にすること。

肺野分割の評価や学習を手元で行う場合は、さらに次が要る。

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install onnxruntime onnx onnxscript
```

---

## 7. 停止と片付け

```bash
docker compose stop            # 止める（データは残る）
docker compose down            # コンテナを消す（データは残る）
docker compose down -v         # ボリュームごと消す（**データが消える**）
```

---

## 8. つまずきやすいところ

**起動しきらない / コンテナが次々落ちる**
: Docker のメモリ割り当てを見る。8 GB 未満だと約 16 個のコンテナが上がりきらない。

**`ModuleNotFoundError: No module named 'cv2'`**
: 素の `python` を使っている。仮想環境を有効にする（§6）。

**`port is already allocated`**
: 8080 / 8000 / 8025 / 15672 / 9090 / 3000 のどれかが使われている。`.env` で
  `FRONTEND_PORT` などを変える。

**migration が入らない**
: `docker compose build db-migrate` を実行したか。migration ファイルはイメージに
  焼き込まれる。

**取り込みが全件失敗する**
: `.env` の次元と DB の列の幅がずれている。[OPERATIONS.md](docs/OPERATIONS.md) §3.1。

**画面が古い動きをする**
: `docker compose build frontend` を実行したか。静的ファイルもイメージに焼き込まれる。

**`invalid option name: pipefail`**
: `sh` で bash スクリプトを実行している。`bash scripts/....sh` と明示する。
