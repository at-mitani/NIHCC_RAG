# EC2（g5.2xlarge / Ubuntu 26.04）構築手順

[English](SETUP_EC2_EN.md) | 日本語

NIHCC 胸部X線画像検索ローカルRAGシステムを AWS EC2 の GPU インスタンス上に一から構築し、
全 112,120 枚を取り込んで停止するまでの手順です。各設定の意味や設計は
[README.md](README.md) を参照してください。

> **手元の PC への構築**は [SETUP_LOCAL.md](SETUP_LOCAL.md)、**構築後の運用**は
> [docs/OPERATIONS.md](docs/OPERATIONS.md) にあります。この文書は EC2 に固有の
> 手順（GPU ドライバ・Container Toolkit・公開範囲）に絞っています。

> Ubuntu 26.04 は新しいリリースのため、NVIDIA ドライバや Docker のパッケージの版は時期によって
> 変わります。本手順では版番号を決め打ちせず、コマンドで検出してから導入します。

## 全体の流れ

| # | 作業 | 目安時間 |
|---|---|---|
| 1 | [AWS 側の準備](#1-aws-側の準備) | 15 分（クォータ申請は別途） |
| 2 | [OS の初期設定](#2-os-の初期設定) | 10 分 |
| 3 | [NVIDIA ドライバ](#3-nvidia-ドライバ) | 15 分 |
| 4 | [Docker Engine / Compose](#4-docker-engine--compose) | 10 分 |
| 5 | [NVIDIA Container Toolkit](#5-nvidia-container-toolkit) | 10 分 |
| 6 | [リポジトリ取得](#6-リポジトリ取得) | 5 分 |
| 7 | [画像データの取得と展開](#7-画像データの取得と展開)（約 42 GB） | 1〜2 時間 |
| 8 | [`.env` の設定](#8-env-の設定) | 10 分 |
| 9 | [コンテナのビルド](#9-コンテナのビルド) | 30〜40 分 |
| 10 | [起動とスキーマ適用](#10-起動とスキーマ適用) | 10 分 |
| 11 | [グローバル IP の確認](#11-グローバル-ip-の確認) | 5 分 |
| 12 | [ブラウザでアクセス](#12-ブラウザでアクセス) | 5 分 |
| 13 | [データ取り込み](#13-データ取り込み)（112,120 枚） | 4〜6 時間 |
| 14 | [コンテナとインスタンスの停止](#14-コンテナとインスタンスの停止) | 5 分 |

**手順の順番には意味があります。** 特に「カーネル更新 → 再起動 → ドライバ導入」の順を守らないと、
ドライバのモジュールと起動中のカーネルがずれて GPU が使えなくなります（[3.4](#34-失敗したときカーネルとモジュールのずれ)）。

---

## 1. AWS 側の準備

### 1.1 vCPU クォータの確認

g5 は「**Running On-Demand G and VT instances**」のクォータ対象です。新しいアカウントでは
0 になっていることがあり、そのままではインスタンスを起動できません。

EC2 コンソール → Service Quotas → Amazon Elastic Compute Cloud (Amazon EC2) →
「Running On-Demand G and VT instances」の適用値が **8 以上**あることを確認します。
足りなければ引き上げを申請してください（承認まで数時間〜数日かかることがあります）。

```bash
# AWS CLI を使う場合
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.Value'
```

### 1.2 セキュリティグループ

| タイプ | ポート | ソース | 用途 |
|---|---:|---|---|
| SSH | 22 | マイ IP | 管理用 SSH |
| カスタム TCP | 8080 | マイ IP | 画面（[12](#12-ブラウザでアクセス) で SSH ポートフォワードを使うなら不要） |

**8080 を `0.0.0.0/0` に開けないでください。** 通信は平文 HTTP です。

### 1.3 インスタンスの作成

| 項目 | 設定値 |
|---|---|
| AMI | Ubuntu Server 26.04 LTS (HVM), SSD Volume Type — 64 ビット (x86) |
| インスタンスタイプ | **g5.2xlarge**（8 vCPU / 32 GiB / NVIDIA A10G 24 GB） |
| キーペア | 新規作成または既存（`.pem` を手元に保存） |
| セキュリティグループ | 1.2 で作成したもの |
| ストレージ | **200 GiB / gp3** |
| Elastic IP（任意） | 停止・起動で IP を変えたくない場合に割り当て |

ストレージの内訳の目安:

| 用途 | 容量 |
|---|---:|
| OS・Docker イメージ（CUDA 版 PyTorch を含む） | 約 25 GB |
| 画像（展開後） | 約 42 GB |
| 画像アーカイブ（展開が終わるまでの一時領域） | 最大 約 42 GB |
| metadata-db（原画像の複製・mask） | 約 50 GB |
| ベクトル DB・WAL・ログ | 約 5 GB（1 画像 1 行） |

### 1.4 SSH 接続

```bash
chmod 400 key.pem
ssh -i key.pem ubuntu@<パブリックIP>
```

Windows の PowerShell からは `ssh -i .\key.pem ubuntu@<パブリックIP>` で接続できます。

---

## 2. OS の初期設定

**ドライバより先に、カーネルを最新にして再起動します。** 先にドライバを入れたあと apt で
カーネルが更新されると、次の再起動でドライバが読み込まれなくなります。

```bash
sudo apt-get update
sudo apt-get -y full-upgrade
sudo reboot
```

再接続したら、必要なパッケージを入れます。

```bash
sudo apt-get install -y git curl ca-certificates gnupg tmux sysstat pciutils \
  python-is-python3 ubuntu-drivers-common
sudo timedatectl set-timezone Asia/Tokyo
```

- `python-is-python3`: `scripts/bootstrap.sh` や README のコマンドは `python` を呼びます。
  Ubuntu は既定で `python3` しか無いため、これで `python` を使えるようにします。
- `tmux`: 長時間の作業（7・9・13）を SSH 切断で止めないために使います。

```bash
tmux new -s work        # セッション作成。切り離しは Ctrl+b → d
tmux attach -t work     # 再接続
```

---

## 3. NVIDIA ドライバ

### 3.1 GPU が見えているか

```bash
lspci | grep -i nvidia
```

`NVIDIA Corporation ... A10G` のような行が出れば問題ありません。何も出ない場合は
g5 インスタンスになっていません（[11](#11-グローバル-ip-の確認) の方法で `instance-type` を確認）。

### 3.2 導入

```bash
sudo ubuntu-drivers list --gpgpu
V=$(sudo ubuntu-drivers list --gpgpu 2>/dev/null | grep -oP 'nvidia-driver-\K[0-9]+(?=-server)' | sort -n | tail -1)
echo "driver version: ${V}"
```

`driver version:` が空なら、一覧に出た `nvidia-driver-XXX-server` の番号を見て `V=XXX` を手で設定します。

```bash
sudo ubuntu-drivers install --gpgpu                 # カーネルモジュールとライブラリ
sudo apt-get install -y nvidia-utils-${V}-server    # nvidia-smi（上のコマンドには含まれない）
sudo reboot
```

> `nvidia-smi` を実行したときに Ubuntu が案内する `sudo apt install nvidia-utils-XXX` だけを
> 入れても動きません（カーネルドライバが無いため `couldn't communicate with the NVIDIA driver`
> になります）。必ず `ubuntu-drivers install --gpgpu` とセットで入れてください。

### 3.3 確認

```bash
nvidia-smi              # A10G と Driver Version（525.60 以上）が表示されること
lsmod | grep nvidia     # nvidia のカーネルモジュールが読み込まれていること
```

### 3.4 失敗したとき（カーネルとモジュールのずれ）

`nvidia-smi` が `couldn't communicate with the NVIDIA driver` になる、または `lsmod` に何も出ない場合です。

```bash
V=$(dpkg -l | grep -oP 'nvidia-utils-\K[0-9]+(?=-server)' | head -1)
uname -r                                            # 起動中のカーネル
dpkg -l | grep -E 'linux-modules-nvidia|nvidia-dkms' | awk '{print $2, $3}'
sudo modprobe nvidia                                # "Module nvidia not found" ならずれている
```

上から順に試します。

```bash
# A. 起動中のカーネル用のモジュールを追加する（何も削除しない）
sudo apt-get install -y linux-modules-nvidia-${V}-server-$(uname -r)
sudo modprobe nvidia && nvidia-smi
```

```bash
# B. A で "Unable to locate package" の場合: 最新カーネルとモジュールを揃えて再起動
sudo apt-get install -y linux-aws linux-modules-nvidia-${V}-server-aws
sudo reboot
```

```bash
# C. それでも合わない場合: DKMS で起動中のカーネル向けにビルドする
sudo apt-get install -y linux-headers-$(uname -r) nvidia-dkms-${V}-server
sudo reboot
```

### 3.5 カーネルの自動更新を止める

Ubuntu は自動更新（unattended-upgrades）で新しいカーネルを入れるため、次の再起動で 3.4 の
ずれが再発することがあります。取り込みを終えるまではカーネルを固定しておきます。

```bash
sudo apt-mark hold linux-aws linux-image-aws linux-headers-aws
```

メンテナンスでカーネルを更新するときは、`sudo apt-mark unhold linux-aws linux-image-aws linux-headers-aws`
→ `full-upgrade` → 再起動 → 3.3 で確認 → [5.3](#53-cdi-の定義ファイル) の CDI 定義を作り直す、の順で行います。

---

## 4. Docker Engine / Compose

**snap 版の Docker は使いません。** snap 版は NVIDIA Container Toolkit の設定
（`/etc/docker/daemon.json`）を読まないため、GPU を使えません。

```bash
snap list docker 2>/dev/null && sudo snap remove docker
```

### 4.1 Docker 公式リポジトリ（推奨）

まず、Docker が Ubuntu 26.04 向けのリポジトリを公開しているか確認します。

```bash
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
curl -fsSL https://download.docker.com/linux/ubuntu/dists/ | grep -q "href=\"${CODENAME}/\"" \
  && echo "docker-ce: ${CODENAME} に対応" || echo "docker-ce: ${CODENAME} は未対応 → 4.2 へ"
```

対応していれば公式パッケージを入れます。

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 4.2 未対応の場合: Ubuntu のパッケージ

```bash
sudo apt-get install -y docker.io docker-compose-v2 docker-buildx
```

### 4.3 共通の設定と確認

```bash
sudo usermod -aG docker ubuntu
exit                     # グループを反映するため、一度ログアウトして SSH で再接続する
```

```bash
docker version --format 'Engine {{.Server.Version}}'
docker compose version   # v2.20 以上であること
docker run --rm hello-world
```

---

## 5. NVIDIA Container Toolkit

### 5.1 導入

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
apt-cache policy nvidia-container-toolkit    # Candidate: に版が出ること
sudo apt-get install -y nvidia-container-toolkit
nvidia-ctk --version
```

### 5.2 Docker への登録

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
cat /etc/docker/daemon.json                  # "runtimes": { "nvidia": ... } があること
```

### 5.3 CDI の定義ファイル

新しい Docker は CDI（Container Device Interface）で GPU を探します。定義ファイルが無いと
`failed to discover GPU vendor from CDI: no known GPU vendor found` になります。

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list                          # nvidia.com/gpu=0 と nvidia.com/gpu=all が出ること
```

定義ファイルにはドライバのライブラリのパスが書き込まれるため、**ドライバを更新したら作り直します**。

### 5.4 確認

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

コンテナ内から A10G の表が出れば、GPU の準備は完了です。

| エラー | 原因と対処 |
|---|---|
| `nvidia-ctk: command not found` | toolkit が未導入。5.1 をやり直す |
| `failed to discover GPU vendor from CDI` | 5.3 の CDI 定義が無い、または toolkit が未導入 |
| `failed to initialize NVML`（5.3 実行時） | ドライバが読み込まれていない。[3.3](#33-確認) / [3.4](#34-失敗したときカーネルとモジュールのずれ) に戻る |
| `could not select device driver "" with capabilities: [[gpu]]` | 5.2 の登録と Docker の再起動をやり直す |

---

## 6. リポジトリ取得

リポジトリの URL は SSH 形式（`git@github.com:at-mitani/NIHCC_RAG.git`）です。
EC2 用に鍵を作り、GitHub に**読み取り専用の Deploy key** として登録します。

```bash
ssh-keygen -t ed25519 -C "ec2-nihcc" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

表示された公開鍵を、GitHub のリポジトリ → Settings → Deploy keys → Add deploy key に登録します
（Allow write access はオフのまま）。

```bash
ssh -o StrictHostKeyChecking=accept-new -T git@github.com    # "successfully authenticated" と出ること
cd ~
git clone git@github.com:at-mitani/NIHCC_RAG.git
cd ~/NIHCC_RAG
```

以降の手順は、特に断りがない限り `~/NIHCC_RAG` で実行します。

---

## 7. 画像データの取得と展開

診断紐づけ CSV（`NIHCC/Data_Entry_2017_v2020.csv`）はリポジトリに含まれています。
画像（`NIHCC/images/`）は Git 管理外なので、NIH の配布元から取得します。

```bash
tmux attach -t work || tmux new -s work
cd ~/NIHCC_RAG/NIHCC
python batch_download_zips.py                # images_01.tar.gz 〜 images_12.tar.gz（合計 約 42 GB）
```

展開は 1 ファイルずつ行い、終わったアーカイブから削除してディスクの使用量を抑えます。
各アーカイブは `images/` ディレクトリへ展開されます。

```bash
for f in images_*.tar.gz; do echo "extract $f"; tar -xzf "$f" && rm -f "$f"; done
ls images | wc -l                            # 112120
ls -l Data_Entry_2017_v2020.csv
df -h /
```

**動作確認だけしたい場合**は、最初の 1 ファイル（4,999 枚）だけ取得します。

```bash
URL=$(grep -oP "https://nihcc\.box\.com/[^']+" batch_download_zips.py | head -1)
curl -L -o images_01.tar.gz "$URL" && tar -xzf images_01.tar.gz && rm -f images_01.tar.gz
ls images | wc -l                            # 4999
```

---

## 8. `.env` の設定

### 8.1 secret と `.env` の生成

```bash
cd ~/NIHCC_RAG
bash scripts/bootstrap.sh
ls -l secrets/                               # ディレクトリ 700 / ファイル 644
```

`secrets/*.txt` は非 root のコンテナ（postgres uid 999 / app uid 10001）から読めるよう 0644 で作られます。
0600 だと `cat: /run/secrets/app_db_password: Permission denied` で DB が起動しません。

### 8.2 GPU と公開範囲の設定

`.env` の値を置き換える関数を定義してから、設定します（新しいシェルでは再定義が必要です）。

```bash
setenv() { if grep -q "^$1=" .env; then sed -i "s#^$1=.*#$1=$2#" .env; else echo "$1=$2" >> .env; fi; }

# GPU 用の compose を重ねて読み込む
setenv COMPOSE_FILE docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.gpu.yml
setenv COMPOSE_PATH_SEPARATOR :
setenv NIHCC_DATA_DIR ./NIHCC

# 管理系のポートはホスト内に閉じる（画面の 8080 だけを外へ出す）
setenv BACKEND_API_PORT 127.0.0.1:8000
setenv RABBITMQ_UI_PORT 127.0.0.1:15672
setenv MAILPIT_UI_PORT 127.0.0.1:8025
setenv PROMETHEUS_PORT 127.0.0.1:9090
setenv GRAFANA_PORT 127.0.0.1:3000
```

#### 2 段検索と所見の推定（既定は無効）

検索は胸部野（肺野＋縦隔＋横隔膜）どうしを比べます。2 段目（胸部野の分類器による
所見プロファイルでの並べ替え）と、画面の「所見の推定」は、学習済みの分類器を置いた
ときだけ有効になります。**既定では無効**で、初回構築ではそのままで構いません。

```bash
grep -E '^(ROI_PROFILE|EMBEDDING_FINE_BACKEND|EMBEDDING_FINE_DIMENSION|CHEST_HEAD_DIR)=' .env
# ROI_PROFILE=nihcc-unet-local-v1  <- U-Net が必須（胸部野には心臓の mask が要る）
# EMBEDDING_FINE_BACKEND=none      <- 1 段検索
# EMBEDDING_FINE_DIMENSION=15      <- 列幅。migration 実行時に固定される
# CHEST_HEAD_DIR=./infra/chest     <- 2 段目を使うときだけ参照される
```

> **`EMBEDDING_FINE_DIMENSION` は [10](#10-起動とスキーマ適用) の migration より前に
> 決めてください。** `embedding_fine` の列幅は migration 実行時の値で固定され、
> あとから変えるには新しい `V00x` の追加が必要になります。既定の 15 は胸部野の
> 分類器のプロファイル（14 所見 + 正常らしさ）用です。
>
> 2 段目を有効にする手順は [docs/runbooks/chest-profile-setup.md](docs/runbooks/chest-profile-setup.md)
> を参照してください（分類器の書き出しと、RAD-DINO の版の固定が必要です）。

### 8.3 確認

```bash
grep -E '^(COMPOSE_FILE|NIHCC_DATA_DIR|EMBEDDING_(DEVICE|DTYPE|MODEL_REPLICAS|TORCH_THREADS)|TORCH_INDEX_URL|INGEST_WORKER_REPLICAS|INGEST_STORE_REGION_CROPS|PREPROCESS_VERSION)=' .env
docker compose config | grep -A5 'reservations:'        # driver: nvidia が出ること
```

g5.2xlarge では次の値になっていれば正しい設定です（`.env.example` の既定値）。

| 変数 | 値 |
|---|---|
| `EMBEDDING_DEVICE` | `cuda` |
| `EMBEDDING_DTYPE` | `float16` |
| `EMBEDDING_MODEL_REPLICAS` | `1` |
| `EMBEDDING_TORCH_THREADS` | `4` |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu121` |
| `INGEST_WORKER_REPLICAS` | `6` |
| `INGEST_STORE_REGION_CROPS` | `false` |
| `PREPROCESS_VERSION` | `prep-v1.1.0` |

画面にログインする初期ユーザーのパスワードを控えておきます（`admin:<パスワード>:admin` の形式）。

```bash
grep '^LOCAL_USERS=' .env
```

---

## 9. コンテナのビルド

DB イメージは pgvector をソースからビルドし、embedding-service は CUDA 版 PyTorch を取得するため時間がかかります。

```bash
tmux attach -t work || tmux new -s work
cd ~/NIHCC_RAG
docker compose build
docker compose build db-migrate      # profile 付きのため、引数なしの build ではビルドされない
docker images | grep -E 'nihcc|citus'
df -h /
```

---

## 10. 起動とスキーマ適用

`.env` が **`HF_HUB_OFFLINE=1`**（モデルを取得しないオフライン運用）の場合は、**起動の前に**モデルを
キャッシュへ取得します。初回の構築時と `docker compose down -v` の後はキャッシュが空のため、これを省くと
embedding-service が起動に失敗して再起動を繰り返し、ingest-worker と frontend が `Created` のまま起動しません。
既定の `HF_HUB_OFFLINE=0` なら不要です（embedding-service が初回起動時に自動で取得します）。

```bash
cd ~/NIHCC_RAG
grep '^HF_HUB_OFFLINE=' .env
bash scripts/fetch_model.sh      # HF_HUB_OFFLINE=1 の場合のみ（実行中だけオンラインにして取得する）
```

```bash
cd ~/NIHCC_RAG
docker compose up -d
```

全コンテナが healthy になるまで待ちます。次のコマンドで**何も表示されなくなれば**起動完了です
（embedding-service は初回にモデルを取得するため数分かかります）。`-a` を付けないと、
`Created` のまま止まったコンテナが一覧に出ないため見落とします。

```bash
docker compose ps -a | grep -E 'unhealthy|starting|Restarting|Exited|Created'
```

起動したらスキーマを適用し、Citus の Worker を登録します。

```bash
# embedding_fine の列幅は実行時の値で固定される。先に確認する
grep '^EMBEDDING_FINE_DIMENSION=' .env      # 既定 15

docker compose run --rm db-migrate
```

metadata へ `V001`〜`V012`、ベクトル用クラスタ（`vector-coordinator` + `vector-worker-1..5`）へ
`V001` が適用され、Worker 5 台に shard が 6 個ずつ載ります。

取り込み済みの EC2 を更新する手順は [docs/OPERATIONS.md](docs/OPERATIONS.md) §2 にあります。

> `git pull` でマイグレーション（`db/metadata-migrations/` など）が増えたときは、**先に
> `docker compose build db-migrate` を実行してから** `run` します。SQL はイメージへコピーされる作りのため、
> 古いイメージのままでは新しいマイグレーションが適用されず、取り込みが全件失敗します。

**GPU が使われているかを必ず確認します。** GPU が見えないと、embedding-service は警告を出して
CPU で動き続けるため、取り込みが極端に遅くなります。

```bash
docker compose exec embedding-service python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`2.5.1+cu121 True NVIDIA A10G` と出れば正常です。

| 結果 | 対処 |
|---|---|
| `2.5.1+cpu` | `TORCH_INDEX_URL` を確認し、`docker compose build embedding-service` をやり直す |
| `2.5.1+cu121 False` | `COMPOSE_FILE` に gpu.yml が無い、または [5](#5-nvidia-container-toolkit) が未完了。直したら `docker compose up -d --force-recreate embedding-service` |
| unhealthy のまま進まない | `docker compose logs --tail=80 <サービス名>` を確認（[docs/OPERATIONS.md](docs/OPERATIONS.md) §3） |

---

## 11. グローバル IP の確認

```bash
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
PUBLIC_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type ; echo
echo "${PUBLIC_IP}"
```

パスワード再設定メールの URL に使われる `APP_BASE_URL` を、実際のアドレスに合わせます。

```bash
setenv() { if grep -q "^$1=" .env; then sed -i "s#^$1=.*#$1=$2#" .env; else echo "$1=$2" >> .env; fi; }
setenv APP_BASE_URL "http://${PUBLIC_IP}:8080"
docker compose up -d backend-api
```

インスタンスを停止・起動するとパブリック IP は変わります（Elastic IP を割り当てた場合を除く）。
そのたびにこの手順をやり直してください。

---

## 12. ブラウザでアクセス

### 方法 A: SSH ポートフォワード（推奨）

セキュリティグループで 8080 を開けずに済み、通信も SSH で暗号化されます。手元の PC で実行します。

```bash
ssh -i key.pem -L 8080:localhost:8080 -L 8025:localhost:8025 ubuntu@<パブリックIP>
```

手元のブラウザで <http://localhost:8080> を開きます（8025 は Mailpit の受信箱）。

### 方法 B: グローバル IP で直接開く

セキュリティグループで 8080 をマイ IP に限定して開けてから、ブラウザで開きます。

```
http://<パブリックIP>:8080          画面
http://<パブリックIP>:8080/docs     OpenAPI
```

> **HTTP は平文です。** ログイン ID・パスワード・アクセストークンが暗号化されずに流れます。
> 方法 B を使う場合もソースは必ず自分の IP に限定し、継続して公開するなら ALB + ACM 証明書を前段に置いてください。

### ログイン

8.3 で控えた `admin` とパスワードでログインします。**ログイン後、メニューからパスワードを変更してください。**

---

## 13. データ取り込み

### 13.1 動作確認（50 枚）

```bash
tmux attach -t work || tmux new -s work
cd ~/NIHCC_RAG
python scripts/load_initial_images.py --limit 50 --yes
```

dry-run 検証 → ジョブ登録 → 完了待ち → 結果サマリの順に表示されます。失敗 0 件で終わることを確認します。

### 13.2 全件の取り込み

```bash
python scripts/load_initial_images.py --yes
```

tmux から切り離しても（Ctrl+b → d）取り込みは続きます。`Ctrl+C` で止めてもサーバー側のジョブは継続します。

画面から取り込む場合は、「データ取り込み」タブで **dry-run 検証** → **取り込み開始** を押します。
プログレスバーに件数・経過時間・残り時間が表示されます。

### 13.3 進み具合と性能の確認

```bash
nvidia-smi -l 5                               # GPU 使用率と python プロセス
docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' | sort -k2 -hr | head -12
iostat -xm 5                                  # ディスクの飽和（%util）
```

1 分あたりの処理枚数を DB から確認します。3 時間で終えるには **約 10.4 枚/秒** が必要です。

```bash
PSQL() { docker compose exec -T -e PGPASSWORD="$(cat secrets/db_password.txt)" metadata-db \
  psql -U postgres -d metadata -h 127.0.0.1 -X -c "$1"; }

PSQL "SELECT status, succeeded_count + failed_count AS done, started_at FROM ingest_job ORDER BY started_at DESC NULLS LAST LIMIT 1;"
sleep 60
PSQL "SELECT status, succeeded_count + failed_count AS done, started_at FROM ingest_job ORDER BY started_at DESC NULLS LAST LIMIT 1;"
```

速度が出ない場合の切り分けは、GPU の確認（[10](#10-起動とスキーマ適用)）→ ROI（CPU）の飽和 → DB の待ち、の順に行います。

---

## 14. コンテナとインスタンスの停止

### 14.1 コンテナの停止

```bash
cd ~/NIHCC_RAG
docker compose down          # 停止（取り込んだデータは Docker ボリュームに残る）
```

**すべてのデータを消す場合のみ**次を使います（取り込み済みの画像・ベクトル・ユーザーに加えて、
**モデルのキャッシュも削除されます**）。

```bash
docker compose down -v
```

`down -v` の後に再び起動するときは、`HF_HUB_OFFLINE=1` なら**先に** `bash scripts/fetch_model.sh` を
実行してから `docker compose up -d` します（[10](#10-起動とスキーマ適用)）。

ソースを修正してデータを取り直すだけなら、`down -v` ではなく次のコマンドでデータだけを削除します。
モデルのキャッシュが残るため、モデルの取り直しは不要です。

```bash
python scripts/reset_data.py --all
```

### 14.2 インスタンスの停止

GPU インスタンスは起動しているだけで課金されます。使わないときは停止してください。

```bash
sudo shutdown -h now         # EBS ボリュームのインスタンスは「停止」になる（削除ではない）
```

EC2 コンソールの「インスタンスの状態 → インスタンスを停止」でも同じです。停止中も EBS（200 GiB）の料金はかかります。

### 14.3 再開するとき

1. EC2 コンソールでインスタンスを開始し、SSH で接続する
2. `nvidia-smi` で GPU を確認する（カーネルを固定していれば通常は問題ない）
3. コンテナを起動する

```bash
cd ~/NIHCC_RAG
docker compose up -d
docker compose ps -a | grep -E 'unhealthy|starting|Restarting|Exited|Created'
```

4. パブリック IP が変わっていれば [11](#11-グローバル-ip-の確認) をやり直す

`docker compose down` をせずにインスタンスを停止した場合、コンテナは `restart: unless-stopped` の設定により
Docker の起動と同時に自動で立ち上がります。

---

## トラブルシュート早見表

| 症状 | 原因 | 参照 |
|---|---|---|
| `Command 'nvidia-smi' not found` | ドライバ未導入、または `nvidia-utils-XXX-server` 未導入 | [3.2](#32-導入) |
| `couldn't communicate with the NVIDIA driver` | カーネルとドライバのモジュールがずれている | [3.4](#34-失敗したときカーネルとモジュールのずれ) |
| `nvidia-ctk: command not found` | Container Toolkit 未導入 | [5.1](#51-導入) |
| `failed to discover GPU vendor from CDI` | CDI 定義が無い | [5.3](#53-cdi-の定義ファイル) |
| `torch.cuda.is_available()` が `False` | gpu.yml 未読み込み、または toolkit 未完了 | [10](#10-起動とスキーマ適用) |
| `cat: /run/secrets/...: Permission denied` | secret が 0600 | [8.1](#81-secret-と-env-の生成) |
| `dependency failed to start: ... is unhealthy` | secret の権限・メモリ・ディスクなど | [OPERATIONS.md](docs/OPERATIONS.md) §3 |
| 取り込みが全件失敗する / 取消で `invalid input value for enum` | `db-migrate` イメージが古く、新しいマイグレーションが未適用 | [10](#10-起動とスキーマ適用)（`docker compose build db-migrate` → `run`） |
| embedding-service が再起動を繰り返す / ingest-worker・frontend が `Created` のまま | `HF_HUB_OFFLINE=1` なのにモデルのキャッシュが空（`down -v` の後など） | [10](#10-起動とスキーマ適用)（`bash scripts/fetch_model.sh` → `docker compose up -d`） |
| 取り込み開始で `crypto.randomUUID is not a function` | 古い画面のキャッシュ | ブラウザでハードリロード（Ctrl+Shift+R） |
| 取り込みが遅い | GPU 未使用 → ROI（CPU）→ DB の順に確認 | [13.3](#133-進み具合と性能の確認) |
| `python: command not found` | `python-is-python3` 未導入 | [2](#2-os-の初期設定) |
| `Permission denied while trying to connect to the Docker daemon` | docker グループが未反映 | [4.3](#43-共通の設定と確認)（再ログイン） |
