# 所見プロファイルを有効にした構築と全量取り込み（EC2）

胸部野の分類器を入れて `EMBEDDING_FINE_BACKEND=chest-profile` を有効にし、全 112,120 枚を
取り込む手順。有効にすると次のようになる（[MULTIMODAL.md](../MULTIMODAL.md) §4.1）。

| 検索 | 進め方 |
|---|---|
| 画像だけ | 所見プロファイル（15 次元）で全件を厳密に検索（`IMAGE_SEARCH_STRATEGY=profile`、既定） |
| 文章・画像＋文章 | BiomedCLIP（胸部野 crop、512 次元）で候補を集め、所見プロファイルで並べ替え |

画面には「所見の推定（研究用・診断ではありません）」が出る。確率は取り込み時に計算して保存する。

- 所要: 1 日（全量の取り込みが約 4 時間）
- 前提: [SETUP_EC2.md](../../SETUP_EC2.md) §1〜8 が済んでいる（GPU・Docker・リポジトリ・画像・`.env`）

---

## 0. 用意するもの

| もの | 場所 | 作り方 |
|---|---|---|
| U-Net のモデル | `infra/lung-unet/lung-unet.onnx` | [lung-unet.md](lung-unet.md) |
| 胸部野の分類器 | `infra/chest/head-pool-chest.npz` | [chest-classifier.md](chest-classifier.md) |
| 分類器を学習したときの RAD-DINO の revision | 控え（40 桁） | [chest-classifier.md](chest-classifier.md) §2.1 |
| 特徴のキャッシュ | `tmp/chest/vec_pool-chest_nihcc-unet-local-v1_112120.npz` | 同上 §4（§4 の検査に使う） |

```bash
cd ~/NIHCC_RAG && source ~/.venv-eval/bin/activate
df -h /                                         # 空きは 120 GB 以上を目安
ls NIHCC/images | wc -l                         # 112120
nvidia-smi                                      # A10G が見えること
ls infra/lung-unet/lung-unet.onnx infra/chest/head-pool-chest.npz
```

ディスクの内訳（見積り）:

| 項目 | 容量 |
|---|---|
| 画像 | 42 GB |
| Docker イメージ | 約 25 GB |
| metadata-db（原画像の複製と mask） | 約 50 GB |
| ベクトル DB | 約 3 GB |

> **既に `EMBEDDING_FINE_BACKEND=none` で取り込んだデータがある場合**、その行には
> プロファイルも所見の推定も無い。`python scripts/reset_data.py` で消してから §6 で
> 取り込み直す（利用者は残る）。

---

## 1. `.env`

```bash
python3 scripts/sync_env.py --write             # .env.example にあって .env に無いキーを足す
setenv() { if grep -q "^$1=" .env; then sed -i "s#^$1=.*#$1=$2#" .env; else echo "$1=$2" >> .env; fi; }
RADDINO_REV=<分類器を学習したときの RAD-DINO の revision（40 桁）>

# §4 の検査のあいだだけ評価用の上書き（roi-service と embedding-service をホストへ公開）を足す
setenv COMPOSE_FILE docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.gpu.yml:infra/compose/docker-compose.eval.yml
setenv COMPOSE_PATH_SEPARATOR :
setenv ROI_SERVICE_REPLICAS 1                   # 検査中はホストへ公開するので 1 台

setenv ROI_PROFILE nihcc-unet-local-v1          # 分類器を学習した profile
setenv EMBEDDING_FINE_BACKEND chest-profile
setenv EMBEDDING_FINE_DIMENSION 15
setenv EMBEDDING_FINE_MODEL_REVISION "$RADDINO_REV"
setenv CHEST_HEAD_ALLOW_UNPINNED false
setenv EMBEDDING_MODEL_REVISION 9f341de24bfb00180f1b847274256e9b65a3a32e   # BiomedCLIP
setenv CITUS_SHARD_COUNT 30
grep -E '^(COMPOSE_FILE|ROI_PROFILE|EMBEDDING_FINE_|EMBEDDING_MODEL_REVISION|CHEST_HEAD|CITUS_SHARD_COUNT|PG_MAJOR)=' .env
```

> **`EMBEDDING_FINE_DIMENSION` と `CITUS_SHARD_COUNT` は migration の前に決める。**
> 列幅とシャード数は migration を実行した時点で固定される。

---

## 2. 作って起動する

```bash
tmux attach -t work || tmux new -s work
docker compose build
docker compose build db-migrate                 # profile 付きのため、引数なしの build には入らない
bash scripts/fetch_model.sh                     # HF_HUB_OFFLINE=1 のときだけ。revision を固定した版を取得する
docker compose up -d
docker compose ps -a | grep -E 'unhealthy|starting|Restarting|Exited|Created'   # 何も出なくなるまで待つ
docker compose run --rm db-migrate
```

embedding-service は起動時に分類器と自分の構成（backbone の revision、pooling の解像度、
mask の設定）を照合し、ずれていれば起動を止める。ingest-worker は胸部野の特徴の作り方
（roi-service の profile と mask の設定）を分類器の記録と照合する。

Citus の構成を確かめる。

```bash
docker compose exec vector-coordinator psql -U postgres -d vector -c \
  "SELECT nodename, isactive FROM pg_dist_node ORDER BY nodename;"
# vector-coordinator と vector-worker-1..5 の 6 行。worker はすべて isactive = t

docker compose exec vector-coordinator psql -U postgres -d vector -c \
  "SELECT nodename, count(*) FROM citus_shards GROUP BY nodename ORDER BY nodename;"
# 各ワーカーに 6 シャードずつ（計 30）
```

---

## 3. 疎通確認

```bash
python scripts/load_initial_users.py
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
# 全項目 PASS（WARN は応答時間のみ許容）
```

---

## 4. 特徴の一致を確かめる

**全量を取り込む前に**、本番の roi-service と embedding-service が分類器の学習と同じ特徴を
作るかを確かめる。分類器は特徴の作り方が 1 画素でも違うと、例外を出さずに確率だけがずれる。
画像ファイルを直接使うので、取り込みは要らない。

```bash
LUNG_MODEL_PATH=infra/lung-unet/lung-unet.onnx \
  python scripts/chest_image_eval.py verify-ingest --limit 200
# モデル入力の不一致: 0 / 200 枚
# 確率の最大の差: 0.0xxx（許容 0.02）
# プロファイルの cosine の最小値: 0.999x（許容 0.999）
# 判定: 一致。全量の取り込みに進んでよい
```

| 検査 | 確かめていること |
|---|---|
| モデル入力の不一致 | roi-service `/roi/chest` の入力が、評価と同じ関数で作る入力と 1 バイトも違わない（U-Net のモデル・profile・前処理） |
| 確率の差・cosine | embedding-service `/embeddings/chest` の確率が、キャッシュの特徴に分類器を当てた確率と一致する（RAD-DINO の revision・pooling・分類器） |

**不一致なら全量を取り込まない**（§8）。

---

## 5. 少量で取り込み、上書きを外す

```bash
python scripts/load_initial_images.py --limit 1000 --yes
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```

通ったら、評価用の上書きを外してポートを閉じる（内部サービスは Docker ネットワーク内に
閉じる方針）。roi-service を 2 台以上にするなら、ここで `ROI_SERVICE_REPLICAS` を上げる。

```bash
setenv COMPOSE_FILE docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.gpu.yml
docker compose up -d --remove-orphans
```

---

## 6. 全量を取り込む

```bash
nohup python scripts/load_initial_images.py --yes > tmp/load.log 2>&1 &   # --yes が無いと確認待ちで止まる
tail -f tmp/load.log
```

- 約 4 時間（g5.2xlarge での実測 231 分）
- 途中で落ちても、同じコマンドを再実行すれば取り込み済みの画像は飛ばす（`on_duplicate=SKIP`）

完了の確認:

```bash
docker compose exec vector-coordinator psql -U postgres -d vector -c "SELECT count(*) FROM image_embedding;"
# 112,066 前後（112,120 から、分割に失敗した 54 枚前後を引いた値）

# reconcile.py は backend-api のイメージの中で実行する（ホストには nihcc_rag が無い）
docker compose run --rm --no-deps --entrypoint python backend-api /srv/scripts/reconcile.py
```

reconcile の `issue_count` のうち、`MISSING_VECTOR` は分割に失敗した画像
（`LUNG_TOO_SMALL`、50 件前後）で想定どおり。`ORPHAN_VECTOR` / `MISSING_FINE_VECTOR` /
`MISSING_FINDING_ESTIMATE` は想定外である。

検索品質を測るなら [chest-evaluation.md](chest-evaluation.md) §1。

---

## 7. 画面で確かめる

`http://<ec2-host>:8080`（または SSH ポートフォワード）を開き、次を確かめる。

- 画像での検索で、1 本の結果一覧と「胸郭の形が似た症例」が出る
- 結果の上に「所見プロファイルで全件検索」の札が出る
- カードに胸部野の輪郭が重なり、「所見の推定（研究用・診断ではありません）」が出る
- 文章での検索（例: 「胸水」）で結果が返る
- 管理画面のクラスタ表が 1 コーディネータ・5 ワーカー

---

## 8. うまくいかなかったとき

| 症状 | 見るところ |
|---|---|
| embedding-service が `胸部野の分類器がありません` で止まる | `infra/chest/head-pool-chest.npz` と `CHEST_HEAD_DIR` / `CHEST_HEAD_PATH` |
| embedding-service が `特徴の作り方と、いまの構成が違います` で止まる | 表示された項目（backbone の版、pooling の解像度、mask の設定）。`EMBEDDING_FINE_MODEL_REVISION` を分類器と揃えるか、分類器を書き出し直す |
| embedding-service が `EMBEDDING_FINE_DIMENSION=... が分類器の出力 15 次元と一致しない` で止まる | `EMBEDDING_FINE_DIMENSION=15` |
| ingest-worker が `vector 列の幅が設定と違います` で止まる | [OPERATIONS.md](../OPERATIONS.md) §3.1 |
| ingest-worker が特徴の作り方の違いで止まる | roi-service の profile・mask の設定と分類器の記録がずれている。`ROI_PROFILE` を分類器を学習した profile に合わせる |
| db-migrate が Worker の登録で止まる | `COMPOSE_FILE` に `docker-compose.vectors.yml` があり、`vector-worker-{1..5}` がすべて起動しているか |
| 特徴一致の検査が「モデル入力の不一致」で落ちる | roi-service の image が古いか、U-Net のモデルが違う。`bash scripts/redeploy.sh` で全イメージを作り直し、`infra/lung-unet/` を確かめる。**全量を取り込まない** |
| 特徴一致の検査が「確率の差」で落ちる | RAD-DINO の版が分類器と違う（`EMBEDDING_FINE_MODEL_REVISION`）か、分類器が別の実行のもの |
| 取り込み中に embedding-service が切断する | `EMBEDDING_POOL_BATCH_SIZE` を下げる。原因は `docker inspect`（OOMKilled）と `dmesg` で確かめる |
| 検索が遅い | [OPERATIONS.md](../OPERATIONS.md) §3.4 |
