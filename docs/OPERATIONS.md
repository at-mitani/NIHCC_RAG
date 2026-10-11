# 運用マニュアル
[English](OPERATIONS_EN.md) | 日本語

日々の運用、更新の反映、障害対応。**運用者向け**である。

構築は [SETUP_LOCAL.md](../SETUP_LOCAL.md) / [SETUP_EC2.md](../SETUP_EC2.md)、
画面の使い方は [README.md](../README.md)、設計は [DESIGN.md](DESIGN.md) を参照。

---

## 1. 日常の操作

### 1.1 起動・停止

```bash
docker compose up -d        # 起動
docker compose stop         # 停止（データは残る）
docker compose ps           # 状態
docker compose logs -f backend-api
```

### 1.2 健全性の確認

```bash
docker compose ps           # 全部が running / healthy か
```

サービスによって確認の仕方が違う。

```bash
# backend-api（外から届く）
curl -s localhost:8000/api/v1/health/ready

# roi-service・embedding-service（ポートを公開していない）
docker compose exec roi-service python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/ready').read().decode())"
```

> **roi-service と embedding-service は外から叩けない。** `app` ネットワークにしか
> 居ないので、`localhost:8000` は backend-api を指す。

### 1.3 全体のスモーク

```bash
# admin のパスワードは .env の LOCAL_USERS にある（user:password:role のカンマ区切り）
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```

> **パスワードが要る。** 無いと `--password または SMOKE_PASSWORD が必要です` と
> 出て終了コード 2 で止まる。`--user` / `--password` で直接渡してもよい。
> 既定の利用者は `admin`、接続先は `http://localhost:8000`（`BASE_URL` で変えられる）。

取り込みと検索を実際に通す。40 項目余り（設定によって増減する）。
**更新のたびに実行すること。**

実際に 30 枚を取り込むので、**本番のデータに行を足す**。試すだけなら `--limit` を
小さくする。

---

## 2. 更新の反映

```bash
git pull --ff-only origin main
python scripts/sync_env.py          # 差分を確認
python scripts/sync_env.py --write  # .env に取り込む
bash scripts/redeploy.sh --yes
python tests/integration/smoke_test.py
```

`redeploy.sh --pull` は `git pull` から始める（`origin main` を取り込む）。

`redeploy.sh` が行うこと。

1. 実行中の評価が無いか確認（あれば止める）
2. 取り込みを停止
3. **`db-migrate` を含む全イメージ**を再構築
4. migration を適用
5. 起動して health を待つ
6. 古いイメージで動いているコンテナを検出して入れ替える

> ## サービスを選んで build しない
>
> **migration ファイルも画面の静的ファイルもイメージに焼き込まれる。**
>
> - `docker compose build`（引数なし）は **`migrate` プロファイルを含まない**
> - `up --force-recreate` は**イメージを作り直さない**
>
> 選んで build すると、新しい migration が適用されないまま「適用済み」だけを表示して
> 終わったり、古い `app.js` が配信され続けて画面が落ちたりする。

### 反映されたか確かめる

```bash
# migration が実際に入ったか
docker compose exec -T metadata-db sh -c \
  'PGPASSWORD=$(cat /run/secrets/db_password) psql -U postgres -d metadata -tAc \
   "SELECT version FROM schema_migration ORDER BY version DESC LIMIT 3"'

# 配信中の画面が新しいか（例）
curl -s localhost:8080/app.js | grep -c findingLabel
```

---

## 3. 障害対応

### 3.1 ingest-worker が `vector 列の幅が設定と違います` で起動しない

**`.env` の次元と DB の列の幅がずれている。** 列（`image_embedding` の `embedding` /
`embedding_fine` / `embedding_lung`）は migration 実行時の値で固定されるので、後から
`.env` だけ変えると起きる。ingest-worker は起動時に列の実際の幅と `.env` を突き合わせ、
ずれていれば取り込みを始めずに止まる。

```bash
grep -E "^EMBEDDING_(DIMENSION|FINE_BACKEND|FINE_DIMENSION|LUNG_DIMENSION)=" .env

docker compose exec -T vector-coordinator sh -c \
  'PGPASSWORD=$(cat /run/secrets/db_password) psql -U postgres -d vector -tAc \
   "SELECT attname, format_type(atttypid, atttypmod) FROM pg_attribute
     WHERE attrelid='"'"'image_embedding'"'"'::regclass AND attname LIKE '"'"'embedding%'"'"'"'
```

対処は 2 通り。

| | 設定 | 結果 |
|---|---|---|
| **推奨** | `.env` を列に合わせる（`chest-profile` / `none` なら `EMBEDDING_FINE_DIMENSION=15`） | そのまま復旧 |
| 非推奨 | 列の幅を変える | `db/vector-migrations` に新しい `V00x` が要り、既存のベクトルを捨てる |

```bash
docker compose up -d --force-recreate embedding-service ingest-worker
```

### 3.2 1 枚ごとの失敗を調べる

失敗の理由は 2 か所に残る。

```bash
# ベクトル書き込みの失敗
docker compose exec -T metadata-db sh -c \
  'PGPASSWORD=$(cat /run/secrets/db_password) psql -U postgres -d metadata -tAc \
   "SELECT image_id, left(last_error, 200) FROM image_vector_state
     WHERE state = '"'"'FAILED'"'"' ORDER BY updated_at DESC LIMIT 5"'

# 取り込み全体
docker compose logs --tail 400 ingest-worker | grep -i "item failed"
```

画面の「取り込み履歴」からも理由ごとに見られ、CSV で落とせる。

### 3.3 embedding-service が起動しない

```bash
docker compose logs --tail 40 embedding-service
```

| メッセージ | 原因 |
|---|---|
| `胸部野の分類器がありません` | `chest-profile` を選んだが `infra/chest/head-pool-chest.npz` が無い |
| `胸部野の分類器が前提とする特徴の作り方と、いまの構成が違います` | RAD-DINO の revision・pooling の解像度・mask の設定が分類器の記録と違う（[runbooks/chest-profile-setup.md](runbooks/chest-profile-setup.md) §8） |
| `vector shape (1, 768) != (1, 15)` | backend と dimension が揃っていない |
| `produces the same model_version as the coarse backend` | 1 段目と同じモデルを指している |

いずれも `EMBEDDING_FINE_BACKEND=none` で 1 段検索として起動できる（所見の推定は出なくなる）。

> 起動に 3 分ほどかかる（モデルの読み込み）。直後の `unhealthy` は待ち不足のことも
> ある。ログに例外が無ければ待つ。

### 3.4 検索が 503 / 504

| 状態 | 意味 | 対処 |
|---|---|---|
| 503 | 依存サービスが応答しない | `docker compose ps` で health を見る |
| 504 | 時間切れ | CPU の取り合い。取り込みや学習と重なっていないか |

画像検索は既定で所見プロファイルの全件検索（`IMAGE_SEARCH_STRATEGY=profile`）。EC2 の
実測でベクトル検索の p95 は 24 ms で、2 段検索（290 ms）より速い。全件検索だけが遅いときは
`IMAGE_SEARCH_STRATEGY=coarse_to_fine`（BiomedCLIP で候補を集めてから並べ替える）にして
`docker compose up -d`（build 不要）で切り分ける。値は `profile` か `coarse_to_fine` のどちらかで、
綴りを誤ると検索ごとにエラーになる（黙って既定へ倒さない）。

2 段検索が遅いときは `HNSW_MAX_SCAN_TUPLES` と `HNSW_EF_SEARCH` を見直す。

### 3.5 画面が古い動きをする

```
Cannot read properties of undefined (reading 'length')
```

**frontend のイメージを作り直していない。** 静的ファイルはイメージに焼き込まれる。

```bash
bash scripts/redeploy.sh --yes      # 全イメージを作り直す。frontend だけなら次の 1 行
docker compose build frontend && docker compose up -d --force-recreate frontend
```

ブラウザ側のキャッシュが残っていることもある。強制再読み込みで確かめる。

### 3.6 migration が `already applied with a different checksum` で止まる

**適用済みの migration ファイルが書き換えられている。** 適用済みの migration は
checksum で固定され、変更は拒否される。ファイルを元に戻し、変更は新しい `V00x` として
足す。

SQL の中身は同じで、記録上の checksum だけがずれている場合に限り、記録だけを更新できる
（再実行はしない）。

```bash
docker compose run --rm -e MIGRATION_REPAIR_CHECKSUMS='<版名>.sql' db-migrate
```

---

## 4. データの管理

### 4.1 取り込み

画面（「データ取り込み」タブ）か、スクリプトで行う。

```bash
python scripts/load_initial_images.py
```

まず dry-run で CSV と画像の突合を確認すること。

### 4.2 突合（日次）

```bash
docker compose run --rm --no-deps --entrypoint python backend-api /srv/scripts/reconcile.py
```

メタデータとベクトルの食い違いを検出し、`reconciliation_report` に残す。

### 4.3 消す

```bash
python scripts/reset_data.py        # 取り込んだデータを消す（利用者は残る）
```

**取り返しがつかない。** 確認を求められる。

### 4.4 検索元の保持

検索元の画像・胸部野の輪郭を重ねた画像は `QUERY_IMAGE_RETENTION_MINUTES`（既定 60 分）で
消える。**患者の画像なので、検索元まわりはすべて同じ期限で消す。**
backend-api の常駐ジョブが削除する。

---

## 5. モデルの入れ替え

### 5.1 肺野分割モデル（U-Net）

roi-service は U-Net で肺野・心臓を分割する（**必須**。規則ベースは心臓を出さない）。
手順は [runbooks/lung-unet.md](runbooks/lung-unet.md)。概略だけ示す。

1. 参照 mask を取り込む（`scripts/fetch_reference_masks.py`）
2. 学習データを作る（`scripts/build_lung_dataset.py`）
3. 学習する（`scripts/train_lung_unet.py`。GPU では専用イメージ `infra/train/Dockerfile.lung-unet`）
4. **配置する前に Dice で測る**（`scripts/eval_roi_dice.py`）
5. `infra/lung-unet/` へ置いて roi-service を作り直す

> **学習に compose のコンテナは使えない。** `read_only: true` で `/tmp` 以外に書けず、
> 実行ユーザーも uid 10001 固定である。`docker run` に `--user` と `--shm-size=2g` を
> 付ける。

モデルや `ROI_PROFILE` を変えると `roi_model_version` が変わるので、検索対象を揃えるには
再取り込みが要る。**胸部野の分類器は学習に使った profile と組になる**ので、profile を
変えるなら分類器も作り直す（§5.2）。

### 5.2 所見プロファイル（胸部野の分類器）

| 作業 | 手順 |
|---|---|
| 分類器を作る（教師データ・学習・評価・書き出し） | [runbooks/chest-classifier.md](runbooks/chest-classifier.md) |
| 分類器を入れて構築・全量取り込み | [runbooks/chest-profile-setup.md](runbooks/chest-profile-setup.md) |
| 検索品質・病変の位置の地図を測る | [runbooks/chest-evaluation.md](runbooks/chest-evaluation.md) |

要点は 3 つ。

1. 分類器は RAD-DINO の版を添えて書き出す（`export_chest_head.py`）
2. `.env` で `EMBEDDING_FINE_BACKEND=chest-profile` と `EMBEDDING_FINE_MODEL_REVISION` を設定する
3. 取り込み経路が学習と同じ特徴を作るかを確かめてから取り込む（`chest_image_eval.py verify-ingest`）

入れると、画像検索は所見プロファイルの全件検索になる。文章を含む検索は BiomedCLIP だけの
1 段検索のまま（プロファイルはテキストを扱えない。[MULTIMODAL.md](MULTIMODAL.md) §4.1）。分類器を差し替えたら、既存行の
`embedding_fine` と所見の推定は古い分類器のものなので、取り込み直しが要る。

### 5.3 埋め込みモデルの取得

```bash
bash scripts/fetch_model.sh
```

閉域環境では、接続できる端末で取得して model volume ごと持ち込む。

---

## 6. 評価結果の記録

### 6.1 なぜ DB に持つのか

`tmp/` は `.gitignore` の対象なので、ファイルに置いた結果は EC2 から手元へ届かず、
消しても気づけず、「前の版と比べてどうか」を問い合わせることもできない。評価結果は
**再生成できない**（同じ実行は二度とできない）わりに**小さい**ので、DB に置く。

### 6.2 何を DB に置き、何を置かないか

| 種類 | 例 | 大きさ | 置き場所 | 理由 |
|---|---|---|---|---|
| **計測結果** | Dice・AUC・破綻率 | 1.7 MB | **DB** | 再生成不可。比較に使う |
| モデル成果物 | `.onnx` `.npz` | 7〜15 MB | ファイル + **DB に所在と SHA** | コンテナが read-only mount で読む |
| 中間キャッシュ | ベクトルの `.npz` | 58 MB | ファイル | NumPy 配列。版が変わると無価値 |
| 学習データ | `data/lung-dataset/` | 539 MB | ファイル | 元画像から決定論的に再生成できる |

**成果物そのものは DB に入れない。** コンテナは read-only mount で読むので、DB に
入れると起動のたびに取り出してファイルへ書く処理が要り、`read_only: true` と
噛み合わない。

### 6.3 評価を記録する

**評価スクリプトはホストで動かす。** どのコンテナの中でも完結しないためである。

| | `cv2`（ROI 抽出） | DB への到達 |
|---|---|---|
| roi-service | あり | **無い** |
| backend-api / ingest-worker | **無い** | あり |

したがってホストから DB へ届く必要がある。評価用の重ね合わせ
（`infra/compose/docker-compose.eval.yml`）が `metadata-db` を **loopback だけに**
公開するので、それを足して起動する。

```bash
# .env
COMPOSE_FILE=docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.eval.yml
COMPOSE_PATH_SEPARATOR=:
```

```bash
docker compose up -d metadata-db

METADATA_DB_HOST=127.0.0.1 METADATA_DB_PORT=55432 \
POSTGRES_PASSWORD="$(cat secrets/app_db_password.txt)" \
EVAL_GIT_REVISION="$(git rev-parse --short HEAD)" \
LUNG_MODEL_PATH="$PWD/infra/lung-unet/lung-unet.onnx" \
  python scripts/eval_roi_dice.py --limit 1000 --profile nihcc-unet-local-v1 --note "<覚書>"
```

`EVAL_RAN_ON` で場所（`ec2` / `local`）を上書きできる。

記録に対応している評価は 3 つ。

| script | `kind` | 主な指標 |
|---|---|---|
| `eval_roi_dice.py` | `roi-dice` | `dice_median` / `breakdown_rate` |
| `region_gold_eval.py` | `gold-box` | `gold_box_accuracy` |
| `region_signal_eval.py` | `region-signal` | `test_a_lift` / `test_b_lift` / `test_c_lift` |

`region_signal_eval.py` は **encoder ごとに 1 実行**として記録する。まとめると
どの encoder の数字かを指標名へ埋め込むことになり、条件で絞れなくなる。
`deterministic` は帰無なので、他の encoder の指標にはその値が `baseline` に入る。

> **公開は評価中だけにすること。** 設計方針では内部サービスは Docker ネットワーク
> 内に閉じる。終わったら `COMPOSE_FILE` から外して起動し直す。
> セキュリティグループでこのポートを外部へ開けないこと。

> **記録は補助である。** DB へ繋げなければ警告を出して諦め、評価そのものは止めない。
> 「評価は成功、記録だけ失敗」と明示するので取り違えない。

### 6.4 結果を見る

```sql
-- 版ごとの推移
SELECT r.target_version, r.ran_on, r.sample_size,
       max(CASE WHEN m.name = 'dice_median' THEN m.value END)    AS dice_median,
       max(CASE WHEN m.name = 'breakdown_rate' THEN m.value END) AS breakdown
  FROM evaluation_run r JOIN evaluation_metric m USING (run_id)
 WHERE r.kind = 'roi-dice'
 GROUP BY r.run_id, r.target_version, r.ran_on, r.sample_size, r.started_at
 ORDER BY r.started_at DESC;
```

`evaluation_metric.baseline` には**帰無（null）の値**を入れる。帰無を測らずに
「効いた」と言わないための欄である。帰無比はこれで出る。

```sql
-- 所見ごとに、帰無を何倍上回ったか
SELECT r.params->>'encoder' AS encoder, m.finding, m.region_code,
       round((m.value / NULLIF(m.baseline, 0))::numeric, 2) AS 帰無比
  FROM evaluation_run r JOIN evaluation_metric m USING (run_id)
 WHERE r.kind = 'region-signal' AND m.name = 'test_b_lift' AND m.baseline IS NOT NULL
 ORDER BY 4 DESC NULLS LAST LIMIT 20;
```

### 6.5 モデル成果物を登録する

```bash
METADATA_DB_HOST=127.0.0.1 METADATA_DB_PORT=55432 \
POSTGRES_PASSWORD="$(cat secrets/app_db_password.txt)" \
  python scripts/register_model_artifact.py \
  --type LUNG_SEGMENTATION --path infra/lung-unet/lung-unet.onnx \
  --version unet-lung-spine-local-v2.0.0 --activate
```

`--version` は profile の `algorithm_version`（`nihcc-unet-local-v1` なら
`unet-lung-spine-local-v2.0.0`）に合わせる。

§6.3 と同じく、評価用の重ね合わせで `metadata-db` を公開しておく。

所在と SHA-256、ONNX なら入力の形も `model_version` へ入る。これが無いと、EC2 と
手元で違う成果物が動いていても気づけない（`roi_model_version` は profile の版で
あって、`.onnx` の中身までは表さない）。

---

## 7. スクリプト一覧

### 運用

| スクリプト | 用途 |
|---|---|
| `bootstrap.sh` | secret と `.env` の生成（**最初に 1 回**） |
| `redeploy.sh` | 更新の反映（**これを使う**） |
| `sync_env.py` | `.env.example` の差分を `.env` へ取り込む |
| `fetch_model.sh` | 埋め込みモデルの取得 |
| `load_initial_users.py` | 初期利用者の登録 |
| `load_initial_images.py` | 画像の取り込み |
| `reset_data.py` | 取り込んだデータの削除 |
| `upgrade_postgres.sh` | PostgreSQL のメジャーバージョンアップ（§8） |

### コンテナから実行するもの

`backend-api` のイメージに入っている（`/srv/scripts/`）。

| スクリプト | 用途 |
|---|---|
| `reconcile.py` | メタデータとベクトルの突合（§4.2） |

### 肺野分割（U-Net）

手順は [runbooks/lung-unet.md](runbooks/lung-unet.md)。

| スクリプト | 用途 |
|---|---|
| `fetch_reference_masks.py` | CheXmask から参照 mask を取り込む |
| `build_lung_dataset.py` | 学習データを作る |
| `train_lung_unet.py` | 学習して ONNX を書き出す |
| `eval_roi_dice.py` | 参照 mask との Dice / IoU で分割の精度を測る |
| `segment_lungs.py` | 一括で mask を作る（任意） |
| `register_model_artifact.py` | モデル成果物の所在と SHA-256 を DB へ記録する（§6.5） |

### 胸部野の分類器・所見プロファイル

| スクリプト | 用途 |
|---|---|
| `chest_image_eval.py` | 胸部野の特徴の生成・分類器の学習と評価（`coverage` / `sheet` / `vectors` / `train`）、取り込み経路の特徴の照合（`verify-ingest`） |
| `run_chest_eval.sh` | `chest_image_eval.py` の `coverage` → `sheet` → `vectors` → `train` の一括実行（手順は [runbooks/chest-classifier.md](runbooks/chest-classifier.md)） |
| `export_chest_head.py` | 学習した分類器を、特徴の作り方を添えて本番の成果物へ書き出す（同上） |
| `chest_search_eval.py` | 取り込んだ DB に対し、実際の検索経路で検索品質（Test B）を測る。所見プロファイルの全件検索・2 段検索・1 段目だけを比べる（手順は [runbooks/chest-evaluation.md](runbooks/chest-evaluation.md)） |
| `chest_lesion_map_eval.py` | 分類器の区画ごとの地図が、病変の位置・大きさを示すかを正解 box で測る（同上） |

構築と全量取り込みの手順は [runbooks/chest-profile-setup.md](runbooks/chest-profile-setup.md)。

### 部位ごとの信号の評価

肺野を部位に分けた表現で、所見の信号を測る評価ツール。取り込み・検索の経路では使わない。
`region_signal_eval.py` と `region_mil_stage_a.py` は、`chest_image_eval.py` が ROI パスと
分類器の学習の共通部品として読み込む。

| スクリプト | 用途 |
|---|---|
| `region_signal_eval.py` | 部位ベクトルが所見を符号化しているかを基準率との比較（Test A / B / C）で測る。ベクトルを `tmp/region-signal/` へキャッシュする |
| `region_gold_eval.py` | 正解 box で、画像ごとに所見の部位を当てられるかを測る |
| `region_contrast_eval.py` | 同一患者の正常画像との対比で、所見のある部位を当てられるかを測る |
| `region_mil_stage_a.py` | 凍結した特徴に MIL ヘッドを学習し、画像単位の AUC と部位の正答率を測る |
| `region_mil_profile.py` | MIL ヘッドの部位別所見プロファイルを検索用ベクトルとして書き出し、`region_signal_eval.py` で測れるようにする |

### 開発

| スクリプト | 用途 |
|---|---|
| `try_roi.py` | ROI 抽出を手早く目視確認する |

> **使っていないスクリプトは置かない。** ここに無いものがあれば、消し忘れである。
> 手法として再利用できる評価ツールは残して用途を書く。

---

## 8. PostgreSQL のバージョンアップ

**メジャーバージョンアップはデータファイルの互換性が無い。** PG16 のボリュームを
PG17 のコンテナには読ませられないので、dump と restore が要る。

### 8.1 どこまで上げられるか

**上限を決めているのは Citus である。** 公式イメージに PG18 版がまだ無い
（2026-09 時点で `citusdata/citus:*-pg18` は存在しない）。pgvector 自体は PG18 に
対応済みで、ソースには PG19 用の分岐も入っているので、**PG18 へ上げられるのは
Citus が追いついてからになる**。

### 8.2 手順

版は `.env` の `PG_MAJOR` 1 か所で決まる。metadata とベクトル用クラスタが同じ版を使う。

```bash
# 0. 更新を取り込む（EC2 の場合）
git pull --ff-only origin main
python3 scripts/sync_env.py --write      # PG_MAJOR などを .env へ

# 1. 今のデータを退避する（tmp/pg-upgrade/ へ）
bash scripts/upgrade_postgres.sh dump

# 2. .env で版を変える（sync_env.py で入っていれば不要）
#    PG_MAJOR=17

# 3. 新しい版で作り直す（**ボリュームを消す**）
bash scripts/upgrade_postgres.sh recreate

# 4. 退避したデータを戻す
bash scripts/upgrade_postgres.sh restore
```

`recreate` が消すのは **DB のボリューム 7 本だけ**（metadata + ベクトル用クラスタの
coordinator 1 + worker 5）。モデル cache・Grafana・Prometheus・キューは残る。
消し漏れがあればその場で止まる。**残ったまま起動すると
`database files are incompatible with server` で落ちる。**

`restore` はアプリを止めてから入れ直し、**権限を付け直し**、行数を突き合わせて、
アプリを戻す（→ §10.4）。最後に実際の版を表示する。

```bash
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```

### 8.3 上げたあとに確かめること

行数が合っただけでは足りない。**分散と内部接続を見る。**

```bash
PGPW="$(cat secrets/db_password.txt)"

# 1. shard が 30 個、Worker 5 台に散っているか
docker compose exec -T -e PGPASSWORD="$PGPW" vector-coordinator \
  psql -U postgres -d vector -c \
  "SELECT nodename, count(*) AS shard数 FROM citus_shards
     WHERE table_name = 'image_embedding'::regclass GROUP BY 1 ORDER BY 1"

# 2. **アプリのロールで分散クエリが通るか**（superuser では気づけない）
docker compose exec -T vector-coordinator \
  psql "host=vector-coordinator user=app_ingest dbname=vector" \
  -tAc "SELECT count(*) FROM image_embedding"

unset PGPW
```

> ## 2 を superuser で代用しないこと
>
> Coordinator → Worker の内部接続は**クエリを実行中のロール**で張られる。
> `postgres` で数えても通るが、`app_ingest` では落ちる状態があり得る。
> 取り込みが全件失敗して初めて判る。
>
> 落ちたときは、まず初回起動の名残を疑う。
>
> ```bash
> docker compose exec -T vector-coordinator sh -c \
>   "tr '\0' '\n' < /proc/1/environ | grep -c '^PGPASSWORD='"
> ```
>
> `1` なら、その postmaster は `PGPASSWORD` を持っていて、**どのロールでも
> postgres のパスワードを送ってしまう**。image に修正が入っているか
> （`infra/postgres/zzz-unset-pgpassword.sh`）確かめ、入っていれば
> `docker compose up -d` でコンテナを入れ替える。

### 8.4 扱いが 2 通りある理由

| | dump するもの | 戻し方 |
|---|---|---|
| metadata | **スキーマ + データ** | そのまま流し込む |
| ベクトル用クラスタ | **データのみ** | migration がスキーマを作ってから入れる |

クラスタ側でスキーマを dump しないのは、**分散テーブルは
`create_distributed_table` を通さないと分散されない**ためである。`pg_dump` は
その呼び出しを出力しないので、スキーマは migration に作らせる。

### 8.5 所要時間と注意

手元（4,999 枚 / dump 4.3 GB）での目安。

| 段 | 時間 |
|---|---|
| `dump` | 約 10 分（大半は metadata） |
| `recreate` | 約 4 分（ビルド + 起動 + migration） |
| `restore` | 約 5 分 |

EC2（11 万枚 / dump 約 52 GB）は**十数倍**を見る。退避先のディスク空きを
**先に**確認すること。dump は元の DB と同じ程度の大きさになる。

> **先に手元で通すこと。** EC2 には 11 万枚が入っており、失敗すると取り込みから
> やり直しになる。

---

## 9. 監視

Grafana（`localhost:3000`）で見る。主な指標。

| 指標 | 見るところ |
|---|---|
| `ingest_stage_duration_seconds{stage="roi"}` | ROI が律速になっていないか |
| ベクトル書き込みの成否 | 失敗が増えていないか |
| 検索の応答時間 | 遅いときは画面の非同期実行（`Prefer: respond-async`）を使う |

`roi` が伸びるようなら `ROI_SERVICE_REPLICAS` を増やす。

---

## 10. バックアップと復元

### 10.1 何を保存するか

| 対象 | 方法 | 大きさ（11 万枚） |
|---|---|---|
| metadata | `pg_dump`（スキーマ + データ） | 約 47 GB |
| ベクトル用クラスタ | **データのみ** `pg_dump`（後述） | 約 3 GB（見積り） |
| モデル成果物 | `infra/chest/` `infra/lung-unet/` をそのまま保存 | 数 MB |
| 設定 | `.env` と `secrets/` | 小 |

> **`secrets/` は秘密情報である。** git に入れず、バックアップの置き場所も限る。

原画像は `image_binary` に入っているので metadata のバックアップに含まれる。
**ここが容量の大半（44 GB）を占める。** データセット本体を別に持っているなら、
`--exclude-table=image_binary` で外して取り込み直す方が軽い。

### 10.2 Citus の分散テーブルは扱いが違う

**ここが要点である。** ベクトル用クラスタの `image_embedding` は Citus の分散テーブル
で、実体は worker 5 台の shard に散っている。

```
  vector-coordinator（テーブル定義と分散の情報）
      └→ vector-worker-1 … vector-worker-5（shard 30 個に散った実データ）
```

| | できる / できない |
|---|---|
| coordinator で `pg_dump` して**データを取る** | **できる。** 全 worker から集めて出力する |
| そのスキーマ dump を復元して**分散を再現する** | **できない** |

`pg_dump` は `CREATE TABLE` は出力するが、**`create_distributed_table()` の呼び出しは
出力しない**。そのまま復元すると、分散されていない普通のテーブルが coordinator に
1 つできるだけで、worker は空のまま。検索は動いても並列に効かず、shard も無い。

**したがってスキーマは migration に作らせる。**

| | dump するもの | 復元の順序 |
|---|---|---|
| metadata | スキーマ + データ | そのまま流し込む |
| ベクトル用クラスタ | **データのみ**（`--data-only --table=image_embedding`） | migration → データ |

migration（`db/vector-migrations/V001`）の末尾が `create_distributed_table` を呼ぶ。
**空のテーブルを先に分散させてから**データを入れると、shard へ正しく散る。

> worker を直接 dump しないこと。shard は `image_embedding_102008` のような名前で
> 散っており、個別に集めても復元できない。**必ず coordinator を通す。**

### 10.3 取る

`scripts/upgrade_postgres.sh dump` がこの手順をそのまま行う。バージョンアップ専用
ではないので、**通常のバックアップにも使える**。

```bash
DUMP_DIR=/backup/$(date +%F) bash scripts/upgrade_postgres.sh dump
```

出力は次のとおり。`counts.txt` に行数が入るので、**復元したときの突き合わせに使う**。

```
metadata.sql   スキーマ + データ
vector.sql     データのみ
counts.txt     各テーブルの行数
```

### 10.4 戻す

**空の環境へ戻す前提である。** 先にスキーマを作る必要がある。

```bash
docker compose up -d
docker compose run --rm db-migrate      # スキーマを作り、分散させる

DUMP_DIR=/backup/<日付> bash scripts/upgrade_postgres.sh restore
```

`restore` が行うこと。

1. アプリを止める（**繋ぎっぱなしだと metadata を DROP できない**）
2. metadata を作り直して流し込む
3. ベクトル用クラスタへデータを入れる
4. **`db-migrate` を流し直して権限を付け直す**
5. `counts.txt` と行数を突き合わせる（合わなければ止まる）
6. アプリを戻す

> ## 権限は dump に入っていない
>
> dump は `--no-privileges` で取る。**権限の定義元は migration の
> `grant_schema` 1 か所**で、dump にも入れると定義が 2 つになる。
>
> metadata を DROP した時点で `app_meta` への GRANT も消えるので、
> **復元しただけでは足りない**。付け直さないと、全サービスが healthy な
> まま**ログインだけが 500 になる**。
>
> ```
> psycopg.errors.InsufficientPrivilege: permission denied for table app_user
> ```
>
> `restore` は 4. で自動的に流し直す。手で復元したときは忘れないこと。
> migration 自体は `schema_migration`（dump に含まれる）を見て飛ばすので、
> スキーマには触らない。

行数だけを数え直したいときは `verify` を使う。**復元は EC2 で数時間かかるので、
確認のためにやり直さない。**

```bash
DUMP_DIR=/backup/<日付> bash scripts/upgrade_postgres.sh verify
```

### 10.5 確かめる

復元しただけで満足しない。**分散が効いているか**を見る。

```bash
PGPW="$(cat secrets/db_password.txt)"
docker compose exec -T -e PGPASSWORD="$PGPW" vector-coordinator \
  psql -U postgres -d vector -c \
  "SELECT logicalrelid AS テーブル, count(*) AS shard数
     FROM pg_dist_shard GROUP BY 1"
unset PGPW
```

`image_embedding` が `30` と出れば分散されている。**0 件なら分散に失敗している**
（migration を通さずにスキーマを復元した場合など）。

最後に通しで確認する。

```bash
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```
