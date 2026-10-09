# 設計仕様書

NIHCC 胸部X線画像検索ローカルRAGシステムの構成と設計。**開発者向け**である。

**略語は [GLOSSARY.md](GLOSSARY.md) にまとめてある。**

マルチモーダル（モデル・ROI・分類器・U-Net・多段検索）の設計は [MULTIMODAL.md](MULTIMODAL.md)
に分けてある。運用手順は [OPERATIONS.md](OPERATIONS.md)。

---

## 1. 何をするシステムか

胸部X線画像の**胸部野**（肺野＋縦隔＋横隔膜）どうしを比べて検索する。画像を投げれば
似た症例が、所見の文を投げれば該当しそうな症例が並ぶ。各候補には、胸部野から学習した
分類器による**所見の推定**（研究用・診断ではない）が付く。

対象は NIH ChestX-ray14（112,120 枚 / 30,805 患者）。

### 1.1 なぜ画像 1 枚の胸部野を単位にするのか

データセットのラベルは**画像単位**で、肺の部位（上葉・下葉など）単位の正解は無い。
部位単位の検索は学習も評価もできない。教師なしの外観埋め込みは部位の形と撮影条件を
捉えるだけで、部位レベルの所見を当てられない。

単位を画像 1 枚に揃えると、画像単位のラベルがそのまま教師データになる。全量の評価で
画像単位 AUC 0.833、類似症例の検索（Test B）は 14 所見すべてで無作為の 1.33 倍以上
である。

枠を肺野だけにせず縦隔と横隔膜まで含めるのは、**心拡大は心臓そのもの、胸水は
肋骨横隔膜角に出る**ためである。正解 box が枠に半分以上入る割合は、肺野だけなら
58.7%、胸部野なら 99.7% である。

その代わり、「どの部位に所見があるか」では検索できない。

### 1.2 研究用途に限る

**診断に使わない。** データセットのラベルは放射線科レポートから NLP で導いたもので、
専門家が画像を見て付けたものではない。検索結果には常にその旨の警告を付ける。

### 1.3 外へ出さない

医療情報を外部 API へ送らない。日本語の所見文も、外部の翻訳 API ではなく
**承認済みのローカル辞書**だけで英語化する（辞書に無い自由文は翻訳せず拒否する）。

---

## 2. システム構成

図（PNG）は [diagrams/](diagrams/README.md) にある（システム構成・プロセス構成・コンテナ関連・
教師データ作成・取り込み・検索のフロー）。

### 2.1 全体

```
                    ┌──────────┐
  利用者 ─────────→ │ frontend │  静的ファイル（nginx）
                    └────┬─────┘
                         │
                    ┌────┴────────┐
                    │ backend-api │  REST API・認証・監査
                    └──┬───┬───┬──┘
           ┌───────────┘   │   └───────────┐
           │               │               │
    ┌──────┴─────┐   ┌─────┴──────┐  ┌─────┴──────┐
    │ metadata-db│   │   queue    │  │   検索・取り込みの
    │ PostgreSQL │   │ RabbitMQ   │  │   下請けサービス群
    └────────────┘   └─────┬──────┘  └─────┬──────┘
                           │               │
              ┌────────────┴───┐    ┌──────┴────────────┐
              │ ingest-worker  │    │ search-worker     │
              │ search-worker  │    │ search-aggregator │
              └────────┬───────┘    └──────┬────────────┘
                       │                   │
         ┌─────────────┴───┬───────────────┴──┐
         │                 │                  │
   ┌─────┴──────┐  ┌───────┴────────┐  ┌──────┴──────────────┐
   │ roi-service│  │embedding-service│ │ ベクトル用クラスタ     │
   │ 胸部野の切出し│ │ ベクトル・推定  │ │ Citus + pgvector × 1 │
   └────────────┘  └────────────────┘  └─────────────────────┘
```

### 2.2 ネットワークを 3 つに分ける

| 名前 | 誰が居るか | 意図 |
|---|---|---|
| `edge` | frontend / backend-api | 外部に開く唯一の面 |
| `app` | サービス間 | 下請けは外から直接叩けない |
| `data` | DB・クラスタ | **DB に到達できるものを絞る** |

`roi-service` と `embedding-service` は `app` だけに居る。**DB へ到達できない。**
DB の読み書きは ingest-worker / backend-api / search-aggregator が行い、
roi-service と embedding-service は受け取った画像を処理して返すだけである。

### 2.3 コンテナ一覧

| サービス | イメージ | 役割 | 公開 |
|---|---|---|---|
| `frontend` | nginx | 画面（静的ファイル） | 8080 |
| `backend-api` | 自作 | REST API・認証・監査 | 8000 |
| `ingest-worker` | 自作 | 取り込み（既定 2 本） | - |
| `search-worker` | 自作 | 非同期検索 | - |
| `search-aggregator` | 自作 | ベクトル用クラスタへの検索（プロファイルの全件検索 / 2 段検索） | - |
| `roi-service` | 自作 | 胸部野（U-Net の肺・心臓から）の切り出し | - |
| `embedding-service` | 自作 | ベクトル生成と所見の推定（torch） | - |
| `metadata-db` | postgres:17 | メタデータ | - |
| `queue` | rabbitmq:3.13 | ジョブ | 15672（管理画面） |
| `db-migrate` | 自作 | スキーマ適用（`migrate` プロファイル） | - |
| `mailpit` | mailpit | 開発用メール受け | 8025 |
| `prometheus` / `grafana` | 公式 | 監視 | 9090 / 3000 |
| `vector-coordinator` / `vector-worker-1..5` | Citus + pgvector | ベクトル（coordinator 1 + worker 5） | - |

**全サービスが `read_only: true`** で、書けるのは `/tmp`（tmpfs）だけである。
`cap_drop: ALL` と `no-new-privileges` も付けている。この堅牢化は学習や backfill の
ような書き込みを伴う作業と噛み合わないので、そちらは別のイメージと `docker run` で
行う（[OPERATIONS.md](OPERATIONS.md)）。

### 2.4 ベクトル用クラスタは 1 つ

検索の単位は画像 1 枚で、1 行 = 1 画像なので、クラスタは 1 つ（coordinator 1 台 +
worker 5 台、shard 30、1 台あたり 6）である。走査は worker 5 台へ分散する。
定義は [infra/compose/docker-compose.vectors.yml](../infra/compose/docker-compose.vectors.yml)。

---

## 3. プロセスフロー

### 3.1 取り込み

```
  利用者 → backend-api        検証（CSV と画像の突合）→ ingest_job / ingest_item
             │
             └→ queue         1 枚 = 1 メッセージ
                  │
                  ↓
             ingest-worker
                  │
                  ├→ roi-service        U-Net の肺・心臓 → 胸部野の mask・crop・形状ベクトル
                  ├→ embedding-service  1 段目（BiomedCLIP）・2 段目（プロファイル）・所見の推定
                  ├→ ベクトル用クラスタ  image_embedding へ UPSERT（1 画像 1 行）
                  └→ metadata-db        chest_asset / finding_estimate / image_vector_state
```

要点。

- **1 枚の失敗で全体を止めない。** 失敗は `ingest_item` に残し、ジョブは続行する
- **冪等。** 同じ画像を再送しても二重に入らない（`idempotency_key`、
  `ON CONFLICT DO UPDATE`）
- **胸部野は roi-service に集約する。** 埋め込みモデルへの入力も roi-service が作り、
  呼び出し側では作り直さない。分類器は評価スクリプトが作った特徴で学習しているので、
  1 画素でも違うと確率が黙ってずれる（評価と本番は `image_pipeline.chest` の同じ関数）
- 起動時に、roi-service の mask の作り方と分類器の前提が一致するかを確かめる
- ベクトル書き込みの前に、**列の幅と `.env` が一致するか**を起動時に確かめる

### 3.2 検索

```
  利用者 → backend-api
             │
             ├→ roi-service         検索元画像の胸部野（画像検索のとき）
             ├→ embedding-service   クエリベクトルと検索元の所見の推定
             │
             └→ search-aggregator → ベクトル用クラスタ
                  │
                  ├→ 画像だけ: 所見プロファイル（15 次元）で全画像を厳密に比べる（既定）
                  ├→ 文章を含む: HNSW（BiomedCLIP）で候補 → 所見プロファイルで再スコア
                  └→ 胸郭の形（embedding_lung）
```

画像だけの検索を全件検索にするのは、2 段検索では 1 段目（BiomedCLIP）が拾わなかった
所見の合う画像が並べ替えの対象にならないためである（結節の lift は 2 段検索 1.91 に
対し全件検索 2.43、ベクトル検索の p95 は 290 ms に対し 24 ms）。`IMAGE_SEARCH_STRATEGY`
で 2 段検索に切り替えられる。

応答が 2 秒を超えると非同期へ切り替わり、`search-worker` が引き取って
`search_result` へ書く。画面は `search_id` で polling する。

多段検索の詳細は [MULTIMODAL.md](MULTIMODAL.md) §4。

### 3.3 認証

JWT。`app_user` に bcrypt のハッシュを持つ。役割は `viewer` / `operator` / `admin`。
パスワード再設定はメール経由（開発環境では mailpit が受ける）。
詳細は [user-management.md](user-management.md)。

---

## 4. データベース構成

### 4.1 2 種類のデータベース

| | metadata | ベクトル用クラスタ |
|---|---|---|
| 実体 | PostgreSQL 17 × 1 | Citus + pgvector（coordinator 1 + worker 5） |
| 持つもの | 画像・メタデータ・胸部野・推定・ジョブ・監査 | ベクトルのみ |
| DDL | [db/metadata-migrations/](../db/metadata-migrations/) | [db/vector-migrations/](../db/vector-migrations/) |

ベクトルの走査（HNSW）を worker へ分散させるため、ベクトルだけ別の DB に置く。

### 4.2 PostgreSQL の版

**PG17 を使う。** 版は `.env` の `PG_MAJOR` 1 か所で決まり、metadata とベクトル用
クラスタが同じ版を使う。

| | 版 | 制約 |
|---|---|---|
| PostgreSQL | 17 | **Citus が対応している最新**（2026-09 時点で PG18 版は無い） |
| Citus | 14.2.0 | |
| pgvector | 0.8.6 | PG13〜18 に対応。版ごとの機能差は無い |

> **上限を決めているのは Citus である。** pgvector は既に PG18 に対応し、ソースには
> PG19 用の分岐も入っている。PG18 へ上げられるのは Citus が対応してからになる。

メジャーバージョンアップは**データファイルの互換性が無い**ので、dump と restore が
要る。手順は `scripts/upgrade_postgres.sh`（[OPERATIONS.md](OPERATIONS.md) §8）。

### 4.3 スキーマの管理

DDL は migration だけに置く。

| | 置き場 | 最初の版 |
|---|---|---|
| metadata | `db/metadata-migrations/` | `V001__initial_schema.sql`（全テーブルと付随ラベルの語彙） |
| ベクトル用クラスタ | `db/vector-migrations/` | `V001__image_embedding.sql`（次元は `.env` の値で置換） |

- `db-migrate` は未適用の版だけを順に当て、`schema_migration` に版名と本文の checksum を
  記録する。再実行しても安全
- **適用済みの版は書き換えない。** スキーマを変えるときは新しい `V00x` を足す

> **migration ファイルは image に焼き込まれる。** `db-migrate` を作り直さずに実行
> すると、新しい migration が無いまま「適用済み」だけを表示して終わる。
> `scripts/redeploy.sh` を使うこと。

### 4.4 metadata のテーブル

#### 画像とメタデータ

| テーブル | 役割 |
|---|---|
| `patient` | 患者。**元の ID は持たず**、ハッシュと内部連番だけ |
| `source_image` | 画像 1 枚のメタデータ（撮影方向・寸法・出典） |
| `image_binary` | 原画像のバイト列と SHA-256 |
| `finding` | 所見のマスタ（14 所見 + `No Finding`） |
| `image_finding` | 画像と所見の対応（データセットのラベル。**画像単位**） |
| `chest_asset` | 胸部野と肺野の mask・bbox・面積・品質。ROI の版を持つ |
| `finding_estimate` | 胸部野の分類器による所見の推定（14 所見の確率）。分類器の版を持つ |
| `region_asset` | **使わない**（肺の部位単位の表。現行の処理は読み書きしない） |

#### 取り込み

| テーブル | 役割 |
|---|---|
| `ingest_job` | ジョブ単位の進行と集計 |
| `ingest_item` | 1 枚ごとの結果。失敗の符号と詳細を残す |
| `ingest_validation` | dry-run の結果 |
| `idempotency_key` | 同じ要求の二重登録を防ぐ |
| `outbox` | キューへの発行を DB 変更と同じトランザクションで確定させる |

#### ベクトルの状態

| テーブル | 役割 |
|---|---|
| `image_vector_state` | 画像ごとの書き込み状態と最後の失敗理由 |
| `model_version` | 使用中のモデルの版（分類器は `EMBEDDING_FINE`） |
| `region_vector_state` | **使わない**（部位単位の表） |

ベクトル本体はここに無い（§4.5）。

#### 検索

| テーブル | 役割 | 保持 |
|---|---|---|
| `search_audit` | いつ誰が何を検索したか。**画像は SHA だけ** | 無期限 |
| `search_result` | 非同期検索の結果 | 無期限 |
| `search_query_image` | 検索元画像の実体 | 期限つき |
| `search_query_chest` | 検索元に胸部野の輪郭を重ねた画像（結果一覧の表示用） | 期限つき |
| `search_query_outline` | 検索元の肺野外周（胸郭ランキング表示用） | 期限つき |
| `search_query_region` | **使わない**（部位単位の表。期限切れの削除だけが走る） | 期限つき |

**検索元まわりは患者の画像なので、すべて同じ期限で消す**
（`QUERY_IMAGE_RETENTION_MINUTES`、既定 60 分）。常駐ジョブが期限切れを削除する。

#### 評価

| テーブル | 役割 |
|---|---|
| `evaluation_run` | 評価の実行 1 回。何をどの条件で測ったか |
| `evaluation_metric` | その結果。指標ごとに 1 行。**帰無の値も持つ** |

**再生成できないので DB に持つ。** `tmp/` は `.gitignore` の対象で、EC2 で測った結果を
ファイルのままでは手元と共有できない。学習データ（539 MB）と
中間キャッシュ（58 MB）は置かない（再生成できる / 版が変わると無価値）。
詳細は [OPERATIONS.md](OPERATIONS.md) §6。

#### 利用者・運用

| テーブル | 役割 |
|---|---|
| `app_user` / `password_reset_token` | 認証 |
| `audit_log` | ログイン・検索・原画像閲覧・取り込み・設定変更 |
| `reconciliation_report` | 日次の突合結果 |
| `schema_migration` | 適用済み migration |

### 4.5 ベクトル用クラスタのテーブル

`image_embedding` の 1 テーブルだけ。1 行 = 1 画像。

| 列 | 内容 | 索引 |
|---|---|---|
| `embedding` | 文章を含む検索の 1 段目（512 次元 / BiomedCLIP、胸部野 crop） | **HNSW** |
| `embedding_fine` | 画像検索の全件検索・2 段目（15 次元 / 胸部野の分類器のプロファイル） | 無し（厳密走査） |
| `embedding_lung` | 胸郭の形（16 次元） | 無し |

主キーに `embedding_model_version` と `preprocess_version` を含むので、モデルや
前処理の版を変えても旧いベクトルと混ざらない。

> **`embedding_fine` に索引を張らないのは意図的である。** 15 次元 × 11 万行は shard
> ごとの順次走査で数十 ms に収まり、HNSW の近似による取りこぼしも無い。2 段検索では
> 1 段目が絞った候補だけを再スコアリングするので、そもそも近似する相手が居ない。
> 詳細は [MULTIMODAL.md](MULTIMODAL.md) §4。

### 4.6 版の扱い

| 版 | 何が変わると変えるか | 影響 |
|---|---|---|
| `roi_model_version` | ROI の profile（`algorithm_version` + パラメータの SHA） | 新旧の ROI が混ざらない |
| `preprocess_version` | crop の作り方 | 旧いベクトルと混ざらない |
| `embedding_model_version` | モデルと revision | 同上 |

**profile 全体の SHA-256 が `parameter_hash`** になる。継承の仕組みを入れていないのは、
再現キーが profile 全体を覆うという性質を保つためである。

---

## 5. 主な設計判断

| 判断 | 理由 |
|---|---|
| 画像 1 枚の胸部野を単位にする | 部位単位の正解が無く学習できない。画像単位なら教師がある |
| 枠に縦隔と横隔膜を含める | 心拡大・胸水の根拠が肺野の外にある（正解 box の 99.7% 対 58.7%） |
| ベクトル用クラスタを 1 つにする | 1 画像 1 行で、分ける単位が無い |
| 画像だけの検索はプロファイルの全件検索 | 2 段検索は 1 段目の取りこぼしで所見の合う画像を逃す。全件でも p95 24 ms |
| `embedding_fine` に索引を張らない | 15 次元は順次走査で足り、近似の取りこぼしが無い |
| 画像の比較は分類器の 15 次元 | 学習済みの所見の確率で比べる。生の外観より類似症例が引ける |
| 分類器は特徴の作り方を持つ | 版や前処理が違うと確率が黙ってずれる。起動時に照合して止める |
| 所見の推定を画面に出す | 利用者の判断。「診断ではない」見出しと、ラベルとの区別を必ず添える |
| 肺野の切り出しは U-Net | 規則ベースは corpus の 23.8% で破綻する |
| 形の比較は肺野全体（胸郭）で行う | 部位は胸郭から幾何規則で切り出すもので、胸郭の形に従属する |
| 日本語の翻訳は辞書のみ | 医療情報を外部 API へ出さない |
| 検索元の保持を期限つきに | 患者の画像である |

---

## 6. 品質と監視

### 6.1 自動品質検査

胸部野の抽出は画像ごとに `quality_status` を付ける。肺野が画像の 2% 未満なら抽出不能
（422、`LUNG_TOO_SMALL`）。片肺が見つからない・左右差が大きい・肺野が広すぎる場合は
旗を立てて `ROI_UNCERTAIN`（信頼度 0.5）にし、取り込みは止めない（気胸や大量胸水の
片肺虚脱でも起きるため）。検索元の信頼度が `ROI_MIN_CONFIDENCE` を下回れば結果を抑制し、
`ROI_UNCERTAIN` なら警告を添える。

### 6.2 監視

Prometheus が各サービスの `/metrics` を集め、Grafana が表示する。主な指標は
取り込みの段階別所要時間、ROI の成否、ベクトル書き込みの成否、検索の応答時間。

### 6.3 テスト

| 種類 | 場所 | 内容 |
|---|---|---|
| 単体 | `tests/unit/` | 契約・前処理・ROI・分類器・スキーマの整合 |
| 結合 | `tests/integration/` | 起動中のシステムに対するスモーク |

結合テストは実際に取り込みと検索を通す。`python tests/integration/smoke_test.py`。
