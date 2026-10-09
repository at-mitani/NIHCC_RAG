-- =============================================================================
-- metadata データベースのスキーマ
--
-- 原画像・付随ラベル・胸部野の mask・所見の推定・ジョブ・検索・評価・利用者・
-- 監査を保持する。ベクトルは別のデータベース（ベクトル用 Citus クラスタ、
-- db/vector-migrations/）にある。
--
-- 変更は、このファイルを書き換えずに新しい V00x を追加して行う。適用済みの版は
-- schema_migration に本文の checksum と共に記録され、db-migrate は未適用の版だけを
-- 当てる（db/migrate/run_migrations.py）。
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE job_status AS ENUM
  ('PENDING','RUNNING','SUCCEEDED','FAILED','PARTIAL','CANCELLED');
-- CANCELLED: 取り消したジョブの未着手 item。Queue に残ったメッセージが再配送されても
-- ingest-worker は処理しない。
CREATE TYPE item_status AS ENUM
  ('PENDING','RUNNING','SUCCEEDED','FAILED','QUARANTINED','CANCELLED');


-- --- 画像とメタデータ --------------------------------------------------------

-- 患者。公開データでも患者 ID は直接保持せず hash 化キーで持つ。
CREATE TABLE patient (
  patient_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_patient_key_hash char(64) NOT NULL UNIQUE,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE source_image (
  image_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  patient_id bigint NOT NULL REFERENCES patient(patient_id),
  source_image_name text NOT NULL UNIQUE,
  sha256 char(64) NOT NULL UNIQUE,
  width integer NOT NULL CHECK (width > 0),
  height integer NOT NULL CHECK (height > 0),
  view_position varchar(8),
  patient_age integer,
  patient_gender varchar(32),
  source_dataset_version text NOT NULL,
  source_metadata jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

-- 画像本体は一覧検索時の I/O を抑えるため専用テーブルへ分離する。
CREATE TABLE image_binary (
  image_id uuid PRIMARY KEY REFERENCES source_image(image_id) ON DELETE CASCADE,
  content bytea NOT NULL,
  mime_type text NOT NULL CHECK (mime_type IN ('image/png','image/jpeg')),
  byte_size bigint NOT NULL CHECK (byte_size > 0)
);

-- ChestX-ray14 の付随ラベル語彙。読影レポートから自然言語処理で抽出された
-- 画像単位の弱教師ラベルであり、確定診断ではない。表示・API・DB では常に
-- 「データセット付随ラベル」と表記する。
CREATE TABLE finding (
  finding_id smallint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  code text NOT NULL UNIQUE,
  display_name text NOT NULL
);

CREATE TABLE image_finding (
  image_id uuid NOT NULL REFERENCES source_image(image_id) ON DELETE CASCADE,
  finding_id smallint NOT NULL REFERENCES finding(finding_id),
  label_source text NOT NULL DEFAULT 'NIHCC_CSV_NLP_DERIVED',
  PRIMARY KEY (image_id, finding_id)
);

CREATE INDEX source_image_patient_idx ON source_image(patient_id);
CREATE INDEX source_image_metadata_gin_idx ON source_image USING gin(source_metadata);


-- --- モデルの版 --------------------------------------------------------------

-- model_type ごとに active は 1 件だけ。種別を分けておくと、1 段目（EMBEDDING）と
-- 2 段目（EMBEDDING_FINE）のように別のモデルを同時に active にでき、管理画面でも
-- 次元と revision を別の行として確認できる。
CREATE TABLE model_version (
  model_version text PRIMARY KEY,
  model_type text NOT NULL CHECK (model_type IN (
    'ROI', 'EMBEDDING', 'EMBEDDING_FINE', 'REGION_MIL', 'TRANSLATION', 'LUNG_SEGMENTATION'
  )),
  model_name text NOT NULL,
  source_revision text NOT NULL,
  artifact_sha256 char(64) NOT NULL,
  embedding_dimension integer,
  config jsonb NOT NULL DEFAULT '{}',
  active boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX one_active_model_per_type
  ON model_version(model_type) WHERE active;


-- --- 胸部野（検索の単位） ----------------------------------------------------

-- 画像ごとの胸部野（肺野＋縦隔＋横隔膜）の mask と品質。
CREATE TABLE chest_asset (
  chest_asset_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  image_id uuid NOT NULL REFERENCES source_image(image_id) ON DELETE CASCADE,
  --: 胸部野と肺野の mask。胸部野の外接矩形で切った作業解像度の PNG
  chest_mask bytea NOT NULL,
  lung_mask bytea NOT NULL,
  --: {"bbox": 作業解像度, "source_bbox": 原画像座標, "work_scale": 縮小率}
  bbox jsonb NOT NULL,
  --: 肺野・胸部野・心臓が画像に占める割合
  areas jsonb NOT NULL,
  flags text[] NOT NULL DEFAULT '{}',
  roi_confidence real NOT NULL CHECK (roi_confidence >= 0 AND roi_confidence <= 1),
  roi_model_version text NOT NULL REFERENCES model_version(model_version),
  preprocess_version text NOT NULL,
  quality_status text NOT NULL CHECK (quality_status IN ('OK', 'ROI_UNCERTAIN')),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (image_id, roi_model_version, preprocess_version)
);

CREATE INDEX chest_asset_image_idx ON chest_asset(image_id, created_at DESC);

-- ベクトル用クラスタへの登録状態。未完了の行を突合と補償に使う。
CREATE TABLE image_vector_state (
  image_id uuid NOT NULL REFERENCES source_image(image_id) ON DELETE CASCADE,
  embedding_model_version text NOT NULL,
  preprocess_version text NOT NULL,
  state text NOT NULL CHECK (state IN ('PENDING', 'STORED', 'FAILED')),
  last_error text,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (image_id, embedding_model_version, preprocess_version)
);

CREATE INDEX image_vector_state_pending_idx
  ON image_vector_state(state, updated_at) WHERE state <> 'STORED';

-- 胸部野の分類器による所見の推定確率（画面に表示する）。
CREATE TABLE finding_estimate (
  image_id uuid NOT NULL REFERENCES source_image(image_id) ON DELETE CASCADE,
  --: 推定に使った分類器（2 段目のモデル版、EMBEDDING_FINE）
  model_version text NOT NULL REFERENCES model_version(model_version),
  --: {"Atelectasis": 0.12, ...}。14 所見すべて
  probabilities jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (image_id, model_version)
);


-- --- 部位単位の表（現在の取り込み・検索では使わない） ------------------------

CREATE TABLE region_asset (
  region_asset_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  image_id uuid NOT NULL REFERENCES source_image(image_id) ON DELETE CASCADE,
  region_code varchar(3) NOT NULL CHECK (region_code IN ('RUL','RML','RLL','LUL','LLL')),
  crop_content bytea,
  mask_content bytea,
  bbox jsonb NOT NULL,
  roi_confidence real NOT NULL CHECK (roi_confidence BETWEEN 0 AND 1),
  roi_model_version text NOT NULL REFERENCES model_version(model_version),
  preprocess_version text NOT NULL,
  quality_status text NOT NULL CHECK (quality_status IN ('OK','ROI_UNCERTAIN','REJECTED')),
  created_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (image_id, region_code, roi_model_version, preprocess_version)
);

CREATE INDEX region_asset_image_region_idx ON region_asset(image_id, region_code);

CREATE TABLE region_vector_state (
  image_id uuid NOT NULL REFERENCES source_image(image_id) ON DELETE CASCADE,
  region_code varchar(3) NOT NULL CHECK (region_code IN ('RUL','RML','RLL','LUL','LLL')),
  embedding_model_version text NOT NULL,
  preprocess_version text NOT NULL,
  state text NOT NULL CHECK (state IN ('PENDING','STORED','FAILED','SKIPPED')),
  last_error text,
  updated_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (image_id, region_code, embedding_model_version, preprocess_version)
);

CREATE INDEX region_vector_state_pending_idx
  ON region_vector_state(state, updated_at) WHERE state <> 'STORED';


-- --- 取り込み ----------------------------------------------------------------

CREATE TABLE ingest_job (
  job_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  status job_status NOT NULL DEFAULT 'PENDING',
  request jsonb NOT NULL,
  total_count integer NOT NULL DEFAULT 0,
  succeeded_count integer NOT NULL DEFAULT 0,
  failed_count integer NOT NULL DEFAULT 0,
  created_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  started_at timestamptz,
  completed_at timestamptz
);

CREATE TABLE ingest_item (
  item_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id uuid NOT NULL REFERENCES ingest_job(job_id) ON DELETE CASCADE,
  source_image_name text NOT NULL,
  image_id uuid REFERENCES source_image(image_id),
  status item_status NOT NULL DEFAULT 'PENDING',
  attempt_count integer NOT NULL DEFAULT 0,
  error_code text,
  error_detail text,
  updated_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE (job_id, source_image_name)
);

CREATE INDEX ingest_item_job_status_idx ON ingest_item(job_id, status);

-- dry-run の検証結果。POST /ingestions は validation_id を参照して起動する。
CREATE TABLE ingest_validation (
  validation_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  request jsonb NOT NULL,
  valid boolean NOT NULL,
  counts jsonb NOT NULL DEFAULT '{}',
  errors jsonb NOT NULL DEFAULT '[]',
  warnings jsonb NOT NULL DEFAULT '[]',
  manifest jsonb NOT NULL DEFAULT '[]',
  created_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  expires_at timestamptz NOT NULL DEFAULT now() + interval '24 hours'
);

-- 取り込み開始の Idempotency-Key。
CREATE TABLE idempotency_key (
  idempotency_key text PRIMARY KEY,
  endpoint text NOT NULL,
  request_fingerprint char(64) NOT NULL,
  job_id uuid REFERENCES ingest_job(job_id) ON DELETE SET NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

-- 登録イベントの outbox（at-least-once 配信）。
CREATE TABLE outbox (
  outbox_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  aggregate_type text NOT NULL,
  aggregate_id text NOT NULL,
  routing_key text NOT NULL,
  payload jsonb NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  published_at timestamptz,
  attempt_count integer NOT NULL DEFAULT 0,
  last_error text
);

CREATE INDEX outbox_unpublished_idx ON outbox(created_at) WHERE published_at IS NULL;


-- --- 検索 --------------------------------------------------------------------

CREATE TABLE search_audit (
  search_id uuid PRIMARY KEY,
  search_mode text NOT NULL CHECK (search_mode IN ('IMAGE_TO_IMAGE','TEXT_TO_IMAGE','HYBRID')),
  query_image_sha256 char(64),
  query_text_original text,
  query_text_normalized_en text,
  embedding_model_version text NOT NULL,
  roi_model_version text,
  request_params jsonb NOT NULL,
  result_summary jsonb,
  duration_ms integer,
  created_by text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX search_audit_created_idx ON search_audit(created_at);

-- 非同期検索の結果（Prefer: respond-async）。
CREATE TABLE search_result (
  search_id uuid PRIMARY KEY REFERENCES search_audit(search_id) ON DELETE CASCADE,
  status text NOT NULL CHECK (status IN ('PENDING','RUNNING','SUCCEEDED','FAILED')),
  response jsonb,
  error jsonb,
  updated_at timestamptz NOT NULL DEFAULT now()
);

-- 検索元まわりの画像。患者の画像なので、すべて QUERY_IMAGE_RETENTION_MINUTES で
-- 期限を切り、期限切れは backend-api の常駐ジョブが削除する。

-- 検索元の画像そのもの。
CREATE TABLE search_query_image (
  search_id uuid PRIMARY KEY REFERENCES search_audit(search_id) ON DELETE CASCADE,
  content bytea NOT NULL,
  mime_type text NOT NULL,
  sha256 char(64) NOT NULL,
  expires_at timestamptz NOT NULL
);

CREATE INDEX search_query_image_expiry_idx ON search_query_image(expires_at);

-- 検索元に胸部野の輪郭を重ねた画像（結果一覧の左端に出す）。
CREATE TABLE search_query_chest (
  search_id uuid PRIMARY KEY REFERENCES search_audit(search_id) ON DELETE CASCADE,
  content bytea NOT NULL,
  roi_confidence real,
  quality_status text,
  expires_at timestamptz NOT NULL
);

CREATE INDEX search_query_chest_expiry_idx ON search_query_chest(expires_at);

-- 検索元に肺野の外周を描いた画像（「胸郭の形が似た症例」の左上に出す）。
-- 検索元の mask は保存しないので、検索の時点で 1 枚だけ描いて置く。
CREATE TABLE search_query_outline (
  search_id uuid PRIMARY KEY REFERENCES search_audit(search_id) ON DELETE CASCADE,
  content bytea NOT NULL,
  expires_at timestamptz NOT NULL
);

CREATE INDEX search_query_outline_expiry_idx ON search_query_outline(expires_at);

-- 部位単位の検索元 crop（現在の検索では書かない。期限切れの削除だけが残っている）。
CREATE TABLE search_query_region (
  search_id uuid NOT NULL REFERENCES search_audit(search_id) ON DELETE CASCADE,
  region_code varchar(3) NOT NULL CHECK (region_code IN ('RUL','RML','RLL','LUL','LLL')),
  crop_content bytea NOT NULL,
  roi_confidence real,
  quality_status text,
  expires_at timestamptz NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (search_id, region_code)
);

CREATE INDEX search_query_region_expiry_idx ON search_query_region (expires_at);


-- --- 評価 --------------------------------------------------------------------
--
-- 評価の結果は再生成できない（同じ実行は二度とできない）わりに小さいので DB に置く。
-- 学習データや中間キャッシュは置かない（元画像から再生成でき、SQL から問えない）。
-- 詳細は docs/OPERATIONS.md「評価結果の記録」。
--
-- 1 回の実行（evaluation_run）と指標（evaluation_metric）を分けるのは、同じ実行に
-- 複数の指標がぶら下がり、かつ指標をまたいで条件で絞りたいためである。

CREATE TABLE evaluation_run (
  run_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),

  --: 何を測ったか。`roi-dice` / `region-signal` / `gold-box` など
  kind text NOT NULL,

  --: 測った対象の版。ROI なら roi_model_version、検索なら embedding_model_version
  target_version text NOT NULL,

  --: 実行の条件（枚数・乱数種・閾値・参照データなど）。再現に要る情報を入れる
  params jsonb NOT NULL DEFAULT '{}',

  --: どこで走らせたか（`ec2` / `local` など）。同じ条件でも機械で差が出るため
  ran_on text,

  --: 実行したときの git の HEAD。あとから何のコードか辿れるようにする
  git_revision text,

  --: 対象の件数。指標の信頼できる幅を判断するのに要る
  sample_size integer,

  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,

  --: 人が読むための覚書
  note text
);

CREATE INDEX evaluation_run_kind_idx ON evaluation_run (kind, started_at DESC);
CREATE INDEX evaluation_run_target_idx ON evaluation_run (target_version);

CREATE TABLE evaluation_metric (
  run_id uuid NOT NULL REFERENCES evaluation_run(run_id) ON DELETE CASCADE,

  --: 指標の名前。`dice_median` / `breakdown_rate` / `gold_box_accuracy` など
  name text NOT NULL,

  --: 部位ごとの指標はここを埋める。画像レベルなら NULL
  region_code varchar(3),

  --: 所見ごとの指標はここを埋める。所見に依らないなら NULL
  finding text,

  value double precision NOT NULL,

  --: 帰無（null）の値。比較対象が無い指標では NULL
  baseline double precision,

  metric_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY
);

-- 同じ実行に同じ指標を二重に入れない。PRIMARY KEY に式は書けないので、
-- NULL を空文字へ寄せた一意索引で表す（region_code / finding は NULL を取る）。
CREATE UNIQUE INDEX evaluation_metric_unique_idx ON evaluation_metric
  (run_id, name, COALESCE(region_code, ''), COALESCE(finding, ''));

CREATE INDEX evaluation_metric_name_idx ON evaluation_metric (name);


-- --- 利用者 ------------------------------------------------------------------
--
-- パスワードは平文で保持せず、KDF（scrypt）の派生値だけを保持する。
-- メールアドレスは大小文字を区別せず一意とする。

CREATE TABLE app_user (
  user_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  username text NOT NULL UNIQUE
    CHECK (char_length(username) BETWEEN 1 AND 64),
  password_hash text NOT NULL,
  role text NOT NULL CHECK (role IN ('viewer','operator','admin')),
  email text NOT NULL CHECK (char_length(email) BETWEEN 3 AND 254),
  memo text CHECK (memo IS NULL OR char_length(memo) <= 2000),
  is_active boolean NOT NULL DEFAULT true,
  -- 発行済みアクセストークンの失効判定に使う（変更時刻より前に発行された token は無効）
  password_changed_at timestamptz NOT NULL DEFAULT now(),
  last_login_at timestamptz,
  created_by text,
  updated_by text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX app_user_email_lower_key ON app_user (lower(email));
CREATE INDEX app_user_role_idx ON app_user (role) WHERE is_active;

-- パスワード再設定 token（既定 10 分有効・1 回限り）。URL に載せる UUID そのもの
-- ではなく SHA-256 を保持し、DB を読んだだけでは有効な URL を再構成できないようにする。
CREATE TABLE password_reset_token (
  token_hash char(64) PRIMARY KEY,
  user_id uuid NOT NULL REFERENCES app_user(user_id) ON DELETE CASCADE,
  expires_at timestamptz NOT NULL,
  used_at timestamptz,
  invalidated_at timestamptz,
  requested_by_ip text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX password_reset_token_user_idx ON password_reset_token (user_id, created_at DESC);
CREATE INDEX password_reset_token_expiry_idx ON password_reset_token (expires_at)
  WHERE used_at IS NULL AND invalidated_at IS NULL;


-- --- 運用 --------------------------------------------------------------------

-- 日次の突合（scripts/reconcile.py）の結果。
CREATE TABLE reconciliation_report (
  report_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  summary jsonb NOT NULL DEFAULT '{}',
  issues jsonb NOT NULL DEFAULT '[]'
);

-- 監査ログ（ログイン・検索・原画像閲覧・取り込み・設定変更）。
CREATE TABLE audit_log (
  audit_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  actor text NOT NULL,
  action text NOT NULL,
  target_type text,
  target_id text,
  trace_id text,
  detail jsonb NOT NULL DEFAULT '{}',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX audit_log_created_idx ON audit_log(created_at);
CREATE INDEX audit_log_actor_idx ON audit_log(actor, created_at);


-- --- 説明 --------------------------------------------------------------------

COMMENT ON TABLE chest_asset IS
  '画像ごとの胸部野（肺野＋縦隔＋横隔膜）の mask と品質。検索の単位。';
COMMENT ON TABLE finding_estimate IS
  '胸部野の分類器による所見の推定確率。データセットのラベル（NLP 由来）とは別物で、診断ではない。';
COMMENT ON TABLE search_query_chest IS
  '検索元に胸部野の輪郭を重ねた画像。保持期間は検索元 crop と同じ。';
COMMENT ON TABLE search_query_outline IS
  '検索元画像に肺野外周を描いたもの（胸郭ランキングの「検索元」表示用）。保持期間は検索元 crop と同じ。';
COMMENT ON TABLE evaluation_run IS
  '評価の実行 1 回ぶん。何をどの条件で測ったか。結果は evaluation_metric。';
COMMENT ON COLUMN evaluation_metric.baseline IS
  '帰無（null）の値。帰無を測らずに「効いた」と言わないために置く。';


-- --- 初期データ --------------------------------------------------------------

INSERT INTO finding (code, display_name) VALUES
  ('Atelectasis',        '無気肺 (Atelectasis)'),
  ('Cardiomegaly',       '心拡大 (Cardiomegaly)'),
  ('Consolidation',      '浸潤影・硬化像 (Consolidation)'),
  ('Edema',              '肺水腫 (Edema)'),
  ('Effusion',           '胸水 (Effusion)'),
  ('Emphysema',          '肺気腫 (Emphysema)'),
  ('Fibrosis',           '線維化 (Fibrosis)'),
  ('Hernia',             'ヘルニア (Hernia)'),
  ('Infiltration',       '浸潤 (Infiltration)'),
  ('Mass',               '腫瘤影 (Mass)'),
  ('No Finding',         '所見なし (No Finding)'),
  ('Nodule',             '結節影 (Nodule)'),
  ('Pleural_Thickening', '胸膜肥厚 (Pleural Thickening)'),
  ('Pneumonia',          '肺炎 (Pneumonia)'),
  ('Pneumothorax',       '気胸 (Pneumothorax)')
ON CONFLICT (code) DO NOTHING;
