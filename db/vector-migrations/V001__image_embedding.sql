-- =============================================================================
-- ベクトル用 Citus クラスタのスキーマ（胸部野 1 単位。deviations §10.36）
--
-- 1 行 = 1 画像。部位ごとに 5 つあったクラスタ（region_embedding）は、
-- 1 クラスタ（1 Coordinator + 5 Worker）の image_embedding にまとめた。
-- 旧クラスタの migration（region_embedding の V001〜V006）は git の履歴にある。
--
-- ${SHARD_COUNT} / ${EMBEDDING_DIMENSION} / ${EMBEDDING_FINE_DIMENSION} /
-- ${EMBEDDING_LUNG_DIMENSION} は migration job が置換する。
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS citus;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS image_embedding (
  image_id uuid NOT NULL,
  embedding_model_version text NOT NULL,
  preprocess_version text NOT NULL,
  roi_model_version text NOT NULL,
  roi_confidence real NOT NULL CHECK (roi_confidence BETWEEN 0 AND 1),
  embedding vector(${EMBEDDING_DIMENSION}) NOT NULL,
  embedding_fine vector(${EMBEDDING_FINE_DIMENSION}),
  embedding_fine_model_version text,
  embedding_lung vector(${EMBEDDING_LUNG_DIMENSION}),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (image_id, embedding_model_version, preprocess_version)
);

SELECT create_distributed_table(
  'image_embedding', 'image_id',
  distribution_type => 'hash', shard_count => ${SHARD_COUNT}
);

CREATE INDEX IF NOT EXISTS image_embedding_hnsw_cos_idx
  ON image_embedding USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 128);

CREATE INDEX IF NOT EXISTS image_embedding_model_idx
  ON image_embedding (embedding_model_version, preprocess_version);
