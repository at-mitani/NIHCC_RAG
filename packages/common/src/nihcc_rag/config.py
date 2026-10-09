"""環境変数と secret file からの設定読み込み。

秘密情報は Git にも image にも置かず、Docker secret（/run/secrets/*）または
環境変数から読む。設計書 11.3。
"""

from __future__ import annotations

import functools
import os
from pathlib import Path


def env(name: str, default: str | None = None) -> str:
    value = os.environ.get(name, default)
    if value is None:
        raise RuntimeError(f"required environment variable is not set: {name}")
    return value


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def optional_secret(name: str, *, file_env: str | None = None) -> str | None:
    """`<NAME>_FILE` があればファイルから、無ければ `<NAME>` から読む。未設定なら None。"""
    file_env = file_env or f"{name}_FILE"
    path = os.environ.get(file_env)
    if path:
        return Path(path).read_text(encoding="utf-8").strip()
    return os.environ.get(name) or None


def secret(name: str, *, file_env: str | None = None, default: str | None = None) -> str:
    value = optional_secret(name, file_env=file_env)
    if value is None:
        value = default
    if value is None:
        raise RuntimeError(
            f"required secret is not set: {name} / {file_env or f'{name}_FILE'}"
        )
    return value


@functools.lru_cache(maxsize=1)
def db_password() -> str:
    return secret("POSTGRES_PASSWORD", default=None)


@functools.lru_cache(maxsize=1)
def jwt_secret() -> str:
    return secret("JWT_SECRET", default=None)


def metadata_dsn(*, user: str | None = None, application_name: str = "nihcc-rag") -> str:
    user = user or env("METADATA_DB_USER", "app_meta")
    host = env("METADATA_DB_HOST", "metadata-db")
    port = env("METADATA_DB_PORT", "5432")
    database = env("METADATA_DB_NAME", "metadata")
    password = optional_secret("METADATA_DB_PASSWORD") or db_password()
    return (
        f"host={host} port={port} dbname={database} user={user} password={password} "
        f"application_name={application_name} connect_timeout=10"
    )


def vector_dsn(*, role: str = "app_search", application_name: str = "nihcc-rag") -> str:
    """ベクトル用 Citus Coordinator への DSN。Worker へは直接接続しない（設計書 11.2）。

    ベクトル用クラスタは 1 つ（1 Coordinator + 5 Worker）。
    """
    host = env("VECTOR_HOST", "vector-coordinator")
    port = env("VECTOR_PORT", "5432")
    database = env("VECTOR_DB", "vector")
    password = optional_secret("VECTOR_DB_PASSWORD") or db_password()
    return (
        f"host={host} port={port} dbname={database} user={role} password={password} "
        f"application_name={application_name} connect_timeout=10"
    )


# --- モデル / 処理 version ---------------------------------------------------


def preprocess_version() -> str:
    return env("PREPROCESS_VERSION", "prep-v1.0.0")


# --- 2 段ベクトル検索（coarse -> fine）--------------------------------------
#
# 1 段目は低次元 coarse ベクトルの HNSW（近似）で候補を広く取り、2 段目は
# 高次元 fine ベクトルの厳密距離で並べ直す。fine ベクトルを生成するかどうかは
# embedding-service の EMBEDDING_FINE_BACKEND だけが決める（単一の真実）。
# backend-api / search-aggregator は「fine ベクトルが返ってきたか」で判断する。


def embedding_dimension() -> int:
    """1 段目（coarse）ベクトルの次元。image_embedding.embedding の幅と一致すること。"""
    return env_int("EMBEDDING_DIMENSION", 512)


def embedding_fine_dimension() -> int:
    """2 段目（fine）ベクトルの次元。image_embedding.embedding_fine の幅と一致すること。"""
    return env_int("EMBEDDING_FINE_DIMENSION", 15)


def rerank_stage1_multiplier() -> int:
    """1 段目で candidate_k の何倍まで多めに取るか。

    再ランキングの効果は 1 段目の recall が上限になる。倍率を上げるほど最終順位は
    厳密解に近づくが、1 段目の HNSW 探索と 2 段目の距離計算がその分増える。
    """
    return max(1, env_int("VECTOR_STAGE1_MULTIPLIER", 4))


def rerank_stage1_max() -> int:
    """1 段目の取得件数の上限。

    pgvector の hnsw.ef_search の上限が 1000 であり、1 段目の LIMIT が
    ef_search を超えると HNSW はその件数を返せない（iterative_scan 無効時）。
    既定はこの上限に合わせる。
    """
    return max(1, env_int("VECTOR_STAGE1_MAX_LIMIT", 1000))


IMAGE_SEARCH_STRATEGIES = ("profile", "coarse_to_fine")


def image_search_strategy() -> str:
    """画像だけの検索の既定の進め方。

    profile: 所見プロファイル（15 次元）で全行を厳密に検索する（既定）。
    coarse_to_fine: BiomedCLIP の HNSW で候補を集め、所見プロファイルで並べ替える。
    綴りの誤りを黙って既定へ倒すと、切り戻したつもりで切り戻っていないことになるので止める。
    """
    value = os.environ.get("IMAGE_SEARCH_STRATEGY", "").strip() or "profile"
    if value not in IMAGE_SEARCH_STRATEGIES:
        raise RuntimeError(
            f"IMAGE_SEARCH_STRATEGY must be one of {IMAGE_SEARCH_STRATEGIES}: {value!r}"
        )
    return value


def roi_service_url() -> str:
    return env("ROI_SERVICE_URL", "http://roi-service:8000").rstrip("/")


def embedding_service_url() -> str:
    return env("EMBEDDING_SERVICE_URL", "http://embedding-service:8000").rstrip("/")


def search_aggregator_url() -> str:
    return env("SEARCH_AGGREGATOR_URL", "http://search-aggregator:8000").rstrip("/")


def amqp_url() -> str:
    host = env("RABBITMQ_HOST", "queue")
    port = env("RABBITMQ_PORT", "5672")
    user = env("RABBITMQ_USER", "nihcc")
    password = optional_secret("RABBITMQ_PASSWORD") or db_password()
    vhost = env("RABBITMQ_VHOST", "/")
    quoted = vhost if vhost != "/" else "%2f"
    return f"amqp://{user}:{password}@{host}:{port}/{quoted}"
