"""PostgreSQL 接続（psycopg3 + connection pool）。

- Metadata DB: 1 プール
- ベクトル用 Citus Coordinator: 1 プール
アプリケーションは superuser を使わない（app_meta / app_ingest / app_search）。
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from . import config

_LOG = logging.getLogger(__name__)

_metadata_pool: ConnectionPool | None = None
_vector_pool: ConnectionPool | None = None


def _make_pool(dsn: str, *, name: str, min_size: int, max_size: int) -> ConnectionPool:
    pool = ConnectionPool(
        conninfo=dsn,
        min_size=min_size,
        max_size=max_size,
        name=name,
        kwargs={"row_factory": dict_row},
        open=False,
        timeout=float(config.env_int("DB_POOL_TIMEOUT_SECONDS", 10)),
        max_lifetime=1800.0,
    )
    pool.open()
    return pool


def init_metadata_pool(*, role: str | None = None, application_name: str = "nihcc-rag") -> None:
    global _metadata_pool
    if _metadata_pool is not None:
        return
    _metadata_pool = _make_pool(
        config.metadata_dsn(user=role, application_name=application_name),
        name="metadata",
        min_size=config.env_int("METADATA_POOL_MIN", 1),
        max_size=config.env_int("METADATA_POOL_MAX", 10),
    )


def init_vector_pool(*, role: str = "app_search", application_name: str = "nihcc-rag") -> None:
    global _vector_pool
    if _vector_pool is not None:
        return
    _vector_pool = _make_pool(
        config.vector_dsn(role=role, application_name=application_name),
        name="vector",
        min_size=config.env_int("VECTOR_POOL_MIN", 1),
        max_size=config.env_int("VECTOR_POOL_MAX", 4),
    )


def close_all() -> None:
    global _metadata_pool, _vector_pool
    if _metadata_pool is not None:
        _metadata_pool.close()
        _metadata_pool = None
    if _vector_pool is not None:
        _vector_pool.close()
        _vector_pool = None


@contextmanager
def metadata_conn() -> Iterator[psycopg.Connection]:
    if _metadata_pool is None:
        raise RuntimeError("metadata pool is not initialised")
    with _metadata_pool.connection() as conn:
        yield conn


@contextmanager
def vector_conn() -> Iterator[psycopg.Connection]:
    if _vector_pool is None:
        raise RuntimeError("vector pool is not initialised")
    with _vector_pool.connection() as conn:
        yield conn


def fetch_all(conn: psycopg.Connection, sql: str, params: Any = None) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def fetch_one(conn: psycopg.Connection, sql: str, params: Any = None) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def execute(conn: psycopg.Connection, sql: str, params: Any = None) -> int:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.rowcount


def check_metadata() -> bool:
    try:
        with metadata_conn() as conn:
            fetch_one(conn, "SELECT 1 AS ok")
        return True
    except Exception:  # noqa: BLE001 - readiness probe
        _LOG.warning("metadata db check failed", exc_info=True)
        return False


def check_vector() -> bool:
    try:
        with vector_conn() as conn:
            fetch_one(conn, "SELECT 1 AS ok")
        return True
    except Exception:  # noqa: BLE001 - readiness probe
        _LOG.warning("vector db check failed", exc_info=True)
        return False


def to_pgvector(values: list[float]) -> str:
    """pgvector のテキスト表現。psycopg の型登録なしで INSERT / 比較に使える。"""
    return "[" + ",".join(f"{float(v):.7g}" for v in values) + "]"
