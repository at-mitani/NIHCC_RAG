"""Search Worker: 非同期検索（Prefer: respond-async）の実行（設計書 5.1 / 9.5）。

高優先度の consumer として動作し、結果を search_result へ保存する。
backend-api と同じ検索関数を使うため、同期経路と結果 schema が一致する。
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from typing import Any

from nihcc_rag import config, db
from nihcc_rag import logging as jsonlog
from nihcc_rag import problem, queueing
from nihcc_rag.contract import SearchRequest
from prometheus_client import Counter, start_http_server

# backend-api の検索実装を再利用する（同一 image で配布する）
from app.services import search as search_service  # type: ignore[import-not-found]

_LOG = logging.getLogger("search-worker")
SEARCHES = Counter("async_searches_total", "Asynchronous searches", ["outcome"])


def _load_query_image(search_id: str) -> bytes | None:
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn, "SELECT content FROM search_query_image WHERE search_id = %s", (search_id,)
        )
    return bytes(row["content"]) if row else None


def _store(search_id: str, status: str, response: Any = None, error: Any = None) -> None:
    with db.metadata_conn() as conn:
        db.execute(
            conn,
            "UPDATE search_result SET status = %s, response = %s, error = %s, updated_at = now() "
            "WHERE search_id = %s",
            (
                status,
                json.dumps(response, ensure_ascii=False, default=str) if response else None,
                json.dumps(error, ensure_ascii=False) if error else None,
                search_id,
            ),
        )


def handle(payload: dict[str, Any]) -> None:
    search_id = payload["search_id"]
    jsonlog.clear()
    jsonlog.bind(search_id=search_id, trace_id=payload.get("trace_id"))
    _store(search_id, "RUNNING")

    try:
        request = SearchRequest.model_validate(payload["request"])
        image = _load_query_image(search_id) if payload.get("has_image") else None
        result = search_service.execute_search(
            request,
            image=image,
            search_id=uuid.UUID(search_id),
            actor=payload.get("actor", "unknown"),
            query_image_name=payload.get("query_image_name"),
        )
    except problem.ProblemError as exc:
        _LOG.warning("async search failed", extra={"fields": {"code": exc.code}})
        _store(search_id, "FAILED", error={"code": exc.code, "detail": exc.detail})
        SEARCHES.labels("failed").inc()
        return
    except Exception as exc:  # noqa: BLE001 - ワーカーを落とさず失敗として記録する
        _LOG.exception("async search raised")
        _store(search_id, "FAILED", error={"code": "INTERNAL_ERROR", "detail": type(exc).__name__})
        SEARCHES.labels("error").inc()
        return

    _store(search_id, "SUCCEEDED", response=result.model_dump(mode="json"))
    search_service.persist_audit(
        uuid.UUID(search_id),
        request,
        result,
        actor=payload.get("actor", "unknown"),
        query_image_sha256=None,
    )
    SEARCHES.labels("succeeded").inc()
    _LOG.info(
        "async search completed",
        extra={"fields": {"duration_ms": result.duration_ms, "partial": result.partial}},
    )


def main() -> int:
    jsonlog.configure("search-worker")
    start_http_server(config.env_int("METRICS_PORT", 9101))
    db.init_metadata_pool(role="app_meta", application_name="search-worker")
    _LOG.info("search worker started")
    try:
        queueing.consume(
            queueing.QUEUE_SEARCH,
            handle,
            client_name="search-worker",
            prefetch=config.env_int("SEARCH_WORKER_PREFETCH", 2),
        )
    finally:
        db.close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
