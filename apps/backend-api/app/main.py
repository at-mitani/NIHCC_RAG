"""Backend API: 認証・入力検証・ジョブ発行・メタデータ取得・結果返却（設計書 5.1 / 9）。"""

from __future__ import annotations

import logging
import threading
import time

from nihcc_rag import auth, config, db, service
from nihcc_rag.contract import RESEARCH_WARNING

from .deps import close_publisher, publisher
from .routers import images, ingestions, searches, system, users
from .services import ingest
from .services import users as user_service

_LOG = logging.getLogger("backend-api")

app = service.create_app(
    "backend-api",
    title="NIHCC Chest X-ray Image Search Local RAG API",
)
app.include_router(system.router)
app.include_router(ingestions.router)
app.include_router(searches.router)
app.include_router(images.router)
app.include_router(users.router)
app.include_router(users.account_router)
app.include_router(users.public_router)

_stop = threading.Event()


def _background_loop() -> None:
    """outbox の再送と一時保存画像の削除（設計書 8.3 / 13）。"""
    interval = config.env_int("BACKGROUND_INTERVAL_SECONDS", 30)
    while not _stop.wait(interval):
        try:
            sent = ingest.publish_outbox(publisher())
            if sent:
                _LOG.info("outbox republished", extra={"fields": {"count": sent}})
        except Exception:  # noqa: BLE001 - 常駐ループを落とさない
            _LOG.warning("outbox republish failed")
        try:
            with db.metadata_conn() as conn:
                deleted = db.execute(
                    conn, "DELETE FROM search_query_image WHERE expires_at < now()"
                )
                deleted += db.execute(
                    conn, "DELETE FROM search_query_region WHERE expires_at < now()"
                )
                # 検索元まわりの成果物は同じ期限で消す（患者の画像であるため）
                deleted += db.execute(
                    conn, "DELETE FROM search_query_outline WHERE expires_at < now()"
                )
                deleted += db.execute(
                    conn, "DELETE FROM search_query_chest WHERE expires_at < now()"
                )
            if deleted:
                _LOG.info("expired query images purged", extra={"fields": {"count": deleted}})
        except Exception:  # noqa: BLE001
            _LOG.warning("query image purge failed")
        try:
            purged = user_service.purge_expired_tokens()
            if purged:
                _LOG.info("expired reset tokens purged", extra={"fields": {"count": purged}})
        except Exception:  # noqa: BLE001
            _LOG.warning("reset token purge failed")


@app.on_event("startup")
def _startup() -> None:
    db.init_metadata_pool(role="app_meta", application_name="backend-api")
    try:
        created = auth.bootstrap_users()
        if created:
            _LOG.info("initial users bootstrapped", extra={"fields": {"count": created}})
    except Exception:  # noqa: BLE001 - 既に利用者がいれば不要。失敗しても API は起動する
        _LOG.exception("user bootstrap failed")
    threading.Thread(target=_background_loop, name="background", daemon=True).start()
    _LOG.info(
        "backend api started",
        extra={
            "fields": {
                "preprocess_version": config.preprocess_version(),
                "max_upload_bytes": config.env_int("MAX_UPLOAD_BYTES", 20971520),
            }
        },
    )


@app.on_event("shutdown")
def _shutdown() -> None:
    _stop.set()
    close_publisher()
    db.close_all()


@app.get("/api/v1", tags=["system"])
def index() -> dict[str, object]:
    return {
        "name": "NIHCC chest X-ray local RAG API",
        "version": app.version,
        "notice": RESEARCH_WARNING,
        "openapi": "/openapi.json",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
