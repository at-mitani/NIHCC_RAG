"""Ingest Worker: Queue から画像 1 件ずつ取り出して取り込む（設計書 5.1 / 10.1）。

複数レプリカで並列消費する。prefetch で 1 ワーカーあたりの同時処理数を制限し、
DB connection pool 上限を超えないようにする。
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from typing import Any

from nihcc_rag import config, db
from nihcc_rag import logging as jsonlog
from nihcc_rag import queueing
from prometheus_client import start_http_server

from .processing import (
    ItemError,
    Services,
    attempts_exhausted,
    process_item,
    record_failure,
    refresh_job_counters,
    verify_chest_pipeline,
    verify_vector_schema,
)

_LOG = logging.getLogger("ingest-worker")
_stop = threading.Event()


def handle(payload: dict[str, Any], services: Services) -> None:
    jsonlog.clear()
    jsonlog.bind(
        job_id=payload.get("job_id"),
        item_id=payload.get("item_id"),
        trace_id=payload.get("trace_id"),
    )
    item_id = payload.get("item_id")
    outcome = None
    try:
        outcome = process_item(payload, services)
    except ItemError as exc:
        _LOG.warning("item failed", extra={"fields": {"error_code": exc.code, "detail": str(exc)}})
        if item_id:
            record_failure(item_id, exc)
    except Exception as exc:  # noqa: BLE001 - 想定外は再配送させる（上限で DLQ）
        _LOG.exception("item raised an unexpected error")
        # 再配送上限に達した回は DLQ 行きになり誰も処理しない。RUNNING のまま
        # 残すとジョブが終了しないので、ここで FAILED にしてから DLQ へ送る。
        if item_id and attempts_exhausted(item_id):
            _LOG.error("delivery attempts exhausted; marking item as failed")
            record_failure(item_id, ItemError("DELIVERY_ATTEMPTS_EXHAUSTED", str(exc)[:500]))
        raise
    finally:
        # 取消済みジョブの読み飛ばしでは集計しない。取消時に集計済みであり、Queue に残った
        # メッセージを読み飛ばすたびに全件を数えると、排出が極端に遅くなる。
        if payload.get("job_id") and outcome != "CANCELLED":
            try:
                refresh_job_counters(payload["job_id"])
            except Exception:  # noqa: BLE001 - 集計失敗で item をロールバックさせない
                _LOG.warning("job counter refresh failed")


def main() -> int:
    jsonlog.configure("ingest-worker")
    start_http_server(config.env_int("METRICS_PORT", 9100))

    db.init_metadata_pool(role="app_meta", application_name="ingest-worker")
    db.init_vector_pool(role="app_ingest", application_name="ingest-worker")
    # **設定と schema を突き合わせてから受け始める。** 列の幅がずれていると
    # 取り込みは全件失敗するが、失敗するのは挿入時なので原因が遠い。
    verify_vector_schema()
    services = Services.create()
    # 胸部野の分類器が前提とする特徴の作り方（mask・解像度・U-Net の profile）と、
    # roi-service の構成を突き合わせる。ずれていると所見の推定が黙ってずれる。
    verify_chest_pipeline(services)

    def _shutdown(signum, _frame):  # pragma: no cover - シグナル経路
        _LOG.info("shutdown requested", extra={"fields": {"signal": signum}})
        _stop.set()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    prefetch = config.env_int("INGEST_WORKER_PREFETCH", 4)
    _LOG.info(
        "ingest worker started",
        extra={"fields": {"prefetch": prefetch, "hostname": os.uname().nodename}},
    )
    try:
        queueing.consume(
            queueing.QUEUE_INGEST,
            lambda payload: handle(payload, services),
            client_name="ingest-worker",
            prefetch=prefetch,
        )
    finally:
        services.close()
        db.close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
