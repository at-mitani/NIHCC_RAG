"""取り込み API（設計書 9.4）。"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import uuid

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import StreamingResponse
from nihcc_rag import db, problem
from nihcc_rag.contract import (
    IngestionAccepted,
    IngestionProgress,
    IngestionRequest,
    IngestionStatus,
    ValidateRequest,
    ValidateResponse,
)

from ..deps import Operator, Viewer, audit, publisher
from ..services import ingest

_LOG = logging.getLogger("backend-api.ingestions")
router = APIRouter(prefix="/api/v1/ingestions", tags=["ingestions"])


@router.post("/validate", response_model=ValidateResponse)
def validate(request: ValidateRequest, principal: Operator, http_request: Request) -> ValidateResponse:
    outcome = ingest.validate(request, actor=principal.subject)
    audit(
        principal.subject,
        "INGEST_VALIDATE",
        target_type="validation",
        target_id=str(outcome.validation_id),
        trace_id=getattr(http_request.state, "trace_id", None),
        detail=outcome.counts,
    )
    return ValidateResponse(
        validation_id=outcome.validation_id,
        valid=outcome.valid,
        counts=outcome.counts,
        errors=outcome.errors,
        warnings=outcome.warnings,
    )


@router.post("", response_model=IngestionAccepted, status_code=202)
def start(
    request: IngestionRequest,
    principal: Operator,
    http_request: Request,
    response: Response,
    idempotency_key: str = Header(alias="Idempotency-Key"),
) -> IngestionAccepted:
    if len(idempotency_key) > 200:
        raise problem.unprocessable("IDEMPOTENCY_KEY_INVALID", "Idempotency-Key is too long")
    trace_id = getattr(http_request.state, "trace_id", str(uuid.uuid4()))
    job_id, status, existed = ingest.create_job(
        request,
        actor=principal.subject,
        idempotency_key=idempotency_key,
        trace_id=trace_id,
        publisher=publisher(),
    )
    if existed:
        response.status_code = 200
    else:
        audit(
            principal.subject,
            "INGEST_START",
            target_type="ingest_job",
            target_id=str(job_id),
            trace_id=trace_id,
            detail={"on_duplicate": request.on_duplicate},
        )
    return IngestionAccepted(
        job_id=job_id, status=status, status_url=f"/api/v1/ingestions/{job_id}"
    )


def _load_status(job_id: uuid.UUID) -> IngestionStatus:
    with db.metadata_conn() as conn:
        job = db.fetch_one(
            conn,
            "SELECT job_id, status, request, total_count, created_by, created_at, started_at, "
            "completed_at FROM ingest_job WHERE job_id = %s",
            (str(job_id),),
        )
        if job is None:
            raise problem.not_found("job_id")
        counts = db.fetch_all(
            conn,
            "SELECT status, count(*) AS count FROM ingest_item WHERE job_id = %s GROUP BY status",
            (str(job_id),),
        )
        errors = db.fetch_all(
            conn,
            "SELECT error_code, count(*) AS count FROM ingest_item "
            "WHERE job_id = %s AND error_code IS NOT NULL GROUP BY error_code ORDER BY 2 DESC LIMIT 20",
            (str(job_id),),
        )
    by_status = {row["status"]: row["count"] for row in counts}
    request_body = job["request"] or {}
    return IngestionStatus(
        job_id=job["job_id"],
        status=job["status"],
        progress=IngestionProgress(
            total=job["total_count"],
            succeeded=by_status.get("SUCCEEDED", 0),
            failed=by_status.get("FAILED", 0) + by_status.get("QUARANTINED", 0),
            pending=by_status.get("PENDING", 0),
            running=by_status.get("RUNNING", 0),
            quarantined=by_status.get("QUARANTINED", 0),
            cancelled=by_status.get("CANCELLED", 0),
        ),
        models={
            "roi": request_body.get("roi_model_version") or "auto",
            "embedding": request_body.get("embedding_model_version") or "auto",
            "preprocess": request_body.get("preprocess_version") or "auto",
        },
        created_by=job["created_by"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        completed_at=job["completed_at"],
        errors_url=f"/api/v1/ingestions/{job_id}/errors",
        error_summary=[dict(row) for row in errors],
    )


@router.get("/{job_id}", response_model=IngestionStatus)
def status(job_id: uuid.UUID, principal: Viewer) -> IngestionStatus:
    return _load_status(job_id)


@router.get("/{job_id}/events")
async def events(job_id: uuid.UUID, principal: Viewer) -> StreamingResponse:
    """SSE による進捗通知（設計書 6.2）。切断後は job_id から再表示できる。"""

    async def stream():
        last = None
        for _ in range(3600):
            current = await asyncio.to_thread(_load_status, job_id)
            payload = current.model_dump(mode="json")
            if payload != last:
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                last = payload
            if current.status in ("SUCCEEDED", "FAILED", "PARTIAL", "CANCELLED"):
                yield "event: done\ndata: {}\n\n"
                return
            await asyncio.sleep(1.0)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}/errors")
def errors(job_id: uuid.UUID, principal: Viewer) -> Response:
    """エラー行の CSV ダウンロード（設計書 6.2）。"""
    with db.metadata_conn() as conn:
        rows = db.fetch_all(
            conn,
            "SELECT source_image_name, status, attempt_count, error_code, error_detail, updated_at "
            "FROM ingest_item WHERE job_id = %s AND status IN ('FAILED','QUARANTINED') "
            "ORDER BY source_image_name",
            (str(job_id),),
        )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["source_image_name", "status", "attempt_count", "error_code", "error_detail", "updated_at"]
    )
    for row in rows:
        writer.writerow(
            [
                row["source_image_name"],
                row["status"],
                row["attempt_count"],
                row["error_code"] or "",
                (row["error_detail"] or "").replace("\n", " ")[:500],
                row["updated_at"].isoformat(),
            ]
        )
    return Response(
        buffer.getvalue().encode("utf-8-sig"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="ingest-errors-{job_id}.csv"'},
    )


@router.post("/{job_id}/retry", response_model=IngestionAccepted, status_code=202)
def retry(job_id: uuid.UUID, principal: Operator, http_request: Request) -> IngestionAccepted:
    trace_id = getattr(http_request.state, "trace_id", str(uuid.uuid4()))
    new_job_id = ingest.retry_failed(
        job_id, actor=principal.subject, trace_id=trace_id, publisher=publisher()
    )
    audit(
        principal.subject,
        "INGEST_RETRY",
        target_type="ingest_job",
        target_id=str(new_job_id),
        trace_id=trace_id,
        detail={"retry_of": str(job_id)},
    )
    return IngestionAccepted(
        job_id=new_job_id, status="PENDING", status_url=f"/api/v1/ingestions/{new_job_id}"
    )


@router.post("/{job_id}/cancel", response_model=IngestionStatus)
def cancel(job_id: uuid.UUID, principal: Operator, http_request: Request) -> IngestionStatus:
    """待機中・実行中のジョブを取り消す。

    未着手の画像は処理せず、取消の時点で処理中だった画像だけはそのまま完了させる。
    終了済み（SUCCEEDED / FAILED / PARTIAL / CANCELLED）のジョブは 409 を返す。
    """
    trace_id = getattr(http_request.state, "trace_id", str(uuid.uuid4()))
    result = ingest.cancel_job(job_id)
    audit(
        principal.subject,
        "INGEST_CANCEL",
        target_type="ingest_job",
        target_id=str(job_id),
        trace_id=trace_id,
        detail=result,
    )
    return _load_status(job_id)
