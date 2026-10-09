"""認証・ヘルスチェック・管理画面向け API（設計書 6.1 / 9.1 / 9.3）。"""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Request
from nihcc_rag import auth, config, db, problem, queueing
from nihcc_rag.contract import TokenRequest, TokenResponse

from ..deps import Admin, Viewer, audit

_LOG = logging.getLogger("backend-api.system")
router = APIRouter(tags=["system"])


@router.post("/api/v1/auth/token", response_model=TokenResponse)
def token(request: TokenRequest, http_request: Request) -> TokenResponse:
    user = auth.authenticate(request.username, request.password)
    access_token, ttl = auth.issue_token(user)
    audit(
        user.username,
        "LOGIN",
        trace_id=getattr(http_request.state, "trace_id", None),
        detail={"role": user.role},
    )
    return TokenResponse(access_token=access_token, expires_in=ttl, role=user.role)


@router.get("/api/v1/auth/me")
def me(principal: Viewer) -> dict[str, str | None]:
    return {
        "subject": principal.subject,
        "role": principal.role,
        "user_id": principal.user_id,
        "email": principal.email,
    }


@router.get("/api/v1/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/api/v1/health/ready")
def ready() -> dict[str, object]:
    checks: dict[str, object] = {
        "metadata_db": db.check_metadata(),
        "queue": queueing.check_ready(),
    }
    for name, url in (
        ("roi_service", config.roi_service_url()),
        ("embedding_service", config.embedding_service_url()),
        ("search_aggregator", config.search_aggregator_url()),
    ):
        try:
            with httpx.Client(base_url=url, timeout=3.0) as client:
                checks[name] = client.get("/api/v1/health/ready").status_code == 200
        except Exception:  # noqa: BLE001 - readiness probe
            checks[name] = False
    ok = all(bool(value) for value in checks.values())
    if not ok:
        raise problem.ProblemError(
            503,
            "NOT_READY",
            "Service unavailable",
            "One or more dependencies are not ready.",
            errors=[{"field": key, "reason": "not ready"} for key, value in checks.items() if not value],
        )
    return {"status": "ok", "checks": checks}


@router.get("/api/v1/admin/status")
def admin_status(principal: Viewer) -> dict[str, object]:
    """管理画面: モデル版・DB 状態・worker 状態・index 状態（設計書 6.1）。"""
    with db.metadata_conn() as conn:
        models = db.fetch_all(
            conn,
            "SELECT model_version, model_type, model_name, source_revision, embedding_dimension, "
            "active, created_at FROM model_version ORDER BY model_type, created_at DESC",
        )
        images = db.fetch_one(conn, "SELECT count(*) AS count FROM source_image")
        chest = db.fetch_one(
            conn,
            "SELECT count(*) AS count, avg(roi_confidence)::numeric(4,3) AS avg_confidence, "
            "count(*) FILTER (WHERE quality_status = 'ROI_UNCERTAIN') AS uncertain "
            "FROM chest_asset",
        )
        vectors = db.fetch_all(
            conn,
            "SELECT state, count(*) AS count FROM image_vector_state "
            "GROUP BY state ORDER BY state",
        )
        estimates = db.fetch_one(
            conn, "SELECT count(*) AS count, count(DISTINCT model_version) AS models "
            "FROM finding_estimate",
        )
        jobs = db.fetch_all(
            conn,
            "SELECT job_id, status, total_count, succeeded_count, failed_count, created_at "
            "FROM ingest_job ORDER BY created_at DESC LIMIT 10",
        )

    cluster: dict[str, object] = {}
    try:
        with httpx.Client(base_url=config.search_aggregator_url(), timeout=5.0) as client:
            cluster = client.get("/api/v1/vector/status").json().get("cluster", {})
    except Exception:  # noqa: BLE001 - 管理画面は縮退表示する
        cluster = {"available": False}

    queue_depth = {}
    for queue in (queueing.QUEUE_INGEST, queueing.QUEUE_SEARCH, queueing.QUEUE_INGEST_DLQ):
        try:
            queue_depth[queue] = queueing.queue_depth(queue, client_name="backend-api-status")
        except Exception:  # noqa: BLE001
            queue_depth[queue] = -1

    return {
        # 検索の単位
        "unit": "chest",
        "models": [dict(row) for row in models],
        "image_count": images["count"] if images else 0,
        "chest_assets": dict(chest) if chest else {},
        "vector_states": [dict(row) for row in vectors],
        "finding_estimates": dict(estimates) if estimates else {},
        "recent_jobs": [dict(row) for row in jobs],
        "cluster": {
            "coordinator": config.env("VECTOR_HOST", "vector-coordinator"),
            "database": config.env("VECTOR_DB", "vector"),
            **cluster,
        },
        "queue_depth": queue_depth,
        "preprocess_version": config.preprocess_version(),
    }


@router.post("/api/v1/admin/models/{model_version}/activate")
def activate_model(model_version: str, principal: Admin, request: Request) -> dict[str, object]:
    """active_model_version の原子的切替（設計書 8.3）。"""
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn, "SELECT model_type FROM model_version WHERE model_version = %s", (model_version,)
        )
        if row is None:
            raise problem.not_found("model_version")
        with conn.transaction():
            db.execute(
                conn,
                "UPDATE model_version SET active = false WHERE model_type = %s AND active",
                (row["model_type"],),
            )
            db.execute(
                conn,
                "UPDATE model_version SET active = true WHERE model_version = %s",
                (model_version,),
            )
    audit(
        principal.subject,
        "MODEL_ACTIVATE",
        target_type="model_version",
        target_id=model_version,
        trace_id=getattr(request.state, "trace_id", None),
        detail={"model_type": row["model_type"]},
    )
    return {"model_version": model_version, "model_type": row["model_type"], "active": True}
