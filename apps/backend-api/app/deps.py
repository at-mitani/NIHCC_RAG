"""認証・RBAC・監査の共通依存（設計書 9.3 / 13）。"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from fastapi import Depends, Request
from nihcc_rag import auth, db, problem, queueing

_LOG = logging.getLogger("backend-api.deps")

_publisher: queueing.Publisher | None = None


def publisher() -> queueing.Publisher:
    global _publisher
    if _publisher is None:
        _publisher = queueing.Publisher(client_name="backend-api")
    return _publisher


def close_publisher() -> None:
    global _publisher
    if _publisher is not None:
        _publisher.close()
        _publisher = None


def current_principal(request: Request) -> auth.Principal:
    header = request.headers.get("Authorization", "")
    if not header.lower().startswith("bearer "):
        raise problem.unauthorized()
    return auth.verify_token(header[7:].strip())


def require_role(role: str):
    def dependency(principal: Annotated[auth.Principal, Depends(current_principal)]) -> auth.Principal:
        if not principal.has_role(role):
            raise problem.forbidden(role)
        return principal

    return dependency


Viewer = Annotated[auth.Principal, Depends(require_role("viewer"))]
Operator = Annotated[auth.Principal, Depends(require_role("operator"))]
Admin = Annotated[auth.Principal, Depends(require_role("admin"))]


def audit(
    actor: str,
    action: str,
    *,
    target_type: str | None = None,
    target_id: str | None = None,
    trace_id: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """監査対象: ログイン、検索、原画像閲覧、取り込み、設定・モデル変更。"""
    try:
        with db.metadata_conn() as conn:
            db.execute(
                conn,
                "INSERT INTO audit_log (actor, action, target_type, target_id, trace_id, detail) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    actor,
                    action,
                    target_type,
                    target_id,
                    trace_id,
                    json.dumps(detail or {}, ensure_ascii=False, default=str),
                ),
            )
    except Exception:  # noqa: BLE001 - 監査失敗で業務処理を止めない（警告として記録）
        _LOG.warning("audit log write failed", extra={"fields": {"action": action}})
