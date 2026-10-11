"""RFC 9457 `application/problem+json` エラー応答（設計書 9.7）。

内部スタック・ファイルパスは露出させない。

**`detail` は英語で書く。** 画面は表示言語を切り替えられるので、利用者向けの文言は
画面の辞書（apps/frontend/static/i18n/*.json の `server.<message_id>`）で訳す。
応答には訳すための `message_id`（既定は `code`）と、文言へ差し込む値の `params` を
付ける。辞書に無い message_id は英語の `detail` がそのまま出る。
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

_LOG = logging.getLogger(__name__)

PROBLEM_BASE = "https://local.example/problems"
CONTENT_TYPE = "application/problem+json"


class ProblemError(Exception):
    """API から返す業務エラー。"""

    def __init__(
        self,
        status: int,
        code: str,
        title: str,
        detail: str,
        *,
        problem_type: str | None = None,
        errors: list[dict[str, Any]] | None = None,
        headers: dict[str, str] | None = None,
        message_id: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        self.problem_type = problem_type or f"{PROBLEM_BASE}/{code.lower().replace('_', '-')}"
        self.errors = errors or []
        self.headers = headers or {}
        #: 画面で訳すときの鍵。同じ code で文言が違う場合（401 の理由など）に分ける
        self.message_id = message_id or code
        self.params = params or {}


def problem_response(request: Request, exc: ProblemError) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", None) or str(uuid.uuid4())
    body: dict[str, Any] = {
        "type": exc.problem_type,
        "title": exc.title,
        "status": exc.status,
        "detail": exc.detail,
        "instance": request.url.path,
        "code": exc.code,
        "message_id": exc.message_id,
        "trace_id": trace_id,
    }
    if exc.params:
        body["params"] = exc.params
    if exc.errors:
        body["errors"] = exc.errors
    headers = dict(exc.headers)
    headers["X-Request-ID"] = trace_id
    return JSONResponse(
        status_code=exc.status, content=body, media_type=CONTENT_TYPE, headers=headers
    )


def install(app: FastAPI) -> None:
    @app.exception_handler(ProblemError)
    async def _problem_handler(request: Request, exc: ProblemError) -> JSONResponse:
        if exc.status >= 500:
            _LOG.error("request failed", extra={"fields": {"code": exc.code}})
        return problem_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {
                "field": ".".join(str(part) for part in err.get("loc", [])[1:]) or "body",
                "reason": err.get("msg", "invalid"),
            }
            for err in exc.errors()
        ][:20]
        return problem_response(
            request,
            ProblemError(
                422,
                "REQUEST_VALIDATION_FAILED",
                "Invalid request",
                "Request body or parameters failed validation.",
                errors=errors,
            ),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        _LOG.exception("unhandled error")
        return problem_response(
            request,
            ProblemError(
                500,
                "INTERNAL_ERROR",
                "Internal server error",
                "An unexpected error occurred. Contact the operator with the trace_id.",
            ),
        )


# --- よく使うエラー ----------------------------------------------------------


def not_found(resource: str) -> ProblemError:
    return ProblemError(
        404, "RESOURCE_NOT_FOUND", "Not found", f"{resource} was not found.",
        params={"resource": resource},
    )


def unauthorized(
    detail: str = "Authentication is required.", *, message_id: str = "UNAUTHENTICATED"
) -> ProblemError:
    """401。code は常に UNAUTHENTICATED。理由ごとの文言は message_id で分ける。"""
    return ProblemError(
        401,
        "UNAUTHENTICATED",
        "Unauthorized",
        detail,
        headers={"WWW-Authenticate": "Bearer"},
        message_id=message_id,
    )


def forbidden(required_role: str) -> ProblemError:
    return ProblemError(
        403,
        "FORBIDDEN",
        "Forbidden",
        f"Role '{required_role}' is required for this operation.",
        params={"role": required_role},
    )


def unprocessable(
    code: str,
    detail: str,
    errors: list[dict[str, Any]] | None = None,
    *,
    params: dict[str, Any] | None = None,
    message_id: str | None = None,
):
    return ProblemError(
        422, code, "Invalid request", detail, errors=errors, params=params, message_id=message_id
    )


def dependency_unavailable(name: str) -> ProblemError:
    return ProblemError(
        503,
        "DEPENDENCY_UNAVAILABLE",
        "Service unavailable",
        f"Dependent service '{name}' is unavailable.",
        params={"name": name},
    )


def gateway_timeout(name: str) -> ProblemError:
    return ProblemError(
        504, "SEARCH_TIMEOUT", "Gateway timeout", f"'{name}' did not respond in time.",
        params={"name": name},
    )
