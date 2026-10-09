"""RFC 9457 `application/problem+json` エラー応答（設計書 9.7）。

内部スタック・ファイルパスは露出させない。
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
    ) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.title = title
        self.detail = detail
        self.problem_type = problem_type or f"{PROBLEM_BASE}/{code.lower().replace('_', '-')}"
        self.errors = errors or []
        self.headers = headers or {}


def problem_response(request: Request, exc: ProblemError) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", None) or str(uuid.uuid4())
    body: dict[str, Any] = {
        "type": exc.problem_type,
        "title": exc.title,
        "status": exc.status,
        "detail": exc.detail,
        "instance": request.url.path,
        "code": exc.code,
        "trace_id": trace_id,
    }
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
    return ProblemError(404, "RESOURCE_NOT_FOUND", "Not found", f"{resource} was not found.")


def unauthorized(detail: str = "Authentication is required.") -> ProblemError:
    return ProblemError(
        401,
        "UNAUTHENTICATED",
        "Unauthorized",
        detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def forbidden(required_role: str) -> ProblemError:
    return ProblemError(
        403,
        "FORBIDDEN",
        "Forbidden",
        f"Role '{required_role}' is required for this operation.",
    )


def unprocessable(code: str, detail: str, errors: list[dict[str, Any]] | None = None):
    return ProblemError(422, code, "Invalid request", detail, errors=errors)


def dependency_unavailable(name: str) -> ProblemError:
    return ProblemError(
        503,
        "DEPENDENCY_UNAVAILABLE",
        "Service unavailable",
        f"Dependent service '{name}' is unavailable.",
    )


def gateway_timeout(name: str) -> ProblemError:
    return ProblemError(
        504, "SEARCH_TIMEOUT", "Gateway timeout", f"'{name}' did not respond in time."
    )
