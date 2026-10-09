"""全 HTTP サービス共通の FastAPI 生成（trace_id、metrics、problem+json）。"""

from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from . import logging as jsonlog
from . import problem

REQUESTS = Counter(
    "http_requests_total", "HTTP requests", ["service", "method", "path", "status"]
)
LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["service", "method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 30),
)


def create_app(service: str, *, title: str, version: str = "1.0.0") -> FastAPI:
    jsonlog.configure(service)
    app = FastAPI(
        title=title,
        version=version,
        docs_url="/docs",
        openapi_url="/openapi.json",
    )
    problem.install(app)

    @app.middleware("http")
    async def _context_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        trace_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.trace_id = trace_id
        jsonlog.clear()
        jsonlog.bind(trace_id=trace_id)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed = time.perf_counter() - started
            route = request.scope.get("route")
            path = getattr(route, "path", request.url.path)
            LATENCY.labels(service, request.method, path).observe(elapsed)
        response.headers["X-Request-ID"] = trace_id
        REQUESTS.labels(service, request.method, path, str(response.status_code)).inc()
        return response

    @app.get("/metrics", include_in_schema=False)
    async def _metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
