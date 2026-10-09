"""JSON 構造化ログ（設計書 14.2）。

trace_id / job_id / search_id / image_id / region_code / model_version を相関可能にする。
画像内容・患者識別子・token・DB credential は出力しない。
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
import time
from typing import Any

_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

#: ログへ出してはならない key（設計書 13）
_FORBIDDEN_KEYS = {
    "password",
    "authorization",
    "token",
    "secret",
    "image_bytes",
    "content",
    "embedding",
    "patient_id",
    "source_patient_key",
}


def _describe_exception(exc_info: Any) -> str:
    """例外を「型名: メッセージの 1 行目」で表す（最大 500 文字）。

    メッセージの最後の行だけを残すと、psycopg のエラーのように位置を示す "^" だけが
    残って原因が分からない。
    """
    exc_type, exc, _ = exc_info if isinstance(exc_info, tuple) else (None, None, None)
    name = exc_type.__name__ if exc_type is not None else "Exception"
    text = str(exc).strip() if exc is not None else ""
    first = text.splitlines()[0].strip() if text else ""
    return (f"{name}: {first}" if first else name)[:500]


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self.service = service

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "service": self.service,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(_context.get())
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update({k: v for k, v in extra.items() if k not in _FORBIDDEN_KEYS})
        if record.exc_info:
            payload["error"] = _describe_exception(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure(service: str) -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter(service))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("uvicorn.access", "pika", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))


def bind(**fields: Any) -> None:
    """以降のログへ付与する相関 ID を設定する。"""
    current = dict(_context.get())
    current.update({k: v for k, v in fields.items() if v is not None})
    _context.set(current)


def clear() -> None:
    _context.set({})


def get_context() -> dict[str, Any]:
    return dict(_context.get())


def log(logger: logging.Logger, level: int, message: str, **fields: Any) -> None:
    logger.log(level, message, extra={"fields": fields})
