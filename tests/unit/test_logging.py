"""JSON 構造化ログの単体テスト（設計書 14.2）。"""

from __future__ import annotations

import json
import logging
import sys

from nihcc_rag.logging import JsonFormatter


def _format_exception(exc: BaseException) -> dict:
    try:
        raise exc
    except BaseException:
        record = logging.LogRecord(
            "test", logging.ERROR, __file__, 1, "item raised an unexpected error", None, sys.exc_info()
        )
    return json.loads(JsonFormatter("test").format(record))


def test_error_keeps_first_line_of_multiline_message() -> None:
    """複数行のメッセージでは本文（1 行目）を残す。

    psycopg のエラーは最後の行が位置を示す "^" だけのため、最後の行を残すと原因が
    分からなくなる。
    """
    message = (
        'invalid input value for enum item_status: "CANCELLED"\n'
        "LINE 1: UPDATE ingest_item SET status = 'CANCELLED'\n"
        "                                        ^"
    )
    payload = _format_exception(ValueError(message))
    assert payload["error"] == 'ValueError: invalid input value for enum item_status: "CANCELLED"'


def test_error_without_message_uses_type_name() -> None:
    payload = _format_exception(RuntimeError())
    assert payload["error"] == "RuntimeError"


def test_error_is_truncated() -> None:
    payload = _format_exception(ValueError("x" * 2000))
    assert payload["error"].startswith("ValueError: x")
    assert len(payload["error"]) <= 500
