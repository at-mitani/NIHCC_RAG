"""時間切れとサービス停止を取り違えないことを固定する。

**「遅かった」と「落ちている」は対処が正反対である。** 前者は負荷か設定の見直し、
後者は死活確認から始める。文ごとタイムアウト（statement_timeout）による打ち切りを
503 DEPENDENCY_UNAVAILABLE へ丸めると、全コンテナが healthy なのに
「依存サービスが停止」と表示され、調査が見当違いの死活確認から始まってしまう。

設計書 9.7 は検索 timeout を 504 と定めている。

ベクトル用クラスタは 1 つなので、「一部の部位だけ失敗」は起きない。
失敗は HTTP 状態で返る。
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _services import aggregator  # noqa: E402

agg = aggregator()

from nihcc_rag.contract import VectorSearchRequest, VectorSearchResponse  # noqa: E402


def request() -> VectorSearchRequest:
    return VectorSearchRequest(
        embedding_model_version="m",
        preprocess_version="p",
        query={"query_vector": [0.0] * 512, "candidate_k": 10},
    )


def run_with(error: Exception | None, monkeypatch):
    """run_search を差し替えて search() を呼ぶ。DB へは触らない。"""

    def fake(_request):
        if error is not None:
            raise error
        return VectorSearchResponse(candidates=[])

    monkeypatch.setattr(agg, "run_search", fake)
    return agg.search(request())


def test_timeout_is_a_gateway_timeout(monkeypatch) -> None:
    """時間切れなら 504。動いているものを停止中と報告しない。"""
    with pytest.raises(Exception) as exc:
        run_with(agg.ClusterTimeout("QueryCanceled"), monkeypatch)
    assert exc.value.status == 504
    assert exc.value.code == "SEARCH_TIMEOUT"


def test_other_failures_are_unavailable(monkeypatch) -> None:
    with pytest.raises(Exception) as exc:
        run_with(agg.ClusterUnavailable("OperationalError"), monkeypatch)
    assert exc.value.status == 503
    assert exc.value.code == "DEPENDENCY_UNAVAILABLE"


def test_success_is_returned_as_is(monkeypatch) -> None:
    response = run_with(None, monkeypatch)
    assert response.candidates == []


# --- 分類の根拠 --------------------------------------------------------------


def test_statement_timeout_raises_query_canceled() -> None:
    """statement_timeout の打ち切りが QueryCanceled であること。

    run_search はこの型で ClusterTimeout へ振り分ける。型名の文字列照合ではなく
    例外の型で判定しているので、ここを固定しておく。
    """
    assert issubclass(psycopg.errors.QueryCanceled, psycopg.Error)
    assert psycopg.errors.QueryCanceled.__name__ == "QueryCanceled"


def test_open_breaker_is_reported_as_unavailable(monkeypatch) -> None:
    """遮断中は DB へ行かずに 503（時間切れではない）。"""
    breaker = agg.CircuitBreaker()
    monkeypatch.setattr(agg, "_breaker", breaker)
    for _ in range(agg._BREAKER_THRESHOLD):
        breaker.record_failure()
    with pytest.raises(agg.ClusterUnavailable):
        agg.run_search(request())
