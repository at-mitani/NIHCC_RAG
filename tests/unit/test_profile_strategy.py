"""画像だけの検索を所見プロファイルの全件検索にすることを固定する。

2 段検索は 1 段目（BiomedCLIP）が拾った候補しか並べ替えられず、所見の合う画像を
取りこぼす。そのため画像だけの検索は既定で embedding_fine の全件厳密検索にする。
- テキストを含む検索・rerank=false では使わない（テキストの意味が順位から消える）
- 綴りを誤った設定は黙って既定へ倒さない（切り戻したつもりで戻っていない事故を防ぐ）
- aggregator は 1 段目の HNSW を使わず、fine の 1 文だけを投げる
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _services import aggregator  # noqa: E402

from nihcc_rag import config, rerank  # noqa: E402
from nihcc_rag.contract import SearchRequest, VectorSearchRequest  # noqa: E402

agg = aggregator()


# --- 進め方の決定 -------------------------------------------------------------


@pytest.mark.parametrize(
    ("requested", "default", "rerank_active", "has_text", "expected"),
    [
        ("auto", "profile", True, False, rerank.STRATEGY_PROFILE),
        ("auto", "coarse_to_fine", True, False, rerank.STRATEGY_COARSE_TO_FINE),
        ("coarse_to_fine", "profile", True, False, rerank.STRATEGY_COARSE_TO_FINE),
        ("profile", "coarse_to_fine", True, False, rerank.STRATEGY_PROFILE),
        # テキストを含む検索では全件検索に使わない
        ("profile", "profile", True, True, rerank.STRATEGY_COARSE_TO_FINE),
        # rerank=false / fine 未構成なら BiomedCLIP だけ
        ("profile", "profile", False, False, rerank.STRATEGY_COARSE_ONLY),
    ],
)
def test_resolve_strategy(requested, default, rerank_active, has_text, expected) -> None:
    assert (
        rerank.resolve_strategy(
            requested, default=default, rerank_active=rerank_active, has_text=has_text
        )
        == expected
    )


def test_every_strategy_has_a_model_label() -> None:
    for strategy in (
        rerank.STRATEGY_PROFILE,
        rerank.STRATEGY_COARSE_TO_FINE,
        rerank.STRATEGY_COARSE_ONLY,
    ):
        assert rerank.STRATEGY_MODEL_LABELS[strategy]


def test_strategy_setting_defaults_to_profile(monkeypatch) -> None:
    monkeypatch.delenv("IMAGE_SEARCH_STRATEGY", raising=False)
    assert config.image_search_strategy() == "profile"
    monkeypatch.setenv("IMAGE_SEARCH_STRATEGY", "coarse_to_fine")
    assert config.image_search_strategy() == "coarse_to_fine"


def test_misspelled_strategy_setting_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("IMAGE_SEARCH_STRATEGY", "coarse-to-fine")
    with pytest.raises(RuntimeError):
        config.image_search_strategy()


def test_search_request_accepts_strategy() -> None:
    assert SearchRequest().strategy == "auto"
    assert SearchRequest(strategy="coarse_to_fine").strategy == "coarse_to_fine"
    with pytest.raises(ValidationError):
        SearchRequest(strategy="hnsw")


# --- 候補列 -------------------------------------------------------------------


def test_profile_exact_orders_by_fine_distance() -> None:
    ranked, status = rerank.profile_exact(
        [
            rerank.FineCandidate("b", 0.30, 0.9),
            rerank.FineCandidate("a", 0.10, 0.8),
            rerank.FineCandidate("c", 0.20, 0.7),
        ],
        limit=2,
        min_similarity=-1.0,
    )
    assert status == rerank.RERANK_PROFILE_EXACT
    assert [item.image_id for item in ranked] == ["a", "c"]
    assert ranked[0].similarity == pytest.approx(0.9)
    # coarse は計算していないので None（0 や番兵を入れると画面に偽の数値が出る）
    assert ranked[0].coarse_distance is None and ranked[0].coarse_similarity is None
    assert not ranked[0].reranked


def test_profile_exact_applies_min_similarity_and_reports_empty() -> None:
    ranked, status = rerank.profile_exact(
        [rerank.FineCandidate("a", 0.6, 0.9)], limit=5, min_similarity=0.5
    )
    assert ranked == [] and status == rerank.RERANK_PROFILE_EXACT
    assert rerank.profile_exact([], limit=5, min_similarity=-1.0)[1] == rerank.RERANK_NO_CANDIDATES


# --- aggregator ---------------------------------------------------------------


def fine_request(**overrides) -> VectorSearchRequest:
    body = {
        "embedding_model_version": "m",
        "preprocess_version": "p",
        "fine_embedding_model_version": "chest-x-d15",
        "stage1": "fine",
        "query": {"query_vector": [0.0] * 512, "query_vector_fine": [0.1] * 15, "candidate_k": 3},
    }
    body.update(overrides)
    return VectorSearchRequest(**body)


def test_fine_stage1_needs_the_fine_query() -> None:
    with pytest.raises(ValidationError):
        fine_request(fine_embedding_model_version=None)
    with pytest.raises(ValidationError):
        fine_request(query={"query_vector": [0.0] * 512, "candidate_k": 3})
    assert VectorSearchRequest(
        embedding_model_version="m", preprocess_version="p", query={"query_vector": [0.0]}
    ).stage1 == "embedding"


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.sql: list[tuple[str, object]] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.sql.append((sql, params))

    def fetchall(self):
        return self.rows


class _Conn:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def transaction(self):
        return self

    def cursor(self):
        return self._cursor


def test_fine_stage1_runs_one_exact_query(monkeypatch) -> None:
    ids = [uuid.uuid4() for _ in range(3)]
    cursor = _Cursor([
        {"image_id": ids[1], "fine_distance": 0.05, "roi_confidence": 0.9},
        {"image_id": ids[0], "fine_distance": 0.02, "roi_confidence": 0.8},
        {"image_id": ids[2], "fine_distance": 0.40, "roi_confidence": 0.7},
    ])
    monkeypatch.setattr(agg.db, "vector_conn", lambda: _Conn(cursor))
    monkeypatch.setattr(agg, "_breaker", agg.CircuitBreaker())

    response = agg.run_search(fine_request())

    queries = [sql for sql, _ in cursor.sql if "SELECT" in sql]
    assert queries == [agg._FINE_EXACT_SQL]          # HNSW の 1 段目も 2 段目も投げない
    params = cursor.sql[-1][1]
    assert params["fine_model"] == "chest-x-d15"     # 同次元の別モデルと突き合わせない
    assert params["limit"] == 3
    assert not any("hnsw" in sql for sql, _ in cursor.sql)
    assert response.rerank_status == rerank.RERANK_PROFILE_EXACT
    assert [c.image_id for c in response.candidates] == [ids[0], ids[1], ids[2]]
    assert response.candidates[0].coarse_similarity is None
    assert response.stage1_count == 3 and response.reranked_count == 0


def test_fine_stage1_failure_is_classified(monkeypatch) -> None:
    import psycopg

    class _Failing(_Cursor):
        def execute(self, sql, params=None):
            if "SELECT" in sql:
                raise psycopg.errors.QueryCanceled("canceling statement due to statement timeout")

    monkeypatch.setattr(agg.db, "vector_conn", lambda: _Conn(_Failing([])))
    monkeypatch.setattr(agg, "_breaker", agg.CircuitBreaker())
    with pytest.raises(agg.ClusterTimeout):
        agg.run_search(fine_request())
