"""2 段ベクトル検索（coarse -> fine）の候補統合の単体テスト。

search-aggregator の SQL は DB が必要だが、順位づけの規則はここで固定する。
- 最終順位は fine 距離で決まること
- fine 未充填の行が混在しても壊れず、1 段目の順位で後方に残ること
- min_similarity は最終順位に使った値へ適用されること（1 段目で切らないこと）
"""

from __future__ import annotations

import pytest
from nihcc_rag import rerank
from nihcc_rag.rerank import Stage1Candidate


def _stage1(*pairs: tuple[str, float]) -> list[Stage1Candidate]:
    """(image_id, coarse_distance) から 1 段目の候補列を作る。"""
    return [
        Stage1Candidate(image_id=image_id, coarse_distance=distance, roi_confidence=0.9)
        for image_id, distance in pairs
    ]


# --- 1 段目の取得件数 --------------------------------------------------------


def test_stage1_limit_multiplies_candidate_k() -> None:
    assert rerank.stage1_limit(50, multiplier=4, max_limit=1000) == 200


def test_stage1_limit_is_capped() -> None:
    assert rerank.stage1_limit(500, multiplier=4, max_limit=1000) == 1000


def test_stage1_limit_never_below_candidate_k() -> None:
    # 上限が candidate_k より小さくても candidate_k は下回らせない。
    # 下回ると再ランキングの有無で返却件数が変わってしまう。
    assert rerank.stage1_limit(300, multiplier=4, max_limit=100) == 300
    assert rerank.stage1_limit(50, multiplier=1, max_limit=1000) == 50


def test_stage1_limit_treats_zero_multiplier_as_one() -> None:
    assert rerank.stage1_limit(50, multiplier=0, max_limit=1000) == 50


# --- 再ランキングの順位 ------------------------------------------------------


def test_fine_distance_decides_the_final_order() -> None:
    # 1 段目は a, b, c の順。fine では順序が c, a, b へ入れ替わる。
    candidates = _stage1(("a", 0.10), ("b", 0.20), ("c", 0.30))
    ranked, status = rerank.merge(
        candidates,
        {"a": 0.50, "b": 0.90, "c": 0.10},
        limit=10,
        min_similarity=-1.0,
    )
    assert [item.image_id for item in ranked] == ["c", "a", "b"]
    assert status == rerank.RERANK_APPLIED
    assert all(item.reranked for item in ranked)


def test_both_similarities_are_reported() -> None:
    ranked, _ = rerank.merge(
        _stage1(("a", 0.25)), {"a": 0.75}, limit=10, min_similarity=-1.0
    )
    item = ranked[0]
    # 最終順位に使った値は fine 空間
    assert item.distance == pytest.approx(0.75)
    assert item.similarity == pytest.approx(0.25)
    # 1 段目の値も残る（2 段目の効き方を追えるようにするため）
    assert item.coarse_distance == pytest.approx(0.25)
    assert item.coarse_similarity == pytest.approx(0.75)


def test_rows_without_fine_vector_fall_back_to_coarse_order_at_the_tail() -> None:
    """fine 未充填の行（再取り込み前の既存データ）が混在する場合。

    coarse 距離と fine 距離は別空間の値なので大小を直接比較できない。
    未充填の行は再スコアリング済みの行より後ろへ置き、その中では 1 段目の順位を保つ。
    """
    candidates = _stage1(("no-fine-near", 0.05), ("fine-far", 0.40), ("no-fine-far", 0.50))
    ranked, status = rerank.merge(
        candidates, {"fine-far": 0.95}, limit=10, min_similarity=-1.0
    )
    assert [item.image_id for item in ranked] == ["fine-far", "no-fine-near", "no-fine-far"]
    assert status == rerank.RERANK_PARTIAL_COVERAGE
    assert [item.reranked for item in ranked] == [True, False, False]


def test_no_fine_coverage_keeps_the_coarse_order() -> None:
    candidates = _stage1(("a", 0.10), ("b", 0.20))
    ranked, status = rerank.merge(candidates, {}, limit=10, min_similarity=-1.0)
    assert [item.image_id for item in ranked] == ["a", "b"]
    assert status == rerank.RERANK_NO_COVERAGE
    assert not any(item.reranked for item in ranked)


def test_empty_stage1_is_distinguished_from_missing_fine_vectors() -> None:
    """1 段目が 0 件なのと fine 未充填なのは別の事象。

    区別しないと「該当なし」の検索で毎回「高次元ベクトルが未登録」の警告が出る。
    """
    ranked, status = rerank.merge([], {}, limit=10, min_similarity=-1.0)
    assert ranked == []
    assert status == rerank.RERANK_NO_CANDIDATES


# --- min_similarity ---------------------------------------------------------


def test_min_similarity_applies_to_the_final_similarity() -> None:
    """1 段目のスコアで切ってはならない。

    coarse では似ていなくても fine で上位に来る候補があり、それを 2 段目より前に
    落としてしまうと再ランキングの意味が無くなる。逆に coarse で似ていても
    fine で閾値を下回る候補は返さない。
    """
    candidates = _stage1(("coarse-bad-fine-good", 0.90), ("coarse-good-fine-bad", 0.05))
    ranked, _ = rerank.merge(
        candidates,
        {"coarse-bad-fine-good": 0.10, "coarse-good-fine-bad": 0.95},
        limit=10,
        min_similarity=0.5,
    )
    # fine similarity は 0.90 と 0.05。閾値 0.5 を超えるのは前者だけ。
    assert [item.image_id for item in ranked] == ["coarse-bad-fine-good"]


def test_limit_is_applied_after_filtering() -> None:
    candidates = _stage1(("a", 0.1), ("b", 0.2), ("c", 0.3))
    ranked, _ = rerank.merge(
        candidates, {"a": 0.1, "b": 0.2, "c": 0.3}, limit=2, min_similarity=-1.0
    )
    assert [item.image_id for item in ranked] == ["a", "b"]


# --- 1 段検索（coarse のみ） -------------------------------------------------


def test_coarse_only_matches_the_single_stage_behaviour() -> None:
    candidates = _stage1(("a", 0.10), ("b", 0.30), ("c", 0.95))
    ranked = rerank.coarse_only(candidates, limit=10, min_similarity=0.5)
    # similarity = 1 - distance。c は 0.05 なので閾値 0.5 で落ちる。
    assert [item.image_id for item in ranked] == ["a", "b"]
    assert not any(item.reranked for item in ranked)
    assert all(item.distance == item.coarse_distance for item in ranked)
