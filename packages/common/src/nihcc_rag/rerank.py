"""2 段ベクトル検索（coarse -> fine）の候補統合。

1 段目は低次元 coarse ベクトルの HNSW（近似）で候補を広く取り、2 段目は高次元
fine ベクトルの厳密距離で並べ直す。本モジュールは DB に依存しない純粋な統合
ロジックだけを持つ（search-aggregator から使い、単体テストで固定する）。

重要な前提:
  - 距離は cosine（pgvector の `<=>`）。値域は 0..2、similarity = 1 - distance。
  - 2 段目の効果は 1 段目の recall が上限である。1 段目が取り落とした画像は
    どれだけ厳密に再計算しても戻らないため、1 段目は candidate_k より多めに取る。
  - fine ベクトルが未充填の行（再取り込み前の既存データ）が混在しうる。
    その行は 1 段目の coarse 距離順で、再スコアリング済みの行より後ろに置く。
    coarse 距離と fine 距離は別空間の値であり、大小を直接比較できないため。
"""

from __future__ import annotations

from dataclasses import dataclass

#: cosine 距離の上限。fine 距離を持たない候補の番兵として使う。
COSINE_DISTANCE_MAX = 2.0

#: 2 段目の適用状況。検索応答へそのまま載せ、画面と監査ログで追跡できるようにする。
RERANK_DISABLED = "DISABLED"  # fine ベクトルの指定が無い（1 段検索）
RERANK_APPLIED = "APPLIED"  # 候補すべてを fine 空間で再スコアリングした
RERANK_PARTIAL_COVERAGE = "PARTIAL_COVERAGE"  # fine 未充填の行が混在した
RERANK_NO_COVERAGE = "NO_COVERAGE"  # 候補のどれも fine ベクトルを持たなかった
RERANK_NO_CANDIDATES = "NO_CANDIDATES"  # 1 段目が 0 件（充填漏れとは無関係）
RERANK_BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"  # 1 段目で timeout 予算を使い切った
#: 1 段目を使わず、fine（所見プロファイル）で全行を厳密に検索した。画像だけの検索で使う。
#: 2 段検索は 1 段目（BiomedCLIP）が拾った候補しか並べ替えられず、所見の合う画像を
#: 取りこぼすため。
RERANK_PROFILE_EXACT = "PROFILE_EXACT"


#: 実際に使った検索の進め方（検索応答の query.strategy と models.rerank に出す）
STRATEGY_PROFILE = "profile"  # 所見プロファイルで全件を厳密に検索
STRATEGY_COARSE_TO_FINE = "coarse_to_fine"  # BiomedCLIP で候補 -> 所見プロファイルで並べ替え
STRATEGY_COARSE_ONLY = "coarse_only"  # BiomedCLIP だけ

STRATEGY_MODEL_LABELS = {
    STRATEGY_PROFILE: "profile-exact-full-scan",
    STRATEGY_COARSE_TO_FINE: "two-stage-coarse-to-fine-exact",
    STRATEGY_COARSE_ONLY: "disabled-single-stage",
}


def resolve_strategy(requested: str, *, default: str, rerank_active: bool, has_text: bool) -> str:
    """検索の進め方を決める。

    requested は要求の strategy（auto / profile / coarse_to_fine）、default は
    IMAGE_SEARCH_STRATEGY。所見プロファイルは画像から作るので、テキストを含む検索で
    全件検索に使うとテキストの意味が順位に入らなくなる。rerank=false や fine 未構成
    （rerank_active=False）なら BiomedCLIP だけ。
    """
    if not rerank_active:
        return STRATEGY_COARSE_ONLY
    if has_text:
        return STRATEGY_COARSE_TO_FINE
    chosen = default if requested == "auto" else requested
    return STRATEGY_PROFILE if chosen == STRATEGY_PROFILE else STRATEGY_COARSE_TO_FINE

@dataclass(frozen=True)
class Stage1Candidate:
    """1 段目（coarse ベクトルの ANN）が返した候補。"""

    image_id: str
    coarse_distance: float
    roi_confidence: float


@dataclass(frozen=True)
class RankedCandidate:
    """2 段目の統合まで終えた候補。

    distance / similarity は最終順位に使った値（再スコアリングできた行は fine
    空間、できなかった行は coarse 空間）。どちらを使ったかは reranked で判る。
    fine で全件を検索したとき（profile_exact）は coarse を計算しないので None。
    """

    image_id: str
    distance: float
    similarity: float
    coarse_distance: float | None
    coarse_similarity: float | None
    roi_confidence: float
    reranked: bool


@dataclass(frozen=True)
class FineCandidate:
    """fine（所見プロファイル）の全件厳密検索が返した候補。"""

    image_id: str
    fine_distance: float
    roi_confidence: float


def stage1_limit(candidate_k: int, *, multiplier: int, max_limit: int) -> int:
    """1 段目の取得件数。

    candidate_k を下回らせない（2 段目で絞る前の母集団が candidate_k より少ないと、
    再ランキングの有無で返却件数が変わってしまう）。
    """
    return max(candidate_k, min(candidate_k * max(1, multiplier), max_limit))


def merge(
    candidates: list[Stage1Candidate],
    fine_distances: dict[str, float],
    *,
    limit: int,
    min_similarity: float,
) -> tuple[list[RankedCandidate], str]:
    """1 段目の候補を fine 距離で並べ直し、(候補列, rerank_status) を返す。

    fine_distances は image_id -> fine 空間の cosine 距離。2 段目のクエリが
    返さなかった image_id（fine 未充填、または fine モデル版が不一致）は
    coarse 距離のまま後方へ置く。

    min_similarity は「最終順位に使った similarity」に対して適用する。1 段目の
    coarse similarity で切ると、再スコアリングで上位に来るはずの候補を
    2 段目に渡す前に落としてしまう。
    """
    ranked: list[RankedCandidate] = []
    for candidate in candidates:
        fine_distance = fine_distances.get(candidate.image_id)
        reranked = fine_distance is not None
        distance = fine_distance if reranked else candidate.coarse_distance
        ranked.append(
            RankedCandidate(
                image_id=candidate.image_id,
                distance=distance,
                similarity=1.0 - distance,
                coarse_distance=candidate.coarse_distance,
                coarse_similarity=1.0 - candidate.coarse_distance,
                roi_confidence=candidate.roi_confidence,
                reranked=reranked,
            )
        )

    # 再スコアリング済みを前に、その中は fine 距離の昇順。未充填は coarse 距離の
    # 昇順で後方へ。bool は False < True なので `not reranked` を第 1 キーにする。
    ranked.sort(key=lambda item: (not item.reranked, item.distance, item.coarse_distance))

    reranked_count = sum(1 for item in ranked if item.reranked)
    if not candidates:
        # 1 段目が 0 件。fine の充填状況とは無関係なので、充填漏れと区別する
        # （区別しないと「該当なし」の検索で毎回 fine 未登録の警告が出てしまう）。
        status = RERANK_NO_CANDIDATES
    elif reranked_count == 0:
        status = RERANK_NO_COVERAGE
    elif reranked_count == len(ranked):
        status = RERANK_APPLIED
    else:
        status = RERANK_PARTIAL_COVERAGE

    kept = [item for item in ranked if item.similarity >= min_similarity][:limit]
    return kept, status


def profile_exact(
    candidates: list[FineCandidate], *, limit: int, min_similarity: float
) -> tuple[list[RankedCandidate], str]:
    """fine の全件厳密検索の結果を候補列にし、(候補列, rerank_status) を返す。

    並びは fine 距離の昇順（SQL の ORDER BY と同じ。Citus が shard ごとの上位を
    まとめた結果を、念のためここでも並べ直す）。
    """
    ranked = sorted(
        (
            RankedCandidate(
                image_id=candidate.image_id,
                distance=candidate.fine_distance,
                similarity=1.0 - candidate.fine_distance,
                coarse_distance=None,
                coarse_similarity=None,
                roi_confidence=candidate.roi_confidence,
                reranked=False,
            )
            for candidate in candidates
        ),
        key=lambda item: (item.distance, item.image_id),
    )
    status = RERANK_PROFILE_EXACT if candidates else RERANK_NO_CANDIDATES
    return [item for item in ranked if item.similarity >= min_similarity][:limit], status


def coarse_only(
    candidates: list[Stage1Candidate], *, limit: int, min_similarity: float
) -> list[RankedCandidate]:
    """2 段目を行わない場合の候補列（1 段目の並びのまま）。"""
    ranked = [
        RankedCandidate(
            image_id=candidate.image_id,
            distance=candidate.coarse_distance,
            similarity=1.0 - candidate.coarse_distance,
            coarse_distance=candidate.coarse_distance,
            coarse_similarity=1.0 - candidate.coarse_distance,
            roi_confidence=candidate.roi_confidence,
            reranked=False,
        )
        for candidate in candidates
    ]
    return [item for item in ranked if item.similarity >= min_similarity][:limit]
