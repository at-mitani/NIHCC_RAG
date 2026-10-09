"""API 契約と日本語正規化の単体テスト（設計書 9.5 / 7.3.2）。"""

from __future__ import annotations

import uuid

import pytest
from nihcc_rag import text_norm
from nihcc_rag.contract import (
    ESTIMATE_LABEL,
    EmbedChestRequest,
    EmbedChestResponse,
    EmbedResponse,
    FindingEstimate,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    VectorCandidate,
    VectorSearchQuery,
    VectorSearchRequest,
)
from nihcc_rag.regions import REGION_CODES, validate_region_codes
from pydantic import ValidationError

# --- 検索リクエストの入力制約（設計書 9.5） ---------------------------------


def test_image_to_image_rejects_text() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(mode="IMAGE_TO_IMAGE", text="nodule")


def test_image_to_image_forces_weight_one() -> None:
    assert SearchRequest(mode="IMAGE_TO_IMAGE").image_weight == 1.0


def test_text_to_image_requires_text_and_forces_weight_zero() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(mode="TEXT_TO_IMAGE")
    assert SearchRequest(mode="TEXT_TO_IMAGE", text="nodule").image_weight == 0.0


def test_hybrid_requires_text_and_strict_weight_range() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(mode="HYBRID", text="nodule", image_weight=1.0)
    with pytest.raises(ValidationError):
        SearchRequest(mode="HYBRID", text="nodule", image_weight=0.0)
    assert SearchRequest(mode="HYBRID", text="nodule", image_weight=0.7).image_weight == 0.7


def test_candidate_k_defaults_to_five_times_top_k() -> None:
    assert SearchRequest(mode="IMAGE_TO_IMAGE", top_k=10).candidate_k == 50
    # 上限 500 を超えない
    assert SearchRequest(mode="IMAGE_TO_IMAGE", top_k=100).candidate_k == 500


def test_candidate_k_must_not_be_below_top_k() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(mode="IMAGE_TO_IMAGE", top_k=20, candidate_k=5)


def test_top_k_upper_bound() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(mode="IMAGE_TO_IMAGE", top_k=101)


def test_search_request_no_longer_carries_regions() -> None:
    """検索の単位は胸部野 1 つ。呼び出し側が regions を送っても壊れないが、
    要求には残らない（検索は部位を選ばない）。"""
    request = SearchRequest.model_validate({"mode": "IMAGE_TO_IMAGE", "regions": ["RUL"]})
    assert "regions" not in request.model_dump()


# --- 2 段検索（coarse -> fine）の契約 ----------------------------------------


def test_rerank_defaults_to_enabled() -> None:
    # fine モデルが構成されていなければ、この指定に関わらず 1 段検索になる。
    # 契約としての既定値は「要求する」側に置く。
    assert SearchRequest(mode="IMAGE_TO_IMAGE").rerank is True


def test_rerank_can_be_disabled_for_comparison() -> None:
    assert SearchRequest(mode="IMAGE_TO_IMAGE", rerank=False).rerank is False


def test_vector_search_query_defaults_to_single_stage() -> None:
    query = VectorSearchQuery(query_vector=[0.0, 1.0])
    assert query.query_vector_fine is None


def test_vector_search_request_takes_one_query() -> None:
    """1 クラスタなので問い合わせは 1 つ（部位ごとの 5 つではない）。"""
    request = VectorSearchRequest(
        embedding_model_version="m", preprocess_version="p",
        query={"query_vector": [0.0, 1.0], "candidate_k": 10},
    )
    assert request.query.candidate_k == 10


def test_vector_candidate_carries_both_spaces() -> None:
    candidate = VectorCandidate(
        image_id=uuid.uuid4(),
        similarity=0.9,
        distance=0.1,
        coarse_similarity=0.7,
        coarse_distance=0.3,
        reranked=True,
        roi_confidence=0.95,
    )
    assert candidate.reranked is True
    assert candidate.coarse_similarity == 0.7


def test_embed_response_fine_fields_are_optional() -> None:
    """既存の呼び出し側が fine を知らなくても壊れないこと（後方互換）。"""
    response = EmbedResponse(
        embedding_model_version="biomedclip-deadbeef-d512",
        embedding_model_name="BiomedCLIP",
        embedding_model_revision="deadbeef",
        dimension=512,
        vectors=[[0.0] * 512],
    )
    assert response.fine_vectors is None
    assert response.fine_embedding_model_version is None
    assert response.fine_dimension is None


def test_embed_chest_request_lists_must_align() -> None:
    """crop・画像・mask は同じ並び・同じ長さ。ずれると別の画像の mask で pooling する。"""
    with pytest.raises(ValidationError):
        EmbedChestRequest(crops_base64=["a", "b"], images_base64=["a"], masks_base64=["a"])
    ok = EmbedChestRequest(crops_base64=["a"], images_base64=["b"], masks_base64=["c"])
    assert len(ok.masks_base64) == 1


def test_embed_chest_response_without_classifier_is_single_stage() -> None:
    response = EmbedChestResponse(
        embedding_model_version="biomedclip-deadbeef-d512",
        embedding_model_name="BiomedCLIP",
        embedding_model_revision="deadbeef",
        dimension=512,
        vectors=[[0.0] * 512],
    )
    assert response.fine_vectors is None
    assert response.probabilities is None


# --- データセットのラベルと所見の推定（別物として扱う） ------------------------


def test_labels_and_estimates_are_separate_fields() -> None:
    """データセットのラベル（NLP 由来）と分類器の推定は別の項目に入る。"""
    item = SearchResultItem(
        rank=1, image_id=uuid.uuid4(), similarity=0.9, distance=0.1, roi_confidence=0.9,
        finding_labels=["Effusion"],
        finding_estimates=[FindingEstimate(finding="Cardiomegaly", probability=0.81)],
    )
    assert item.finding_labels == ["Effusion"]
    assert item.finding_estimates[0].finding == "Cardiomegaly"


def test_estimates_default_to_empty() -> None:
    """分類器の無い構成では推定を出さない（空）。断定しないのが既定。"""
    item = SearchResultItem(
        rank=1, image_id=uuid.uuid4(), similarity=0.9, distance=0.1, roi_confidence=0.9
    )
    assert item.finding_estimates == []
    assert item.chest_url is None


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_estimate_probability_is_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        FindingEstimate(finding="Nodule", probability=value)


def test_response_carries_the_not_a_diagnosis_label() -> None:
    """推定には「診断ではない」見出しが必ず付く（推定の表示規則）。"""
    response = SearchResponse(search_id=uuid.uuid4(), status="SUCCEEDED", mode="IMAGE_TO_IMAGE")
    assert "診断ではありません" in response.estimate_label
    assert response.estimate_label == ESTIMATE_LABEL
    assert response.rerank_status is None
    assert response.stage1_count == 0
    assert response.results == []


def test_return_alias_is_accepted() -> None:
    request = SearchRequest.model_validate(
        {"mode": "IMAGE_TO_IMAGE", "return": {"thumbnail": False, "roi_overlay": False}}
    )
    assert request.return_options.thumbnail is False


def test_validate_region_codes_rejects_unknown() -> None:
    """部位コードは評価スクリプトと内部契約（部位 mask pooling）にだけ残る。"""
    assert validate_region_codes(["LLL", "RUL"]) == ["RUL", "LLL"]
    with pytest.raises(ValueError):
        validate_region_codes(["RUL", "XXX"])
    assert len(REGION_CODES) == 5


# --- 日本語テキストの英語正規化（設計書 7.3.2） -----------------------------


def test_registered_japanese_phrase_is_translated() -> None:
    result = text_norm.normalize("右上肺野の結節影", "ja")
    assert result.english == "pulmonary nodule in the right upper lung field"
    assert result.language == "ja"
    assert result.translation_model_version


def test_english_input_passes_through() -> None:
    result = text_norm.normalize("pulmonary nodule in the right upper lobe", "en")
    assert result.english == "pulmonary nodule in the right upper lobe"


def test_untranslatable_japanese_is_rejected_not_guessed() -> None:
    """外部翻訳 API へ送らず、辞書で処理できない自由文はエラーとする。"""
    with pytest.raises(text_norm.UntranslatableText) as excinfo:
        text_norm.normalize("昨日から続く強い咳と発熱があります", "ja")
    assert excinfo.value.unresolved


def test_empty_text_is_rejected() -> None:
    with pytest.raises(text_norm.UntranslatableText):
        text_norm.normalize("   ", "ja")


def test_known_phrases_exclude_particles() -> None:
    phrases = text_norm.known_phrases()
    assert all(len(p["ja"]) >= 2 and p["en"] for p in phrases)
    assert any(p["ja"] == "結節影" for p in phrases)
