"""API 契約（設計書 9 章）。pydantic モデルから OpenAPI を生成する。"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, model_validator

#: 肺の 5 部位のコード。検索の単位ではない（検索は胸部野 1 単位）。
#: 評価スクリプトと内部契約（EmbedImagesRequest.region_codes、部位 mask pooling）が使う。
RegionCode = Literal["RUL", "RML", "RLL", "LUL", "LLL"]
#: 部位 mask pooling に渡せる mask の名前。5 部位に加え、画像 1 枚を 1 単位にする
#: 評価のための 3 つ:
#: LUNG（肺野のみ）/ CHEST（肺野＋縦隔＋横隔膜）/ WHOLE（画像全体）。
PoolMaskCode = Literal["RUL", "RML", "RLL", "LUL", "LLL", "LUNG", "CHEST", "WHOLE"]
SearchMode = Literal["IMAGE_TO_IMAGE", "TEXT_TO_IMAGE", "HYBRID"]

RESEARCH_WARNING = "研究用途です。検索結果および付随ラベルは診断を示しません。"


# --- 認証 -------------------------------------------------------------------


class TokenRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    role: str


# --- 取り込み ---------------------------------------------------------------


class ValidateRequest(BaseModel):
    image_root: str = Field(max_length=1024)
    csv_path: str = Field(max_length=1024)
    dataset_version: str = Field(max_length=128)
    csv_schema_version: str = Field(default="nihcc-data-entry-2017-v1", max_length=128)
    sample_limit: int | None = Field(default=None, ge=1, le=200000)


class ValidationIssue(BaseModel):
    code: str
    source_image_name: str | None = None
    detail: str | None = None


class ValidateResponse(BaseModel):
    validation_id: uuid.UUID
    valid: bool
    counts: dict[str, int]
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []


class IngestionRequest(BaseModel):
    validation_id: uuid.UUID
    roi_model_version: str | None = None
    embedding_model_version: str | None = None
    preprocess_version: str | None = None
    on_duplicate: Literal["SKIP", "REPLACE"] = "SKIP"
    dry_run: bool = False


class IngestionAccepted(BaseModel):
    job_id: uuid.UUID
    status: str
    status_url: str


class IngestionProgress(BaseModel):
    total: int
    succeeded: int
    failed: int
    pending: int
    running: int = 0
    quarantined: int = 0
    #: ジョブの取消により処理しなかった item 数
    cancelled: int = 0


class IngestionStatus(BaseModel):
    job_id: uuid.UUID
    status: str
    progress: IngestionProgress
    models: dict[str, str] = {}
    created_by: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    errors_url: str
    error_summary: list[dict[str, Any]] = []


# --- 検索 -------------------------------------------------------------------


class SearchFilters(BaseModel):
    finding_labels_any: list[str] = []
    view_positions: list[str] = []
    exclude_image_ids: list[uuid.UUID] = []


class SearchReturnOptions(BaseModel):
    thumbnail: bool = True
    roi_overlay: bool = True
    include_source_metadata: bool = False


class SearchRequest(BaseModel):
    mode: SearchMode = "IMAGE_TO_IMAGE"
    text: str | None = Field(default=None, max_length=2000)
    text_language: Literal["ja", "en"] = "en"
    image_weight: float = Field(default=0.70, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_k: int | None = Field(default=None, ge=1, le=500)
    min_similarity: float = Field(default=-1.0, ge=-1.0, le=1.0)
    #: 2 段検索の 2 段目（高次元ベクトルによる厳密な再ランキング）を使うか。
    #: fine モデルが構成されていない場合は指定しても無効（応答の models.rerank で判る）。
    #: 精度評価で 1 段のみと比較するために false を指定できる。
    rerank: bool = True
    #: 画像だけの検索の進め方。
    #:   auto: サーバの設定（IMAGE_SEARCH_STRATEGY）に従う
    #:   profile: 所見プロファイルで全件を厳密に検索する
    #:   coarse_to_fine: BiomedCLIP で候補を集めてから所見プロファイルで並べ替える
    #: テキストを含む検索と rerank=false では使われない（常に BiomedCLIP から）。
    strategy: Literal["auto", "profile", "coarse_to_fine"] = "auto"
    filters: SearchFilters = SearchFilters()
    return_options: SearchReturnOptions = Field(default=SearchReturnOptions(), alias="return")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def _mode_constraints(self) -> "SearchRequest":
        # 設計書 9.5 の入力制約
        if self.mode == "TEXT_TO_IMAGE":
            if not (self.text or "").strip():
                raise ValueError("TEXT_TO_IMAGE requires text")
            self.image_weight = 0.0
        elif self.mode == "IMAGE_TO_IMAGE":
            if (self.text or "").strip():
                raise ValueError("IMAGE_TO_IMAGE does not accept text")
            self.image_weight = 1.0
        else:  # HYBRID
            if not (self.text or "").strip():
                raise ValueError("HYBRID requires text")
            if not 0.0 < self.image_weight < 1.0:
                raise ValueError("HYBRID requires 0.0 < image_weight < 1.0")
        if self.candidate_k is None:
            self.candidate_k = min(500, self.top_k * 5)
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be greater than or equal to top_k")
        return self


#: 所見の推定を画面に出すときの見出し。**診断ではない**ことを常に添える。
ESTIMATE_LABEL = "所見の推定（研究用・診断ではありません）"


class FindingEstimate(BaseModel):
    """胸部野の分類器が出した所見ごとの確率。

    **データセットのラベル（NLP 由来）とは別物**であり、画面では両者を並べて
    区別できるように示す。取り込み時に計算して保存した値で、検索のたびには
    計算し直さない。
    """

    finding: str
    probability: float = Field(ge=0.0, le=1.0)


class SearchResultItem(BaseModel):
    rank: int
    image_id: uuid.UUID
    #: 最終順位に使った類似度（再ランキングできた候補は fine 空間、できなければ coarse 空間）
    similarity: float
    distance: float
    #: 1 段目（coarse 空間）の類似度。2 段目の効き方を確認するために返す。
    coarse_similarity: float | None = None
    #: この候補が fine ベクトルで再スコアリングされたか
    reranked: bool = False
    roi_confidence: float
    quality_status: str | None = None
    #: **データセットのラベル**（放射線科レポートから NLP で導いたもの、画像単位）
    finding_labels: list[str] = []
    label_source: str | None = None
    #: 胸部野の分類器による所見の推定（確率の高い順、上位のみ）。ラベルとは別物
    finding_estimates: list[FindingEstimate] = []
    view_position: str | None = None
    patient_ref: str | None = None
    source_image_name: str | None = None
    thumbnail_url: str | None = None
    #: 原画像へ胸部野（肺野＋縦隔＋横隔膜）の輪郭を重ねた画像。何を比べたかを示す
    chest_url: str | None = None
    #: 肺野全体の外周を描いた確認用画像。胸郭ランキングで、何を「肺野」として
    #: 比較したかを目で確かめるために返す（脊柱・心陰影・横隔膜が外側にあるか）。
    lung_outline_url: str | None = None
    image_url: str | None = None
    source_metadata: dict[str, Any] | None = None


class SearchQueryEcho(BaseModel):
    text_original: str | None = None
    text_normalized_en: str | None = None
    image_weight: float
    top_k: int
    candidate_k: int
    #: 1 段目が実際に取得した件数の上限（candidate_k x VECTOR_STAGE1_MULTIPLIER）
    stage1_candidate_k: int | None = None
    min_similarity: float
    rerank: bool = True
    #: 実際に使った進め方（profile / coarse_to_fine / coarse_only）
    strategy: str | None = None


class SearchResponse(BaseModel):
    search_id: uuid.UUID
    status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED"]
    mode: SearchMode
    query: SearchQueryEcho | None = None
    models: dict[str, str] = {}
    #: 胸部野どうしを比べた 1 本の順位
    results: list[SearchResultItem] = []
    #: 検索元の胸部野の品質と、輪郭を重ねた画像
    query_quality_status: str | None = None
    query_roi_confidence: float | None = None
    query_chest_url: str | None = None
    #: 検索元の所見の推定（画像検索のときだけ）
    query_finding_estimates: list[FindingEstimate] = []
    #: 推定に使った分類器の版（2 段目のモデル版）。どの分類器の推定かを追うため
    estimate_model_version: str | None = None
    estimate_label: str = ESTIMATE_LABEL
    timing_ms: int = 0
    #: 2 段目の適用状況（nihcc_rag.rerank の RERANK_* のいずれか）
    rerank_status: str | None = None
    #: 1 段目が返した候補数と、そのうち 2 段目で再スコアリングできた件数
    stage1_count: int = 0
    reranked_count: int = 0
    partial: bool = False
    warnings: list[str] = [RESEARCH_WARNING]
    duration_ms: int = 0
    #: **胸郭の形が似た症例**（画像単位）。
    #: 所見の判別ではない。
    #: 検索元画像に肺野外周を描いたものの URL。胸郭ランキングの左上に置く。
    #: 候補と同じ見せ方にするためで、無い場合（テキスト検索・描画失敗）は None。
    query_lung_outline_url: str | None = None
    lung_shape_results: list[SearchResultItem] = []
    #: AVAILABLE / EMPTY / UNAVAILABLE / NOT_APPLICABLE
    lung_shape_status: str = "NOT_APPLICABLE"


class SearchAccepted(BaseModel):
    search_id: uuid.UUID
    status: str
    status_url: str


# --- 画像 -------------------------------------------------------------------


class ChestAssetInfo(BaseModel):
    """画像の胸部野（肺野＋縦隔＋横隔膜）。"""

    bbox: dict[str, Any]
    #: 肺野・胸部野・心臓が画像に占める割合
    areas: dict[str, float] = {}
    roi_confidence: float
    quality_status: str
    flags: list[str] = []
    roi_model_version: str
    preprocess_version: str
    has_embedding: bool = False


class ImageMetadata(BaseModel):
    image_id: uuid.UUID
    source_image_name: str
    patient_ref: str
    sha256: str
    width: int
    height: int
    view_position: str | None
    patient_age: int | None
    patient_gender: str | None
    source_dataset_version: str
    #: **データセットのラベル**（放射線科レポートから NLP で導いたもの）
    finding_labels: list[str]
    label_source: str
    chest: ChestAssetInfo | None = None
    #: 胸部野の分類器による所見の推定（14 所見すべて、確率の高い順）
    finding_estimates: list[FindingEstimate] = []
    estimate_model_version: str | None = None
    estimate_label: str = ESTIMATE_LABEL
    source_metadata: dict[str, Any] | None = None
    warnings: list[str] = [RESEARCH_WARNING]


# --- 内部サービス間契約 ------------------------------------------------------


class ChestRoiRequest(BaseModel):
    """胸部野の抽出要求（roi-service `/api/v1/roi/chest`）。"""

    image_base64: str
    profile: str | None = None


class ChestRoiResponse(BaseModel):
    """胸部野の抽出結果と、埋め込みモデルへそのまま渡す入力。

    mask は**胸部野の外接矩形で切った**作業解像度の PNG で返す（全画面より小さく、
    原画像への重ね合わせは `source_bbox` で位置を戻す）。肺野 mask も同じ矩形で
    切ってある。

    `pool_image` / `pool_mask` / `crop` は `image_pipeline.chest_model_inputs` の
    出力そのままである。**呼び出し側で作り直さない**（分類器が前提とする特徴と
    1 画素でも違うと、確率が黙ってずれる）。
    """

    roi_model_version: str
    preprocess_version: str
    profile_name: str
    parameter_hash: str
    #: OK / ROI_UNCERTAIN（抽出できない場合は 422）
    quality_status: str
    roi_confidence: float
    flags: list[str] = []
    #: 肺野・胸部野・心臓が画像に占める割合
    areas: dict[str, float] = {}
    #: 胸部野の外接矩形（作業解像度）と、原画像の座標へ戻したもの
    bbox: list[int]
    source_bbox: list[int]
    work_scale: float = 1.0
    chest_mask_base64: str
    lung_mask_base64: str
    #: 胸郭（肺野全体）の形状ベクトル（nihcc_rag.shape.describe_lung、16 次元）
    lung_vector: list[float] | None = None
    #: 埋め込みモデルへの入力（chest_model_inputs）
    pool_image_base64: str
    pool_mask_base64: str
    crop_base64: str


class EmbedChestRequest(BaseModel):
    """胸部野の埋め込み要求（embedding-service `/api/v1/embeddings/chest`）。

    3 つのリストは同じ並び・同じ長さで、`ChestRoiResponse` の値をそのまま渡す。

      crops   … 1 段目（BiomedCLIP）へ。胸部野の外を黒塗りした正方形
      images  … 2 段目（RAD-DINO の pooling）へ。全体画像
      masks   … 同上。胸部野 mask（images と同じ寸法）
    """

    crops_base64: list[str] = Field(min_length=1, max_length=64)
    images_base64: list[str] = Field(min_length=1, max_length=64)
    masks_base64: list[str] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _aligned(self) -> "EmbedChestRequest":
        sizes = {len(self.crops_base64), len(self.images_base64), len(self.masks_base64)}
        if len(sizes) != 1:
            raise ValueError(
                "crops_base64 / images_base64 / masks_base64 must have the same length "
                f"({len(self.crops_base64)}, {len(self.images_base64)}, {len(self.masks_base64)})"
            )
        return self


class EmbedChestResponse(BaseModel):
    """1 段目のベクトル、2 段目のプロファイル、所見の推定確率。

    2 段目の backend が胸部野の分類器（chest-profile）でない構成では、fine_* と
    findings / probabilities は None になり、呼び出し側は 1 段検索になる。
    """

    embedding_model_version: str
    embedding_model_name: str
    embedding_model_revision: str
    dimension: int
    normalization: str = "L2"
    vectors: list[list[float]]
    fine_embedding_model_version: str | None = None
    fine_embedding_model_name: str | None = None
    fine_embedding_model_revision: str | None = None
    fine_dimension: int | None = None
    fine_vectors: list[list[float]] | None = None
    #: probabilities の列の並び（14 所見）
    findings: list[str] | None = None
    probabilities: list[list[float]] | None = None


class EmbedImagesRequest(BaseModel):
    """ROI crop の埋め込み要求。

    `region_codes` は推論に部位コードを要する backend（needs_region_codes）を
    2 段目に置くときに必須になる。部位ごとの標準化とバイアスの両方が
    「どの部位の crop か」を必要とし、crop の並びだけでは推論できないためである。
    1 段目の外観埋め込みには不要なので任意項目とする。
    """

    images_base64: list[str] = Field(min_length=1, max_length=256)
    #: images_base64 と同じ並び・同じ長さの部位コード
    region_codes: list[RegionCode] | None = None

    @model_validator(mode="after")
    def _region_codes_align(self) -> "EmbedImagesRequest":
        if self.region_codes is not None and len(self.region_codes) != len(self.images_base64):
            raise ValueError(
                "region_codes must have the same length as images_base64 "
                f"({len(self.region_codes)} != {len(self.images_base64)})"
            )
        return self


class RegionPoolRequest(BaseModel):
    """画像 1 回の順伝播 + mask pooling の要求（評価スクリプト用）。

    `EmbedImagesRequest` は「切り出した crop を N 枚渡し、N 本のベクトルを受け取る」
    形で、crop ごとに別々に符号化するため、黒塗り crop の分布差がそのまま部位固有の
    偏りになる。

    ここでは**画像 1 枚を 1 回だけ順伝播し、patch token を mask で pooling**
    する。どの mask のベクトルも同一の順伝播から出るので、この偏りが入らない。

    mask は画像と**同じ画素寸法**で渡す。こうしておけば、モデル側の入力解像度や
    patch grid がいくつであっても、両者を同じ枠で縮約でき、切り取り位置のずれが
    原理的に起きない。
    """

    #: 画像 1 枚ぶんずつの base64。全体の順伝播を行うため 1 件が重く、上限は小さい。
    images_base64: list[str] = Field(min_length=1, max_length=64)
    #: images_base64 と同じ並び・同じ長さ。各要素は {部位コード: mask PNG の base64}。
    masks_base64: list[dict[PoolMaskCode, str]] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def _masks_align(self) -> "RegionPoolRequest":
        if len(self.masks_base64) != len(self.images_base64):
            raise ValueError(
                "masks_base64 must have the same length as images_base64 "
                f"({len(self.masks_base64)} != {len(self.images_base64)})"
            )
        for index, item in enumerate(self.masks_base64):
            if not item:
                raise ValueError(f"masks_base64[{index}] has no region")
        return self


class RegionPoolItem(BaseModel):
    """1 枚 1 部位ぶんの結果。

    mask が patch grid 上で空になった部位は返さない。返す件数が要求より少なく
    なりうるため、呼び出し側は index と部位コードで突き合わせる。
    """

    image_index: int
    #: 要求の mask 名をそのまま返す（5 部位とは限らない。PoolMaskCode）
    region_code: PoolMaskCode
    vector: list[float]


class RegionPoolResponse(BaseModel):
    model_version: str
    model_name: str
    model_revision: str
    dimension: int
    normalization: str = "L2"
    #: patch grid の一辺（RAD-DINO @518 / patch 14 なら 37）。
    #: mask をどれだけ粗く縮約したかを呼び出し側が記録できるようにする。
    patch_grid: int
    items: list[RegionPoolItem]


class EmbedTextsRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class EmbedResponse(BaseModel):
    """埋め込み応答。

    vectors / dimension / embedding_model_* は 1 段目（coarse）の空間を表す。
    fine_* は 2 段検索の 2 段目に使う高次元空間で、fine backend が構成されて
    いないとき、および fine モデルが text encoder を持たないときは None になる。
    fine_* を使わない呼び出し側は無視してよい。
    """

    embedding_model_version: str
    embedding_model_name: str
    embedding_model_revision: str
    dimension: int
    normalization: str = "L2"
    vectors: list[list[float]]
    fine_embedding_model_version: str | None = None
    fine_embedding_model_name: str | None = None
    fine_embedding_model_revision: str | None = None
    fine_dimension: int | None = None
    fine_vectors: list[list[float]] | None = None


class ChestMapRequest(BaseModel):
    """区画ごとの所見の地図の要求（embedding-service `/api/v1/embeddings/chest-map`）。

    評価用（`scripts/chest_lesion_map_eval.py`）。images / masks は `ChestRoiResponse` の
    pool_image / pool_mask と同じもの。所見を絞ると応答が小さくなる。
    """

    images_base64: list[str] = Field(min_length=1, max_length=16)
    masks_base64: list[str] = Field(min_length=1, max_length=16)
    findings: list[str] | None = None

    @model_validator(mode="after")
    def _aligned(self) -> "ChestMapRequest":
        if len(self.images_base64) != len(self.masks_base64):
            raise ValueError(
                "images_base64 and masks_base64 must have the same length "
                f"({len(self.images_base64)} != {len(self.masks_base64)})"
            )
        return self


class ChestMapResponse(BaseModel):
    """maps[画像][所見] は grid × grid を行優先で平らにした寄与（加重前）。"""

    fine_embedding_model_version: str
    grid: int
    findings: list[str]
    maps: list[list[list[float]]]
    probabilities: list[list[float]]


class VectorSearchQuery(BaseModel):
    #: 1 段目（coarse 空間）のクエリベクトル
    query_vector: list[float]
    #: 2 段目（fine 空間）のクエリベクトル。None なら 1 段検索になる。
    query_vector_fine: list[float] | None = None
    candidate_k: int = Field(default=50, ge=1, le=500)
    min_similarity: float = -1.0
    exclude_image_ids: list[uuid.UUID] = []


class VectorSearchRequest(BaseModel):
    embedding_model_version: str
    preprocess_version: str
    #: 2 段目で突き合わせる fine モデル版。行の embedding_fine_model_version と
    #: 一致する行だけを再スコアリング対象にする（同次元の別モデルとの誤突合を防ぐ）。
    fine_embedding_model_version: str | None = None
    query: VectorSearchQuery
    ef_search: int | None = Field(default=None, ge=10, le=1000)
    timeout_ms: int | None = Field(default=None, ge=100, le=60000)
    #: 1 段目に使う列。fine なら embedding_fine で全行を厳密に検索し、2 段目は無い
    #: （query_vector_fine と fine_embedding_model_version が必須）。
    stage1: Literal["embedding", "fine"] = "embedding"

    @model_validator(mode="after")
    def _fine_stage1_needs_fine_query(self) -> "VectorSearchRequest":
        if self.stage1 == "fine" and not (
            self.query.query_vector_fine and self.fine_embedding_model_version
        ):
            raise ValueError("stage1=fine requires query_vector_fine and fine_embedding_model_version")
        return self


class VectorCandidate(BaseModel):
    image_id: uuid.UUID
    #: 最終順位に使った値（再スコアリング済みなら fine 空間、そうでなければ coarse 空間。
    #: stage1=fine なら fine 空間）
    similarity: float
    distance: float
    #: stage1=fine では coarse を計算しないので None
    coarse_similarity: float | None = None
    coarse_distance: float | None = None
    reranked: bool = False
    roi_confidence: float


class VectorSearchResponse(BaseModel):
    """1 クラスタへの 2 段検索の結果。

    失敗は応答の項目ではなく HTTP 状態で返す（時間切れは 504、それ以外は 503）。
    検索先は 1 クラスタなので「一部だけ成功」は無く、partial は持たない。
    """

    candidates: list[VectorCandidate] = []
    timing_ms: int = 0
    #: 1 段目 / 2 段目それぞれの所要時間（内訳の可視化用）
    stage1_ms: int = 0
    stage2_ms: int = 0
    #: 2 段目の適用状況（nihcc_rag.rerank の RERANK_* のいずれか）
    rerank_status: str = "DISABLED"
    #: 1 段目が返した候補数と、そのうち再スコアリングできた件数
    stage1_count: int = 0
    reranked_count: int = 0


# **形状検索は胸郭（画像単位）だけを持つ。** 部位は胸郭を幾何規則（正規化高さ u と
# 側方位置 w の固定境界）で切り分けたものなので、その形は胸郭の形の従属変数でしかなく、
# 独立した情報を持たない。


class LungShapeQuery(BaseModel):
    """肺野全体の形状検索（画像単位）。"""

    query_vector: list[float] = Field(min_length=1)
    candidate_k: int = Field(default=10, ge=1, le=500)
    exclude_image_ids: list[str] = []


class LungShapeRequest(BaseModel):
    """胸郭の形が似た症例を探す。

    **所見の判別ではない。** 形態の似た症例を並べる道具として提供する。
    """

    preprocess_version: str
    query: LungShapeQuery
    timeout_ms: int | None = Field(default=None, ge=100, le=60000)


class LungShapeResponse(BaseModel):
    candidates: list[ShapeCandidate] = []
    timing_ms: int = 0
    #: 肺野ベクトルが入っている行が無い（backfill 未了）場合に立てる
    empty_space: bool = False


class ShapeCandidate(BaseModel):
    image_id: uuid.UUID
    similarity: float
    distance: float
    roi_confidence: float


# --- ユーザー管理（機能追加） -------------------------------------------------

Role = Literal["viewer", "operator", "admin"]

#: パスワード入力規則（8 文字以上 128 文字以下）
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128


class UserSummary(BaseModel):
    user_id: uuid.UUID
    username: str
    role: Role
    email: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: datetime | None = None


class UserDetail(UserSummary):
    memo: str | None = None
    password_changed_at: datetime
    created_by: str | None = None
    updated_by: str | None = None


class UserListResponse(BaseModel):
    users: list[UserSummary]
    total: int


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._@-]+$")
    password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH)
    password_confirmation: str = Field(
        min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH
    )
    role: Role
    email: EmailStr
    memo: str | None = Field(default=None, max_length=2000)
    is_active: bool = True

    @model_validator(mode="after")
    def _passwords_match(self) -> "UserCreateRequest":
        if self.password != self.password_confirmation:
            raise ValueError("password_confirmation does not match password")
        return self


class UserUpdateRequest(BaseModel):
    """指定した項目だけを更新する。パスワードは任意（指定時は 2 回入力）。"""

    role: Role | None = None
    email: EmailStr | None = None
    memo: str | None = Field(default=None, max_length=2000)
    is_active: bool | None = None
    password: str | None = Field(
        default=None, min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH
    )
    password_confirmation: str | None = Field(
        default=None, min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH
    )

    @model_validator(mode="after")
    def _passwords_match(self) -> "UserUpdateRequest":
        if (self.password is None) != (self.password_confirmation is None):
            raise ValueError("password and password_confirmation must be provided together")
        if self.password is not None and self.password != self.password_confirmation:
            raise ValueError("password_confirmation does not match password")
        return self


class PasswordChangeRequest(BaseModel):
    """ログイン済み利用者による自身のパスワード変更。"""

    current_password: str = Field(min_length=1, max_length=PASSWORD_MAX_LENGTH)
    new_password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH)
    new_password_confirmation: str = Field(
        min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH
    )

    @model_validator(mode="after")
    def _passwords_match(self) -> "PasswordChangeRequest":
        if self.new_password != self.new_password_confirmation:
            raise ValueError("new_password_confirmation does not match new_password")
        return self


class PasswordResetRequest(BaseModel):
    """パスワード忘れによる再設定要求。"""

    email: EmailStr


class PasswordResetConfirmRequest(BaseModel):
    """仮パスワード変更 URL からの新パスワード設定。"""

    token: uuid.UUID
    new_password: str = Field(min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH)
    new_password_confirmation: str = Field(
        min_length=PASSWORD_MIN_LENGTH, max_length=PASSWORD_MAX_LENGTH
    )

    @model_validator(mode="after")
    def _passwords_match(self) -> "PasswordResetConfirmRequest":
        if self.new_password != self.new_password_confirmation:
            raise ValueError("new_password_confirmation does not match new_password")
        return self


class AcceptedResponse(BaseModel):
    status: str = "accepted"
    detail: str
