"""原画像・胸部野の取得 API（設計書 9.6）。

binary endpoint は Cache-Control: private とし、認可・監査の対象とする。
存在しない ID と閲覧権限のない ID を区別せず情報漏洩させない。
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict

from fastapi import APIRouter, Query, Request, Response
from nihcc_rag import config, db, imaging, problem
from prometheus_client import Counter
from nihcc_rag.contract import ChestAssetInfo, ImageMetadata

from ..deps import Viewer, audit
from ..services import chest as chest_store

_LOG = logging.getLogger("backend-api.images")
router = APIRouter(prefix="/api/v1/images", tags=["images"])

PRIVATE_CACHE = {"Cache-Control": "private, max-age=300"}

# 表示用派生画像（サムネイル・ROI overlay）の短時間キャッシュ（設計書 10.3）。
# プロセス内メモリのみで、TTL 経過後に破棄する。取得は認証・認可済みの
# エンドポイント経由に限られ、ディスクへは書き出さない。
_CACHE_TTL_SECONDS = config.env_int("DERIVED_IMAGE_CACHE_TTL_SECONDS", 300)
_CACHE_MAX_ENTRIES = config.env_int("DERIVED_IMAGE_CACHE_ENTRIES", 256)
_derived_cache: "OrderedDict[str, tuple[float, str, bytes]]" = OrderedDict()
_cache_lock = threading.Lock()

DERIVED_CACHE = Counter(
    "derived_image_cache_total", "Derived image cache outcome", ["kind", "outcome"]
)


def _cache_get(key: str) -> tuple[str, bytes] | None:
    with _cache_lock:
        entry = _derived_cache.get(key)
        if entry is None:
            return None
        expires_at, etag, payload = entry
        if expires_at < time.monotonic():
            _derived_cache.pop(key, None)
            return None
        _derived_cache.move_to_end(key)
        return etag, payload


def _cache_put(key: str, etag: str, payload: bytes) -> None:
    with _cache_lock:
        _derived_cache[key] = (time.monotonic() + _CACHE_TTL_SECONDS, etag, payload)
        _derived_cache.move_to_end(key)
        while len(_derived_cache) > _CACHE_MAX_ENTRIES:
            _derived_cache.popitem(last=False)


def _conditional(request: Request, etag: str) -> Response | None:
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, **PRIVATE_CACHE})
    return None


def _load_binary(image_id: uuid.UUID) -> dict:
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn,
            "SELECT ib.content, ib.mime_type, si.sha256 FROM image_binary ib "
            "JOIN source_image si USING (image_id) WHERE ib.image_id = %s",
            (str(image_id),),
        )
    if row is None:
        raise problem.not_found("image")
    return row


@router.get("/{image_id}", response_model=ImageMetadata)
def metadata(image_id: uuid.UUID, principal: Viewer) -> ImageMetadata:
    with db.metadata_conn() as conn:
        image = db.fetch_one(
            conn,
            """
            SELECT si.*, COALESCE(array_agg(f.code ORDER BY f.code)
                     FILTER (WHERE f.code IS NOT NULL), '{}') AS finding_labels,
                   MIN(imf.label_source) AS label_source
              FROM source_image si
              LEFT JOIN image_finding imf ON imf.image_id = si.image_id
              LEFT JOIN finding f ON f.finding_id = imf.finding_id
             WHERE si.image_id = %s
             GROUP BY si.image_id
            """,
            (str(image_id),),
        )
        if image is None:
            raise problem.not_found("image")
        chest = chest_store.latest_chest(conn, str(image_id))
        stored = db.fetch_one(
            conn,
            "SELECT EXISTS (SELECT 1 FROM image_vector_state "
            "WHERE image_id = %s AND state = 'STORED') AS ok",
            (str(image_id),),
        )
        estimates = chest_store.finding_estimates(conn, [str(image_id)]).get(str(image_id))
    return ImageMetadata(
        image_id=image["image_id"],
        source_image_name=image["source_image_name"],
        patient_ref=f"P{image['patient_id']:07d}",
        sha256=image["sha256"],
        width=image["width"],
        height=image["height"],
        view_position=image["view_position"],
        patient_age=image["patient_age"],
        patient_gender=image["patient_gender"],
        source_dataset_version=image["source_dataset_version"],
        finding_labels=list(image["finding_labels"] or []),
        label_source=image["label_source"] or "NIHCC_CSV_NLP_DERIVED",
        chest=(
            ChestAssetInfo(
                bbox=chest["bbox"],
                areas=dict(chest["areas"] or {}),
                roi_confidence=chest["roi_confidence"],
                quality_status=chest["quality_status"],
                flags=list(chest["flags"] or []),
                roi_model_version=chest["roi_model_version"],
                preprocess_version=chest["preprocess_version"],
                has_embedding=bool(stored and stored["ok"]),
            )
            if chest
            else None
        ),
        # 画像の詳細では 14 所見すべてを返す（一覧は上位 3 件）
        finding_estimates=chest_store.ranked_estimates(estimates[1]) if estimates else [],
        estimate_model_version=estimates[0] if estimates else None,
        source_metadata=image["source_metadata"],
    )


@router.get("/{image_id}/content")
def content(image_id: uuid.UUID, principal: Viewer, request: Request) -> Response:
    row = _load_binary(image_id)
    etag = f'"{row["sha256"]}"'
    not_modified = _conditional(request, etag)
    if not_modified is not None:
        return not_modified
    audit(
        principal.subject,
        "IMAGE_VIEW",
        target_type="image",
        target_id=str(image_id),
        trace_id=getattr(request.state, "trace_id", None),
    )
    return Response(
        bytes(row["content"]),
        media_type=row["mime_type"],
        headers={"ETag": etag, **PRIVATE_CACHE},
    )


@router.get("/{image_id}/thumbnail")
def thumbnail(
    image_id: uuid.UUID,
    principal: Viewer,
    request: Request,
    max_size: int = Query(default=512, ge=64, le=1024),
) -> Response:
    key = f"thumb:{image_id}:{max_size}"
    cached = _cache_get(key)
    if cached is not None:
        DERIVED_CACHE.labels("thumbnail", "hit").inc()
        etag, payload = cached
        return _conditional(request, etag) or Response(
            payload, media_type="image/png", headers={"ETag": etag, **PRIVATE_CACHE}
        )

    DERIVED_CACHE.labels("thumbnail", "miss").inc()
    row = _load_binary(image_id)
    payload = imaging.make_thumbnail(bytes(row["content"]), max_size)
    etag = f'"{row["sha256"]}-thumb{max_size}"'
    _cache_put(key, etag, payload)
    return _conditional(request, etag) or Response(
        payload, media_type="image/png", headers={"ETag": etag, **PRIVATE_CACHE}
    )


def _derived(request: Request, key: str, kind: str, build) -> Response:
    """派生画像（外周・胸部野）を短時間キャッシュ付きで返す。build は (etag, payload)。"""
    cached = _cache_get(key)
    if cached is not None:
        DERIVED_CACHE.labels(kind, "hit").inc()
        etag, payload = cached
        return _conditional(request, etag) or Response(
            payload, media_type="image/png", headers={"ETag": etag, **PRIVATE_CACHE}
        )
    DERIVED_CACHE.labels(kind, "miss").inc()
    etag, payload = build()
    _cache_put(key, etag, payload)
    return _conditional(request, etag) or Response(
        payload, media_type="image/png", headers={"ETag": etag, **PRIVATE_CACHE}
    )


@router.get("/{image_id}/lung-outline")
def lung_outline_image(
    image_id: uuid.UUID,
    principal: Viewer,
    request: Request,
) -> Response:
    """肺野全体の外周を原画像へ描いた確認用画像。

    胸郭ランキング（`shape.describe_lung`）が何を「肺野」として扱ったかを
    目で確かめるための表示。脊柱・心陰影・横隔膜より下が輪郭の外側にあるかを
    見れば、肺野が正しく抜けているかが判る。
    """

    def build() -> tuple[str, bytes]:
        with db.metadata_conn() as conn:
            row = chest_store.latest_chest(conn, str(image_id))
        if row is None:
            raise problem.not_found("lung masks")
        parent = _load_binary(image_id)
        try:
            payload = imaging.lung_outline(
                bytes(parent["content"]),
                [(bytes(row["lung_mask"]), chest_store.source_bbox(row["bbox"]))],
            )
        except ValueError as exc:
            raise problem.not_found("lung masks") from exc
        return f'"{parent["sha256"]}-lung-outline-v2"', payload

    return _derived(request, f"lung:{image_id}", "lung", build)


@router.get("/{image_id}/chest")
def chest_image(
    image_id: uuid.UUID,
    principal: Viewer,
    request: Request,
) -> Response:
    """原画像へ胸部野（肺野＋縦隔＋横隔膜）の輪郭を描いた画像。

    検索は胸部野どうしを比べる。結果の各候補で
    何を比べたかを目で確かめられるようにする。
    """

    def build() -> tuple[str, bytes]:
        with db.metadata_conn() as conn:
            row = chest_store.latest_chest(conn, str(image_id))
        if row is None:
            raise problem.not_found("chest mask")
        parent = _load_binary(image_id)
        try:
            payload = chest_store.chest_overlay(
                bytes(parent["content"]),
                bytes(row["chest_mask"]),
                chest_store.source_bbox(row["bbox"]),
            )
        except ValueError as exc:
            raise problem.not_found("chest mask") from exc
        return f'"{parent["sha256"]}-chest-v1"', payload

    return _derived(request, f"chest:{image_id}", "chest", build)
