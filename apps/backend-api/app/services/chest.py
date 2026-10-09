"""胸部野（肺野＋縦隔＋横隔膜）の保存値の読み出しと、表示用の補助。

画像 API と検索の両方が使う。検索の単位は画像 1 枚の胸部野なので、
部位ごとの表（region_asset）は読まない。
"""

from __future__ import annotations

from typing import Any

from nihcc_rag import db, imaging
from nihcc_rag.contract import FindingEstimate

#: 画面で示す推定の件数
ESTIMATE_TOP = 3
#: 胸部野の輪郭の色（評価の目視一覧・roi-service の確認画像と同じ橙）
CHEST_COLOR = (255, 140, 0)


def latest_chest(conn, image_id: str) -> dict[str, Any] | None:
    """最新の ROI 版の胸部野。再取り込みの前後で版を混ぜない。"""
    return db.fetch_one(
        conn,
        "SELECT chest_mask, lung_mask, bbox, areas, flags, roi_confidence, "
        "quality_status, roi_model_version, preprocess_version "
        "FROM chest_asset WHERE image_id = %s ORDER BY created_at DESC LIMIT 1",
        (image_id,),
    )


def quality_by_image(conn, image_ids: list[str]) -> dict[str, str]:
    """image_id -> 胸部野の品質（最新の ROI 版）。"""
    if not image_ids:
        return {}
    rows = db.fetch_all(
        conn,
        "SELECT DISTINCT ON (image_id) image_id, quality_status FROM chest_asset "
        "WHERE image_id = ANY(%s::uuid[]) ORDER BY image_id, created_at DESC",
        (image_ids,),
    )
    return {str(r["image_id"]): r["quality_status"] for r in rows}


def finding_estimates(conn, image_ids: list[str]) -> dict[str, tuple[str, dict[str, float]]]:
    """image_id -> (分類器の版, {所見: 確率})。画像ごとに最新の分類器の推定を採る。"""
    if not image_ids:
        return {}
    rows = db.fetch_all(
        conn,
        "SELECT DISTINCT ON (image_id) image_id, model_version, probabilities "
        "FROM finding_estimate WHERE image_id = ANY(%s::uuid[]) "
        "ORDER BY image_id, created_at DESC",
        (image_ids,),
    )
    return {str(r["image_id"]): (r["model_version"], dict(r["probabilities"])) for r in rows}


def ranked_estimates(
    probabilities: dict[str, float], limit: int | None = None
) -> list[FindingEstimate]:
    """確率の高い順に並べる。limit を指定すると上位だけ返す。"""
    ordered = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    if limit is not None:
        ordered = ordered[:limit]
    return [FindingEstimate(finding=f, probability=round(float(p), 4)) for f, p in ordered]


def source_bbox(meta: dict[str, Any] | None) -> list[int] | None:
    """保存した bbox 情報から、原画像座標の外接矩形を取り出す。"""
    meta = meta or {}
    bbox = meta.get("bbox")
    scale = float(meta.get("work_scale") or 1.0)
    return meta.get("source_bbox") or ([int(round(v / scale)) for v in bbox] if bbox else None)


def chest_overlay(parent_png: bytes, chest_mask_png: bytes, bbox: list[int] | None) -> bytes:
    """原画像へ胸部野の輪郭を描いた画像。何を比べたかを目で確かめるための表示。"""
    return imaging.lung_outline(parent_png, [(chest_mask_png, bbox)], color=CHEST_COLOR)
