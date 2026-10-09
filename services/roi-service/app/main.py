"""ROI Service: 胸部野（肺野＋縦隔＋横隔膜）の抽出（設計書 5.1 / 7.1.3）。

検索の単位は画像 1 枚の胸部野である。

返すもの:
  - 胸部野と肺野の mask（胸部野の外接矩形で切った作業解像度の PNG）
  - 品質（OK / ROI_UNCERTAIN）と旗。抽出できない画像は 422
  - 胸郭の形状ベクトル（16 次元）
  - 埋め込みモデルへそのまま渡す入力（pooling 用の画像と mask、1 段目用の crop）

**埋め込みモデルへの入力はここで作り、呼び出し側では作り直さない。** 胸部野の
分類器は評価スクリプトが作った特徴で学習している。入力の作り方が 1 画素でも違うと
確率が黙ってずれるため、評価スクリプトと同じ `image_pipeline.chest_model_inputs` を
通す。

推論結果はすべて roi_model_version（algorithm version + parameter hash）付きで返し、
再現可能にする。
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import time

import numpy as np
from fastapi import Response
from image_pipeline import chest as chest_spec
from image_pipeline import segmenter as segmentation
from image_pipeline import (
    ExtractionError,
    available_profiles,
    chest_model_inputs,
    encode_png,
    extract_chest,
    load_profile,
)
from nihcc_rag import config, problem, service, shape
from nihcc_rag.contract import ChestRoiRequest, ChestRoiResponse
from prometheus_client import Counter, Histogram

_LOG = logging.getLogger("roi-service")

PROFILE_NAME = os.environ.get("ROI_PROFILE", "nihcc-unet-local-v1")
PROFILE_DIR = os.environ.get("ROI_PROFILE_DIR") or None
MAX_IMAGE_BYTES = config.env_int("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)

EXTRACTIONS = Counter("roi_extractions_total", "Chest extractions", ["status"])
DURATION = Histogram(
    "roi_extraction_duration_seconds",
    "Chest extraction duration",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

app = service.create_app("roi-service", title="NIHCC RAG ROI Service")


@app.on_event("startup")
def _startup() -> None:
    profile = load_profile(PROFILE_NAME, PROFILE_DIR)
    _LOG.info(
        "roi profile loaded",
        extra={
            "fields": {
                "profile": profile.name,
                "algorithm_version": profile.algorithm_version,
                "parameter_hash": profile.parameter_hash[:12],
            }
        },
    )
    _self_test(profile)


def _self_test(profile) -> None:
    """起動時 self-test（設計書 15.4 相当）。

    胸部野には U-Net の心臓 mask が要る。規則ベースの profile は心臓を出さない
    ので、**ここで起動を止める**（取り込みの 1 件目で全件 422 になるより早く気づく）。

    U-Net は人工的な図に肺野を見つけないのが正しい挙動なので、合成画像で抽出まで
    通すことはしない。配備で壊れやすいところを直接見る。

      - モデルのファイルが在って読めるか（mount の失敗・持ち込み漏れ）
      - onnxruntime が動くか
      - 出力の形が profile の input_size と噛み合うか
    """
    segmenter = segmentation.from_profile(profile)
    if segmenter is None:
        raise RuntimeError(
            f"ROI_PROFILE={profile.name} は U-Net を使わない。胸部野には心臓の mask が要るので "
            "U-Net の profile（nihcc-unet-local-v1 / nihcc-unet-v1）を指定すること"
        )
    size = segmenter.input_size
    probe = np.full((size, size), 128, dtype=np.uint8)
    result = segmenter.predict(probe)
    if result.right.shape != probe.shape:
        raise RuntimeError(
            f"self-test: segmentation returned {result.right.shape}, expected {probe.shape}"
        )
    _LOG.info(
        "roi self-test passed",
        extra={"fields": {"backend": "unet", "model": str(segmenter.model_path),
                          "input_size": size}},
    )


@app.get("/api/v1/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/health/ready")
def ready() -> dict[str, object]:
    profile = load_profile(PROFILE_NAME, PROFILE_DIR)
    return {
        "status": "ok",
        "profile": profile.name,
        "roi_model_version": profile.roi_model_version,
        "available_profiles": available_profiles(PROFILE_DIR),
    }


@app.get("/api/v1/model")
def model_info() -> dict[str, object]:
    profile = load_profile(PROFILE_NAME, PROFILE_DIR)
    return {
        "roi_model_version": profile.roi_model_version,
        "algorithm_version": profile.algorithm_version,
        "profile_name": profile.name,
        "parameter_hash": profile.parameter_hash,
        "preprocess_version": config.preprocess_version(),
        "unit": "chest",
        # 胸部野 mask と埋め込み入力の作り方。ingest-worker が embedding-service の
        # 分類器（chest_head）の申告と突き合わせる。ずれていると確率が黙ってずれる
        "chest_mask": {
            "diaphragm_ratio": chest_spec.DIAPHRAGM_RATIO,
            "margin_ratio": chest_spec.MARGIN_RATIO,
            "pool_side": chest_spec.POOL_SIDE,
        },
        "config": profile.data,
    }


def _decode_image(image_base64: str) -> bytes:
    try:
        data = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise problem.unprocessable("IMAGE_BASE64_INVALID", "image_base64 is not valid") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise problem.ProblemError(
            413, "IMAGE_TOO_LARGE", "Payload too large", "Image exceeds the configured limit."
        )
    return data


def _profile_name(requested: str | None) -> str:
    if requested and requested != PROFILE_NAME:
        # API 入力で任意の設定値を本番適用させない（設計書 6.8）
        if requested not in available_profiles(PROFILE_DIR):
            raise problem.unprocessable(
                "ROI_PROFILE_NOT_ALLOWED", f"profile '{requested}' is not registered"
            )
    return requested or PROFILE_NAME


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@app.post("/api/v1/roi/chest", response_model=ChestRoiResponse)
def extract_chest_roi(request: ChestRoiRequest) -> ChestRoiResponse:
    data = _decode_image(request.image_base64)
    started = time.perf_counter()
    try:
        extraction = extract_chest(
            data, profile_name=_profile_name(request.profile), profile_dir=PROFILE_DIR
        )
    except ExtractionError as exc:
        EXTRACTIONS.labels("rejected").inc()
        raise problem.unprocessable(exc.code, str(exc)) from exc
    inputs = chest_model_inputs(extraction)
    DURATION.observe(time.perf_counter() - started)
    EXTRACTIONS.labels(extraction.quality_status).inc()

    x0, y0, x1, y1 = extraction.chest_bbox
    try:
        lung_vector = shape.lung_from_masks(extraction.right, extraction.left)
    except Exception as exc:  # noqa: BLE001 - 形状が無くても取り込み・検索は成立する
        _LOG.warning("lung shape failed", extra={"fields": {"error": f"{type(exc).__name__}: {exc}"}})
        lung_vector = None

    profile = extraction.profile
    return ChestRoiResponse(
        roi_model_version=profile.roi_model_version,
        preprocess_version=config.preprocess_version(),
        profile_name=profile.name,
        parameter_hash=profile.parameter_hash,
        quality_status=extraction.quality_status,
        roi_confidence=extraction.confidence,
        flags=extraction.flags,
        areas={k: round(v, 5) for k, v in extraction.areas.items()},
        bbox=[x0, y0, x1, y1],
        source_bbox=extraction.source_bbox,
        work_scale=extraction.scale,
        chest_mask_base64=_b64(encode_png(extraction.chest[y0:y1, x0:x1])),
        lung_mask_base64=_b64(encode_png(extraction.lung[y0:y1, x0:x1])),
        lung_vector=(
            [round(float(v), 7) for v in lung_vector]
            if lung_vector is not None and lung_vector.any()
            else None
        ),
        pool_image_base64=_b64(inputs.pool_image),
        pool_mask_base64=_b64(inputs.pool_masks["CHEST"]),
        crop_base64=_b64(inputs.crop),
    )


@app.post("/api/v1/roi/preview")
def preview(request: ChestRoiRequest) -> Response:
    """胸部野（橙）と肺野（青）の輪郭を重ねた確認用 PNG（確認画面・障害調査用）。"""
    import cv2

    data = _decode_image(request.image_base64)
    try:
        extraction = extract_chest(
            data, profile_name=_profile_name(request.profile), profile_dir=PROFILE_DIR
        )
    except ExtractionError as exc:
        raise problem.unprocessable(exc.code, str(exc)) from exc

    canvas = cv2.cvtColor(extraction.normalized, cv2.COLOR_GRAY2BGR)
    for mask, colour in ((extraction.lung, (255, 160, 0)), (extraction.chest, (0, 165, 255))):
        contours, _ = cv2.findContours(
            (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, colour, 2)
    label = f"{extraction.quality_status} {','.join(extraction.flags)}".strip()
    cv2.putText(canvas, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
                cv2.LINE_AA)
    ok, buffer = cv2.imencode(".png", canvas)
    if not ok:
        raise problem.ProblemError(500, "PNG_ENCODE_FAILED", "Encoding failed", "preview failed")
    return Response(buffer.tobytes(), media_type="image/png")
