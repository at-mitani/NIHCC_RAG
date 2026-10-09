"""Embedding Service: ROI 画像・英語テキストの埋め込み生成（設計書 7.3）。

出力は必ず L2 正規化し、cosine 距離で検索できるようにする。
モデル名・revision・次元は応答へ常に含め、DB 側の version 列と一致させる。

2 段検索（coarse -> fine）に対応するため、backend を 2 つ持てる。

  coarse: 必須。HNSW index を張る低次元空間（image_embedding.embedding）。
  fine  : 任意。index を張らず厳密な再スコアリングにのみ使う空間
          （image_embedding.embedding_fine）。EMBEDDING_FINE_BACKEND で有効化する。
          採用構成は chest-profile（胸部野の分類器のプロファイル）。

取り込みと画像検索の入口は `/api/v1/embeddings/chest`（胸部野 1 単位）。

fine が無効なとき応答の fine_* は None であり、呼び出し側は 1 段検索になる。
すなわち「fine ベクトルが返ってきたか」が 2 段検索の可否の単一の判断基準である。
"""

from __future__ import annotations

import base64
import binascii
import logging
import time

import numpy as np
from nihcc_rag import config, problem, service
from nihcc_rag.contract import (
    ChestMapRequest,
    ChestMapResponse,
    EmbedChestRequest,
    EmbedChestResponse,
    EmbedImagesRequest,
    EmbedResponse,
    EmbedTextsRequest,
    RegionPoolItem,
    RegionPoolRequest,
    RegionPoolResponse,
)
from prometheus_client import Counter, Histogram

from .backends import EmbeddingBackend, build_backend, build_fine_backend

_LOG = logging.getLogger("embedding-service")

EMBEDDINGS = Counter("embeddings_total", "Generated embeddings", ["kind"])
DURATION = Histogram(
    "embedding_duration_seconds",
    "Embedding duration",
    ["kind"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

app = service.create_app("embedding-service", title="NIHCC RAG Embedding Service")

_backend: EmbeddingBackend | None = None
_fine_backend: EmbeddingBackend | None = None
_ready = False


def backend() -> EmbeddingBackend:
    if _backend is None:
        raise problem.dependency_unavailable("embedding model")
    return _backend


@app.on_event("startup")
def _startup() -> None:
    global _backend, _fine_backend, _ready
    started = time.perf_counter()
    _backend = build_backend()
    _backend.self_test()
    _backend.warmup(config.env_int("MODEL_WARMUP_BATCHES", 3))

    _fine_backend = build_fine_backend()
    if _fine_backend is not None:
        # coarse と同じモデル・同じ次元では 2 段目に情報が増えず、推論を 2 回
        # 行うだけになる。設定間違いを黙って受け入れると「精度が上がったつもりで
        # 遅くなっただけ」の状態になるため、起動時に落とす。
        if _fine_backend.model_version == _backend.model_version:
            raise RuntimeError(
                "EMBEDDING_FINE_BACKEND produces the same model_version as the coarse "
                f"backend ({_backend.model_version}). a fine space that carries no extra "
                "information only doubles inference cost; configure a different model or "
                "set EMBEDDING_FINE_BACKEND=none"
            )
        _fine_backend.self_test()
        _fine_backend.warmup(config.env_int("MODEL_WARMUP_BATCHES", 3))
        if _fine_backend.dimension <= _backend.dimension:
            # 致命的ではない（2 段目の厳密計算自体は成立する）ため警告に留める。
            _LOG.warning(
                "fine embedding dimension is not larger than the coarse one; "
                "the second stage adds exact re-scoring but no extra representation capacity",
                extra={
                    "fields": {
                        "coarse_dimension": _backend.dimension,
                        "fine_dimension": _fine_backend.dimension,
                    }
                },
            )
    _ready = True
    _LOG.info(
        "embedding backend ready",
        extra={
            "fields": {
                "backend": _backend.name,
                "model_version": _backend.model_version,
                "model_name": _backend.model_name,
                "revision": _backend.model_revision,
                "dimension": _backend.dimension,
                "fine_backend": _fine_backend.name if _fine_backend else None,
                "fine_model_version": _fine_backend.model_version if _fine_backend else None,
                "fine_dimension": _fine_backend.dimension if _fine_backend else None,
                "fine_supports_text": _fine_backend.supports_text if _fine_backend else None,
                "startup_seconds": round(time.perf_counter() - started, 2),
            }
        },
    )


@app.get("/api/v1/health/live")
def live() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/health/ready")
def ready() -> dict[str, object]:
    if not _ready or _backend is None:
        raise problem.dependency_unavailable("embedding model")
    return {
        "status": "ok",
        "backend": _backend.name,
        "embedding_model_version": _backend.model_version,
        "dimension": _backend.dimension,
        # 2 段検索の可否は呼び出し側がこの値で判断できる
        "rerank_available": _fine_backend is not None,
        "fine_backend": _fine_backend.name if _fine_backend else None,
        "fine_embedding_model_version": _fine_backend.model_version if _fine_backend else None,
        "fine_dimension": _fine_backend.dimension if _fine_backend else None,
    }


@app.get("/api/v1/model")
def model_info() -> dict[str, object]:
    engine = backend()
    fine = _fine_backend
    return {
        "backend": engine.name,
        "embedding_model_version": engine.model_version,
        "embedding_model_name": engine.model_name,
        "embedding_model_revision": engine.model_revision,
        "embedding_dimension": engine.dimension,
        "embedding_normalization": "L2",
        "distance_metric": "cosine",
        "image_input": "RGB 224x224 (official preprocess)" if engine.name == "biomedclip" else "grayscale statistics",
        "text_language": "en",
        "hybrid_image_weight_default": config.env_float("HYBRID_IMAGE_WEIGHT", 0.70),
        # --- 2 段検索の 2 段目 ---
        "rerank_available": fine is not None,
        "fine_backend": fine.name if fine else None,
        "fine_embedding_model_version": fine.model_version if fine else None,
        "fine_embedding_model_name": fine.model_name if fine else None,
        "fine_embedding_model_revision": fine.model_revision if fine else None,
        "fine_dimension": fine.dimension if fine else None,
        "fine_supports_text": fine.supports_text if fine else None,
        # --- 胸部野の分類器（2 段目が chest-profile のとき）。ingest-worker が
        # roi-service の申告と突き合わせる（特徴の作り方の一致）
        "chest_head": fine.describe() if fine is not None and fine.supports_chest else None,
        # --- 画像 1 回の順伝播 + mask pooling（評価スクリプト用） ---
        "region_pool_backend": _pool_backend().name if _pool_backend() else None,
        "region_pool_patch_grid": (
            _pool_backend().patch_grid if _pool_backend() else None
        ),
    }


def _pool_backend() -> EmbeddingBackend | None:
    """部位 mask pooling に使う backend を選ぶ。

    2 段目（fine）を優先する。評価では fine に RAD-DINO を置く構成で動かすため。
    fine が pooling を持たない構成でも 1 段目が持っていれば使えるようにする。
    """
    for engine in (_fine_backend, _backend):
        if engine is not None and engine.supports_region_pool:
            return engine
    return None


def _decode(images_base64: list[str]) -> list[bytes]:
    decoded: list[bytes] = []
    for index, item in enumerate(images_base64):
        try:
            decoded.append(base64.b64decode(item, validate=True))
        except (binascii.Error, ValueError) as exc:
            raise problem.unprocessable(
                "IMAGE_BASE64_INVALID", f"images_base64[{index}] is not valid base64"
            ) from exc
    return decoded


def _round(vectors: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 7) for v in row] for row in vectors]


def _response(
    engine: EmbeddingBackend,
    vectors: np.ndarray,
    *,
    fine_engine: EmbeddingBackend | None = None,
    fine_vectors: np.ndarray | None = None,
) -> EmbedResponse:
    return EmbedResponse(
        embedding_model_version=engine.model_version,
        embedding_model_name=engine.model_name,
        embedding_model_revision=engine.model_revision,
        dimension=engine.dimension,
        vectors=_round(vectors),
        fine_embedding_model_version=fine_engine.model_version if fine_engine else None,
        fine_embedding_model_name=fine_engine.model_name if fine_engine else None,
        fine_embedding_model_revision=fine_engine.model_revision if fine_engine else None,
        fine_dimension=fine_engine.dimension if fine_engine else None,
        fine_vectors=_round(fine_vectors) if fine_vectors is not None else None,
    )


def _encode(
    engine: EmbeddingBackend, kind: str, encode, payload, *, extra=None
) -> np.ndarray:
    """1 つの backend で推論し、所要時間と件数を記録する。

    推論失敗は 503 とし、上位（backend-api / ingest-worker）のリトライへ委ねる。
    """
    started = time.perf_counter()
    try:
        vectors = encode(payload, extra) if extra is not None else encode(payload)
    except problem.ProblemError:
        raise
    except Exception as exc:  # noqa: BLE001 - 推論失敗は 503 として上位でリトライ
        _LOG.exception("%s embedding failed", kind)
        raise problem.dependency_unavailable("embedding model") from exc
    DURATION.labels(kind).observe(time.perf_counter() - started)
    EMBEDDINGS.labels(kind).inc(len(payload))
    return vectors


@app.post("/api/v1/embeddings/images", response_model=EmbedResponse)
def embed_images(request: EmbedImagesRequest) -> EmbedResponse:
    engine = backend()
    payload = _decode(request.images_base64)
    codes = list(request.region_codes) if request.region_codes else None
    vectors = _encode(
        engine, "image", engine.encode_images, payload,
        extra=codes if engine.needs_region_codes else None,
    )
    fine = _fine_backend
    fine_vectors = None
    if fine is not None and fine.supports_chest:
        # 胸部野の分類器は mask を要するので crop だけでは推論できない。
        # 2 段目は /embeddings/chest で行う（このエンドポイントは 1 段目だけ返す）。
        fine = None
    if fine is not None:
        if fine.needs_region_codes and codes is None:
            # 部位コードが要る 2 段目モデルに対し、呼び出し側が渡していない。
            # 1 段目は成立するので全体を失敗させず、2 段目だけ落として明示する。
            _LOG.warning(
                "fine backend requires region_codes but none were provided; "
                "returning coarse vectors only",
                extra={"fields": {"fine_backend": fine.name, "images": len(payload)}},
            )
        else:
            fine_vectors = _encode(
                fine, "image_fine", fine.encode_images, payload,
                extra=codes if fine.needs_region_codes else None,
            )
    return _response(engine, vectors, fine_engine=fine, fine_vectors=fine_vectors)


@app.post("/api/v1/embeddings/chest", response_model=EmbedChestResponse)
def embed_chest(request: EmbedChestRequest) -> EmbedChestResponse:
    """胸部野の埋め込み（取り込みと画像検索の入口）。

    1 段目: 胸部野 crop を coarse backend（BiomedCLIP）へ
    2 段目: 全体画像 + 胸部野 mask を chest-profile backend へ。プロファイルと
            14 所見の確率を返す

    2 段目が chest-profile でない構成では 1 段目だけを返し、呼び出し側は 1 段検索に
    なる（fine_* が None）。
    """
    engine = backend()
    crops = _decode(request.crops_base64)
    images = _decode(request.images_base64)
    masks = _decode(request.masks_base64)
    vectors = _encode(engine, "chest", engine.encode_images, crops)

    fine = _fine_backend
    if fine is None or not fine.supports_chest:
        return EmbedChestResponse(
            embedding_model_version=engine.model_version,
            embedding_model_name=engine.model_name,
            embedding_model_revision=engine.model_revision,
            dimension=engine.dimension,
            vectors=_round(vectors),
        )

    started = time.perf_counter()
    try:
        profiles, probs = fine.encode_chest(images, masks)
    except problem.ProblemError:
        raise
    except Exception as exc:  # noqa: BLE001 - 推論失敗は 503 として上位でリトライ
        _LOG.exception("chest profile embedding failed")
        raise problem.dependency_unavailable("embedding model") from exc
    DURATION.labels("chest_fine").observe(time.perf_counter() - started)
    EMBEDDINGS.labels("chest_fine").inc(len(images))
    return EmbedChestResponse(
        embedding_model_version=engine.model_version,
        embedding_model_name=engine.model_name,
        embedding_model_revision=engine.model_revision,
        dimension=engine.dimension,
        vectors=_round(vectors),
        fine_embedding_model_version=fine.model_version,
        fine_embedding_model_name=fine.model_name,
        fine_embedding_model_revision=fine.model_revision,
        fine_dimension=fine.dimension,
        fine_vectors=_round(profiles),
        findings=list(fine.head.findings),
        probabilities=[[round(float(v), 5) for v in row] for row in probs],
    )


@app.post("/api/v1/embeddings/chest-map", response_model=ChestMapResponse)
def embed_chest_map(request: ChestMapRequest) -> ChestMapResponse:
    """区画ごとの所見の地図（評価用。scripts/chest_lesion_map_eval.py）。

    分類器のロジットを区画ごとの寄与へ正確に分解したもの。病変の位置を示すかは
    正解 box で測ってから使う。
    """
    fine = _fine_backend
    if fine is None or not fine.supports_chest:
        raise problem.unprocessable(
            "CHEST_MAP_UNAVAILABLE",
            "the fine backend is not chest-profile; set EMBEDDING_FINE_BACKEND=chest-profile",
        )
    images = _decode(request.images_base64)
    masks = _decode(request.masks_base64)
    started = time.perf_counter()
    try:
        maps, probs, findings = fine.chest_map(images, masks, request.findings)
    except ValueError as exc:
        raise problem.unprocessable("FINDING_UNKNOWN", str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - 推論失敗は 503 として上位でリトライ
        _LOG.exception("chest map failed")
        raise problem.dependency_unavailable("embedding model") from exc
    DURATION.labels("chest_map").observe(time.perf_counter() - started)
    return ChestMapResponse(
        fine_embedding_model_version=fine.model_version,
        grid=int(maps.shape[-1]),
        findings=findings,
        maps=[[[round(float(v), 4) for v in m.ravel()] for m in image] for image in maps],
        probabilities=[[round(float(v), 5) for v in row] for row in probs],
    )


@app.post("/api/v1/embeddings/region-pool", response_model=RegionPoolResponse)
def embed_region_pool(request: RegionPoolRequest) -> RegionPoolResponse:
    """画像 1 回の順伝播で mask ごとのベクトルを返す（評価スクリプト用）。

    既存の `/embeddings/images` とは入出力の個数の関係が違う。あちらは
    「crop N 枚 -> ベクトル N 本」だが、ここは「画像 N 枚 -> 最大 5N 本」であり、
    mask が空の部位は返さない。呼び出し側が突き合わせられるよう、結果には
    画像 index と部位コードを必ず添える。
    """
    engine = _pool_backend()
    if engine is None:
        raise problem.unprocessable(
            "REGION_POOL_UNAVAILABLE",
            "no configured backend supports region pooling; "
            "set EMBEDDING_FINE_BACKEND=rad-dino",
        )
    images = _decode(request.images_base64)
    masks: list[dict[str, bytes]] = []
    for index, item in enumerate(request.masks_base64):
        decoded: dict[str, bytes] = {}
        for code, value in item.items():
            try:
                decoded[code] = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise problem.unprocessable(
                    "IMAGE_BASE64_INVALID",
                    f"masks_base64[{index}][{code}] is not valid base64",
                ) from exc
        masks.append(decoded)

    started = time.perf_counter()
    try:
        pooled = engine.encode_region_pool(images, masks)
    except problem.ProblemError:
        raise
    except Exception as exc:  # noqa: BLE001 - 推論失敗は 503 として上位でリトライ
        _LOG.exception("region pool embedding failed")
        raise problem.dependency_unavailable("embedding model") from exc
    DURATION.labels("region_pool").observe(time.perf_counter() - started)
    EMBEDDINGS.labels("region_pool").inc(len(pooled))

    return RegionPoolResponse(
        model_version=engine.model_version,
        model_name=engine.model_name,
        model_revision=engine.model_revision,
        dimension=engine.dimension,
        patch_grid=engine.patch_grid,
        items=[
            RegionPoolItem(
                image_index=index,
                region_code=code,
                vector=[round(float(v), 7) for v in vector],
            )
            for index, code, vector in pooled
        ],
    )


@app.post("/api/v1/embeddings/texts", response_model=EmbedResponse)
def embed_texts(request: EmbedTextsRequest) -> EmbedResponse:
    engine = backend()
    vectors = _encode(engine, "text", engine.encode_texts, request.texts)
    fine = _fine_backend
    # fine モデルが text encoder を共有空間に持たない場合、テキストは fine 空間へ
    # 写せない。その場合 fine_vectors は None を返し、呼び出し側は
    # TEXT_TO_IMAGE / HYBRID で 2 段目を行わない（画像モードのみ 2 段になる）。
    if fine is not None and not fine.supports_text:
        return _response(engine, vectors, fine_engine=fine, fine_vectors=None)
    fine_vectors = (
        _encode(fine, "text_fine", fine.encode_texts, request.texts) if fine is not None else None
    )
    return _response(engine, vectors, fine_engine=fine, fine_vectors=fine_vectors)
