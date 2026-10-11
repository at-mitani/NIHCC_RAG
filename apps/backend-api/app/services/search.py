"""検索処理の中核（設計書 7.2 / 7.3.3）。

同期経路（API 直接実行）と非同期経路（search-worker）で同じ関数を使う。

検索の単位は画像 1 枚の**胸部野**（肺野＋縦隔＋横隔膜）であり、1 本の順位を返す。
各候補には、胸部野の分類器による**所見の推定**（上位 3 件）を添える。推定は診断ではなく、
データセットのラベル（NLP 由来）とも別物である。

2 段検索（coarse -> fine）
--------------------------
embedding-service が fine ベクトル（胸部野の分類器のプロファイル）を返したときだけ
2 段検索になる。すなわち「fine ベクトルが返ってきたか」が唯一の判断基準であり、
本サービス側に有効・無効のフラグは持たない（設定が 2 箇所にあると食い違うため）。

fine 空間が使えない場合は 1 段検索へ自動的に縮退し、その旨を応答の
models.rerank・rerank_status・warnings で明示する。縮退する条件:

  - EMBEDDING_FINE_BACKEND が未設定（既定）
  - 要求が rerank=false（1 段のみとの精度比較用）
  - テキストを含む検索（TEXT_TO_IMAGE / HYBRID）。プロファイルは画像から作るもので、
    テキストを同じ空間へ写せない

画像だけの検索の進め方
----------------------
画像だけの検索（fine ベクトルがあり、テキストを含まない）は、既定で 1 段目を使わず
所見プロファイルで全件を厳密に検索する（strategy = profile）。
2 段検索（coarse_to_fine）は 1 段目の BiomedCLIP が拾った候補しか並べ替えられず、
所見の合う画像を取りこぼすため。既定は IMAGE_SEARCH_STRATEGY、要求ごとに
strategy で上書きできる（品質の比較用）。
"""

from __future__ import annotations

import base64
import json
import logging
import statistics
import time
import uuid
from collections import Counter
from typing import Any

import httpx
import numpy as np
from nihcc_rag import config, csvspec, db, imaging, problem, rerank, text_norm
from nihcc_rag.contract import (
    RESEARCH_MESSAGE,
    SearchQueryEcho,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    ServerMessage,
)

from . import chest as chest_store

_LOG = logging.getLogger("backend-api.search")

#: aggregator へ渡す照会上限(ms)。contract の VectorSearchRequest が
#: 100〜60000 に制限しているため、範囲外の設定はここで丸める。
#: 丸めないと aggregator が要求そのものを 422 で弾き、利用者には原因の判らない
#: 500 INTERNAL_ERROR だけが返る（設定を直す手がかりが何も出ない）。
_VECTOR_TIMEOUT_MIN_MS, _VECTOR_TIMEOUT_MAX_MS = 100, 60000


def _vector_query_timeout_ms() -> int:
    raw = config.env_int("VECTOR_QUERY_TIMEOUT_MS", 2000)
    clamped = max(_VECTOR_TIMEOUT_MIN_MS, min(_VECTOR_TIMEOUT_MAX_MS, raw))
    if clamped != raw:
        _LOG.warning(
            "VECTOR_QUERY_TIMEOUT_MS is out of range; clamping",
            extra={
                "fields": {
                    "configured": raw,
                    "used": clamped,
                    "allowed": f"{_VECTOR_TIMEOUT_MIN_MS}-{_VECTOR_TIMEOUT_MAX_MS}",
                }
            },
        )
    return clamped


#: 検索の警告。文言は英語で、画面は id（辞書の server.<id>）で表示言語に訳す。
LOW_CONFIDENCE_WARNING = ServerMessage(
    id="QUERY_ROI_UNCERTAIN",
    text="Chest region extraction for the query image is uncertain (e.g. one lung was not "
    "found, or the left-right difference is large). Treat the results as reference only.",
)
RERANK_PARTIAL_WARNING = ServerMessage(
    id="RERANK_PARTIAL",
    text="Some candidates have no finding profile registered, so they are returned in their "
    "stage-1 order. Re-ingesting them resolves this.",
)
RERANK_TEXT_UNAVAILABLE_WARNING = ServerMessage(
    id="RERANK_TEXT_UNAVAILABLE",
    text="Finding profiles are built from images, so re-ranking is not applied to searches "
    "that include text (it is applied to image-only searches).",
)
SELF_MATCH_EXCLUDED_WARNING = ServerMessage(
    id="SELF_MATCH_EXCLUDED",
    text="The same image as the query (same file name, similarity 1.0) is excluded from "
    "the results.",
)
BUDGET_EXHAUSTED_WARNING = ServerMessage(
    id="RERANK_BUDGET_EXHAUSTED",
    text="The stage-1 search used up the timeout budget, so re-ranking was skipped. "
    "Review VECTOR_QUERY_TIMEOUT_MS or VECTOR_STAGE1_MULTIPLIER.",
)


def _suppressed_warning(confidence: float, threshold: float) -> ServerMessage:
    params = {"confidence": f"{confidence:.2f}", "threshold": f"{threshold:.2f}"}
    return ServerMessage(
        id="RESULTS_SUPPRESSED_LOW_CONFIDENCE",
        text=f"Results were suppressed because the chest region confidence of the query image "
        f"({params['confidence']}) is below the threshold ({params['threshold']}).",
        params=params,
    )


def _warning_fields(warnings: list[ServerMessage]) -> dict[str, list]:
    """応答の warnings（英語の文）と warning_details（訳すための詳細）を作る。"""
    unique = list({message.text: message for message in warnings}.values())
    return {"warnings": [message.text for message in unique], "warning_details": unique}


def _image_name(name: str | None) -> str:
    """比較用のファイル名。ブラウザによってはパス付きで送られるので末尾だけを使う。"""
    return (name or "").replace("\\", "/").rsplit("/", 1)[-1].strip().casefold()


def is_query_itself(query_image_name: str | None, candidate_name: str | None, similarity: float) -> bool:
    """候補が検索元そのもの（取り込み済みの同じ画像）か。

    同じ画像を検索しても比べる相手が自分になるだけなので、結果から外す。
    類似度は表示と同じ桁（小数 4 桁）で 1.0 のものに限る。ファイル名が同じでも
    中身の違う画像（加工した画像など）は 1.0 にならないので残る。
    """
    query = _image_name(query_image_name)
    return bool(query) and query == _image_name(candidate_name) and round(float(similarity), 4) >= 1.0


def _client(url: str, timeout: float) -> httpx.Client:
    return httpx.Client(base_url=url, timeout=timeout)


def _call(client: httpx.Client, method: str, path: str, *, name: str, **kwargs) -> httpx.Response:
    """依存サービス呼び出し。timeout / 接続失敗を problem+json へ写像する。

    設計書 9.7: 503（依存サービス停止）、504（検索 timeout）。
    """
    try:
        return client.request(method, path, **kwargs)
    except httpx.TimeoutException as exc:
        _LOG.warning("dependency timed out", extra={"fields": {"dependency": name}})
        raise problem.gateway_timeout(name) from exc
    except httpx.HTTPError as exc:
        _LOG.warning(
            "dependency call failed",
            extra={"fields": {"dependency": name, "error_type": type(exc).__name__}},
        )
        raise problem.dependency_unavailable(name) from exc


def _item(
    image_id: str,
    meta: dict[str, Any],
    request: SearchRequest,
    *,
    rank: int,
    similarity: float,
    distance: float,
    roi_confidence: float,
    estimates: tuple[str, dict[str, float]] | None,
    quality_status: str | None = None,
    coarse_similarity: float | None = None,
    reranked: bool = False,
    lung_outline: bool = False,
) -> SearchResultItem:
    return SearchResultItem(
        rank=rank,
        image_id=uuid.UUID(image_id),
        similarity=round(float(similarity), 4),
        distance=round(float(distance), 4),
        coarse_similarity=round(float(coarse_similarity), 4) if coarse_similarity is not None else None,
        reranked=reranked,
        roi_confidence=round(float(roi_confidence), 4),
        quality_status=quality_status,
        finding_labels=meta["finding_labels"],
        label_source=meta["label_source"],
        finding_estimates=(
            chest_store.ranked_estimates(estimates[1], chest_store.ESTIMATE_TOP) if estimates else []
        ),
        view_position=meta["view_position"],
        patient_ref=meta["patient_ref"],
        source_image_name=meta["source_image_name"],
        source_metadata=meta["source_metadata"],
        thumbnail_url=(
            f"/api/v1/images/{image_id}/thumbnail" if request.return_options.thumbnail else None
        ),
        # 何を比べたかを示すため、原画像へ胸部野の輪郭を重ねる
        chest_url=f"/api/v1/images/{image_id}/chest" if request.return_options.roi_overlay else None,
        lung_outline_url=f"/api/v1/images/{image_id}/lung-outline" if lung_outline else None,
        image_url=f"/api/v1/images/{image_id}",
    )


def _lung_items(
    fanout: dict | None,
    metadata: dict,
    estimates: dict,
    request: SearchRequest,
    query_image_name: str | None = None,
) -> tuple[list, str]:
    """胸郭ランキングを組み立て、その状態を返す。「候補が無い」「未充填」「失敗」を区別する。"""
    if fanout is None:
        return [], "NOT_APPLICABLE"
    if not fanout:
        return [], "UNAVAILABLE"
    if fanout.get("empty_space"):
        return [], "EMPTY"

    items = []
    for candidate in fanout.get("candidates", []):
        image_id = str(candidate["image_id"])
        meta = metadata.get(image_id)
        if meta is None:
            continue
        if is_query_itself(query_image_name, meta["source_image_name"], candidate["similarity"]):
            continue
        items.append(
            _item(
                image_id,
                meta,
                request,
                rank=len(items) + 1,
                similarity=candidate["similarity"],
                distance=candidate["distance"],
                roi_confidence=candidate["roi_confidence"],
                estimates=estimates.get(image_id),
                # 何を「肺野」として比較したかを目で確かめられるようにする
                lung_outline=True,
            )
        )
        if len(items) >= request.top_k:
            break
    return items, "AVAILABLE" if items else "EMPTY"


def _lung_shape_search(
    vector: list[float] | None, preprocess_version: str, timeout: float, top_k: int
) -> dict | None:
    """胸郭の形が似た症例を探す。

    **所見の判別ではない**。形態の似た症例を
    並べる道具として返す。失敗しても検索全体は落とさない。
    """
    if not vector:
        return None
    try:
        with _client(config.search_aggregator_url(), timeout) as client:
            response = client.post(
                "/api/v1/vector/lung-shape",
                json={
                    "preprocess_version": preprocess_version,
                    "query": {"query_vector": vector, "candidate_k": max(top_k, 10)},
                    "timeout_ms": _vector_query_timeout_ms(),
                },
            )
        if response.status_code >= 400:
            _LOG.warning(
                "lung shape search failed", extra={"fields": {"status": response.status_code}}
            )
            return {}
        return response.json()
    except Exception as exc:  # noqa: BLE001 - 無くても検索は成立する
        _LOG.warning(
            "lung shape search unavailable",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"}},
        )
        return {}


def _normalise_text(request: SearchRequest) -> tuple[str | None, str | None, str | None]:
    """(原文, 英語正規化文, 翻訳器 version) を返す。"""
    if not request.text:
        return None, None, None
    try:
        normalized = text_norm.normalize(request.text, request.text_language)
    except text_norm.UntranslatableText as exc:
        raise problem.unprocessable(
            "TEXT_NOT_TRANSLATABLE",
            "Japanese free text could not be translated to English with the approved local "
            "dictionary. Enter English, or one of the registered standard Japanese findings.",
            errors=[{"field": "text", "reason": f"unresolved: {', '.join(exc.unresolved[:5])}"}],
        ) from exc
    return normalized.original, normalized.english, normalized.translation_model_version


def _roi_for_query(image: bytes, timeout: float) -> dict[str, Any]:
    with _client(config.roi_service_url(), timeout) as client:
        response = _call(
            client,
            "POST",
            "/api/v1/roi/chest",
            name="roi-service",
            json={"image_base64": base64.b64encode(image).decode()},
        )
    if response.status_code == 422:
        body = response.json()
        # 画面は抽出できなかった理由（code）ごとに訳す。辞書に無い理由は
        # ROI_EXTRACTION_FAILED の文言に英語の detail を添えて出す
        raise problem.unprocessable(
            body.get("code", "ROI_EXTRACTION_FAILED"),
            f"Could not extract the chest region from the query image: {body.get('detail', '')}",
            params={"reason": body.get("detail", "")},
        )
    if response.status_code >= 500:
        raise problem.dependency_unavailable("roi-service")
    response.raise_for_status()
    return response.json()


def _embed(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    with _client(config.embedding_service_url(), timeout) as client:
        response = _call(client, "POST", path, name="embedding-service", json=payload)
    if response.status_code >= 500:
        raise problem.dependency_unavailable("embedding-service")
    response.raise_for_status()
    return response.json()


def _hybrid(image_vector: list[float], text_vector: list[float], alpha: float) -> list[float]:
    """L2 正規化済み vector の加重合成を再度 L2 正規化する（設計書 7.3.3）。

    coarse 空間と fine 空間それぞれで独立に行う。空間をまたいだ合成はできない
    （同じ画像・同じテキストでも、別モデルの座標系には対応関係が無い）。
    """
    combined = alpha * np.asarray(image_vector, dtype=np.float32) + (1.0 - alpha) * np.asarray(
        text_vector, dtype=np.float32
    )
    norm = float(np.linalg.norm(combined))
    if norm == 0.0:
        raise problem.unprocessable(
            "HYBRID_VECTOR_DEGENERATE", "image and text vectors cancelled each other out"
        )
    return (combined / norm).tolist()


def _hydrate(image_ids: list[uuid.UUID], request: SearchRequest) -> dict[str, dict[str, Any]]:
    """候補 image_id のメタデータと付随ラベルを一括取得する（設計書 7.2.2）。"""
    if not image_ids:
        return {}
    conditions = ["si.image_id = ANY(%(ids)s::uuid[])"]
    params: dict[str, Any] = {"ids": [str(value) for value in image_ids]}
    if request.filters.view_positions:
        conditions.append("si.view_position = ANY(%(views)s)")
        params["views"] = request.filters.view_positions

    sql = f"""
        SELECT si.image_id, si.source_image_name, si.view_position, si.patient_id,
               si.source_metadata,
               COALESCE(array_agg(f.code ORDER BY f.code) FILTER (WHERE f.code IS NOT NULL), '{{}}')
                 AS finding_labels,
               MIN(imf.label_source) AS label_source
          FROM source_image si
          LEFT JOIN image_finding imf ON imf.image_id = si.image_id
          LEFT JOIN finding f ON f.finding_id = imf.finding_id
         WHERE {' AND '.join(conditions)}
         GROUP BY si.image_id
    """
    with db.metadata_conn() as conn:
        rows = db.fetch_all(conn, sql, params)

    # 所見の絞り込みは日本語でも受ける。**解決できない語は黙って捨てず 400 で返す。**
    # 捨てると「絞り込みが効かない」のか「綴りが違う」のかが利用者に分からない。
    wanted_labels, unresolved = csvspec.resolve_findings(request.filters.finding_labels_any)
    if unresolved:
        params = {
            "unresolved": ", ".join(unresolved),
            "allowed": ", ".join(csvspec.ALLOWED_FINDINGS),
        }
        raise problem.unprocessable(
            "FINDING_LABEL_UNKNOWN",
            f"Could not resolve the finding names: {params['unresolved']}. "
            f"Allowed values: {params['allowed']} (Japanese names are also accepted).",
            errors=[{"field": "filters.finding_labels_any", "value": value}
                    for value in unresolved],
            params=params,
        )
    wanted = set(wanted_labels)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        labels = list(row["finding_labels"] or [])
        if wanted and not wanted.intersection(labels):
            continue
        result[str(row["image_id"])] = {
            "source_image_name": row["source_image_name"],
            "view_position": row["view_position"],
            # 患者 ID は直接返さず内部 ID の仮名を返す（設計書 13）
            "patient_ref": f"P{row['patient_id']:07d}",
            # データセットのラベル（画像単位、NLP 由来）
            "finding_labels": labels,
            "label_source": row["label_source"] or "NIHCC_CSV_NLP_DERIVED",
            "source_metadata": row["source_metadata"] if request.return_options.include_source_metadata else None,
        }
    return result


def _render(label: str, render) -> bytes | None:
    """表示補助の画像を作る。失敗しても検索は続ける。"""
    try:
        return render()
    except Exception as exc:  # noqa: BLE001 - 表示補助
        _LOG.warning(f"query {label} failed", extra={"fields": {"error": f"{type(exc).__name__}: {exc}"}})
        return None


def store_query_assets(
    search_id: uuid.UUID,
    *,
    request: SearchRequest,
    actor: str,
    model_version: str,
    chest_image: bytes | None,
    lung_outline: bytes | None,
    roi_confidence: float | None,
    quality_status: str | None,
) -> tuple[bool, bool]:
    """検索元の表示用画像（胸部野の輪郭・肺野の外周）を保持する。

    保持期間は検索入力画像と同じ設定に従い、期限切れは常駐ジョブが削除する。
    **患者の画像であるため、検索元まわりの成果物はすべて同じ期限で消す。**
    同期検索では search_audit 行が後から作られるため、参照整合性を満たすよう
    最小限の行を先に作っておく（後続の persist_audit が同じ行を更新する）。

    戻り値: (胸部野を保持したか, 外周を保持したか)
    """
    if not chest_image and not lung_outline:
        return False, False
    retention = config.env_int("QUERY_IMAGE_RETENTION_MINUTES", 60)
    try:
        with db.metadata_conn() as conn:
            with conn.transaction():
                db.execute(
                    conn,
                    """
                    INSERT INTO search_audit
                      (search_id, search_mode, embedding_model_version, request_params, created_by)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (search_id) DO NOTHING
                    """,
                    (
                        str(search_id),
                        request.mode,
                        model_version,
                        json.dumps(request.model_dump(mode="json", by_alias=True), ensure_ascii=False),
                        actor,
                    ),
                )
                if chest_image:
                    db.execute(
                        conn,
                        """
                        INSERT INTO search_query_chest
                          (search_id, content, roi_confidence, quality_status, expires_at)
                        VALUES (%s, %s, %s, %s, now() + make_interval(mins => %s))
                        ON CONFLICT (search_id) DO UPDATE
                          SET content = EXCLUDED.content, expires_at = EXCLUDED.expires_at
                        """,
                        (str(search_id), chest_image, roi_confidence, quality_status, retention),
                    )
                if lung_outline:
                    db.execute(
                        conn,
                        """
                        INSERT INTO search_query_outline (search_id, content, expires_at)
                        VALUES (%s, %s, now() + make_interval(mins => %s))
                        ON CONFLICT (search_id) DO UPDATE
                          SET content = EXCLUDED.content, expires_at = EXCLUDED.expires_at
                        """,
                        (str(search_id), lung_outline, retention),
                    )
        return bool(chest_image), bool(lung_outline)
    except Exception:  # noqa: BLE001 - 表示補助のため、失敗しても検索自体は継続する
        _LOG.warning("failed to store query display images")
        return False, False


def execute_search(
    request: SearchRequest,
    *,
    image: bytes | None,
    search_id: uuid.UUID,
    actor: str,
    roi_min_confidence: float | None = None,
    query_image_name: str | None = None,
) -> SearchResponse:
    started = time.perf_counter()
    timeout = config.env_int("API_SEARCH_TIMEOUT_MS", 5000) / 1000.0
    min_confidence = (
        roi_min_confidence
        if roi_min_confidence is not None
        else config.env_float("ROI_MIN_CONFIDENCE", 0.35)
    )
    warnings: list[ServerMessage] = [RESEARCH_MESSAGE]

    original_text, english_text, translation_version = _normalise_text(request)

    # 1. クエリ画像の胸部野抽出（設計書 7.2）
    roi: dict[str, Any] | None = _roi_for_query(image, timeout) if image is not None else None

    # 2. ベクトル生成（coarse = 1 段目 / fine = 2 段目。fine は無い場合がある）
    text_vector: list[float] | None = None
    text_vector_fine: list[float] | None = None
    embedding_meta: dict[str, Any] = {}
    if english_text:
        embedded = _embed("/api/v1/embeddings/texts", {"texts": [english_text]}, timeout)
        text_vector = embedded["vectors"][0]
        fine_text_vectors = embedded.get("fine_vectors")
        text_vector_fine = fine_text_vectors[0] if fine_text_vectors else None
        embedding_meta = embedded

    query_vector: list[float] | None = text_vector
    query_vector_fine: list[float] | None = text_vector_fine
    query_probs: dict[str, float] | None = None
    if roi is not None:
        embedded = _embed(
            "/api/v1/embeddings/chest",
            {
                # roi-service が作った入力をそのまま渡す（作り直すと確率が黙ってずれる）
                "crops_base64": [roi["crop_base64"]],
                "images_base64": [roi["pool_image_base64"]],
                "masks_base64": [roi["pool_mask_base64"]],
            },
            timeout,
        )
        embedding_meta = embedded
        image_vector = embedded["vectors"][0]
        fine_vectors = embedded.get("fine_vectors")
        fine_image_vector = fine_vectors[0] if fine_vectors else None
        if embedded.get("probabilities") and embedded.get("findings"):
            query_probs = dict(zip(embedded["findings"], embedded["probabilities"][0], strict=True))
        query_vector = (
            image_vector if text_vector is None else _hybrid(image_vector, text_vector, request.image_weight)
        )
        # fine 空間。テキストを混ぜる検索で fine のテキストベクトルが無い場合は合成
        # できないため 1 段検索にする（coarse と fine は別空間なので、片方だけテキストを
        # 混ぜると両段のクエリの意味が食い違う）。
        if fine_image_vector is None or (text_vector is not None and text_vector_fine is None):
            query_vector_fine = None
        elif text_vector_fine is None:
            query_vector_fine = fine_image_vector
        else:
            query_vector_fine = _hybrid(fine_image_vector, text_vector_fine, request.image_weight)

    if not embedding_meta or query_vector is None:
        raise problem.unprocessable("SEARCH_INPUT_MISSING", "no query vector could be produced")

    model_version = embedding_meta["embedding_model_version"]
    fine_model_version = embedding_meta.get("fine_embedding_model_version")
    preprocess_version = config.preprocess_version()

    # 2 段検索を行うか。要求が rerank=false なら fine ベクトルを捨てて 1 段にする。
    if request.rerank and fine_model_version and english_text and text_vector_fine is None:
        warnings.append(RERANK_TEXT_UNAVAILABLE_WARNING)
    # 応答には「要求したか」ではなく「実際に 2 段になったか」を書く。
    rerank_active = bool(request.rerank and fine_model_version and query_vector_fine is not None)
    strategy = rerank.resolve_strategy(
        request.strategy,
        default=config.image_search_strategy(),
        rerank_active=rerank_active,
        has_text=text_vector is not None,
    )

    query_confidence = float(roi["roi_confidence"]) if roi else None
    query_quality = roi["quality_status"] if roi else None
    has_query_chest = has_query_outline = False
    if roi is not None and image is not None:
        chest_png = base64.b64decode(roi["chest_mask_base64"])
        lung_png = base64.b64decode(roi["lung_mask_base64"])
        has_query_chest, has_query_outline = store_query_assets(
            search_id,
            request=request,
            actor=actor,
            model_version=model_version,
            chest_image=_render(
                "chest", lambda: chest_store.chest_overlay(image, chest_png, roi["source_bbox"])
            ),
            lung_outline=_render(
                "lung outline", lambda: imaging.lung_outline(image, [(lung_png, roi["source_bbox"])])
            ),
            roi_confidence=query_confidence,
            quality_status=query_quality,
        )

    # 3. ベクトル検索（1 クラスタ・2 段）
    with _client(config.search_aggregator_url(), timeout) as client:
        response = _call(
            client,
            "POST",
            "/api/v1/vector/search",
            name="search-aggregator",
            json={
                "embedding_model_version": model_version,
                "preprocess_version": preprocess_version,
                "fine_embedding_model_version": fine_model_version if rerank_active else None,
                "query": {
                    "query_vector": query_vector,
                    "query_vector_fine": query_vector_fine if rerank_active else None,
                    "candidate_k": request.candidate_k,
                    "min_similarity": request.min_similarity,
                    "exclude_image_ids": [str(v) for v in request.filters.exclude_image_ids],
                },
                "timeout_ms": _vector_query_timeout_ms(),
                "stage1": "fine" if strategy == rerank.STRATEGY_PROFILE else "embedding",
            },
        )
    if response.status_code == 504:
        # aggregator が「時間内に終わらなかった」と判断した場合。
        # 503 へ丸めると、動いているサービスを停止中と報告することになる。
        raise problem.gateway_timeout("vector cluster")
    if response.status_code >= 500:
        raise problem.dependency_unavailable("search-aggregator")
    response.raise_for_status()
    found = response.json()

    lung_found = _lung_shape_search(
        roi.get("lung_vector") if roi else None, preprocess_version, timeout, request.top_k
    )

    # 4. メタデータ・所見の推定・品質を結合し、フィルタを適用する
    ids = [str(c["image_id"]) for c in found["candidates"]]
    ids += [str(c["image_id"]) for c in (lung_found or {}).get("candidates", [])]
    ids = list(dict.fromkeys(ids))
    metadata = _hydrate([uuid.UUID(value) for value in ids], request)
    with db.metadata_conn() as conn:
        estimates = chest_store.finding_estimates(conn, ids)
        quality = chest_store.quality_by_image(conn, ids)

    items: list[SearchResultItem] = []
    self_excluded = False
    for candidate in found["candidates"]:
        image_id = str(candidate["image_id"])
        meta = metadata.get(image_id)
        if meta is None:
            continue
        # 検索元そのものは比べる意味が無いので外す
        if is_query_itself(query_image_name, meta["source_image_name"], candidate["similarity"]):
            self_excluded = True
            continue
        items.append(
            _item(
                image_id,
                meta,
                request,
                rank=len(items) + 1,
                similarity=candidate["similarity"],
                distance=candidate["distance"],
                roi_confidence=candidate["roi_confidence"],
                estimates=estimates.get(image_id),
                quality_status=quality.get(image_id),
                coarse_similarity=candidate.get("coarse_similarity"),
                reranked=bool(candidate.get("reranked")),
            )
        )
        if len(items) >= request.top_k:
            break

    if query_confidence is not None and query_confidence < min_confidence:
        # 低信頼度の検索元は結果を抑制し、警告付きで返す（設計書 7.2.2）
        warnings.append(_suppressed_warning(query_confidence, min_confidence))
        items = []
    elif query_quality == "ROI_UNCERTAIN":
        warnings.append(LOW_CONFIDENCE_WARNING)
    if found.get("rerank_status") in {"PARTIAL_COVERAGE", "NO_COVERAGE"}:
        warnings.append(RERANK_PARTIAL_WARNING)
    if found.get("rerank_status") == "BUDGET_EXHAUSTED":
        warnings.append(BUDGET_EXHAUSTED_WARNING)

    if self_excluded:
        warnings.append(SELF_MATCH_EXCLUDED_WARNING)

    lung_items, lung_status = _lung_items(
        lung_found, metadata, estimates, request, query_image_name
    )

    # 推定に使った分類器の版。検索元の推定があればその版、無ければ候補の多数派
    estimate_versions = Counter(v for v, _ in estimates.values())
    estimate_model_version = (
        fine_model_version
        if query_probs is not None
        else (estimate_versions.most_common(1)[0][0] if estimate_versions else None)
    )

    duration_ms = int((time.perf_counter() - started) * 1000)
    return SearchResponse(
        search_id=search_id,
        status="SUCCEEDED",
        mode=request.mode,
        query=SearchQueryEcho(
            text_original=original_text,
            text_normalized_en=english_text,
            image_weight=request.image_weight,
            top_k=request.top_k,
            candidate_k=request.candidate_k or request.top_k * 5,
            stage1_candidate_k=(
                rerank.stage1_limit(
                    request.candidate_k or request.top_k * 5,
                    multiplier=config.rerank_stage1_multiplier(),
                    max_limit=config.rerank_stage1_max(),
                )
                if strategy == rerank.STRATEGY_COARSE_TO_FINE
                else None
            ),
            min_similarity=request.min_similarity,
            rerank=request.rerank,
            strategy=strategy,
        ),
        models={
            "unit": "chest",
            "embedding": model_version,
            "embedding_name": embedding_meta.get("embedding_model_name", ""),
            "embedding_revision": embedding_meta.get("embedding_model_revision", ""),
            # 2 段検索の 2 段目。未構成なら n/a（= 1 段検索）
            "embedding_fine": (fine_model_version or "n/a") if rerank_active else "n/a",
            "embedding_fine_dimension": str(embedding_meta.get("fine_dimension") or "")
            if rerank_active
            else "",
            "rerank": rerank.STRATEGY_MODEL_LABELS[strategy],
            "roi": (roi or {}).get("roi_model_version") or "n/a",
            "preprocess": preprocess_version,
            "translation": translation_version or "n/a",
        },
        results=items,
        query_quality_status=query_quality,
        query_roi_confidence=query_confidence,
        query_chest_url=f"/api/v1/searches/{search_id}/chest" if has_query_chest else None,
        query_finding_estimates=(
            chest_store.ranked_estimates(query_probs, chest_store.ESTIMATE_TOP) if query_probs else []
        ),
        estimate_model_version=estimate_model_version,
        timing_ms=int(found.get("timing_ms", 0)),
        rerank_status=found.get("rerank_status"),
        stage1_count=int(found.get("stage1_count", 0)),
        reranked_count=int(found.get("reranked_count", 0)),
        query_lung_outline_url=(
            f"/api/v1/searches/{search_id}/lung-outline" if has_query_outline else None
        ),
        lung_shape_results=lung_items,
        lung_shape_status=lung_status,
        # 同じ警告は 1 度だけ。英語の文と訳すための詳細を同じ順で返す
        **_warning_fields(warnings),
        duration_ms=duration_ms,
    )


def persist_audit(
    search_id: uuid.UUID,
    request: SearchRequest,
    response: SearchResponse | None,
    *,
    actor: str,
    query_image_sha256: str | None,
    error: dict[str, Any] | None = None,
) -> None:
    """検索監査（設計書 7.2.2 / 13）。検索画像そのものは監査ログに保存しない。"""
    summary = None
    if response is not None:
        summary = {
            "unit": "chest",
            "partial": response.partial,
            "duration_ms": response.duration_ms,
            "returned": len(response.results),
            "timing_ms": response.timing_ms,
            "rerank_status": response.rerank_status,
            "strategy": response.query.strategy if response.query else None,
            "stage1_count": response.stage1_count,
            "reranked_count": response.reranked_count,
            "top_image_ids": [str(item.image_id) for item in response.results[:5]],
            "top_distances": [item.distance for item in response.results[:5]],
        }
    with db.metadata_conn() as conn:
        db.execute(
            conn,
            """
            INSERT INTO search_audit
              (search_id, search_mode, query_image_sha256, query_text_original,
               query_text_normalized_en, embedding_model_version, roi_model_version,
               request_params, result_summary, duration_ms, created_by)
            VALUES (%(id)s, %(mode)s, %(sha)s, %(text)s, %(text_en)s, %(model)s, %(roi)s,
                    %(params)s, %(summary)s, %(duration)s, %(actor)s)
            ON CONFLICT (search_id) DO UPDATE
              SET result_summary = EXCLUDED.result_summary,
                  duration_ms = EXCLUDED.duration_ms
            """,
            {
                "id": str(search_id),
                "mode": request.mode,
                "sha": query_image_sha256,
                "text": request.text,
                "text_en": response.query.text_normalized_en if response and response.query else None,
                "model": (response.models.get("embedding") if response else None) or "unknown",
                "roi": response.models.get("roi") if response else None,
                "params": json.dumps(
                    request.model_dump(mode="json", by_alias=True), ensure_ascii=False
                ),
                "summary": json.dumps(summary or error or {}, ensure_ascii=False, default=str),
                "duration": response.duration_ms if response else None,
                "actor": actor,
            },
        )


def median_duration_ms() -> float | None:
    """同期/非同期の切替判断に使う直近の検索所要時間中央値（設計書 9.5）。"""
    with db.metadata_conn() as conn:
        rows = db.fetch_all(
            conn,
            "SELECT duration_ms FROM search_audit WHERE duration_ms IS NOT NULL "
            "ORDER BY created_at DESC LIMIT 20",
        )
    values = [row["duration_ms"] for row in rows if row["duration_ms"]]
    return statistics.median(values) if values else None
