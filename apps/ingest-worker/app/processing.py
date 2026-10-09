"""画像 1 件の取り込み処理（設計書 7.1）。

検証 → SHA-256 → メタデータ UPSERT → 胸部野の抽出 → ベクトル化 →
ベクトルクラスタへ UPSERT → 状態更新、までを冪等に実行する。

検索の単位は画像 1 枚の**胸部野**（肺野＋縦隔＋横隔膜）である。
1 枚につきベクトルは 1 行になる。

2 段検索（coarse -> fine）
--------------------------
embedding-service の 2 段目が胸部野の分類器（chest-profile）なら、プロファイルを
同じ行の embedding_fine へ、所見の推定確率を metadata の finding_estimate へ書く。
coarse と fine は 1 文の UPSERT で書くので、片方だけが入った行は生じない。

冪等性（設計書 7.1.5）:
  - 少なくとも 1 回配信を前提に、UPSERT と一意制約で二重登録を防ぐ
  - メタデータ登録後にベクトル登録が失敗した場合は image_vector_state へ
    未完了状態を残し、補償ジョブで再投入できるようにする
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from nihcc_rag import chest_head, config, db, imaging, queueing
from prometheus_client import Counter, Histogram

_LOG = logging.getLogger("ingest-worker")

ITEMS = Counter("ingest_items_total", "Processed ingest items", ["outcome"])
STAGE_DURATION = Histogram(
    "ingest_stage_duration_seconds",
    "Ingest stage duration",
    ["stage"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)
VECTORS = Counter("ingest_vectors_total", "Stored vectors", ["outcome"])

IMPORT_ROOT = Path(os.environ.get("IMPORT_ROOT", "/imports")).resolve()
HTTP_TIMEOUT = float(config.env_int("WORKER_HTTP_TIMEOUT_SECONDS", 120))


class ItemError(Exception):
    """1 件の取り込み失敗。error_code とともに ingest_item へ記録する。"""

    def __init__(self, code: str, message: str, *, quarantine: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.quarantine = quarantine


@dataclass
class Services:
    roi: httpx.Client
    embedding: httpx.Client

    @classmethod
    def create(cls) -> "Services":
        return cls(
            roi=httpx.Client(base_url=config.roi_service_url(), timeout=HTTP_TIMEOUT),
            embedding=httpx.Client(base_url=config.embedding_service_url(), timeout=HTTP_TIMEOUT),
        )

    def close(self) -> None:
        self.roi.close()
        self.embedding.close()


def resolve_import_path(raw_path: str) -> Path:
    """path traversal を防ぎ、必ず IMPORT_ROOT 配下を指すことを保証する（設計書 13）。"""
    candidate = Path(raw_path)
    resolved = (candidate if candidate.is_absolute() else IMPORT_ROOT / candidate).resolve()
    if not resolved.is_relative_to(IMPORT_ROOT):
        raise ItemError("PATH_OUTSIDE_IMPORT_ROOT", "path escapes the import root", quarantine=True)
    if not resolved.is_file():
        raise ItemError("IMAGE_MISSING", f"image file was not found: {resolved.name}")
    return resolved


# --- メタデータ登録 ----------------------------------------------------------


def upsert_model_version(
    conn, model_version: str, model_type: str, model_name: str, revision: str, config_json: dict
) -> None:
    artifact_sha = config_json.get("artifact_sha256") or ("0" * 64)
    db.execute(
        conn,
        """
        INSERT INTO model_version
          (model_version, model_type, model_name, source_revision, artifact_sha256,
           embedding_dimension, config, active)
        VALUES (%(version)s, %(type)s, %(name)s, %(revision)s, %(sha)s, %(dim)s, %(config)s, false)
        ON CONFLICT (model_version) DO UPDATE
          SET config = EXCLUDED.config, model_name = EXCLUDED.model_name
        """,
        {
            "version": model_version,
            "type": model_type,
            "name": model_name,
            "revision": revision or "unpinned",
            "sha": artifact_sha,
            "dim": config_json.get("embedding_dimension"),
            "config": json.dumps(config_json, ensure_ascii=False, default=str),
        },
    )


def upsert_image(conn, payload: dict[str, Any], facts: imaging.ImageFacts, content: bytes) -> tuple[str, bool]:
    """患者・原画像・画像本体・付随ラベルを UPSERT し、(image_id, 既存かどうか) を返す。"""
    existing = db.fetch_one(
        conn, "SELECT image_id FROM source_image WHERE sha256 = %s", (facts.sha256,)
    )
    if existing:
        return str(existing["image_id"]), True

    patient = db.fetch_one(
        conn,
        """
        INSERT INTO patient (source_patient_key_hash) VALUES (%s)
        ON CONFLICT (source_patient_key_hash) DO UPDATE SET source_patient_key_hash = EXCLUDED.source_patient_key_hash
        RETURNING patient_id
        """,
        (payload["patient_key_hash"],),
    )
    row = db.fetch_one(
        conn,
        """
        INSERT INTO source_image
          (patient_id, source_image_name, sha256, width, height, view_position,
           patient_age, patient_gender, source_dataset_version, source_metadata)
        VALUES (%(patient_id)s, %(name)s, %(sha)s, %(width)s, %(height)s, %(view)s,
                %(age)s, %(gender)s, %(dataset)s, %(metadata)s)
        ON CONFLICT (source_image_name) DO UPDATE
          SET source_metadata = EXCLUDED.source_metadata
        RETURNING image_id
        """,
        {
            "patient_id": patient["patient_id"],
            "name": payload["source_image_name"],
            "sha": facts.sha256,
            "width": facts.width,
            "height": facts.height,
            "view": payload.get("view_position"),
            "age": payload.get("patient_age"),
            "gender": payload.get("patient_sex"),
            "dataset": payload["dataset_version"],
            "metadata": json.dumps(payload.get("source_metadata", {}), ensure_ascii=False),
        },
    )
    image_id = str(row["image_id"])

    db.execute(
        conn,
        """
        INSERT INTO image_binary (image_id, content, mime_type, byte_size)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (image_id) DO UPDATE
          SET content = EXCLUDED.content, mime_type = EXCLUDED.mime_type,
              byte_size = EXCLUDED.byte_size
        """,
        (image_id, content, facts.mime_type, facts.byte_size),
    )

    for label in payload.get("finding_labels", []):
        db.execute(
            conn,
            """
            INSERT INTO image_finding (image_id, finding_id, label_source)
            SELECT %s, finding_id, %s FROM finding WHERE code = %s
            ON CONFLICT (image_id, finding_id) DO NOTHING
            """,
            (image_id, payload.get("label_source", "NIHCC_CSV_NLP_DERIVED"), label),
        )
    return image_id, False


def upsert_chest_asset(conn, image_id: str, roi: dict[str, Any], preprocess_version: str) -> None:
    db.execute(
        conn,
        """
        INSERT INTO chest_asset
          (image_id, chest_mask, lung_mask, bbox, areas, flags, roi_confidence,
           roi_model_version, preprocess_version, quality_status)
        VALUES (%(image_id)s, %(chest)s, %(lung)s, %(bbox)s, %(areas)s, %(flags)s,
                %(confidence)s, %(roi_version)s, %(preprocess)s, %(status)s)
        ON CONFLICT (image_id, roi_model_version, preprocess_version) DO UPDATE
          SET chest_mask = EXCLUDED.chest_mask,
              lung_mask = EXCLUDED.lung_mask,
              bbox = EXCLUDED.bbox,
              areas = EXCLUDED.areas,
              flags = EXCLUDED.flags,
              roi_confidence = EXCLUDED.roi_confidence,
              quality_status = EXCLUDED.quality_status
        """,
        {
            "image_id": image_id,
            "chest": base64.b64decode(roi["chest_mask_base64"]),
            "lung": base64.b64decode(roi["lung_mask_base64"]),
            "bbox": json.dumps(
                {
                    "bbox": roi["bbox"],
                    "source_bbox": roi["source_bbox"],
                    "work_scale": roi.get("work_scale", 1.0),
                }
            ),
            "areas": json.dumps(roi.get("areas", {})),
            "flags": list(roi.get("flags", [])),
            "confidence": max(0.0, min(1.0, float(roi["roi_confidence"]))),
            "roi_version": roi["roi_model_version"],
            "preprocess": preprocess_version,
            "status": roi["quality_status"],
        },
    )


def upsert_finding_estimate(conn, image_id: str, model_version: str,
                            findings: list[str], probabilities: list[float]) -> None:
    """胸部野の分類器による所見の推定を残す（画面の「所見の推定」）。"""
    db.execute(
        conn,
        """
        INSERT INTO finding_estimate (image_id, model_version, probabilities)
        VALUES (%s, %s, %s)
        ON CONFLICT (image_id, model_version) DO UPDATE
          SET probabilities = EXCLUDED.probabilities
        """,
        (
            image_id,
            model_version,
            json.dumps({f: round(float(p), 5) for f, p in zip(findings, probabilities, strict=True)}),
        ),
    )


def set_vector_state(conn, image_id: str, model_version: str, preprocess_version: str,
                     state: str, error: str | None = None) -> None:
    db.execute(
        conn,
        """
        INSERT INTO image_vector_state
          (image_id, embedding_model_version, preprocess_version, state, last_error, updated_at)
        VALUES (%s, %s, %s, %s, %s, now())
        ON CONFLICT (image_id, embedding_model_version, preprocess_version) DO UPDATE
          SET state = EXCLUDED.state, last_error = EXCLUDED.last_error, updated_at = now()
        """,
        (image_id, model_version, preprocess_version, state, error),
    )


def store_vector(image_id: str, vector: list[float], roi_confidence: float,
                 model_version: str, preprocess_version: str, roi_model_version: str,
                 fine_vector: list[float] | None = None,
                 fine_model_version: str | None = None,
                 lung_vector: list[float] | None = None) -> None:
    """1 画像のベクトルをクラスタへ UPSERT する。

    coarse（embedding）と fine（embedding_fine）を 1 文で書く。fine が無い構成では
    NULL を入れる。fine だけを後から足すことはしない（どのモデルの coarse と
    組んだ fine なのかが行から読めなくなるため）。
    """
    with db.vector_conn() as conn:
        db.execute(
            conn,
            """
            INSERT INTO image_embedding
              (image_id, embedding_model_version, preprocess_version,
               roi_model_version, roi_confidence, embedding,
               embedding_fine, embedding_fine_model_version, embedding_lung)
            VALUES (%(image_id)s, %(model)s, %(preprocess)s, %(roi_version)s,
                    %(confidence)s, %(embedding)s::vector,
                    %(embedding_fine)s::vector, %(fine_model)s,
                    %(embedding_lung)s::vector)
            -- Citus は分散テーブルの DO UPDATE SET に IMMUTABLE でない関数を許さないため
            -- now() は使わない。created_at は初回登録時刻を保持する。
            ON CONFLICT (image_id, embedding_model_version, preprocess_version) DO UPDATE
              SET embedding = EXCLUDED.embedding,
                  embedding_fine = EXCLUDED.embedding_fine,
                  embedding_fine_model_version = EXCLUDED.embedding_fine_model_version,
                  embedding_lung = EXCLUDED.embedding_lung,
                  roi_confidence = EXCLUDED.roi_confidence,
                  roi_model_version = EXCLUDED.roi_model_version
            """,
            {
                "image_id": image_id,
                "model": model_version,
                "preprocess": preprocess_version,
                "roi_version": roi_model_version,
                "confidence": max(0.0, min(1.0, float(roi_confidence))),
                "embedding": db.to_pgvector(vector),
                "embedding_fine": db.to_pgvector(fine_vector) if fine_vector else None,
                "fine_model": fine_model_version if fine_vector else None,
                "embedding_lung": db.to_pgvector(lung_vector) if lung_vector else None,
            },
        )


# --- 外部サービス呼び出し ----------------------------------------------------


def call_roi(services: Services, content: bytes) -> dict[str, Any]:
    started = time.perf_counter()
    response = services.roi.post(
        "/api/v1/roi/chest", json={"image_base64": base64.b64encode(content).decode()}
    )
    STAGE_DURATION.labels("roi").observe(time.perf_counter() - started)
    if response.status_code == 422:
        body = response.json()
        raise ItemError(body.get("code", "ROI_EXTRACTION_FAILED"), body.get("detail", "ROI failed"))
    response.raise_for_status()
    return response.json()


def call_embedding(services: Services, roi: dict[str, Any]) -> dict[str, Any]:
    """胸部野を埋め込む。入力は roi-service が作ったものを**そのまま**渡す。

    作り直すと、分類器が学習した特徴と 1 画素でもずれて確率が黙ってずれる
    （image_pipeline.chest_model_inputs）。
    """
    started = time.perf_counter()
    response = services.embedding.post(
        "/api/v1/embeddings/chest",
        json={
            "crops_base64": [roi["crop_base64"]],
            "images_base64": [roi["pool_image_base64"]],
            "masks_base64": [roi["pool_mask_base64"]],
        },
    )
    STAGE_DURATION.labels("embedding").observe(time.perf_counter() - started)
    response.raise_for_status()
    embedded = response.json()
    _check_dimensions(embedded)
    return embedded


#: `image_embedding` の vector 列と、その幅を決める環境変数。
_VECTOR_COLUMNS = (
    ("embedding", "EMBEDDING_DIMENSION"),
    ("embedding_fine", "EMBEDDING_FINE_DIMENSION"),
    ("embedding_lung", "EMBEDDING_LUNG_DIMENSION"),
)


def verify_vector_schema() -> None:
    """**列の実際の幅**が設定と一致することを、起動時に確かめる。

    `_check_dimensions` は「embedding-service が返した次元」と「環境変数」を
    比べている。どちらも同じ `.env` から来るので、**列だけが取り残されていても
    検出できない。**

    例えば `.env` の `EMBEDDING_FINE_DIMENSION` を変えて列を広げる migration が無いと、
    取り込みは挿入時に pgvector の

        DataException: expected 15 dimensions, not 768

    で全件失敗する。**設定どうしではなく、設定と schema を突き合わせる。**

    列の幅は `atttypmod` に入っている（pgvector は次元をそのまま持つ）。
    """
    expected = {column: config.env_int(env_name, 0) for column, env_name in _VECTOR_COLUMNS}
    try:
        with db.vector_conn() as conn:
            rows = db.fetch_all(
                conn,
                """
                SELECT attname, atttypmod AS dimension
                  FROM pg_attribute
                 WHERE attrelid = 'image_embedding'::regclass
                   AND attname = ANY(%s)
                """,
                ([column for column, _ in _VECTOR_COLUMNS],),
            )
    except Exception as exc:  # noqa: BLE001 - クラスタが未起動でも worker は上げる
        _LOG.warning(
            "vector schema check skipped",
            extra={"fields": {"error": f"{type(exc).__name__}: {exc}"}},
        )
        return
    problems = []
    for row in rows:
        column = row["attname"]
        actual = int(row["dimension"])
        want = expected.get(column, 0)
        if want and actual != want:
            problems.append(
                f"{column} は vector({actual}) だが {dict(_VECTOR_COLUMNS)[column]}={want}"
            )
    if problems:
        raise RuntimeError(
            "vector 列の幅が設定と違います。取り込みは全件失敗します:\n  "
            + "\n  ".join(problems)
            + "\n設定を列に合わせるか、db/vector-migrations に新しい V00x を足して"
            " 列を作り直してください（適用済み migration の checksum 変更は拒否されます）。"
        )
    _LOG.info(
        "vector schema verified",
        extra={"fields": {column: width for column, width in expected.items() if width}},
    )


def verify_chest_pipeline(services: Services, *, attempts: int = 60, wait: float = 5.0) -> None:
    """起動時に、胸部野の特徴の作り方が分類器の前提と一致するかを確かめる。

    **照合できるまで取り込みを始めない。** embedding-service はモデルの読み込みに
    数分かかるので、起動直後は応答しない。黙って飛ばすと照合しないまま取り込みが
    始まり、食い違っていれば全件の推定がずれたまま保存される。待っても応答が
    無ければ起動を止める（restart で再試行される）。
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            roi_response = services.roi.get("/api/v1/model")
            roi_response.raise_for_status()
            embedding_response = services.embedding.get("/api/v1/model")
            embedding_response.raise_for_status()
            roi_model, embedding_model = roi_response.json(), embedding_response.json()
            break
        except Exception as exc:  # noqa: BLE001 - 依存先の起動を待つ
            last = exc
            if attempt == attempts:
                raise RuntimeError(
                    "roi-service / embedding-service の構成を確認できないため取り込みを"
                    f"始めません（{attempts * wait:.0f} 秒待った）: {type(exc).__name__}: {exc}"
                ) from exc
            if attempt == 1 or attempt % 12 == 0:
                _LOG.info("waiting for roi-service / embedding-service",
                          extra={"fields": {"attempt": attempt, "error": type(exc).__name__}})
            time.sleep(wait)
    else:  # pragma: no cover - attempts >= 1 なら到達しない
        raise RuntimeError("chest pipeline check did not run") from last
    problems = chest_head.pipeline_problems(roi_model, embedding_model)
    if problems:
        raise RuntimeError(
            "胸部野の分類器が前提とする特徴の作り方と、roi-service の構成が違います。"
            "このまま取り込むと所見の推定が黙ってずれます:\n  - " + "\n  - ".join(problems)
        )
    _LOG.info(
        "chest pipeline verified",
        extra={"fields": {"chest_head": bool(embedding_model.get("chest_head"))}},
    )


def _check_dimensions(embedded: dict[str, Any]) -> None:
    """モデルの出力次元が DB の列幅と一致することを確認する。

    列幅（vector(N)）は migration 時に EMBEDDING_DIMENSION /
    EMBEDDING_FINE_DIMENSION で確定している。食い違うと pgvector が
    「different vector dimensions」で全件失敗するため、意味のある error_code で
    1 件目から止める。quarantine にはしない（画像ではなく構成の問題であり、
    直せば同じ item をそのまま再実行できる）。

    **ここは設定どうしの比較でしかない。** 列の実際の幅との突き合わせは
    起動時の `verify_vector_schema` が行う。
    """
    expected = config.embedding_dimension()
    actual = int(embedded.get("dimension", 0))
    if actual != expected:
        raise ItemError(
            "EMBEDDING_DIMENSION_MISMATCH",
            f"embedding-service returned {actual}-dim vectors but image_embedding.embedding "
            f"is vector({expected}); align EMBEDDING_DIMENSION with the model and re-run "
            "the migration",
        )
    if not embedded.get("fine_vectors"):
        return
    expected_fine = config.embedding_fine_dimension()
    actual_fine = int(embedded.get("fine_dimension") or 0)
    if actual_fine != expected_fine:
        raise ItemError(
            "EMBEDDING_FINE_DIMENSION_MISMATCH",
            f"embedding-service returned {actual_fine}-dim fine vectors but "
            f"image_embedding.embedding_fine is vector({expected_fine}); align "
            "EMBEDDING_FINE_DIMENSION with the fine model and re-run the migration",
        )


# --- 1 件処理 ----------------------------------------------------------------


def process_item(payload: dict[str, Any], services: Services) -> str:
    """戻り値: 'SUCCEEDED' | 'SKIPPED' | 'CANCELLED'"""
    job_id = payload["job_id"]
    item_id = payload["item_id"]
    preprocess_version = payload.get("preprocess_version") or config.preprocess_version()

    # 取消済みのジョブ・item は処理しない。着手の判定と RUNNING への更新を 1 文で行い、
    # 取消（backend-api の cancel_job）と同時に走っても二重に扱わないようにする。
    with db.metadata_conn() as conn:
        claimed = db.execute(
            conn,
            "UPDATE ingest_item i SET status = 'RUNNING', attempt_count = i.attempt_count + 1, "
            "updated_at = now() FROM ingest_job j "
            "WHERE i.item_id = %s AND j.job_id = i.job_id "
            "AND j.status <> 'CANCELLED' AND i.status <> 'CANCELLED'",
            (item_id,),
        )
        if not claimed:
            db.execute(
                conn,
                "UPDATE ingest_item SET status = 'CANCELLED', error_code = 'JOB_CANCELLED', "
                "updated_at = now() WHERE item_id = %s AND status IN ('PENDING','RUNNING')",
                (item_id,),
            )
    if not claimed:
        ITEMS.labels("cancelled").inc()
        return "CANCELLED"

    path = resolve_import_path(payload["image_path"])
    content = path.read_bytes()
    try:
        facts = imaging.inspect(content, max_bytes=config.env_int("MAX_UPLOAD_BYTES", 20971520))
    except imaging.ImageValidationError as exc:
        raise ItemError(exc.code, str(exc), quarantine=True) from exc

    with db.metadata_conn() as conn:
        with conn.transaction():
            image_id, existed = upsert_image(conn, payload, facts, content)

    if existed and payload.get("on_duplicate", "SKIP") == "SKIP":
        with db.metadata_conn() as conn:
            db.execute(
                conn,
                "UPDATE ingest_item SET status = 'SUCCEEDED', image_id = %s, "
                "error_code = 'DUPLICATE_SKIPPED', updated_at = now() WHERE item_id = %s",
                (image_id, item_id),
            )
        ITEMS.labels("skipped").inc()
        return "SKIPPED"

    roi = call_roi(services, content)
    roi_model_version = roi["roi_model_version"]
    embedded = call_embedding(services, roi)
    model_version = embedded["embedding_model_version"]
    # 2 段検索の 2 段目。fine backend 未構成なら None のまま（1 段検索）。
    fine_model_version = embedded.get("fine_embedding_model_version")
    fine_vectors = embedded.get("fine_vectors")
    probabilities = embedded.get("probabilities")

    with db.metadata_conn() as conn:
        with conn.transaction():
            upsert_model_version(
                conn,
                roi_model_version,
                "ROI",
                roi.get("profile_name", "unknown"),
                roi.get("parameter_hash", ""),
                {
                    "artifact_sha256": roi.get("parameter_hash"),
                    "unit": "chest",
                    "preprocess_version": preprocess_version,
                },
            )
            upsert_model_version(
                conn,
                model_version,
                "EMBEDDING",
                embedded["embedding_model_name"],
                embedded["embedding_model_revision"],
                {
                    "embedding_dimension": embedded["dimension"],
                    "normalization": embedded.get("normalization", "L2"),
                    "distance_metric": "cosine",
                    "search_stage": "coarse",
                    "input": "chest-crop",
                },
            )
            if fine_vectors and fine_model_version:
                # fine は別種のモデルとして登録する（model_type = EMBEDDING_FINE）。
                # 同じ EMBEDDING にすると one_active_model_per_type により
                # coarse と fine の片方しか active にできない。
                upsert_model_version(
                    conn,
                    fine_model_version,
                    "EMBEDDING_FINE",
                    embedded.get("fine_embedding_model_name") or "unknown",
                    embedded.get("fine_embedding_model_revision") or "unpinned",
                    {
                        "embedding_dimension": embedded.get("fine_dimension"),
                        "normalization": embedded.get("normalization", "L2"),
                        "distance_metric": "cosine",
                        "search_stage": "fine",
                        "coarse_model_version": model_version,
                        "findings": embedded.get("findings"),
                    },
                )
            upsert_chest_asset(conn, image_id, roi, preprocess_version)
            if probabilities and fine_model_version and embedded.get("findings"):
                upsert_finding_estimate(
                    conn, image_id, fine_model_version, embedded["findings"], probabilities[0]
                )
            set_vector_state(conn, image_id, model_version, preprocess_version, "PENDING")

    started = time.perf_counter()
    error: str | None = None
    try:
        store_vector(
            image_id,
            embedded["vectors"][0],
            roi["roi_confidence"],
            model_version,
            preprocess_version,
            roi_model_version,
            fine_vector=fine_vectors[0] if fine_vectors else None,
            fine_model_version=fine_model_version,
            lung_vector=roi.get("lung_vector"),
        )
    except Exception as exc:  # noqa: BLE001 - 状態で追跡し補償ジョブへ委ねる
        # 種別だけでは障害調査ができないため、DB からのメッセージも記録する。
        # 認証情報・画像内容は含まれない（設計書 13）。
        detail = str(exc).splitlines()[0][:300] if str(exc) else ""
        error = f"{type(exc).__name__}: {detail}"
    STAGE_DURATION.labels("vector_store").observe(time.perf_counter() - started)
    VECTORS.labels("error" if error else "ok").inc()

    with db.metadata_conn() as conn:
        set_vector_state(
            conn, image_id, model_version, preprocess_version,
            "FAILED" if error else "STORED", error,
        )
    if error:
        _LOG.warning("vector store failed", extra={"fields": {"image_id": image_id, "error": error}})
        raise ItemError("VECTOR_STORE_FAILED", f"vector store failed: {error}")

    with db.metadata_conn() as conn:
        db.execute(
            conn,
            "UPDATE ingest_item SET status = 'SUCCEEDED', image_id = %s, error_code = NULL, "
            "error_detail = NULL, updated_at = now() WHERE item_id = %s",
            (image_id, item_id),
        )
    _LOG.info(
        "item ingested",
        extra={
            "fields": {
                "job_id": job_id,
                "image_id": image_id,
                "roi_status": roi["quality_status"],
                "model_version": model_version,
                "fine_model_version": fine_model_version,
            }
        },
    )
    ITEMS.labels("succeeded").inc()
    return "SUCCEEDED"


def record_failure(item_id: str, error: ItemError) -> None:
    status = "QUARANTINED" if error.quarantine else "FAILED"
    with db.metadata_conn() as conn:
        db.execute(
            conn,
            "UPDATE ingest_item SET status = %s, error_code = %s, error_detail = %s, "
            "updated_at = now() WHERE item_id = %s",
            (status, error.code, str(error)[:1000], item_id),
        )
    ITEMS.labels(status.lower()).inc()


def attempts_exhausted(item_id: str) -> bool:
    """このメッセージが次に nack されると DLQ 行きになるかを判定する。

    DLQ に落ちたメッセージはもう誰も処理しないため、ingest_item を RUNNING の
    まま残すとジョブが永久に終了しない。最終試行と判った時点で FAILED にする。
    """
    with db.metadata_conn() as conn:
        row = db.fetch_one(
            conn, "SELECT attempt_count FROM ingest_item WHERE item_id = %s", (item_id,)
        )
    return bool(row) and row["attempt_count"] >= queueing.MAX_DELIVERY_ATTEMPTS


def refresh_job_counters(job_id: str) -> None:
    with db.metadata_conn() as conn:
        db.execute(
            conn,
            """
            WITH counts AS (
              SELECT
                count(*) FILTER (WHERE status = 'SUCCEEDED') AS succeeded,
                count(*) FILTER (WHERE status IN ('FAILED','QUARANTINED')) AS failed,
                count(*) FILTER (WHERE status IN ('PENDING','RUNNING')) AS remaining
              FROM ingest_item WHERE job_id = %(job_id)s
            )
            UPDATE ingest_job j
               SET succeeded_count = counts.succeeded,
                   failed_count = counts.failed,
                   started_at = COALESCE(j.started_at, now()),
                   -- 取消済みのジョブは状態と完了時刻を変えない（取消時点で処理中だった
                   -- item が後から完了しても、RUNNING や SUCCEEDED へ戻さないため）
                   status = CASE
                     WHEN j.status = 'CANCELLED' THEN j.status
                     WHEN counts.remaining > 0 THEN 'RUNNING'::job_status
                     WHEN counts.failed = 0 THEN 'SUCCEEDED'::job_status
                     WHEN counts.succeeded = 0 THEN 'FAILED'::job_status
                     ELSE 'PARTIAL'::job_status
                   END,
                   completed_at = CASE
                     WHEN j.status = 'CANCELLED' THEN j.completed_at
                     WHEN counts.remaining > 0 THEN NULL
                     ELSE now()
                   END
              FROM counts
             WHERE j.job_id = %(job_id)s
            """,
            {"job_id": job_id},
        )
