"""取り込みの dry-run 検証とジョブ発行（設計書 7.1.1 / 9.4）。"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nihcc_rag import config, csvspec, db, problem, queueing
from nihcc_rag.contract import IngestionRequest, ValidateRequest, ValidationIssue

_LOG = logging.getLogger("backend-api.ingest")

IMPORT_ROOT = Path(os.environ.get("IMPORT_ROOT", "/imports")).resolve()
MAX_ERRORS_IN_RESPONSE = 50


def safe_path(raw: str, *, must_be_dir: bool = False) -> Path:
    """利用者指定パスを IMPORT_ROOT 配下に限定する（設計書 13）。"""
    candidate = Path(raw)
    resolved = (candidate if candidate.is_absolute() else IMPORT_ROOT / candidate).resolve()
    if not resolved.is_relative_to(IMPORT_ROOT):
        raise problem.unprocessable(
            "PATH_OUTSIDE_IMPORT_ROOT",
            f"path must be under the mounted import root ({IMPORT_ROOT}).",
        )
    if must_be_dir and not resolved.is_dir():
        raise problem.unprocessable("IMAGE_ROOT_NOT_FOUND", f"directory not found: {raw}")
    if not must_be_dir and not resolved.is_file():
        raise problem.unprocessable("CSV_NOT_FOUND", f"file not found: {raw}")
    return resolved


@dataclass
class ValidationOutcome:
    validation_id: uuid.UUID
    valid: bool
    counts: dict[str, int]
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    manifest: list[dict[str, Any]] = field(default_factory=list)


def index_images(image_root: Path) -> dict[str, Path]:
    """画像ディレクトリを再帰探索し、ファイル名 → パスの索引を作る。"""
    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.suffix.lower() in (".png", ".jpg", ".jpeg") and path.is_file():
            index.setdefault(path.name, path)
    return index


def validate(request: ValidateRequest, *, actor: str) -> ValidationOutcome:
    if request.csv_schema_version != csvspec.SCHEMA_VERSION:
        raise problem.unprocessable(
            "CSV_SCHEMA_UNSUPPORTED",
            f"csv_schema_version must be {csvspec.SCHEMA_VERSION}",
        )
    image_root = safe_path(request.image_root, must_be_dir=True)
    csv_path = safe_path(request.csv_path)

    images = index_images(image_root)
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    manifest: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    csv_rows = 0

    try:
        rows = list(csvspec.iter_rows(csv_path))
    except csvspec.CsvSchemaError as exc:
        raise problem.unprocessable("CSV_SCHEMA_MISMATCH", str(exc)) from exc

    for row in rows:
        csv_rows += 1
        name = row.source_image_name
        if not csvspec.is_valid_image_name(name):
            errors.append(ValidationIssue(code="IMAGE_NAME_INVALID", source_image_name=name))
            continue
        if name in seen_names:
            warnings.append(ValidationIssue(code="CSV_DUPLICATE_ROW", source_image_name=name))
            continue
        seen_names.add(name)
        if row.unknown_labels:
            warnings.append(
                ValidationIssue(
                    code="UNKNOWN_LABEL",
                    source_image_name=name,
                    detail=",".join(row.unknown_labels),
                )
            )
        path = images.get(name)
        if path is None:
            # 分割配布（images_001 のみ等）では大半が未配置になり得るため警告として扱う
            warnings.append(ValidationIssue(code="IMAGE_MISSING", source_image_name=name))
            continue
        manifest.append(
            {
                "source_image_name": name,
                "image_path": str(path),
                "patient_key_hash": row.patient_key_hash,
                "finding_labels": row.finding_labels,
                "patient_age": row.patient_age,
                "patient_sex": row.patient_sex,
                "view_position": row.view_position,
                "source_metadata": row.source_metadata(),
            }
        )
        if request.sample_limit and len(manifest) >= request.sample_limit:
            break

    extra_images = [name for name in images if name not in seen_names]
    for name in extra_images[:MAX_ERRORS_IN_RESPONSE]:
        warnings.append(ValidationIssue(code="IMAGE_WITHOUT_CSV_ROW", source_image_name=name))

    counts = {
        "csv_rows": csv_rows,
        "images": len(images),
        "matched": len(manifest),
        "missing_images": sum(1 for w in warnings if w.code == "IMAGE_MISSING"),
        "images_without_csv_row": len(extra_images),
    }
    valid = bool(manifest) and not errors

    validation_id = uuid.uuid4()
    with db.metadata_conn() as conn:
        db.execute(
            conn,
            """
            INSERT INTO ingest_validation
              (validation_id, request, valid, counts, errors, warnings, manifest, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                str(validation_id),
                json.dumps(request.model_dump(mode="json"), ensure_ascii=False),
                valid,
                json.dumps(counts),
                json.dumps([e.model_dump() for e in errors[:MAX_ERRORS_IN_RESPONSE]]),
                json.dumps([w.model_dump() for w in warnings[:MAX_ERRORS_IN_RESPONSE]]),
                json.dumps(manifest, ensure_ascii=False),
                actor,
            ),
        )
    _LOG.info("validation stored", extra={"fields": {"validation_id": str(validation_id), **counts}})
    return ValidationOutcome(
        validation_id=validation_id,
        valid=valid,
        counts=counts,
        errors=errors[:MAX_ERRORS_IN_RESPONSE],
        warnings=warnings[:MAX_ERRORS_IN_RESPONSE],
        manifest=manifest,
    )


def request_fingerprint(request: IngestionRequest) -> str:
    payload = json.dumps(request.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_job(
    request: IngestionRequest,
    *,
    actor: str,
    idempotency_key: str,
    trace_id: str,
    publisher: queueing.Publisher,
) -> tuple[uuid.UUID, str, bool]:
    """(job_id, status, 既存かどうか) を返す。Idempotency-Key で再送を吸収する。"""
    fingerprint = request_fingerprint(request)
    with db.metadata_conn() as conn:
        existing = db.fetch_one(
            conn,
            "SELECT job_id, request_fingerprint FROM idempotency_key WHERE idempotency_key = %s",
            (idempotency_key,),
        )
        if existing:
            if existing["request_fingerprint"] != fingerprint:
                raise problem.ProblemError(
                    409,
                    "IDEMPOTENCY_KEY_REUSED",
                    "Conflict",
                    "The Idempotency-Key was already used with a different request body.",
                )
            job = db.fetch_one(
                conn, "SELECT status FROM ingest_job WHERE job_id = %s", (existing["job_id"],)
            )
            return existing["job_id"], (job["status"] if job else "PENDING"), True

        validation = db.fetch_one(
            conn,
            "SELECT manifest, valid, request, expires_at FROM ingest_validation WHERE validation_id = %s",
            (str(request.validation_id),),
        )
    if validation is None:
        raise problem.not_found("validation_id")
    if not validation["valid"]:
        raise problem.unprocessable(
            "VALIDATION_NOT_PASSED", "the referenced validation did not pass dry-run checks"
        )
    manifest = validation["manifest"] or []
    if not manifest:
        raise problem.unprocessable("VALIDATION_EMPTY", "the referenced validation has no items")

    dataset_version = (validation["request"] or {}).get("dataset_version", "unknown")
    preprocess_version = request.preprocess_version or config.preprocess_version()

    job_id = uuid.uuid4()
    with db.metadata_conn() as conn:
        with conn.transaction():
            db.execute(
                conn,
                """
                INSERT INTO ingest_job (job_id, status, request, total_count, created_by)
                VALUES (%s, 'PENDING', %s, %s, %s)
                """,
                (
                    str(job_id),
                    json.dumps(
                        {
                            **request.model_dump(mode="json"),
                            "dataset_version": dataset_version,
                            "preprocess_version": preprocess_version,
                        },
                        ensure_ascii=False,
                    ),
                    len(manifest),
                    actor,
                ),
            )
            db.execute(
                conn,
                "INSERT INTO idempotency_key (idempotency_key, endpoint, request_fingerprint, job_id) "
                "VALUES (%s, %s, %s, %s)",
                (idempotency_key, "POST /api/v1/ingestions", fingerprint, str(job_id)),
            )
            with conn.cursor() as cur:
                for entry in manifest:
                    cur.execute(
                        "INSERT INTO ingest_item (item_id, job_id, source_image_name, status) "
                        "VALUES (%s, %s, %s, 'PENDING') "
                        "ON CONFLICT (job_id, source_image_name) DO NOTHING",
                        (str(uuid.uuid4()), str(job_id), entry["source_image_name"]),
                    )
            # outbox 経由で at-least-once 配信する（設計書 8.3）
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT item_id, source_image_name FROM ingest_item WHERE job_id = %s",
                    (str(job_id),),
                )
                item_ids = {row["source_image_name"]: str(row["item_id"]) for row in cur.fetchall()}
                for entry in manifest:
                    payload = {
                        "job_id": str(job_id),
                        "item_id": item_ids[entry["source_image_name"]],
                        "trace_id": trace_id,
                        "dataset_version": dataset_version,
                        "preprocess_version": preprocess_version,
                        "on_duplicate": request.on_duplicate,
                        "label_source": "NIHCC_CSV_NLP_DERIVED",
                        **entry,
                    }
                    cur.execute(
                        "INSERT INTO outbox (aggregate_type, aggregate_id, routing_key, payload) "
                        "VALUES ('ingest_item', %s, %s, %s)",
                        (
                            payload["item_id"],
                            queueing.ROUTING_INGEST,
                            json.dumps(payload, ensure_ascii=False),
                        ),
                    )

    if request.dry_run:
        with db.metadata_conn() as conn:
            db.execute(
                conn,
                "UPDATE ingest_job SET status = 'CANCELLED', completed_at = now() WHERE job_id = %s",
                (str(job_id),),
            )
        return job_id, "CANCELLED", False

    publish_outbox(publisher)
    return job_id, "PENDING", False


def publish_outbox(publisher: queueing.Publisher, *, batch: int = 1000) -> int:
    """未送信 outbox を Queue へ送る。少なくとも 1 回配信・冪等 UPSERT 前提。"""
    published = 0
    while True:
        with db.metadata_conn() as conn:
            rows = db.fetch_all(
                conn,
                "SELECT outbox_id, routing_key, payload FROM outbox "
                "WHERE published_at IS NULL ORDER BY outbox_id LIMIT %s",
                (batch,),
            )
        if not rows:
            break
        for row in rows:
            try:
                publisher.publish(row["routing_key"], row["payload"])
            except Exception as exc:  # noqa: BLE001 - 次回の publish で再試行する
                with db.metadata_conn() as conn:
                    db.execute(
                        conn,
                        "UPDATE outbox SET attempt_count = attempt_count + 1, last_error = %s "
                        "WHERE outbox_id = %s",
                        (type(exc).__name__, row["outbox_id"]),
                    )
                _LOG.warning("outbox publish failed", extra={"fields": {"outbox_id": row["outbox_id"]}})
                return published
            with db.metadata_conn() as conn:
                db.execute(
                    conn,
                    "UPDATE outbox SET published_at = now() WHERE outbox_id = %s",
                    (row["outbox_id"],),
                )
            published += 1
        if len(rows) < batch:
            break
    return published


#: 取り消せる（まだ終わっていない）ジョブの状態
CANCELLABLE_JOB_STATUSES = ("PENDING", "RUNNING")


def cancel_job(job_id: uuid.UUID) -> dict[str, int]:
    """取り込みジョブを取り消す。

    - ジョブを CANCELLED にし、まだ着手していない item（PENDING）を CANCELLED にする
    - 未送信の outbox は送らないよう送信済み扱いにする（理由を last_error に残す）
    - 実行中（RUNNING）の item は途中で止めず、そのまま完了させる。
      Queue に残ったメッセージは ingest-worker が取消済みとして読み飛ばす

    戻り値は取り消した item 数と、取消時点で実行中だった item 数。
    """
    with db.metadata_conn() as conn:
        with conn.transaction():
            job = db.fetch_one(
                conn, "SELECT status FROM ingest_job WHERE job_id = %s FOR UPDATE", (str(job_id),)
            )
            if job is None:
                raise problem.not_found("job_id")
            if job["status"] not in CANCELLABLE_JOB_STATUSES:
                raise problem.ProblemError(
                    409,
                    "JOB_NOT_CANCELLABLE",
                    "Conflict",
                    f"The job is already {job['status']} and cannot be cancelled.",
                )
            cancelled = db.execute(
                conn,
                "UPDATE ingest_item SET status = 'CANCELLED', error_code = 'JOB_CANCELLED', "
                "updated_at = now() WHERE job_id = %s AND status = 'PENDING'",
                (str(job_id),),
            )
            db.execute(
                conn,
                "UPDATE outbox SET published_at = now(), last_error = 'JOB_CANCELLED' "
                "WHERE published_at IS NULL AND aggregate_type = 'ingest_item' "
                "AND payload->>'job_id' = %s",
                (str(job_id),),
            )
            counts = db.fetch_one(
                conn,
                """
                SELECT
                  count(*) FILTER (WHERE status = 'SUCCEEDED') AS succeeded,
                  count(*) FILTER (WHERE status IN ('FAILED','QUARANTINED')) AS failed,
                  count(*) FILTER (WHERE status = 'RUNNING') AS running
                FROM ingest_item WHERE job_id = %s
                """,
                (str(job_id),),
            )
            db.execute(
                conn,
                "UPDATE ingest_job SET status = 'CANCELLED', completed_at = now(), "
                "succeeded_count = %s, failed_count = %s WHERE job_id = %s",
                (counts["succeeded"], counts["failed"], str(job_id)),
            )
    return {"cancelled": cancelled, "running": counts["running"]}


def retry_failed(job_id: uuid.UUID, *, actor: str, trace_id: str, publisher: queueing.Publisher) -> uuid.UUID:
    """失敗分のみを新しいジョブとして再実行する（設計書 9.1 / 6.2）。"""
    with db.metadata_conn() as conn:
        job = db.fetch_one(conn, "SELECT request FROM ingest_job WHERE job_id = %s", (str(job_id),))
        if job is None:
            raise problem.not_found("job_id")
        failed = db.fetch_all(
            conn,
            "SELECT source_image_name FROM ingest_item WHERE job_id = %s "
            "AND status IN ('FAILED','QUARANTINED')",
            (str(job_id),),
        )
    if not failed:
        raise problem.unprocessable("NO_FAILED_ITEMS", "this job has no failed items to retry")

    original = job["request"] or {}
    validation_id = original.get("validation_id")
    with db.metadata_conn() as conn:
        validation = db.fetch_one(
            conn,
            "SELECT manifest FROM ingest_validation WHERE validation_id = %s",
            (str(validation_id),),
        )
    if validation is None:
        raise problem.unprocessable(
            "VALIDATION_EXPIRED", "the original validation is no longer available; run dry-run again"
        )

    names = {row["source_image_name"] for row in failed}
    manifest = [entry for entry in (validation["manifest"] or []) if entry["source_image_name"] in names]

    new_job_id = uuid.uuid4()
    with db.metadata_conn() as conn:
        with conn.transaction():
            db.execute(
                conn,
                "INSERT INTO ingest_job (job_id, status, request, total_count, created_by) "
                "VALUES (%s, 'PENDING', %s, %s, %s)",
                (
                    str(new_job_id),
                    json.dumps({**original, "retry_of": str(job_id)}, ensure_ascii=False),
                    len(manifest),
                    actor,
                ),
            )
            with conn.cursor() as cur:
                for entry in manifest:
                    item_id = str(uuid.uuid4())
                    cur.execute(
                        "INSERT INTO ingest_item (item_id, job_id, source_image_name, status) "
                        "VALUES (%s, %s, %s, 'PENDING')",
                        (item_id, str(new_job_id), entry["source_image_name"]),
                    )
                    payload = {
                        "job_id": str(new_job_id),
                        "item_id": item_id,
                        "trace_id": trace_id,
                        "dataset_version": original.get("dataset_version", "unknown"),
                        "preprocess_version": original.get("preprocess_version")
                        or config.preprocess_version(),
                        "on_duplicate": original.get("on_duplicate", "SKIP"),
                        "label_source": "NIHCC_CSV_NLP_DERIVED",
                        **entry,
                    }
                    cur.execute(
                        "INSERT INTO outbox (aggregate_type, aggregate_id, routing_key, payload) "
                        "VALUES ('ingest_item', %s, %s, %s)",
                        (item_id, queueing.ROUTING_INGEST, json.dumps(payload, ensure_ascii=False)),
                    )
    publish_outbox(publisher)
    return new_job_id
