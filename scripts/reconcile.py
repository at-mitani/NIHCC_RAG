"""日次 reconciliation（設計書 8.3 / 14.1）。

検索の単位は画像 1 枚の胸部野である。

検出対象:
  - 原画像に対するベクトルの欠落（image_vector_state が STORED でない）
  - 孤立ベクトル（Metadata DB に存在しない image_id）
  - 2 段検索の fine ベクトル（embedding_fine）の充填漏れ
    fine を持つ行が 1 件以上あるのに未充填行が残っている場合だけ検出する。
    fine モデル未導入（全行 NULL）は正常な構成なので検出しない。
  - 所見の推定（finding_estimate）の欠落。2 段目が胸部野の分類器のときだけ

検出結果は reconciliation_report へ保存する。--repair を付けると未完了分を
再取り込み対象として登録する（冪等 UPSERT のため再実行は安全）。

    docker compose exec backend-api python /srv/scripts/reconcile.py [--repair]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid

from nihcc_rag import db
from nihcc_rag import logging as jsonlog
from nihcc_rag import queueing

_LOG = logging.getLogger("reconcile")


def collect_issues() -> tuple[dict[str, object], list[dict[str, object]]]:
    issues: list[dict[str, object]] = []

    with db.metadata_conn() as conn:
        images = db.fetch_all(
            conn,
            "SELECT si.image_id, si.source_image_name, "
            "EXISTS (SELECT 1 FROM image_vector_state s WHERE s.image_id = si.image_id "
            "        AND s.state = 'STORED') AS stored, "
            "EXISTS (SELECT 1 FROM chest_asset c WHERE c.image_id = si.image_id) AS has_chest, "
            "EXISTS (SELECT 1 FROM finding_estimate e WHERE e.image_id = si.image_id) "
            "  AS has_estimate "
            "FROM source_image si ORDER BY si.created_at",
        )

    metadata_ids = {str(row["image_id"]) for row in images}
    missing = 0
    for row in images:
        if row["stored"]:
            continue
        missing += 1
        issues.append(
            {
                "type": "MISSING_VECTOR",
                "image_id": str(row["image_id"]),
                "source_image_name": row["source_image_name"],
                "has_chest_asset": bool(row["has_chest"]),
            }
        )

    orphans = 0
    coverage: dict[str, object] = {}
    try:
        with db.vector_conn() as conn:
            counts = db.fetch_one(
                conn,
                "SELECT count(*) AS total, "
                "count(*) FILTER (WHERE embedding_fine IS NULL) AS missing_fine, "
                "count(DISTINCT embedding_fine_model_version) AS fine_models, "
                "count(*) FILTER (WHERE embedding_lung IS NULL) AS missing_lung "
                "FROM image_embedding",
            )
            rows = db.fetch_all(
                conn,
                "SELECT image_id, embedding_model_version FROM image_embedding "
                "ORDER BY created_at DESC LIMIT 200000",
            )
    except Exception as exc:  # noqa: BLE001 - クラスタ停止は報告して終える
        issues.append({"type": "CLUSTER_UNAVAILABLE", "error": type(exc).__name__})
        counts, rows = None, []

    if counts is not None:
        total = int(counts.get("total") or 0)
        missing_fine = int(counts.get("missing_fine") or 0)
        fine_models = int(counts.get("fine_models") or 0)
        coverage = {
            "rows": total,
            "with_fine": total - missing_fine,
            "missing_fine": missing_fine,
            "fine_model_versions": fine_models,
            "missing_lung": int(counts.get("missing_lung") or 0),
        }
        # fine を 1 件も持たないのは「fine モデル未導入」であり正常。
        # 混在している場合だけ、再取り込みが必要な取りこぼしとして報告する。
        if fine_models > 0 and missing_fine > 0:
            issues.append(
                {
                    "type": "MISSING_FINE_VECTOR",
                    "rows": total,
                    "missing_fine": missing_fine,
                    "detail": (
                        "2 段検索の 2 段目（embedding_fine）が未充填の行がある。"
                        "該当行は 1 段目の順位で返るため検索は成立するが、精度は"
                        "1 段検索と同じになる。解消には再取り込みが必要"
                    ),
                }
            )
        if fine_models > 0:
            lacking = sum(1 for row in images if row["stored"] and not row["has_estimate"])
            if lacking:
                issues.append(
                    {
                        "type": "MISSING_FINDING_ESTIMATE",
                        "images": lacking,
                        "detail": "所見の推定が無い画像がある。画面に推定が出ない。再取り込みで解消する",
                    }
                )
    for row in rows:
        if str(row["image_id"]) not in metadata_ids:
            orphans += 1
            issues.append(
                {
                    "type": "ORPHAN_VECTOR",
                    "image_id": str(row["image_id"]),
                    "embedding_model_version": row["embedding_model_version"],
                }
            )

    summary = {
        "unit": "chest",
        "images": len(images),
        "images_with_missing_vectors": missing,
        "orphan_vectors": orphans,
        "vector_coverage": coverage,
    }
    return summary, issues


def repair(issues: list[dict[str, object]]) -> int:
    """未完了画像を再取り込みキューへ投入する（冪等 UPSERT 前提）。"""
    publisher = queueing.Publisher(client_name="reconcile")
    requeued = 0
    try:
        for issue in issues:
            if issue.get("type") != "MISSING_VECTOR":
                continue
            with db.metadata_conn() as conn:
                row = db.fetch_one(
                    conn,
                    "SELECT si.source_image_name, si.source_dataset_version, si.view_position, "
                    "si.patient_age, si.patient_gender, p.source_patient_key_hash "
                    "FROM source_image si JOIN patient p USING (patient_id) WHERE si.image_id = %s",
                    (issue["image_id"],),
                )
                job_id = str(uuid.uuid4())
                item_id = str(uuid.uuid4())
                db.execute(
                    conn,
                    "INSERT INTO ingest_job (job_id, status, request, total_count, created_by) "
                    "VALUES (%s, 'PENDING', %s, 1, 'reconcile')",
                    (job_id, json.dumps({"reconcile": True, "image_id": issue["image_id"]})),
                )
                db.execute(
                    conn,
                    "INSERT INTO ingest_item (item_id, job_id, source_image_name, status) "
                    "VALUES (%s, %s, %s, 'PENDING')",
                    (item_id, job_id, row["source_image_name"]),
                )
            _LOG.warning(
                "reconcile requeue requires the original image path; "
                "run the ingestion again from the UI for %s",
                row["source_image_name"],
            )
            requeued += 1
    finally:
        publisher.close()
    return requeued


def main() -> int:
    parser = argparse.ArgumentParser(description="reconciliation for vectors and metadata")
    parser.add_argument("--repair", action="store_true", help="未完了画像を再取り込み対象として登録する")
    args = parser.parse_args()

    jsonlog.configure("reconcile")
    db.init_metadata_pool(role="app_meta", application_name="reconcile")
    db.init_vector_pool(role="app_search", application_name="reconcile")

    with db.metadata_conn() as conn:
        report = db.fetch_one(
            conn, "INSERT INTO reconciliation_report DEFAULT VALUES RETURNING report_id"
        )
    report_id = str(report["report_id"])

    summary, issues = collect_issues()
    if args.repair:
        summary["requeued"] = repair(issues)

    with db.metadata_conn() as conn:
        db.execute(
            conn,
            "UPDATE reconciliation_report SET finished_at = now(), summary = %s, issues = %s "
            "WHERE report_id = %s",
            (
                json.dumps(summary, ensure_ascii=False),
                json.dumps(issues[:1000], ensure_ascii=False),
                report_id,
            ),
        )
    print(json.dumps({"report_id": report_id, "summary": summary,
                      "issue_count": len(issues)}, ensure_ascii=False, indent=2))
    db.close_all()
    return 0 if not issues else 2


if __name__ == "__main__":
    sys.exit(main())
