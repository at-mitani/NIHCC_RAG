"""起動中のシステムに対する結合スモークテスト（設計書 15.1）。

    python tests/integration/smoke_test.py [--base-url http://localhost:8000] [--limit 30]

確認する内容:
  1. ログインと RBAC
  2. dry-run 検証（CSV と画像の突合）
  3. 取り込みジョブの登録と完了待ち（冪等性の再送確認を含む）
  4. 胸部野とベクトルが登録され、ベクトル用クラスタが 1 Coordinator + 5 Worker であること
  5. 画像検索（IMAGE_TO_IMAGE / TEXT_TO_IMAGE / HYBRID）
  6. 原画像・サムネイル・胸部野の輪郭画像の取得
  7. 入力制約違反が problem+json で拒否されること
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
_results: list[tuple[str, str, str]] = []

# Windows のコンソールは既定が cp932 のため、UTF-8 で出力できるよう再設定する
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def check(name: str, condition: bool, detail: str = "", *, warn_only: bool = False) -> bool:
    """warn_only: 環境依存で判定が変わる項目（GPU 前提の性能目標など）。

    未達でもテスト全体は失敗させず、実測値を必ず表示する。
    """
    state = PASS if condition else (WARN if warn_only else FAIL)
    _results.append((state, name, detail))
    marker = {"PASS": "  ok ", "WARN": "WARN ", "FAIL": "FAIL "}[state]
    print(f"{marker}{name}" + (f" - {detail}" if detail else ""), flush=True)
    return condition


class Client:
    def __init__(self, base_url: str) -> None:
        self.base = base_url.rstrip("/")
        self.token: str | None = None

    def request(self, method: str, path: str, *, body=None, headers=None, raw=False, timeout=120):
        url = f"{self.base}{path}"
        data = None
        request_headers = dict(headers or {})
        if self.token:
            request_headers["Authorization"] = f"Bearer {self.token}"
        if isinstance(body, tuple):  # (content_type, bytes)
            request_headers["Content-Type"] = body[0]
            data = body[1]
        elif body is not None:
            request_headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers=request_headers, method=method)
        # HTTP ヘッダ名は大文字小文字を区別しないため、小文字へ正規化して返す
        def norm(headers) -> dict[str, str]:
            return {key.lower(): value for key, value in dict(headers).items()}

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = response.read()
                if raw:
                    return response.status, payload, norm(response.headers)
                return response.status, (json.loads(payload) if payload else {}), norm(response.headers)
        except urllib.error.HTTPError as error:
            payload = error.read()
            try:
                return error.code, json.loads(payload), norm(error.headers)
            except json.JSONDecodeError:
                return error.code, {"raw": payload[:200].decode(errors="replace")}, norm(error.headers)


def multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[str, bytes]:
    boundary = f"----nihcc{uuid.uuid4().hex}"
    buffer = io.BytesIO()
    for name, value in fields.items():
        buffer.write(f"--{boundary}\r\n".encode())
        buffer.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        buffer.write(value.encode() + b"\r\n")
    for name, (filename, content, content_type) in files.items():
        buffer.write(f"--{boundary}\r\n".encode())
        buffer.write(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        )
        buffer.write(f"Content-Type: {content_type}\r\n\r\n".encode())
        buffer.write(content + b"\r\n")
    buffer.write(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", buffer.getvalue()


def _check_two_stage(result: dict, label: str = "") -> None:
    """検索の進め方（profile / 2 段 / 1 段）の整合性を確認する。

    fine モデルは既定で無効なので「2 段になっていること」は要求しない。
    要求するのは「どのモードでも矛盾がないこと」である:
      - models.rerank が 3 モードのいずれかで、query.strategy と食い違わないこと
      - 無効なら rerank_status も DISABLED で、reranked な候補が無いこと
      - profile なら rerank_status が PROFILE_EXACT で、1 段目の件数上限を持たないこと
      - 2 段なら 1 段目の件数 >= 再スコアリング件数、かつ再スコアリング済みの
        候補には 1 段目の類似度が付いていること
    """
    models = result.get("models", {})
    mode = models.get("rerank")
    expected_strategy = {
        "profile-exact-full-scan": "profile",
        "two-stage-coarse-to-fine-exact": "coarse_to_fine",
        "disabled-single-stage": "coarse_only",
    }
    if not check(
        f"{label}models.rerank が検索の進め方を示す",
        mode in expected_strategy,
        f"rerank={mode!r}",
    ):
        return
    strategy = result.get("query", {}).get("strategy")
    check(
        f"{label}query.strategy が models.rerank と一致する",
        strategy == expected_strategy[mode],
        f"strategy={strategy!r} rerank={mode!r}",
    )

    if mode == "profile-exact-full-scan":
        check(
            f"{label}全件検索では rerank_status が PROFILE_EXACT",
            result.get("rerank_status") in ("PROFILE_EXACT", "NO_CANDIDATES"),
            f"rerank_status={result.get('rerank_status')!r}",
        )
        check(
            f"{label}全件検索では fine モデル版が応答に含まれる",
            bool(models.get("embedding_fine")) and models.get("embedding_fine") != "n/a",
            f"embedding_fine={models.get('embedding_fine')!r}",
        )
        check(
            f"{label}全件検索では 1 段目の件数上限が無い",
            result.get("query", {}).get("stage1_candidate_k") is None,
            f"stage1={result.get('query', {}).get('stage1_candidate_k')}",
        )
        check(
            f"{label}全件検索では検索元と候補に所見の推定が付く",
            bool(result.get("query_finding_estimates"))
            and all(item.get("finding_estimates") for item in result.get("results", [])),
            f"query={len(result.get('query_finding_estimates') or [])}",
        )
        return

    status = result.get("rerank_status")
    reranked = [item for item in result.get("results", []) if item.get("reranked")]

    if mode == "disabled-single-stage":
        check(
            f"{label}1 段検索では rerank_status が DISABLED",
            status in ("DISABLED", None),
            f"rerank_status={status!r}",
        )
        check(f"{label}1 段検索では reranked な候補が無い", not reranked, f"reranked={len(reranked)}")
        check(
            f"{label}1 段検索では models.embedding_fine が n/a",
            models.get("embedding_fine") == "n/a",
            f"embedding_fine={models.get('embedding_fine')!r}",
        )
        return

    check(
        f"{label}2 段検索では fine モデル版が応答に含まれる",
        bool(models.get("embedding_fine")) and models.get("embedding_fine") != "n/a",
        f"embedding_fine={models.get('embedding_fine')!r}",
    )
    check(
        f"{label}1 段目の候補数が再スコアリング件数を下回らない",
        result.get("stage1_count", 0) >= result.get("reranked_count", 0),
        f"stage1={result.get('stage1_count')} reranked={result.get('reranked_count')}",
    )
    check(
        f"{label}2 段検索では検索元と候補に所見の推定が付く",
        bool(result.get("query_finding_estimates"))
        and all(item.get("finding_estimates") for item in result.get("results", [])),
        f"query={len(result.get('query_finding_estimates') or [])}",
    )
    check(
        f"{label}所見の推定に「診断ではない」見出しが付く",
        "診断ではありません" in (result.get("estimate_label") or ""),
        result.get("estimate_label", ""),
    )
    check(
        f"{label}1 段目は candidate_k より多く候補を取る",
        (result.get("query", {}).get("stage1_candidate_k") or 0)
        >= (result.get("query", {}).get("candidate_k") or 0),
        f"stage1={result.get('query', {}).get('stage1_candidate_k')} "
        f"candidate_k={result.get('query', {}).get('candidate_k')}",
    )
    check(
        f"{label}再スコアリング済みの候補に 1 段目の類似度が付く",
        all(item.get("coarse_similarity") is not None for item in reranked),
        f"reranked={len(reranked)}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000"))
    parser.add_argument("--user", default=os.environ.get("SMOKE_USER", "admin"))
    parser.add_argument("--password", default=os.environ.get("SMOKE_PASSWORD", ""))
    parser.add_argument("--image-root", default="/imports/nihcc/images")
    parser.add_argument("--csv-path", default="/imports/nihcc/Data_Entry_2017_v2020.csv")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--query-image", default="", help="ローカルの検索用 PNG パス")
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    if not args.password:
        print("--password または SMOKE_PASSWORD が必要です（.env の LOCAL_USERS を参照）")
        return 2

    client = Client(args.base_url)
    print(f"\n=== NIHCC RAG smoke test @ {client.base} ===\n")

    # 1. ヘルスチェックと認証 ------------------------------------------------
    status, body, _ = client.request("GET", "/api/v1/health/ready")
    check("readiness が ok", status == 200, json.dumps(body.get("checks", body), ensure_ascii=False))

    status, body, _ = client.request("GET", "/api/v1/admin/status")
    check("未認証アクセスが 401 で拒否される", status == 401, f"status={status}")

    status, body, _ = client.request(
        "POST", "/api/v1/auth/token", body={"username": args.user, "password": args.password}
    )
    if not check("ログイン成功", status == 200 and "access_token" in body, f"status={status}"):
        return 1
    client.token = body["access_token"]
    check("role が返る", body.get("role") in ("viewer", "operator", "admin"), body.get("role", ""))

    status, body, _ = client.request(
        "POST", "/api/v1/auth/token", body={"username": args.user, "password": "wrong-password"}
    )
    check("誤ったパスワードは 401", status == 401)

    # 2. dry-run 検証 --------------------------------------------------------
    status, validation, _ = client.request(
        "POST",
        "/api/v1/ingestions/validate",
        body={
            "image_root": args.image_root,
            "csv_path": args.csv_path,
            "dataset_version": "ChestX-ray14-2017",
            "csv_schema_version": "nihcc-data-entry-2017-v1",
            "sample_limit": args.limit,
        },
        timeout=600,
    )
    if not check(
        "dry-run 検証が成功", status == 200 and validation.get("valid"),
        json.dumps(validation.get("counts", validation), ensure_ascii=False)[:200],
    ):
        return 1
    validation_id = validation["validation_id"]
    check("CSV 行数が読み取れている", validation["counts"]["csv_rows"] > 0,
          f"csv_rows={validation['counts']['csv_rows']}")
    check("画像と CSV が突合されている", validation["counts"]["matched"] > 0,
          f"matched={validation['counts']['matched']}")

    status, body, _ = client.request(
        "POST",
        "/api/v1/ingestions/validate",
        body={
            "image_root": "/etc",
            "csv_path": args.csv_path,
            "dataset_version": "x",
        },
    )
    check("インポートルート外のパスが拒否される", status == 422 and
          body.get("code") == "PATH_OUTSIDE_IMPORT_ROOT", body.get("code", ""))

    # 3. 取り込み ------------------------------------------------------------
    key = str(uuid.uuid4())
    payload = {"validation_id": validation_id, "on_duplicate": "REPLACE"}
    status, job, _ = client.request(
        "POST", "/api/v1/ingestions", body=payload, headers={"Idempotency-Key": key}
    )
    if not check("取り込みジョブが 202 で受理される", status == 202 and "job_id" in job,
                 f"status={status} {json.dumps(job, ensure_ascii=False)[:160]}"):
        return 1
    job_id = job["job_id"]

    status, repeat, _ = client.request(
        "POST", "/api/v1/ingestions", body=payload, headers={"Idempotency-Key": key}
    )
    check("同一 Idempotency-Key の再送で同じ job_id が返る",
          repeat.get("job_id") == job_id, f"status={status}")

    status, conflict, _ = client.request(
        "POST",
        "/api/v1/ingestions",
        body={**payload, "on_duplicate": "SKIP"},
        headers={"Idempotency-Key": key},
    )
    check("異なる本文での Idempotency-Key 再利用は 409", status == 409, conflict.get("code", ""))

    print(f"\n  取り込み待機中… job_id={job_id}")
    deadline = time.time() + args.timeout
    progress = {}
    while time.time() < deadline:
        status, current, _ = client.request("GET", f"/api/v1/ingestions/{job_id}")
        progress = current.get("progress", {})
        print(f"    {current.get('status'):<10} "
              f"succeeded={progress.get('succeeded')} failed={progress.get('failed')} "
              f"pending={progress.get('pending')} running={progress.get('running')}", flush=True)
        if current.get("status") in ("SUCCEEDED", "FAILED", "PARTIAL", "CANCELLED"):
            break
        time.sleep(5)
    check("取り込みジョブが完了", current.get("status") in ("SUCCEEDED", "PARTIAL"),
          f"status={current.get('status')} errors={current.get('error_summary')}")
    check("1 件以上取り込み成功", progress.get("succeeded", 0) > 0,
          f"succeeded={progress.get('succeeded')}")

    status, _, headers = client.request("GET", f"/api/v1/ingestions/{job_id}/errors", raw=True)
    check("エラー CSV がダウンロードできる", status == 200 and "csv" in headers.get("content-type", ""),
          f"status={status} type={headers.get('content-type')}")

    # 3b. 取消 ---------------------------------------------------------------
    status, body, _ = client.request("POST", f"/api/v1/ingestions/{job_id}/cancel")
    check("終了済みジョブの取消は 409", status == 409 and body.get("code") == "JOB_NOT_CANCELLABLE",
          f"status={status} code={body.get('code', '')}")

    status, created, _ = client.request(
        "POST", "/api/v1/ingestions",
        body={**payload, "on_duplicate": "SKIP"},
        headers={"Idempotency-Key": str(uuid.uuid4())},
    )
    if check("取消確認用のジョブが受理される", status == 202, f"status={status}"):
        cancel_id = created["job_id"]
        status, cancelled, _ = client.request("POST", f"/api/v1/ingestions/{cancel_id}/cancel")
        # 件数が少ないと取消より先に完了することがあるため、その場合は警告にとどめる
        finished_first = status == 409
        check("待機中・実行中のジョブを取り消せる",
              status == 200 and cancelled.get("status") == "CANCELLED",
              f"status={status} job_status={cancelled.get('status')}", warn_only=finished_first)
        if not finished_first:
            status, _, _ = client.request("POST", f"/api/v1/ingestions/{cancel_id}/cancel")
            check("取消済みジョブの再取消は 409", status == 409, f"status={status}")
            time.sleep(10)
            status, after, _ = client.request("GET", f"/api/v1/ingestions/{cancel_id}")
            check("取消後に処理中の item が完了しても CANCELLED のまま",
                  after.get("status") == "CANCELLED", f"status={after.get('status')}")

    # 4. 胸部野とベクトルの登録状況 ------------------------------------------
    status, admin, _ = client.request("GET", "/api/v1/admin/status")
    chest = admin.get("chest_assets", {})
    check("胸部野が登録されている", (chest.get("count") or 0) > 0, json.dumps(chest))
    cluster = admin.get("cluster", {})
    check("ベクトル用クラスタにベクトルがある", (cluster.get("row_count") or 0) > 0,
          f"row_count={cluster.get('row_count')}")
    check("Worker が 5 台登録されている", len(cluster.get("workers", [])) == 5,
          json.dumps(cluster.get("workers", [])))
    placement = cluster.get("shard_placement", {})
    check("shard が Worker 5 台に均等に載っている",
          len(placement) == 5 and len(set(placement.values())) == 1, json.dumps(placement))
    states = {row["state"]: row["count"] for row in admin.get("vector_states", [])}
    check("ベクトルの登録に失敗した画像が無い", not states.get("FAILED"), json.dumps(states))

    # 5. 検索 ----------------------------------------------------------------
    query_bytes: bytes | None = None
    if args.query_image and Path(args.query_image).is_file():
        query_bytes = Path(args.query_image).read_bytes()
    else:
        status, images, _ = client.request("GET", "/api/v1/admin/status")
        # 取り込み済み画像の 1 件を検索クエリとして使う
        status, job_status, _ = client.request("GET", f"/api/v1/ingestions/{job_id}")
        status, search_probe, _ = client.request(
            "POST",
            "/api/v1/searches",
            body=("application/json", b"{}"),
        )
        # 画像は API 経由で取得する（取り込み済み画像を検索クエリに再利用）
        status, admin2, _ = client.request("GET", "/api/v1/admin/status")

    if query_bytes is None:
        # 取り込み済みの原画像を 1 件取得して検索クエリにする
        status, body, _ = client.request(
            "POST",
            "/api/v1/searches",
            body=multipart({"request": json.dumps({"mode": "TEXT_TO_IMAGE",
                                                   "text": "chest radiograph", "top_k": 1})}, {}),
        )
        image_id = None
        for item in body.get("results", []):
            image_id = item["image_id"]
            break
        if image_id:
            status, content, _ = client.request(
                "GET", f"/api/v1/images/{image_id}/content", raw=True
            )
            if status == 200:
                query_bytes = content

    if query_bytes:
        content_type, body_bytes = multipart(
            {"request": json.dumps({"mode": "IMAGE_TO_IMAGE", "top_k": 5, "min_similarity": -1.0})},
            {"image": ("query.png", query_bytes, "image/png")},
        )
        # 設計書 4.2 の応答目標は warm 状態の値のため、1 回目は暖機として計測しない
        client.request("POST", "/api/v1/searches", body=(content_type, body_bytes), timeout=180)
        samples = []
        for _ in range(3):
            started = time.time()
            status, result, _ = client.request(
                "POST", "/api/v1/searches", body=(content_type, body_bytes), timeout=180
            )
            samples.append(time.time() - started)
        elapsed = sorted(samples)[len(samples) // 2]
        if check("IMAGE_TO_IMAGE 検索が成功", status == 200,
                 f"status={status} {json.dumps(result, ensure_ascii=False)[:200]}"):
            check("検索の単位が胸部野", result.get("models", {}).get("unit") == "chest",
                  str(result.get("models", {}).get("unit")))
            hits = len(result.get("results", []))
            check("候補が 1 件以上ある", hits > 0, f"hits={hits}")
            check("検索元の胸部野の画像が返る", bool(result.get("query_chest_url")),
                  str(result.get("query_chest_url")))
            check("研究用途の警告が含まれる",
                  any("研究用途" in warning for warning in result.get("warnings", [])))
            check("モデル版が応答に含まれる",
                  bool(result.get("models", {}).get("embedding")),
                  result.get("models", {}).get("embedding", ""))
            _check_two_stage(result)
            # 既定（profile）とは別に、2 段検索（coarse_to_fine）の経路も確かめる
            if result.get("query", {}).get("strategy") == "profile":
                content_type_2, body_2 = multipart(
                    {"request": json.dumps({"mode": "IMAGE_TO_IMAGE", "top_k": 5,
                                            "min_similarity": -1.0,
                                            "strategy": "coarse_to_fine"})},
                    {"image": ("query.png", query_bytes, "image/png")},
                )
                status_2, result_2, _ = client.request(
                    "POST", "/api/v1/searches", body=(content_type_2, body_2), timeout=180
                )
                if check("2 段検索（strategy=coarse_to_fine）が成功", status_2 == 200,
                         f"status={status_2}"):
                    _check_two_stage(result_2, "2 段: ")
                    check(
                        "2 段: 進め方が coarse_to_fine",
                        result_2.get("query", {}).get("strategy") == "coarse_to_fine",
                        str(result_2.get("query", {}).get("strategy")),
                    )
            # 設計書 4.2 の目標「warm 状態、Top-10 で API P95 3 秒以内」は
            # GPU 推論を含む目標であり、実機試験で確定するものと明記されている。
            # CPU 推論環境では未達になり得るため警告として扱い、実測値を記録する。
            on_cpu = "cuda" not in str(result.get("models", {})).lower()
            check(
                f"warm 検索応答が 3 秒以内（中央値 {elapsed:.2f}s）",
                elapsed <= 3.0,
                f"samples={[round(v, 2) for v in samples]}"
                + ("（CPU 推論。設計書 4.2 の目標は GPU 前提）" if on_cpu else ""),
                warn_only=on_cpu,
            )

            item = result["results"][0] if result.get("results") else None
            if item:
                status, _, headers = client.request(
                    "GET", f"/api/v1/images/{item['image_id']}/thumbnail", raw=True
                )
                check("サムネイルが取得できる", status == 200 and headers.get("content-type") == "image/png",
                      f"status={status} type={headers.get('content-type')}")
                status, _, headers = client.request(
                    "GET", f"/api/v1/images/{item['image_id']}/chest", raw=True,
                )
                check("胸部野の輪郭画像が取得できる", status == 200)
                status, _, _ = client.request(
                    "GET", f"/api/v1/searches/{result['search_id']}/chest", raw=True
                )
                check("検索元の胸部野の画像が取得できる", status == 200, f"status={status}")
                status, meta, _ = client.request("GET", f"/api/v1/images/{item['image_id']}")
                check("画像メタデータに付随ラベル出典が含まれる",
                      meta.get("label_source") == "NIHCC_CSV_NLP_DERIVED", meta.get("label_source", ""))
                check("患者 ID が仮名化されている",
                      str(meta.get("patient_ref", "")).startswith("P"), meta.get("patient_ref", ""))
                check("画像メタデータに胸部野が含まれる", bool(meta.get("chest")),
                      str(meta.get("chest", {}))[:120])

        # HYBRID
        content_type, body_bytes = multipart(
            {"request": json.dumps({"mode": "HYBRID", "text": "右上肺野の結節影",
                                    "text_language": "ja", "image_weight": 0.7, "top_k": 5,
                                    "min_similarity": -1.0})},
            {"image": ("query.png", query_bytes, "image/png")},
        )
        status, result, _ = client.request(
            "POST", "/api/v1/searches", body=(content_type, body_bytes), timeout=180
        )
        check("HYBRID 検索が成功", status == 200, f"status={status}")
        if status == 200:
            check("日本語が英語へ正規化されて記録される",
                  result.get("query", {}).get("text_normalized_en", "").startswith("pulmonary nodule"),
                  result.get("query", {}).get("text_normalized_en", ""))
    else:
        check("検索クエリ画像を用意できた", False, "取り込み済み画像を取得できませんでした")

    # TEXT_TO_IMAGE
    content_type, body_bytes = multipart(
        {"request": json.dumps({"mode": "TEXT_TO_IMAGE",
                                "text": "pulmonary nodule in the right upper lung field",
                                "top_k": 5, "min_similarity": -1.0})},
        {},
    )
    status, result, _ = client.request("POST", "/api/v1/searches", body=(content_type, body_bytes))
    check("TEXT_TO_IMAGE 検索が成功", status == 200, f"status={status}")

    # 6. 入力制約 ------------------------------------------------------------
    content_type, body_bytes = multipart(
        {"request": json.dumps({"mode": "HYBRID", "text": "nodule"})}, {}
    )
    status, body, _ = client.request("POST", "/api/v1/searches", body=(content_type, body_bytes))
    check("HYBRID で画像なしは 422", status == 422 and body.get("code") == "SEARCH_INPUT_MISSING",
          body.get("code", ""))
    check("エラーが problem+json 形式", "type" in body and "trace_id" in body)

    content_type, body_bytes = multipart(
        {"request": json.dumps({"mode": "TEXT_TO_IMAGE", "text": "昨日から咳が続いています",
                                "text_language": "ja"})}, {}
    )
    status, body, _ = client.request("POST", "/api/v1/searches", body=(content_type, body_bytes))
    check("辞書外の日本語自由文は翻訳せず 422", status == 422 and
          body.get("code") == "TEXT_NOT_TRANSLATABLE", body.get("code", ""))

    content_type, body_bytes = multipart(
        {"request": json.dumps({"mode": "IMAGE_TO_IMAGE"})},
        {"image": ("bad.png", b"not-an-image-at-all", "image/png")},
    )
    status, body, _ = client.request("POST", "/api/v1/searches", body=(content_type, body_bytes))
    check("不正な画像は 415/422 で拒否", status in (415, 422), f"status={status} {body.get('code','')}")

    # --- 結果 ---------------------------------------------------------------
    failed = [row for row in _results if row[0] == FAIL]
    print(f"\n=== {len(_results) - len(failed)} passed / {len(failed)} failed ===")
    for _, name, detail in failed:
        print(f"  FAIL {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
