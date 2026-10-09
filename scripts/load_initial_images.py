"""NIHCC フォルダのデータ（胸部X線画像と診断紐づけ CSV）を取り込む。

    python scripts/load_initial_images.py [--limit N] [--yes]

前提:
  1. bash scripts/bootstrap.sh   （secret と .env の生成）
  2. docker compose build && docker compose up -d
  3. docker compose run --rm db-migrate
  4. python scripts/load_initial_users.py   （取り込み実行者のアカウントが必要）
  5. .env の NIHCC_DATA_DIR がデータセットを指していること
     （コンテナ内では /imports/nihcc に read-only でマウントされる）

処理内容:
  dry-run 検証 → 取り込みジョブ登録 → 完了までの進捗表示 → 結果サマリ

パスはすべてコンテナ内のパスで指定する。既定は NIHCC/images の全画像。
CPU 推論では 1 枚あたり約 2 秒かかるため、全 4,999 枚では 3 時間以上かかる。
動作確認だけなら --limit 50 程度から始めるとよい。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
TERMINAL = ("SUCCEEDED", "FAILED", "PARTIAL", "CANCELLED")

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def request(method: str, url: str, *, body=None, token: str | None = None,
            extra_headers: dict[str, str] | None = None, timeout: int = 900):
    headers = dict(extra_headers or {})
    data = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
            return response.status, (json.loads(payload) if payload else {})
    except urllib.error.HTTPError as error:
        payload = error.read()
        try:
            return error.code, json.loads(payload)
        except json.JSONDecodeError:
            return error.code, {"detail": payload[:300].decode(errors="replace")}
    except urllib.error.URLError as error:
        return 0, {"detail": str(error.reason)}


def admin_credentials(user: str | None, password: str | None) -> tuple[str, str]:
    """--user/--password が無ければ .env の LOCAL_USERS から admin を取り出す。"""
    if user and password:
        return user, password
    env = ROOT / ".env"
    if not env.is_file():
        raise SystemExit(".env がありません。--user と --password を指定してください。")
    for line in env.read_text(encoding="utf-8").splitlines():
        if not line.startswith("LOCAL_USERS="):
            continue
        for entry in line.split("=", 1)[1].split(","):
            parts = entry.strip().split(":")
            if len(parts) == 3 and parts[2] == "admin":
                return user or parts[0], password or parts[1]
    raise SystemExit(".env の LOCAL_USERS に admin が見つかりません。--password を指定してください。")


def wait_ready(api: str, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last = ""
    while time.time() < deadline:
        status, body = request("GET", f"{api}/api/v1/health/ready", timeout=15)
        if status == 200:
            print("API 準備完了")
            return
        last = json.dumps(body.get("errors", body), ensure_ascii=False)[:160]
        print(f"  依存サービス待機中… {last}")
        time.sleep(10)
    raise SystemExit(f"API が {timeout_seconds} 秒以内に準備完了になりませんでした: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description="NIHCC フォルダのデータを取り込む")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--image-root", default="/imports/nihcc/images",
                        help="コンテナ内の画像ディレクトリ")
    parser.add_argument("--csv", default="/imports/nihcc/Data_Entry_2017_v2020.csv",
                        help="コンテナ内の診断紐づけ CSV")
    parser.add_argument("--dataset-version", default="ChestX-ray14-2017")
    parser.add_argument("--limit", type=int, default=0,
                        help="取り込む枚数の上限。0 で全件（既定）")
    parser.add_argument("--on-duplicate", choices=["SKIP", "REPLACE"], default="SKIP",
                        help="既存画像の扱い。SKIP は再処理しない（既定）")
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--yes", "-y", action="store_true", help="確認プロンプトを省略する")
    parser.add_argument("--ready-timeout", type=int, default=300)
    parser.add_argument("--seconds-per-image", type=float, default=2.2,
                        help="所要時間の見積りに使う 1 枚あたりの秒数")
    args = parser.parse_args()
    api = args.base_url.rstrip("/")

    print("=== NIHCC データの取り込み ===")
    wait_ready(api, args.ready_timeout)

    user, password = admin_credentials(args.user, args.password)
    status, body = request("POST", f"{api}/api/v1/auth/token",
                           body={"username": user, "password": password})
    if status != 200:
        raise SystemExit(f"ログインに失敗しました（status={status}）: {body.get('detail', '')}")
    token = body["access_token"]
    print(f"ログイン: {user}（{body.get('role')}）")

    # --- dry-run 検証 --------------------------------------------------------
    payload = {
        "image_root": args.image_root,
        "csv_path": args.csv,
        "dataset_version": args.dataset_version,
        "csv_schema_version": "nihcc-data-entry-2017-v1",
    }
    if args.limit > 0:
        payload["sample_limit"] = args.limit
    print(f"\ndry-run 検証: {args.image_root}")
    status, validation = request("POST", f"{api}/api/v1/ingestions/validate",
                                 body=payload, token=token)
    if status != 200:
        raise SystemExit(f"検証に失敗しました（status={status}）: "
                         f"{validation.get('code', '')} {validation.get('detail', '')}")
    counts = validation["counts"]
    print(f"  CSV 行数        : {counts['csv_rows']}")
    print(f"  画像ファイル数  : {counts['images']}")
    print(f"  突合できた件数  : {counts['matched']}")
    if counts.get("missing_images"):
        print(f"  画像が無い CSV 行: {counts['missing_images']}（警告。分割配布では正常）")
    if validation.get("errors"):
        for issue in validation["errors"][:5]:
            print(f"  エラー: {issue['code']} {issue.get('source_image_name', '')}")
    if not validation["valid"]:
        raise SystemExit("検証に合格しませんでした。パスと CSV を確認してください。")

    total = counts["matched"]
    estimate = int(total * args.seconds_per_image / 60)
    print(f"\n取り込み対象 {total} 件（推定所要時間 約 {estimate} 分 / CPU 推論の実測に基づく）")
    if not args.yes:
        answer = input("開始しますか？ [yes/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("中止しました。")
            return 1

    # --- 取り込み ------------------------------------------------------------
    status, job = request(
        "POST", f"{api}/api/v1/ingestions",
        body={"validation_id": validation["validation_id"], "on_duplicate": args.on_duplicate},
        token=token,
        # 同じ検証 ID で再実行しても二重にジョブを作らない（設計書 9.4）
        extra_headers={"Idempotency-Key": f"load-initial-{validation['validation_id']}"},
    )
    if status not in (200, 202):
        raise SystemExit(f"ジョブ登録に失敗しました（status={status}）: {job}")
    job_id = job["job_id"]
    print(f"\njob_id: {job_id}")

    started = time.time()
    last_done = -1
    while True:
        status, current = request("GET", f"{api}/api/v1/ingestions/{job_id}", token=token)
        if status != 200:
            print(f"  進捗取得に失敗（status={status}）。再試行します。")
            time.sleep(10)
            continue
        progress = current["progress"]
        done = progress["succeeded"] + progress["failed"]
        if done != last_done:
            elapsed = time.time() - started
            rate = elapsed / done if done else 0
            remain = int((total - done) * rate / 60) if done else estimate
            print(f"  {current['status']:<9} {done}/{total} "
                  f"成功={progress['succeeded']} 失敗={progress['failed']} "
                  f"実行中={progress['running']} 残り約 {remain} 分", flush=True)
            last_done = done
        if current["status"] in TERMINAL:
            break
        time.sleep(15)

    # --- 結果 ----------------------------------------------------------------
    progress = current["progress"]
    print(f"\n=== {current['status']} ===")
    print(f"  成功 {progress['succeeded']} / 失敗 {progress['failed']} / 総数 {progress['total']}")
    print(f"  所要 {int((time.time() - started) / 60)} 分")
    for row in current.get("error_summary", []):
        # DUPLICATE_SKIPPED は on_duplicate=SKIP による正常な読み飛ばし（失敗ではない）
        label = "スキップ" if row["error_code"] == "DUPLICATE_SKIPPED" else "エラー"
        print(f"  {label} {row['error_code']}: {row['count']} 件")
    if progress["failed"]:
        print(f"\n失敗した画像の一覧: {api}/api/v1/ingestions/{job_id}/errors")
        print("  抽出できない画像（過露出・高度な陰影など）は設計どおり手動確認へ回されます。")

    status, admin = request("GET", f"{api}/api/v1/admin/status", token=token)
    if status == 200:
        print(f"\n登録画像 {admin['image_count']} 件")
        print("  ベクトル:", {row["state"]: row["count"] for row in admin.get("vector_states", [])})
        print("  所見の推定:", admin.get("finding_estimates", {}).get("count", 0), "件")
    return 0 if current["status"] == "SUCCEEDED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
