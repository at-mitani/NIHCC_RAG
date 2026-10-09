"""データベースの全データを削除する（Git 公開前の容量削減・再現手順のリセット用）。

    python scripts/reset_data.py [--all] [--yes] [--dry-run]

削除対象:
  Metadata DB   取り込み・検索に関するデータ（原画像 BYTEA、胸部野の mask、ジョブ、監査対象外の履歴）
  ベクトル用クラスタ image_embedding（全ベクトル）
  Queue          ingest / search キューと dead-letter queue

既定で保持するもの:
  app_user（利用者）、audit_log（監査ログ）、finding（所見マスタ）、
  model_version（モデル登録）、schema_migration（適用済み migration）

--all を付けると利用者・監査ログ・モデル登録も削除し、初回起動直後の状態へ戻す。
このとき backend-api を再起動して LOCAL_USERS から初期利用者を作り直す
（作り直さないとログインできなくなるため）。

削除後の再投入:
  python scripts/load_initial_users.py    利用者・権限
  python scripts/load_initial_images.py   画像と診断紐づけデータ

DB のデータ本体は Docker ボリューム上にあり Git には含まれない。本スクリプトは
ローカルのボリューム使用量を減らすためのもので、リポジトリ容量には影響しない。
ボリュームごと破棄する場合は `docker compose down -v` を使う。
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: ベクトル用クラスタ（1 Coordinator + 5 Worker）
VECTOR_SERVICE = "vector-coordinator"
VECTOR_DATABASE = "vector"

#: 既定で削除する（CASCADE で従属テーブルも消える）
INGEST_TABLES = (
    "ingest_job",
    "ingest_validation",
    "idempotency_key",
    "outbox",
    "search_audit",
    "reconciliation_report",
    "source_image",
    "patient",
)
#: --all のときに追加で削除する
FULL_RESET_TABLES = ("app_user", "audit_log", "model_version")

QUEUES = ("ingest.items", "ingest.items.dlq", "search.requests", "search.requests.dlq")

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def run(args: list[str], *, stdin: str | None = None, check: bool = True) -> str:
    result = subprocess.run(
        args, cwd=ROOT, input=stdin, capture_output=True, text=True, encoding="utf-8"
    )
    if check and result.returncode != 0:
        sys.stderr.write((result.stderr or result.stdout or "")[:2000] + "\n")
        raise SystemExit(f"コマンドが失敗しました: {' '.join(args[:6])} ...")
    return result.stdout or ""


def secret(name: str) -> str:
    path = ROOT / "secrets" / f"{name}.txt"
    if not path.is_file():
        raise SystemExit(
            f"{path} がありません。先に `bash scripts/bootstrap.sh` を実行してください。"
        )
    return path.read_text(encoding="utf-8").strip()


def psql(service: str, database: str, sql: str, password: str, *, quiet: bool = True) -> str:
    flags = ["-tAc"] if quiet else ["-c"]
    return run(
        [
            "docker", "compose", "exec", "-T",
            "-e", f"PGPASSWORD={password}",
            service, "psql", "-h", "127.0.0.1", "-U", "postgres",
            "-d", database, "-v", "ON_ERROR_STOP=1", *flags, sql,
        ]
    )


def database_size(service: str, database: str, password: str) -> str:
    return psql(service, database, f"SELECT pg_size_pretty(pg_database_size('{database}'))",
                password).strip()


COUNT_SQL = """
SELECT 'source_image=' || (SELECT count(*) FROM source_image)
    || ' chest_asset=' || (SELECT count(*) FROM chest_asset)
    || ' ingest_item=' || (SELECT count(*) FROM ingest_item)
    || ' search_audit=' || (SELECT count(*) FROM search_audit)
    || ' app_user=' || (SELECT count(*) FROM app_user)
    || ' audit_log=' || (SELECT count(*) FROM audit_log)
"""


def report(password: str, title: str) -> None:
    print(f"\n--- {title} ---")
    counts = psql("metadata-db", "metadata", COUNT_SQL, password)
    print(f"metadata  : {counts.strip()}  ({database_size('metadata-db', 'metadata', password)})")
    rows = psql(VECTOR_SERVICE, VECTOR_DATABASE,
                "SELECT count(*) FROM image_embedding", password).strip()
    size = database_size(VECTOR_SERVICE, VECTOR_DATABASE, password)
    print(f"vector    : image_embedding={rows}  ({size})")


def main() -> int:
    parser = argparse.ArgumentParser(description="データベースの全データを削除する")
    parser.add_argument("--all", action="store_true",
                        help="利用者・監査ログ・モデル登録も削除して初期状態へ戻す")
    parser.add_argument("--yes", "-y", action="store_true", help="確認プロンプトを省略する")
    parser.add_argument("--dry-run", action="store_true", help="現在の件数と容量だけを表示する")
    args = parser.parse_args()

    db_password = secret("db_password")

    print("=== 削除前の状態 ===" if not args.dry_run else "=== 現在の状態 ===")
    report(db_password, "件数と容量")

    if args.dry_run:
        print("\n--dry-run のため削除は行いませんでした。")
        return 0

    scope = "利用者・監査ログを含む全データ" if args.all else "取り込み・検索データ（利用者と監査ログは保持）"
    print(f"\n削除対象: {scope}")
    if not args.yes:
        answer = input("この操作は取り消せません。削除しますか？ [yes/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("中止しました。")
            return 1

    # 1) キューを空にする（削除済みデータを参照する滞留メッセージを残さない）
    print("\n[1/3] キューを空にします")
    for queue in QUEUES:
        out = run(["docker", "compose", "exec", "-T", "queue",
                   "rabbitmqctl", "purge_queue", queue], check=False)
        print(f"  {queue}: {'ok' if 'Purging' in out else 'skip（キュー未作成）'}")

    # 2) Metadata DB
    print("[2/3] Metadata DB のデータを削除します")
    tables = list(INGEST_TABLES) + (list(FULL_RESET_TABLES) if args.all else [])
    psql("metadata-db", "metadata",
         f"TRUNCATE {', '.join(tables)} RESTART IDENTITY CASCADE", db_password)
    print(f"  TRUNCATE {len(tables)} テーブル（CASCADE で従属テーブルも削除）")

    # 3) ベクトル
    print("[3/3] ベクトル用クラスタのベクトルを削除します")
    psql(VECTOR_SERVICE, VECTOR_DATABASE, "TRUNCATE image_embedding", db_password)
    print("  image_embedding を削除")

    if args.all:
        # 利用者を消したままだとログインできないため、起動時ブートストラップで作り直す
        print("\n利用者を作り直すため backend-api を再起動します")
        run(["docker", "compose", "up", "-d", "--no-deps", "--force-recreate", "backend-api"])

    report(db_password, "削除後の件数と容量")
    print("\n完了しました。データを入れ直す手順:")
    if args.all:
        print("  1. python scripts/load_initial_users.py    ユーザー・権限")
        print("  2. python scripts/load_initial_images.py   画像と紐づけデータ")
    else:
        print("  python scripts/load_initial_images.py      画像と紐づけデータ")
    print("  ボリュームごと破棄する場合: docker compose down -v")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
