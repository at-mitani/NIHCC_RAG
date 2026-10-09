"""初期ユーザー（アカウント・権限・メールアドレス）を取り込む。

    python scripts/load_initial_users.py [--file users.json] [--yes]

前提:
  1. bash scripts/bootstrap.sh   （secret と .env の生成）
  2. docker compose up -d && docker compose run --rm db-migrate

取り込み元:
  --file 未指定       .env の LOCAL_USERS（`名前:パスワード:権限` のカンマ区切り）
                      メールアドレスは `名前@BOOTSTRAP_EMAIL_DOMAIN` を補います。
  --file users.json   アカウント名・パスワード・権限・メールアドレス・メモを明示する
                      （書式は scripts/initial_users.example.json を参照）

パスワードは backend-api コンテナ内で scrypt ハッシュ化してから登録します。
平文は DB にも標準出力にも残りません。既存アカウントは既定で読み飛ばし、
`--update-existing` を付けたときだけ権限・メール・パスワードを上書きします。

app_user が 0 件のときは backend-api が起動時に LOCAL_USERS から同じ初期ユーザーを
自動作成します。本スクリプトは、その自動作成を待たずに明示的に投入したい場合や、
`scripts/reset_data.py --all` の後にユーザーを入れ直す場合に使います。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ROLES = ("viewer", "operator", "admin")
RESULT_MARKER = "@@RESULT@@"

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

# backend-api コンテナ内で実行する登録処理。共有モジュール（nihcc_rag.passwords）を
# 使うことで、画面から作成したユーザーとハッシュ方式を完全に一致させる。
# ユーザー定義は argv ではなく stdin で渡す（平文パスワードをプロセス一覧に出さない）。
CONTAINER_PROGRAM = r"""
import json, sys
from nihcc_rag import db, passwords

spec = json.load(sys.stdin)
db.init_metadata_pool(role="app_meta", application_name="load-initial-users")
created, updated, skipped = [], [], []
with db.metadata_conn() as conn:
    for user in spec["users"]:
        row = db.fetch_one(
            conn, "SELECT user_id FROM app_user WHERE username = %s", (user["username"],)
        )
        if row is not None and not spec["update_existing"]:
            skipped.append(user["username"])
            continue
        password_hash = passwords.hash_password(user["password"])
        if row is not None:
            db.execute(
                conn,
                "UPDATE app_user SET password_hash = %s, role = %s, email = %s, memo = %s, "
                "is_active = true, password_changed_at = now(), "
                "updated_by = 'load_initial_users', updated_at = now() WHERE username = %s",
                (password_hash, user["role"], user["email"], user.get("memo"), user["username"]),
            )
            updated.append(user["username"])
        else:
            db.execute(
                conn,
                "INSERT INTO app_user (username, password_hash, role, email, memo, "
                "  created_by, updated_by) "
                "VALUES (%s, %s, %s, %s, %s, 'load_initial_users', 'load_initial_users')",
                (user["username"], password_hash, user["role"], user["email"], user.get("memo")),
            )
            created.append(user["username"])
    rows = db.fetch_all(
        conn,
        "SELECT username, role, email, is_active FROM app_user ORDER BY role DESC, username",
    )
print("@@RESULT@@" + json.dumps(
    {"created": created, "updated": updated, "skipped": skipped,
     "users": [dict(r) for r in rows]}, default=str))
"""


def env_values() -> dict[str, str]:
    path = ROOT / ".env"
    if not path.is_file():
        raise SystemExit(f"{path} がありません。先に `bash scripts/bootstrap.sh` を実行してください。")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def users_from_env() -> list[dict[str, str | None]]:
    values = env_values()
    domain = values.get("BOOTSTRAP_EMAIL_DOMAIN", "local.example")
    users: list[dict[str, str | None]] = []
    for entry in values.get("LOCAL_USERS", "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) != 3 or parts[2] not in ROLES:
            raise SystemExit(
                f"LOCAL_USERS の書式が不正です: {parts[0] if parts else entry!r} "
                "（`名前:パスワード:viewer|operator|admin`）"
            )
        users.append(
            {
                "username": parts[0],
                "password": parts[1],
                "role": parts[2],
                "email": f"{parts[0]}@{domain}",
                "memo": "初期ユーザー。運用開始前にメールアドレスとパスワードを更新してください。",
            }
        )
    if not users:
        raise SystemExit(".env の LOCAL_USERS が空です。--file でユーザー定義を指定してください。")
    return users


def users_from_file(path: pathlib.Path) -> list[dict[str, str | None]]:
    if not path.is_file():
        raise SystemExit(f"{path} がありません。")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"{path} を JSON として読めません: {error}") from error
    users = payload["users"] if isinstance(payload, dict) else payload
    if not isinstance(users, list) or not users:
        raise SystemExit(f"{path} に users 配列がありません。")
    return [dict(user) for user in users]


def validate(users: list[dict[str, str | None]]) -> None:
    """DB へ送る前に必須項目と権限・パスワード長を検査する（設計書 9.7 と同じ制約）。"""
    seen: set[str] = set()
    for index, user in enumerate(users, start=1):
        for field in ("username", "password", "role", "email"):
            if not user.get(field):
                raise SystemExit(f"{index} 件目: {field} は必須です。")
        if user["role"] not in ROLES:
            raise SystemExit(f"{user['username']}: 権限は {' / '.join(ROLES)} のいずれかです。")
        if not 8 <= len(str(user["password"])) <= 128:
            raise SystemExit(f"{user['username']}: パスワードは 8 文字以上 128 文字以下です。")
        if "@" not in str(user["email"]):
            raise SystemExit(f"{user['username']}: メールアドレスの書式が不正です。")
        if user["username"] in seen:
            raise SystemExit(f"{user['username']}: アカウント名が重複しています。")
        seen.add(str(user["username"]))
    if not any(user["role"] == "admin" for user in users):
        raise SystemExit("admin が 1 件もありません。管理機能を使えなくなるため中止します。")


def main() -> int:
    parser = argparse.ArgumentParser(description="初期ユーザーを取り込む")
    parser.add_argument("--file", type=pathlib.Path, default=None,
                        help="ユーザー定義 JSON。未指定なら .env の LOCAL_USERS を使う")
    parser.add_argument("--update-existing", action="store_true",
                        help="既存アカウントの権限・メール・パスワードを上書きする")
    parser.add_argument("--yes", "-y", action="store_true", help="確認プロンプトを省略する")
    parser.add_argument("--dry-run", action="store_true", help="登録内容の確認だけ行う")
    args = parser.parse_args()

    users = users_from_file(args.file) if args.file else users_from_env()
    validate(users)

    source = str(args.file) if args.file else ".env の LOCAL_USERS"
    print("=== 初期ユーザーの取り込み ===")
    print(f"取り込み元: {source}")
    for user in users:
        # パスワードは表示しない
        print(f"  {user['username']:<12} {user['role']:<9} {user['email']}")
    print(f"既存アカウント: {'上書きする' if args.update_existing else '読み飛ばす'}")

    if args.dry_run:
        print("\n--dry-run のため登録は行いませんでした。")
        return 0
    if not args.yes:
        answer = input("登録しますか？ [yes/N]: ").strip().lower()
        if answer not in ("y", "yes"):
            print("中止しました。")
            return 1

    payload = json.dumps({"users": users, "update_existing": args.update_existing})
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "backend-api", "python", "-c", CONTAINER_PROGRAM],
        cwd=ROOT, input=payload, capture_output=True, text=True, encoding="utf-8",
    )
    marker = [line for line in (result.stdout or "").splitlines() if line.startswith(RESULT_MARKER)]
    if result.returncode != 0 or not marker:
        sys.stderr.write((result.stderr or result.stdout or "")[:2000] + "\n")
        raise SystemExit("登録に失敗しました。backend-api が起動しているか確認してください。")
    outcome = json.loads(marker[-1][len(RESULT_MARKER):])

    print(f"\n新規 {len(outcome['created'])} 件 / 更新 {len(outcome['updated'])} 件 "
          f"/ 読み飛ばし {len(outcome['skipped'])} 件")
    if outcome["skipped"]:
        print(f"  既存のため読み飛ばし: {', '.join(outcome['skipped'])}")
        print("  上書きする場合は --update-existing を付けてください。")

    print("\n--- 登録済みユーザー ---")
    for row in outcome["users"]:
        state = "" if row["is_active"] else "  (無効)"
        print(f"  {row['username']:<12} {row['role']:<9} {row['email']}{state}")

    print("\n完了しました。")
    print("  画像データを取り込む: python scripts/load_initial_images.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
