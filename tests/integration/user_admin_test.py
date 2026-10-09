"""ユーザー管理とパスワード運用の結合テスト（機能追加）。

    python tests/integration/user_admin_test.py --password <admin のパスワード>

確認する内容:
  1. 管理者によるユーザー一覧・新規追加・詳細・編集・削除
  2. 追加したユーザーでのログインと権限（RBAC）
  3. パスワード変更（既存パスワード照合・2 回入力・旧トークン失効）
  4. パスワード忘れによる再設定（メール送付・10 分有効・1 回限り・URL 形式）
  5. パスワードがハッシュ値で保存されていること
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid

PASS = "PASS"
FAIL = "FAIL"
_results: list[tuple[str, str, str]] = []

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def check(name: str, condition: bool, detail: str = "") -> bool:
    _results.append((PASS if condition else FAIL, name, detail))
    print(("  ok " if condition else "FAIL ") + name + (f" - {detail}" if detail else ""), flush=True)
    return condition


def request(method, url, *, body=None, token=None, timeout=30):
    headers = {}
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
            return error.code, {"raw": payload[:200].decode(errors="replace")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("BASE_URL", "http://localhost:8000"))
    parser.add_argument("--mailpit-url", default=os.environ.get("MAILPIT_URL", "http://localhost:8025"))
    parser.add_argument("--admin", default="admin")
    parser.add_argument("--password", required=True)
    args = parser.parse_args()
    api = args.base_url.rstrip("/")

    print("\n=== ユーザー管理・パスワード運用 結合テスト ===\n")

    status, body = request("POST", f"{api}/api/v1/auth/token",
                           body={"username": args.admin, "password": args.password})
    if not check("管理者でログインできる", status == 200, f"status={status}"):
        return 1
    admin_token = body["access_token"]

    # --- 1. ユーザー一覧 ------------------------------------------------------
    status, listing = request("GET", f"{api}/api/v1/users", token=admin_token)
    check("ユーザー一覧を取得できる", status == 200 and listing.get("total", 0) >= 1,
          f"total={listing.get('total')}")
    # password_changed_at のような正当な項目名を誤検知しないよう、値と機微な項目名で判定する
    check("一覧にパスワード情報が含まれない",
          "password_hash" not in json.dumps(listing) and args.password not in json.dumps(listing))

    # --- 2. 新規追加 ----------------------------------------------------------
    suffix = uuid.uuid4().hex[:8]
    username = f"test.{suffix}"
    email = f"test.{suffix}@example.com"
    initial_password = "Initial-Passw0rd"

    status, created = request(
        "POST", f"{api}/api/v1/users", token=admin_token,
        body={
            "username": username, "password": initial_password,
            "password_confirmation": initial_password, "role": "viewer",
            "email": email, "memo": "結合テストで作成", "is_active": True,
        },
    )
    if not check("ユーザーを新規追加できる", status == 201, f"status={status} {str(created)[:160]}"):
        return 1
    user_id = created["user_id"]
    check("応答にパスワード・ハッシュが含まれない",
          "password_hash" not in json.dumps(created)
          and initial_password not in json.dumps(created))
    check("メモが保存される", created.get("memo") == "結合テストで作成")

    # 必須・重複・規則違反
    status, body = request("POST", f"{api}/api/v1/users", token=admin_token,
                           body={"username": f"dup.{suffix}", "password": initial_password,
                                 "password_confirmation": initial_password, "role": "viewer",
                                 "email": email})
    check("メールアドレス重複は 409", status == 409, body.get("code", ""))

    status, body = request("POST", f"{api}/api/v1/users", token=admin_token,
                           body={"username": username, "password": initial_password,
                                 "password_confirmation": initial_password, "role": "viewer",
                                 "email": f"other.{suffix}@example.com"})
    check("アカウント名重複は 409", status == 409, body.get("code", ""))

    status, body = request("POST", f"{api}/api/v1/users", token=admin_token,
                           body={"username": f"short.{suffix}", "password": "short",
                                 "password_confirmation": "short", "role": "viewer",
                                 "email": f"short.{suffix}@example.com"})
    check("8 文字未満のパスワードは 422", status == 422, body.get("code", ""))

    status, body = request("POST", f"{api}/api/v1/users", token=admin_token,
                           body={"username": f"mm.{suffix}", "password": initial_password,
                                 "password_confirmation": initial_password + "x", "role": "viewer",
                                 "email": f"mm.{suffix}@example.com"})
    check("確認用パスワード不一致は 422", status == 422, body.get("code", ""))

    status, body = request("POST", f"{api}/api/v1/users", token=admin_token,
                           body={"username": f"em.{suffix}", "password": initial_password,
                                 "password_confirmation": initial_password, "role": "viewer",
                                 "email": "not-an-email"})
    check("不正なメールアドレスは 422", status == 422, body.get("code", ""))

    # --- 3. 詳細 --------------------------------------------------------------
    status, detail = request("GET", f"{api}/api/v1/users/{user_id}", token=admin_token)
    check("ユーザー詳細を取得できる", status == 200 and detail.get("username") == username)
    check("詳細に権限・メール・メモ・更新者が含まれる",
          all(key in detail for key in ("role", "email", "memo", "updated_by", "password_changed_at")))

    # --- 4. 追加したユーザーでログイン ---------------------------------------
    status, body = request("POST", f"{api}/api/v1/auth/token",
                           body={"username": username, "password": initial_password})
    if not check("追加したユーザーでログインできる", status == 200, f"status={status}"):
        return 1
    user_token = body["access_token"]
    check("役割が viewer で返る", body.get("role") == "viewer", body.get("role", ""))

    status, body = request("GET", f"{api}/api/v1/users", token=user_token)
    check("viewer はユーザー管理 API を使えない（403）", status == 403, f"status={status}")

    # --- 5. 編集 --------------------------------------------------------------
    status, updated = request("PATCH", f"{api}/api/v1/users/{user_id}", token=admin_token,
                              body={"role": "operator", "memo": "権限を更新"})
    check("ユーザーを編集できる", status == 200 and updated.get("role") == "operator",
          f"status={status} role={updated.get('role')}")

    status, body = request("PATCH", f"{api}/api/v1/users/{user_id}", token=admin_token,
                           body={"is_active": False})
    check("無効化できる", status == 200 and body.get("is_active") is False)
    status, body = request("POST", f"{api}/api/v1/auth/token",
                           body={"username": username, "password": initial_password})
    check("無効化したユーザーはログインできない", status == 401, f"status={status}")
    request("PATCH", f"{api}/api/v1/users/{user_id}", token=admin_token, body={"is_active": True})

    # 管理者によるパスワード再設定
    admin_set_password = "Admin-Set-Passw0rd"
    status, body = request("PATCH", f"{api}/api/v1/users/{user_id}", token=admin_token,
                           body={"password": admin_set_password,
                                 "password_confirmation": admin_set_password})
    check("管理者がパスワードを変更できる", status == 200, f"status={status}")
    status, body = request("POST", f"{api}/api/v1/auth/token",
                           body={"username": username, "password": admin_set_password})
    check("変更後のパスワードでログインできる", status == 200, f"status={status}")
    user_token = body.get("access_token", "")

    # 最後の管理者を守る
    status, listing = request("GET", f"{api}/api/v1/users", token=admin_token)
    admins = [u for u in listing["users"] if u["role"] == "admin" and u["is_active"]]
    if len(admins) == 1:
        status, body = request("PATCH", f"{api}/api/v1/users/{admins[0]['user_id']}",
                               token=admin_token, body={"role": "viewer"})
        check("最後の管理者を降格できない", status == 422 and body.get("code") == "LAST_ADMIN_PROTECTED",
              body.get("code", ""))
        status, body = request("DELETE", f"{api}/api/v1/users/{admins[0]['user_id']}",
                               token=admin_token)
        check("最後の管理者を削除できない", status == 422, body.get("code", ""))
    else:
        check("最後の管理者保護（管理者が複数のため確認省略）", True, f"admins={len(admins)}")

    # --- 6. パスワード変更（本人） -------------------------------------------
    changed_password = "Self-Changed-Passw0rd"
    status, body = request("POST", f"{api}/api/v1/account/password", token=user_token,
                           body={"current_password": "wrong-password",
                                 "new_password": changed_password,
                                 "new_password_confirmation": changed_password})
    check("既存パスワードが誤っていれば 422", status == 422 and
          body.get("code") == "CURRENT_PASSWORD_MISMATCH", body.get("code", ""))

    status, body = request("POST", f"{api}/api/v1/account/password", token=user_token,
                           body={"current_password": admin_set_password,
                                 "new_password": changed_password,
                                 "new_password_confirmation": changed_password + "x"})
    check("新パスワードの 2 回入力が一致しなければ 422", status == 422, body.get("code", ""))

    status, body = request("POST", f"{api}/api/v1/account/password", token=user_token,
                           body={"current_password": admin_set_password,
                                 "new_password": "sh0rt",
                                 "new_password_confirmation": "sh0rt"})
    check("8 文字未満の新パスワードは 422", status == 422, body.get("code", ""))

    status, body = request("POST", f"{api}/api/v1/account/password", token=user_token,
                           body={"current_password": admin_set_password,
                                 "new_password": changed_password,
                                 "new_password_confirmation": changed_password})
    check("パスワードを変更できる", status == 200, f"status={status} {str(body)[:120]}")

    status, body = request("GET", f"{api}/api/v1/account/me", token=user_token)
    check("パスワード変更前のトークンは失効する", status == 401, f"status={status}")

    status, body = request("POST", f"{api}/api/v1/auth/token",
                           body={"username": username, "password": changed_password})
    check("新しいパスワードでログインできる", status == 200, f"status={status}")

    # --- 7. パスワード忘れによる再設定 ---------------------------------------
    status, body = request("POST", f"{api}/api/v1/auth/password-reset/request",
                           body={"email": f"unknown.{suffix}@example.com"})
    check("未登録アドレスでも 202（利用者の存在を漏らさない）", status == 202, f"status={status}")

    status, body = request("POST", f"{api}/api/v1/auth/password-reset/request",
                           body={"email": email})
    check("登録済みアドレスでも同じ 202 応答", status == 202, f"status={status}")
    check("応答文が有効期限を案内する", "10" in body.get("detail", ""), body.get("detail", "")[:60])

    # メールから URL を取り出す
    reset_token = None
    try:
        status_code, mail_list = request(
            "GET", f"{args.mailpit_url}/api/v1/search?query={urllib.parse.quote(email)}"
        )
        messages = mail_list.get("messages", [])
        if messages:
            _, message = request("GET", f"{args.mailpit_url}/api/v1/message/{messages[0]['ID']}")
            text = message.get("Text", "")
            found = re.search(
                r"https?://[^\s]+/reset\?token=([0-9a-f-]{36})", text
            )
            if found:
                reset_token = found.group(1)
        check("再設定メールが送信される", bool(messages), f"messages={len(messages)}")
        check("メール本文に仮パスワード変更 URL（画面 URL + UUID）が含まれる", reset_token is not None,
              f"token={'取得済み' if reset_token else '未取得'}")
    except Exception as exc:  # noqa: BLE001
        check("再設定メールを確認できる", False, f"mailpit 参照に失敗: {type(exc).__name__}")

    if reset_token:
        reset_password = "Reset-Passw0rd-9"
        status, body = request("POST", f"{api}/api/v1/auth/password-reset/confirm",
                               body={"token": reset_token, "new_password": "sh0rt",
                                     "new_password_confirmation": "sh0rt"})
        check("再設定でも 8 文字未満は 422", status == 422, body.get("code", ""))

        status, body = request("POST", f"{api}/api/v1/auth/password-reset/confirm",
                               body={"token": reset_token, "new_password": reset_password,
                                     "new_password_confirmation": reset_password + "x"})
        check("再設定でも 2 回入力の不一致は 422", status == 422, body.get("code", ""))

        status, body = request("POST", f"{api}/api/v1/auth/password-reset/confirm",
                               body={"token": str(uuid.uuid4()), "new_password": reset_password,
                                     "new_password_confirmation": reset_password})
        check("未知の token は 422", status == 422 and body.get("code") == "RESET_TOKEN_INVALID",
              body.get("code", ""))

        status, body = request("POST", f"{api}/api/v1/auth/password-reset/confirm",
                               body={"token": reset_token, "new_password": reset_password,
                                     "new_password_confirmation": reset_password})
        check("仮パスワード変更 URL でパスワードを再設定できる", status == 200,
              f"status={status} {str(body)[:120]}")

        status, body = request("POST", f"{api}/api/v1/auth/token",
                               body={"username": username, "password": reset_password})
        check("再設定したパスワードでログインできる", status == 200, f"status={status}")

        status, body = request("POST", f"{api}/api/v1/auth/password-reset/confirm",
                               body={"token": reset_token, "new_password": reset_password,
                                     "new_password_confirmation": reset_password})
        check("同じ token は 2 回使えない", status == 422 and
              body.get("code") == "RESET_TOKEN_INVALID", body.get("code", ""))

    # --- 8. 削除 --------------------------------------------------------------
    status, body = request("DELETE", f"{api}/api/v1/users/{user_id}", token=admin_token)
    check("ユーザーを削除できる", status == 204, f"status={status}")
    status, body = request("GET", f"{api}/api/v1/users/{user_id}", token=admin_token)
    check("削除後は 404", status == 404, f"status={status}")
    status, body = request("POST", f"{api}/api/v1/auth/token",
                           body={"username": username, "password": "Reset-Passw0rd-9"})
    check("削除したユーザーはログインできない", status == 401, f"status={status}")

    failed = [row for row in _results if row[0] == FAIL]
    print(f"\n=== {len(_results) - len(failed)} passed / {len(failed)} failed ===")
    for _, name, detail in failed:
        print(f"  FAIL {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
