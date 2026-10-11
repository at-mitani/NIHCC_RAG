"""ユーザー管理・パスワード変更・再設定の画面確認（機能追加）。

    python tests/integration/ui_users_check.py --password <admin のパスワード>

ログイン → ユーザー管理（一覧・追加・詳細・編集・削除）→ パスワード変更画面 →
パスワード忘れ画面 → 仮パスワード変更 URL での再設定、まで実際に操作する。
スクリーンショットは tmp/ui-users/ へ保存する。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("tmp/ui-users")


def fetch_reset_token(mailpit_url: str, email: str) -> str | None:
    query = urllib.parse.quote(email)
    with urllib.request.urlopen(f"{mailpit_url}/api/v1/search?query={query}", timeout=15) as res:
        messages = json.loads(res.read()).get("messages", [])
    if not messages:
        return None
    with urllib.request.urlopen(
        f"{mailpit_url}/api/v1/message/{messages[0]['ID']}", timeout=15
    ) as res:
        text = json.loads(res.read()).get("Text", "")
    found = re.search(r"https?://[^\s]+/reset\?token=([0-9a-f-]{36})", text)
    return found.group(1) if found else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--mailpit-url", default="http://localhost:8025")
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    suffix = uuid.uuid4().hex[:8]
    username = f"ui.{suffix}"
    email = f"ui.{suffix}@example.com"
    password = "Ui-Test-Passw0rd"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        # 画面は日本語の文言を検査するので、ブラウザの言語を日本語に固定する
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, locale="ja-JP")
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

        # --- ログイン画面のパスワード忘れ導線 --------------------------------
        page.goto(args.base_url, wait_until="networkidle")
        page.screenshot(path=str(OUT / "01-login-with-forgot-link.png"))
        if not page.locator("#go-forgot").is_visible():
            failures.append("ログイン画面にパスワード忘れの導線がない")

        page.click("#go-forgot")
        page.wait_for_selector("#view-forgot:not([hidden])", timeout=10000)
        page.screenshot(path=str(OUT / "02-forgot-form.png"))
        page.click("#back-to-login")
        page.wait_for_selector("#view-login:not([hidden])", timeout=10000)

        # --- 管理者ログイン ---------------------------------------------------
        page.fill("#login-user", args.user)
        page.fill("#login-password", args.password)
        page.click("#login-form button[type=submit]")
        page.wait_for_selector("#view-main:not([hidden])", timeout=20000)

        # --- ユーザー管理: 一覧 -----------------------------------------------
        page.click('[data-tab="users"]')
        page.wait_for_selector("#users-table table", timeout=20000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "03-user-list.png"), full_page=True)
        rows = page.locator("#users-table table tr").count()
        if rows < 2:
            failures.append(f"ユーザー一覧が空: rows={rows}")

        # --- ユーザー管理: 新規追加 -------------------------------------------
        page.click("#new-user")
        page.wait_for_selector("#user-form-card:not([hidden])", timeout=10000)
        page.fill("#user-username", username)
        page.select_option("#user-role", "operator")
        page.fill("#user-email", email)
        page.fill("#user-password", password)
        page.fill("#user-password-confirm", password)
        page.fill("#user-memo", "画面確認で作成したユーザー")
        page.screenshot(path=str(OUT / "04-user-create-form.png"), full_page=True)
        page.click("#user-save")
        page.wait_for_selector("#users-notice:not([hidden])", timeout=20000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "05-user-created.png"), full_page=True)
        if username not in page.content():
            failures.append("追加したユーザーが一覧に現れない")

        # 確認用パスワード不一致が画面で弾かれること
        page.click("#new-user")
        page.fill("#user-username", f"ng.{suffix}")
        page.fill("#user-email", f"ng.{suffix}@example.com")
        page.fill("#user-password", password)
        page.fill("#user-password-confirm", password + "x")
        page.click("#user-save")
        page.wait_for_selector("#user-form-error:not([hidden])", timeout=10000)
        if "一致しません" not in page.locator("#user-form-error").inner_text():
            failures.append("確認用パスワード不一致のエラーが表示されない")
        page.screenshot(path=str(OUT / "06-password-mismatch.png"), full_page=True)
        page.click("#user-cancel")

        # --- ユーザー管理: 詳細と編集 -----------------------------------------
        row = page.locator("#users-table table tr", has_text=username)
        row.locator("button", has_text="詳細").click()
        page.wait_for_selector("#user-detail-card:not([hidden])", timeout=10000)
        page.wait_for_timeout(400)
        page.screenshot(path=str(OUT / "07-user-detail.png"), full_page=True)
        detail_text = page.locator("#user-detail").inner_text()
        for expected in (username, email, "画面確認で作成したユーザー"):
            if expected not in detail_text:
                failures.append(f"詳細に {expected} が表示されない")

        page.click("#user-detail-edit")
        page.wait_for_selector("#user-form-card:not([hidden])", timeout=10000)
        page.select_option("#user-role", "viewer")
        page.fill("#user-memo", "権限を viewer へ変更")
        page.screenshot(path=str(OUT / "08-user-edit-form.png"), full_page=True)
        page.click("#user-save")
        page.wait_for_selector("#users-notice:not([hidden])", timeout=20000)
        page.wait_for_timeout(800)
        if "viewer" not in page.locator("#users-table table tr", has_text=username).inner_text():
            failures.append("編集後の権限が一覧へ反映されない")
        page.screenshot(path=str(OUT / "09-user-updated.png"), full_page=True)

        # --- パスワード変更画面（表示確認） -----------------------------------
        page.click('[data-tab="password"]')
        page.wait_for_selector("#password-form", timeout=10000)
        page.wait_for_timeout(300)
        page.screenshot(path=str(OUT / "10-password-change.png"))
        if "8 文字以上" not in page.content():
            failures.append("パスワード変更画面に入力規則が表示されていない")

        # --- パスワード忘れ → メール → 再設定 --------------------------------
        page.click("#logout")
        page.wait_for_selector("#view-login:not([hidden])", timeout=10000)
        page.click("#go-forgot")
        page.fill("#forgot-email", email)
        page.click("#forgot-form button[type=submit]")
        page.wait_for_selector("#forgot-notice:not([hidden])", timeout=20000)
        page.screenshot(path=str(OUT / "11-forgot-sent.png"))

        token = fetch_reset_token(args.mailpit_url, email)
        if not token:
            failures.append("再設定メールから URL を取得できない")
        else:
            new_password = "Ui-Reset-Passw0rd"
            page.goto(f"{args.base_url}/reset?token={token}", wait_until="networkidle")
            page.wait_for_selector("#view-reset:not([hidden])", timeout=10000)
            page.screenshot(path=str(OUT / "12-reset-form.png"))
            page.fill("#reset-password", new_password)
            page.fill("#reset-password-confirm", new_password + "x")
            page.click("#reset-form button[type=submit]")
            page.wait_for_selector("#reset-error:not([hidden])", timeout=10000)
            page.fill("#reset-password-confirm", new_password)
            page.click("#reset-form button[type=submit]")
            page.wait_for_selector("#view-login:not([hidden])", timeout=20000)
            page.wait_for_timeout(400)
            page.screenshot(path=str(OUT / "13-reset-done.png"))
            if "再設定" not in page.locator("#login-notice").inner_text():
                failures.append("再設定完了の案内がログイン画面に表示されない")

            # 再設定したパスワードでログインできること
            page.fill("#login-user", username)
            page.fill("#login-password", new_password)
            page.click("#login-form button[type=submit]")
            page.wait_for_selector("#view-main:not([hidden])", timeout=20000)
            page.wait_for_timeout(400)
            page.screenshot(path=str(OUT / "14-login-after-reset.png"))
            # viewer はユーザー管理タブを見られない
            if page.locator('[data-tab="users"]').is_visible():
                failures.append("viewer にユーザー管理タブが見えている")

        browser.close()

    for message in errors[:10]:
        if "favicon" not in message.lower():
            failures.append(f"ブラウザエラー: {message[:160]}")

    if failures:
        print("FAILURES:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"UI check passed. screenshots: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
