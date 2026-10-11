"""画面の受入確認（設計書 6 章）。Playwright で実際にブラウザ操作を行う。

    python tests/integration/ui_check.py --password <admin のパスワード>

ログイン → 検索実行 → 結果表示（胸部野の類似症例・所見の推定）→ 管理画面の順に操作し、
各段階のスクリーンショットを tmp/ui/ へ保存する。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("tmp/ui")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", required=True)
    parser.add_argument("--query-image", default="tmp/q.png")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        # 画面は日本語の文言を検査するので、ブラウザの言語を日本語に固定する
        page = browser.new_page(viewport={"width": 1440, "height": 1000}, locale="ja-JP")
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

        page.goto(args.base_url, wait_until="networkidle")
        page.screenshot(path=str(OUT / "01-login.png"))
        if "ログイン" not in page.content():
            failures.append("ログイン画面が表示されない")

        page.fill("#login-user", args.user)
        page.fill("#login-password", args.password)
        page.click("#login-form button[type=submit]")
        page.wait_for_selector("#view-main:not([hidden])", timeout=15000)
        page.screenshot(path=str(OUT / "02-search-form.png"))

        # 検索対象画像を設定して検索する
        if Path(args.query_image).is_file():
            page.set_input_files("#file-input", args.query_image)
            page.wait_for_selector("#preview-wrap:not([hidden])", timeout=10000)
            page.fill("#min-similarity", "-1")
            page.screenshot(path=str(OUT / "03-query-image.png"))
            page.click("#run-search")
            page.wait_for_selector("#results:not([hidden])", timeout=120000)
            page.wait_for_timeout(6000)
            # 遅延読み込み（同時 4 本）が終わるのを待ってから撮影する
            page.wait_for_timeout(8000)
            page.screenshot(path=str(OUT / "04-results.png"), full_page=True)
            hits = page.locator("#chest-results .hit").count()
            if hits == 0:
                failures.append("胸部野の候補カードが 1 件も描画されていない")
            if "部位別" in page.inner_text("#results"):
                failures.append("結果に部位別の表示が残っている")
            # 所見の推定は 2 段目（分類器）があるときだけ出る。出ているなら見出しも出ていること
            if page.locator(".estimate").count() and "診断ではありません" not in page.inner_text("#results"):
                failures.append("所見の推定に「診断ではありません」の見出しが無い")
        else:
            failures.append(f"検索用画像がない: {args.query_image}")

        # 管理画面
        page.click('[data-tab="admin"]')
        page.wait_for_selector("#admin-content table", timeout=30000)
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "06-admin.png"), full_page=True)
        if "vector-coordinator" not in page.content():
            failures.append("管理画面にベクトル用クラスタが表示されていない")

        # 取り込み画面
        page.click('[data-tab="ingest"]')
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "07-ingest.png"))

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
