"""システム構成の説明用の図（docs/diagrams/stage3-current.png / .svg）を作り直す。

    python -B docs/diagrams/build_samples.py

Playwright（headless Chromium）が要る:
    pip install playwright && python -m playwright install chromium

内容は docker-compose.yml・infra/compose/docker-compose.vectors.yml と DESIGN.md に
合わせてある。胸部野を画像 1 単位にし、教師データ作成と取り込み・検索の流れを含む。
発表のスライドで読めるよう、文字は大きく・少なく・余白は詰めてある。
「フロントエンド ｜ バックエンド ｜ データベース」を左から並べる。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE
sys.path.insert(0, str(HERE))

from svgkit import Svg  # noqa: E402

SEARCH = "#2f6fdb"   # 検索の流れ
INGEST = "#d27a0a"   # 取り込みの流れ
TRAIN = "#2f9e5b"    # 教師データ作成の流れ

T = 22   # 箱の見出し
L = 17   # 箱の本文
Z = 20   # 枠の見出し


def page(width, height, title, subtitle) -> Svg:
    return Svg(width, height, title, subtitle, title_size=34, subtitle_size=19)


def note(s: Svg, x, y, w, text) -> None:
    s.note(x, y, w, [text], size=19)


def zone(s: Svg, x, y, w, h, label, color, fill) -> None:
    s.zone(x, y, w, h, label, color=color, dashed=False, fill=fill, label_size=Z)


# ============================================================================ システム構成


def stage3() -> str:
    s = page(1800, 1160, "教師データによる学習を含むシステム構成",
             "胸部野（肺野＋縦隔＋横隔膜）を画像 1 単位にし、画像単位のラベルで分類器を学習")

    s.box(90, 104, 220, 70, "利用者", [], kind="user", title_size=T)
    s.arrow([(200, 174), (200, 238)], width=2.2, color=SEARCH)

    s.zone(30, 212, 1740, 622, "", color="#b59b2b")
    s.text(1765, 204, "EC2 1 台（GPU）・Docker Compose", size=Z, weight="bold", color="#b59b2b",
           anchor="end")
    for i, (label, color) in enumerate([("検索", SEARCH), ("取り込み", INGEST), ("教師データ作成", TRAIN)]):
        x = 1120 + i * 200
        s.raw(f'<line x1="{x}" y1="122" x2="{x + 44}" y2="122" stroke="{color}" stroke-width="4"/>')
        s.text(x + 52, 129, label, size=19, color="#1f2937")

    # 1 行目
    s.box(50, 240, 300, 110, "frontend", ["取り込み・検索の画面"], kind="edge", title_size=T, line_size=L)
    s.box(410, 240, 330, 110, "backend-api", ["REST API"], kind="edge", title_size=T, line_size=L)
    s.box(800, 240, 420, 110, "roi-service", ["胸部野の切り出し（U-Net）"], kind="ai", title_size=T,
          line_size=L)
    s.cylinder(1280, 236, 470, 118, "metadata-db", ["画像・ラベル・所見の推定"], title_size=T, line_size=L)
    # 2 行目
    s.box(410, 400, 150, 110, "queue", ["ジョブ"], kind="worker", title_size=T - 2, line_size=L)
    s.box(590, 400, 150, 210, "ingest", ["worker", "（取り込み）"], kind="worker", badge="×N",
          title_size=T - 2, line_size=L)
    s.box(800, 400, 420, 210, "embedding-service", ["BiomedCLIP（512）", "RAD-DINO → 分類器",
                                                   "→ 所見プロファイル（15）"],
          kind="ai", badge="GPU", title_size=T, line_size=L)
    s.zone(1280, 400, 470, 400, "Citus（1 ＋ 5 台）", color="#6d55c9", dashed=False, fill="#f6f4fe",
           label_size=Z)
    s.box(1365, 455, 300, 80, "コーディネータ", [], kind="data", title_size=T - 2)
    for j in range(5):
        s.box(1298 + j * 89, 620, 80, 80, f"W{j + 1}", [], kind="data", title_size=19)
    s.arrow([(1515, 535), (1515, 616)], color="#6d55c9", width=2)
    s.text(1300, 760, "1 画像 1 行・30 shard", size=L, color="#3b2a86")
    # 3 行目
    s.box(50, 660, 300, 130, "infra/chest", ["分類器（学習済み）", "→ embedding-service"], kind="file",
          title_size=T, line_size=L)
    s.box(800, 670, 420, 120, "search-aggregator", ["画像: 全件を厳密に比較", "文章: BiomedCLIP の HNSW"],
          kind="worker", title_size=T, line_size=L)

    # 検索（青）
    s.arrow([(350, 295), (408, 295)], color=SEARCH, width=2.2)
    s.arrow([(740, 270), (798, 270)], color=SEARCH, width=2.2)
    s.arrow([(740, 300), (772, 300), (772, 450), (798, 450)], color=SEARCH, width=2.2)
    s.arrow([(740, 330), (756, 330), (756, 730), (798, 730)], color=SEARCH, width=2.2)
    s.arrow([(1220, 730), (1250, 730), (1250, 495), (1363, 495)], color=SEARCH, width=2.2)
    # 取り込み（橙）
    s.arrow([(485, 350), (485, 398)], color=INGEST, width=2.2)
    s.arrow([(560, 455), (588, 455)], color=INGEST, width=2.2)
    s.arrow([(740, 570), (798, 570)], color=INGEST, width=2.2)
    s.arrow([(665, 610), (665, 816), (1760, 816), (1760, 495), (1667, 495)], color=INGEST, width=2.2)
    # メタデータ（灰）
    s.arrow([(575, 240), (575, 226), (1515, 226), (1515, 234)], color="#6b7280", width=1.8)

    # 教師データ作成
    zone(s, 30, 860, 1740, 200, "教師データ作成（EC2 で 1 回）", TRAIN, "#f1faf4")
    steps = [("胸部野の抽出", "U-Net"), ("特徴", "RAD-DINO"), ("教師データ", "画像単位のラベル"),
             ("分類器の学習", "15 次元"), ("評価・配置", "AUC 0.833")]
    for i, (name, sub) in enumerate(steps):
        x = 60 + i * 345
        s.box(x, 905, 300, 125, name, [sub], kind="host", title_size=T, line_size=L)
        if i < len(steps) - 1:
            s.arrow([(x + 302, 967), (x + 343, 967)], color=TRAIN, width=2.2)
    s.arrow([(1590, 905), (1590, 847), (200, 847), (200, 792)], color=TRAIN, dashed=True, width=2.2)

    note(s, 30, 1085, 1740, "要点: 画像単位のラベルで分類器を学習する。検索は画像 1 枚の胸部野単位（部位ごとには探さない）")
    return s.render()


SAMPLES = {"stage3-current": stage3}


def main() -> int:
    from playwright.sync_api import sync_playwright

    names = sys.argv[1:] or list(SAMPLES)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page_ = browser.new_page(device_scale_factor=1.5)
        for name in names:
            svg = SAMPLES[name]()
            (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
            page_.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
            page_.locator("svg").first.screenshot(path=str(OUT / f"{name}.png"))
            print(f"wrote {name}.png / .svg")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
