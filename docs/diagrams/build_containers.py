"""プロセスの関連を含むコンテナ構成図（docs/diagrams/containers-processes.png / .svg）を作り直す。

    python -B docs/diagrams/build_containers.py

Playwright（headless Chromium）が要る:
    pip install playwright && python -m playwright install chromium

内容は docker-compose.yml・infra/compose/docker-compose.vectors.yml と各サービスの
Dockerfile（CMD）・backend-api の常駐スレッドに合わせてある。
スライドで読めるよう、文字は大きく・少なくしてある（build_preprocess.py と同じ寸法）。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE
sys.path.insert(0, str(HERE))

from svgkit import PALETTE, Svg  # noqa: E402

HTTP, AMQP, SQL = "#2f6fdb", "#d27a0a", "#6d55c9"

COL = {"A": (40, 320), "B": (430, 400), "C": (900, 400), "D": (1370, 400)}
ROW = {1: 210, 2: 450, 3: 690}
H = 180


def container(s: Svg, x, y, w, h, name, procs, kind, *, badge=None, name_size=25) -> None:
    """コンテナの枠（見出し）と、その中のプロセスの箱。"""
    fill, stroke, head = PALETTE[kind]
    s.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" fill-opacity="0.45" '
          f'stroke="{stroke}" stroke-width="2.2"/>')
    s.text(x + 14, y + 34, name, size=name_size, weight="bold", color=head)
    if badge:
        bw = 16 * len(badge) + 24
        s.raw(f'<rect x="{x + w - bw - 10}" y="{y + 12}" width="{bw}" height="30" rx="15" fill="{stroke}"/>')
        s.text(x + w - bw / 2 - 10, y + 33, badge, size=17, weight="bold", color="#fff", anchor="middle")
    n = len(procs)
    gap = 10
    ph = (h - 58 - (n - 1) * gap - 14) / n
    for i, p in enumerate(procs):
        py = y + 50 + i * (ph + gap)
        s.raw(f'<rect x="{x + 14}" y="{py:.0f}" width="{w - 28}" height="{ph:.0f}" rx="8" fill="#ffffff" '
              f'stroke="{stroke}" stroke-width="1.4"/>')
        s.text(x + w / 2, py + ph / 2 + 7, p, size=20, color="#1f2937", anchor="middle")


def cell(col, row, rows=1):
    x, w = COL[col]
    return x, ROW[row], w, H * rows + (ROW[2] - ROW[1] - H) * (rows - 1)


def arrow(s: Svg, pts, color) -> None:
    s.arrow(pts, color=color, width=3)


def label(s: Svg, x, y, text, color, anchor="start") -> None:
    w = 19 * len(text) * 0.9 + 10
    bx = x - 5 if anchor == "start" else x - w / 2
    s.raw(f'<rect x="{bx:.0f}" y="{y - 19}" width="{w:.0f}" height="25" rx="4" fill="#fff" fill-opacity="0.92"/>')
    s.text(x, y, text, size=19, weight="bold", color=color, anchor=anchor)


def containers() -> str:
    s = Svg(1810, 990, "コンテナ構成とプロセスの関連", "各コンテナで動くプロセスと、HTTP・AMQP・SQL のつながり",
            title_size=38, subtitle_size=22)

    s.box(40, 128, 320, 58, "ブラウザ", [], kind="user", title_size=24)

    container(s, *cell("A", 1), "frontend", ["nginx"], "edge")
    container(s, *cell("A", 3), "metadata-db", ["PostgreSQL 17"], "data")
    container(s, *cell("B", 1), "backend-api", ["uvicorn（REST API）", "常駐スレッド（再送・期限切れ削除）"], "edge")
    container(s, *cell("B", 2), "queue", ["RabbitMQ"], "worker")
    container(s, 430, ROW[3], 195, H, "search-worker", ["キュー消費"], "worker", name_size=19)
    container(s, 635, ROW[3], 195, H, "ingest-worker", ["キュー消費 ×N"], "worker", name_size=19)
    container(s, *cell("C", 1), "roi-service", ["uvicorn", "U-Net（ONNX）"], "ai")
    container(s, *cell("C", 2), "embedding-service", ["uvicorn", "BiomedCLIP・RAD-DINO・分類器"], "ai",
              badge="GPU")
    container(s, *cell("C", 3), "search-aggregator", ["uvicorn（全件検索・2 段検索）"], "worker")

    # Citus（D 列の 2〜3 行目）
    x, y, w, h = cell("D", 2, rows=2)
    s.raw(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#ece8fb" fill-opacity="0.45" '
          f'stroke="#6d55c9" stroke-width="2.2"/>')
    s.text(x + 16, y + 34, "Citus（1 ＋ 5 台）", size=25, weight="bold", color="#3b2a86")
    s.box(x + 25, y + 60, w - 50, 80, "coordinator", ["PostgreSQL＋pgvector"], kind="data", title_size=21,
          line_size=17)
    for j in range(5):
        s.box(x + 18 + j * 74, y + 230, 66, 92, f"w{j + 1}", ["6"], kind="data", title_size=19, line_size=15)
    s.arrow([(x + w / 2, y + 140), (x + w / 2, y + 226)], color=SQL, width=2.6)
    s.text(x + w / 2 + 14, y + 190, "30 shard に分散", size=18, color="#3b2a86")
    s.text(x + 20, y + 365, "PostgreSQL＋pgvector", size=17, color="#3b2a86")

    # 凡例（D 列 1 行目）
    lx, ly = 1370, 210
    s.raw(f'<rect x="{lx}" y="{ly}" width="400" height="180" rx="12" fill="#f8fafc" stroke="#cbd5e1"/>')
    for i, (name, color, desc) in enumerate([("HTTP", HTTP, "サービスの呼び出し"), ("AMQP", AMQP, "ジョブ（キュー）"),
                                             ("SQL", SQL, "データベース")]):
        yy = ly + 45 + i * 48
        s.raw(f'<line x1="{lx + 22}" y1="{yy}" x2="{lx + 82}" y2="{yy}" stroke="{color}" stroke-width="4"/>')
        s.text(lx + 96, yy + 7, f"{name}: {desc}", size=20, color="#1f2937")

    # HTTP（青）
    arrow(s, [(200, 186), (200, 208)], HTTP)
    arrow(s, [(360, 300), (428, 300)], HTTP)
    arrow(s, [(830, 262), (898, 262)], HTTP)
    arrow(s, [(830, 318), (866, 318), (866, 520), (898, 520)], HTTP)
    arrow(s, [(830, 360), (850, 360), (850, 780), (898, 780)], HTTP)
    label(s, 852, 248, "検索", HTTP)
    arrow(s, [(830, 718), (882, 718), (882, 600), (898, 600)], HTTP)
    # AMQP（橙）
    arrow(s, [(630, 390), (630, 448)], AMQP)
    arrow(s, [(530, 630), (530, 688)], AMQP)
    arrow(s, [(732, 630), (732, 688)], AMQP)
    # SQL（紫）
    arrow(s, [(430, 375), (400, 375), (400, 655), (200, 655), (200, 688)], SQL)
    arrow(s, [(430, 790), (362, 790)], SQL)
    arrow(s, [(732, 870), (732, 900), (200, 900), (200, 872)], SQL)
    arrow(s, [(732, 900), (1570, 900), (1570, 872)], SQL)
    label(s, 1150, 896, "取り込み: ベクトルを UPSERT", SQL, anchor="middle")
    arrow(s, [(1300, 780), (1335, 780), (1335, 550), (1393, 550)], SQL)

    s.note(40, 925, 1730, ["search-worker は backend-api と同じ検索処理（roi・embedding・aggregator を呼ぶ）。"
                           "アプリのコンテナは read_only"], size=21)
    return s.render()


def main() -> int:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page_ = browser.new_page(device_scale_factor=1.5)
        svg = containers()
        (OUT / "containers-processes.svg").write_text(svg, encoding="utf-8")
        page_.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
        page_.locator("svg").first.screenshot(path=str(OUT / "containers-processes.png"))
        print("wrote containers-processes.png / .svg")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
