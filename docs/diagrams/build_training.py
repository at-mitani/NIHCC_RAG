"""教師データの作成と利用の処理フロー図（docs/diagrams/training-*.png / .svg）を作り直す。

    python -B docs/diagrams/build_training.py

Playwright（headless Chromium）が要る:
    pip install playwright && python -m playwright install chromium

内容は scripts/chest_image_eval.py（vectors / train）・export_chest_head.py・
nihcc_rag.chest_head / mil.profile_vectors と docs/runbooks/chest-classifier.md・
chest-evaluation.md に合わせてある。
スライドで読めるよう、文字は大きく・少なくしてある（build_preprocess.py と同じ寸法）。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE
sys.path.insert(0, str(HERE))

from svgkit import Svg  # noqa: E402

T, L = 27, 21
STEP = "#1f9e7a"
TRAIN_C, VAL_C, TEST_C = "#2f6fdb", "#d27a0a", "#7c3aed"


def page(width, height, title, subtitle) -> Svg:
    return Svg(width, height, title, subtitle, title_size=38, subtitle_size=22)


def steps(s: Svg, x, w, items) -> None:
    for i, (y, h, title, lines, kind) in enumerate(items):
        s.box(x, y, w, h, title, lines, kind=kind, title_size=T, line_size=L)
        if i < len(items) - 1:
            s.arrow([(x + w / 2, y + h + 2), (x + w / 2, items[i + 1][0] - 3)], color=STEP, width=3)


def link(s: Svg, x1, x2, y, color=STEP) -> None:
    s.arrow([(x1, y), (x2, y)], color=color, width=3)


# ============================================================================ 教師データの作成


def teacher_flow() -> str:
    s = page(1700, 905, "教師データの作成", "画像 1 枚ごとに「特徴」と「所見ラベル」を組にする（EC2 で 1 回）")
    X, W = 30, 700
    steps(s, X, W, [
        (120, 92, "① 画像を集める", ["NIH ChestX-ray14・112,120 枚"], "file"),
        (240, 92, "② 胸部野の抽出", ["U-Net（肺野が小さい 54 枚は除外）"], "app"),
        (360, 92, "③ 特徴を作る", ["RAD-DINO → 768 次元"], "ai"),
        (480, 92, "④ ラベルを付ける", ["14 所見の有無（画像単位）"], "app"),
        (600, 92, "⑤ 患者単位で分ける", ["train・val・test"], "app"),
        (720, 92, "⑥ 教師データ", ["112,066 枚（特徴＋ラベル＋分割）"], "edge"),
    ])

    # ③ 特徴: 37×37 の区画を胸部野で平均
    s.raw('<rect x="780" y="120" width="890" height="212" rx="10" fill="#fdf2f8" stroke="#f3c4dd"/>')
    gx, gy, cell, n = 810, 140, 15, 11
    for r in range(n):
        for c in range(n):
            inside = ((c - 5) / 5.2) ** 2 + ((r - 5.4) / 5.6) ** 2 <= 1.0
            s.raw(f'<rect x="{gx + c * cell}" y="{gy + r * cell}" width="{cell - 2}" height="{cell - 2}" '
                  f'fill="{"#f19ab0" if inside else "#e5e7eb"}"/>')
    s.text(1000, 190, "37 × 37 の区画のうち", size=22, color="#374151")
    s.text(1000, 226, "胸部野の区画だけを平均", size=22, color="#374151")
    s.text(1000, 272, "→ 768 次元の特徴", size=24, weight="bold", color="#831843")
    s.text(1000, 308, "（画像統計量の帰無と比べる）", size=19, color="#6b7280")
    s.arrow([(730, 406), (754, 406), (754, 226), (778, 226)], color=STEP, width=3)

    # ④ ラベルの出典
    s.box(780, 380, 890, 192, "ラベルの出典", ["Data_Entry CSV（レポートから NLP で抽出）",
                                                "1 枚に複数の所見あり（多ラベル）",
                                                "No Finding → 14 所見すべて 0"],
          kind="file", title_size=T - 2, line_size=L)
    link(s, 730, 778, 526)

    # ⑤ 患者単位の分割
    s.raw('<rect x="780" y="600" width="890" height="212" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>')
    bx, by, bw = 810, 640, 830
    parts = [("train", 77648, TRAIN_C), ("val", 11435, VAL_C), ("test", 22983, TEST_C)]
    total = sum(p[1] for p in parts)
    x = bx
    for name, count, color in parts:
        w = bw * count / total
        s.raw(f'<rect x="{x:.1f}" y="{by}" width="{w - 3:.1f}" height="56" rx="6" fill="{color}" '
              f'fill-opacity="0.85"/>')
        s.text(x + w / 2, by + 36, name, size=22, weight="bold", color="#fff", anchor="middle")
        s.text(x + w / 2, by + 90, f"{count:,}", size=21, color="#1f2937", anchor="middle")
        x += w
    s.text(1225, 790, "同じ患者は 1 つの組だけに（成績の水増しを防ぐ）", size=21, color="#374151",
           anchor="middle")
    link(s, 730, 778, 646)

    s.note(30, 840, 1640, ["ラベルは画像単位で部位単位の正解は無い。だから検索の単位も画像 1 枚の胸部野にしている"], size=22)
    return s.render()


# ============================================================================ 教師データの利用


def usage_flow() -> str:
    s = page(1800, 1000, "教師データの利用", "分類器を学習・評価し、合格したら本番で使う")

    # 学習（EC2 で 1 回）
    s.zone(30, 110, 1740, 400, "学習と評価（EC2 で 1 回）", color="#475569", dashed=False, fill="#f8fafc",
           label_size=22)
    s.box(50, 220, 260, 180, "教師データ", ["特徴＋ラベル"], kind="edge", title_size=T, line_size=L)
    rows = [
        (160, "train 70%", TRAIN_C, "分類器を学習", ["線形 × 14 所見（稀な所見に重み）"]),
        (280, "val 10%", VAL_C, "止め時を決める", ["平均 AUC が最大の回を採用"]),
        (400, "test 20%", TEST_C, "評価", ["AUC 0.833・検索 lift ≥ 1.33"]),
    ]
    for y, name, color, title, lines in rows:
        s.raw(f'<rect x="370" y="{y}" width="200" height="90" rx="10" fill="{color}" fill-opacity="0.12" '
              f'stroke="{color}" stroke-width="2"/>')
        s.text(470, y + 54, name, size=24, weight="bold", color=color, anchor="middle")
        s.arrow([(310, 310), (340, 310), (340, y + 45), (368, y + 45)], color=color, width=2.6)
        s.box(620, y, 640, 90, title, lines, kind="app" if name != "test 20%" else "judge",
              title_size=T - 2, line_size=L)
        link(s, 570, 618, y + 45, color)
    s.box(1320, 220, 430, 180, "分類器", ["head-pool-chest.npz", "（合格したら配置）"], kind="file",
          title_size=T, line_size=L)
    s.arrow([(1260, 205), (1290, 205), (1290, 290), (1318, 290)], color=STEP, width=3)
    s.arrow([(1260, 445), (1290, 445), (1290, 330), (1318, 330)], color=STEP, width=3)

    # 本番での利用
    s.zone(30, 545, 1740, 370, "本番での利用（取り込み・検索のたび）", color="#c2418a", dashed=False,
           fill="#fff7fb", label_size=22)
    s.box(560, 590, 680, 100, "embedding-service（GPU）", ["画像の特徴 → 分類器 → 14 所見の確率"], kind="ai",
          title_size=T, line_size=L)
    s.arrow([(1535, 400), (1535, 640), (1242, 640)], color=STEP, width=3)

    outs = [
        (50, "所見プロファイル", ["14 確率＋正常らしさ（15 次元）", "→ 似た画像を全件から探す"], "data"),
        (620, "所見の推定", ["上位 3 件を画面に表示", "研究用・診断ではない"], "edge"),
        (1190, "病変の位置（評価用）", ["確率を区画ごとに分解（chest-map）", "位置を示したのは腫瘤だけ"], "ops"),
    ]
    for x, title, lines, kind in outs:
        s.box(x, 745, 560, 140, title, lines, kind=kind, title_size=T - 1, line_size=L)
        s.arrow([(900, 690), (900, 715), (x + 280, 715), (x + 280, 743)], color="#c2418a", width=2.6)

    s.note(30, 935, 1740, ["教師データ自体は本番の DB に入れない。本番に置くのは学習した分類器だけ"], size=22)
    return s.render()


DIAGRAMS = {"training-teacher-data": teacher_flow, "training-usage": usage_flow}


def main() -> int:
    from playwright.sync_api import sync_playwright

    names = sys.argv[1:] or list(DIAGRAMS)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page_ = browser.new_page(device_scale_factor=1.5)
        for name in names:
            svg = DIAGRAMS[name]()
            (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")
            page_.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
            page_.locator("svg").first.screenshot(path=str(OUT / f"{name}.png"))
            print(f"wrote {name}.png / .svg")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
