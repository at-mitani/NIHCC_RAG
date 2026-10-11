"""画像の正規化と肺野の抽出（ROI）の処理フロー図（docs/diagrams/preprocess-*.png / .svg）を作り直す。

    python -B docs/diagrams/build_preprocess.py

Playwright（headless Chromium）が要る:
    pip install playwright && python -m playwright install chromium

内容は packages/image-pipeline の pipeline.normalize・chest.extract_chest・
segmenter.LungSegmenter・chest.chest_model_inputs と、profile nihcc-unet-local-v1 に
合わせてある。値を変えたらここも直すこと。スライドで読めるよう、文字は大きく・少なくしてある。
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE
sys.path.insert(0, str(HERE))

from svgkit import Svg  # noqa: E402

T, L = 27, 21          # 箱の見出し・本文
STEP = "#1f9e7a"
BAD = "#c0392b"


def page(width, height, title, subtitle) -> Svg:
    return Svg(width, height, title, subtitle, title_size=38, subtitle_size=22)


def steps(s: Svg, x, w, items) -> None:
    """縦に並ぶ処理の箱と矢印。items = [(y, h, title, lines, kind)]"""
    for i, (y, h, title, lines, kind) in enumerate(items):
        s.box(x, y, w, h, title, lines, kind=kind, title_size=T, line_size=L)
        if i < len(items) - 1:
            s.arrow([(x + w / 2, y + h + 2), (x + w / 2, items[i + 1][0] - 3)], color=STEP, width=3)


def link(s: Svg, x1, x2, y, color=STEP) -> None:
    s.arrow([(x1, y), (x2, y)], color=color, width=3)


# ============================================================================ 正規化


def normalize_flow() -> str:
    s = page(1600, 900, "画像の正規化", "装置や露出の違いを揃える")
    X, W = 30, 640
    steps(s, X, W, [
        (120, 92, "① 読み込み", ["濃淡 1 チャネルへ"], "app"),
        (242, 92, "② 濃度を切る", ["下 0.5%・上 99.5%"], "app"),
        (364, 92, "③ 0〜255 へ伸ばす", ["8 bit に揃える"], "app"),
        (486, 92, "④ 白黒反転の判定", ["肺野が明るければ反転"], "app"),
        (608, 92, "⑤ 縮小", ["長辺 1,024 px 以下"], "app"),
        (730, 80, "→ 肺野の抽出（ROI）へ", [], "edge"),
    ])

    # ① の横: 却下
    s.box(730, 120, 840, 92, "却下", ["濃淡画像でない・濃淡の幅が無い"], kind="judge", title_size=T - 2,
          line_size=L)
    link(s, 670, 728, 166, BAD)

    # ② の横: ヒストグラム
    hx, hy, hw, hh = 760, 268, 780, 120
    s.raw(f'<rect x="730" y="232" width="840" height="224" rx="10" fill="#fbfdfc" stroke="#cfe9df"/>')
    n = 50
    for i in range(n):
        t = i / (n - 1)
        v = 0.15 + 0.85 * math.exp(-((t - 0.42) / 0.2) ** 2) + 0.35 * math.exp(-((t - 0.78) / 0.1) ** 2)
        if t < 0.04 or t > 0.96:
            v = 0.12
        bh = v * hh
        cut = t < 0.05 or t > 0.95
        s.raw(f'<rect x="{hx + i * hw / n:.1f}" y="{hy + hh - bh:.1f}" width="{hw / n - 3:.1f}" '
              f'height="{bh:.1f}" fill="{"#d1d5db" if cut else "#7fc4ab"}"/>')
    for frac, label in ((0.05, "0.5%"), (0.95, "99.5%")):
        lx = hx + frac * hw
        s.raw(f'<line x1="{lx}" y1="{hy - 4}" x2="{lx}" y2="{hy + hh + 4}" stroke="{BAD}" '
              f'stroke-width="3" stroke-dasharray="7 5"/>')
        s.text(lx + (12 if frac < 0.5 else -12), hy + 18, label, size=22, weight="bold", color=BAD,
               anchor="start" if frac < 0.5 else "end")
    s.text(1150, 440, "両端を切り、中央を 0〜255 に", size=22, color="#374151", anchor="middle")
    link(s, 670, 728, 288)

    # ④ の横: 白黒反転
    def chest(x, y, lateral, central, label, verdict, color):
        s.text(x + 170, y - 12, label, size=21, weight="bold", color="#374151", anchor="middle")
        s.raw(f'<rect x="{x}" y="{y}" width="340" height="170" rx="8" fill="#9ca3af"/>')
        by, bh = y + 50, 60
        for lx in (x + 30, x + 250):
            s.raw(f'<rect x="{lx}" y="{by}" width="60" height="{bh}" fill="{lateral}"/>')
        s.raw(f'<rect x="{x + 140}" y="{by}" width="60" height="{bh}" fill="{central}"/>')
        for cx, name in ((x + 60, "側方"), (x + 170, "中央"), (x + 280, "側方")):
            s.text(cx, by + bh + 30, name, size=19, color="#111827", anchor="middle")
        s.text(x + 170, y + 205, verdict, size=23, weight="bold", color=color, anchor="middle")

    chest(745, 515, "#1f2937", "#f3f4f6", "通常", "そのまま", STEP)
    chest(1215, 515, "#f3f4f6", "#1f2937", "反転して保存", "白黒を反転", BAD)
    s.text(1150, 610, "→", size=36, color="#6b7280", anchor="middle")
    link(s, 670, 728, 532)

    s.note(30, 838, 1540, ["評価と本番で同じ関数（1 画素違うと分類器の確率がずれる）"], size=22)
    return s.render()


# ============================================================================ 肺野の抽出（ROI）


def _lungs(s: Svg, x, y, k, *, hull=False, band=False, margin=False) -> None:
    """胸部野の作り方の模式図（基準 190 × 200 を k 倍）。"""
    pts = [(22, 98), (28, 46), (52, 30), (136, 30), (160, 46), (166, 98), (162, 150),
           (140, 168), (100, 170), (46, 168), (26, 150)]
    if band:
        pts = [(px, py + 20 if py > 140 else py) for px, py in pts]
    poly = " ".join(f"{px},{py}" for px, py in pts)
    g = [f'<g transform="translate({x},{y}) scale({k})">']
    if margin:
        g.append(f'<polygon points="{poly}" fill="#fde7c8" stroke="#f0a64a" stroke-width="16" '
                 f'stroke-linejoin="round"/>')
    if hull or band or margin:
        g.append(f'<polygon points="{poly}" fill="#fde7c8" stroke="#d27a0a" stroke-width="2.5"/>')
    g.append('<ellipse cx="52" cy="96" rx="25" ry="60" fill="#7aa7e8" stroke="#2f6fdb" stroke-width="2"/>')
    g.append('<ellipse cx="136" cy="96" rx="24" ry="58" fill="#7aa7e8" stroke="#2f6fdb" stroke-width="2"/>')
    g.append('<ellipse cx="100" cy="142" rx="28" ry="22" fill="#f19ab0" stroke="#c2418a" stroke-width="2"/>')
    g.append("</g>")
    s.raw("".join(g))


def roi_flow() -> str:
    s = page(1700, 905, "肺野の抽出（ROI）", "U-Net で肺と心臓を切り出し、胸部野（肺野＋縦隔＋横隔膜）を作る")
    X, W = 30, 700
    steps(s, X, W, [
        (120, 92, "① U-Net で分割", ["右肺・左肺・心臓（160 px）"], "ai"),
        (240, 92, "② 二値化", ["確率で拡大 → 閾値 0.5"], "app"),
        (360, 92, "③ 最大の塊だけ残す", ["離れた誤検出を消す"], "app"),
        (480, 92, "④ 品質の判定", ["肺野が 2% 未満なら却下"], "judge"),
        (600, 92, "⑤ 胸部野 mask", ["凸包 → 下 10% → 全周 2%"], "app"),
        (720, 92, "⑥ モデルへの入力", ["518 px の画像＋mask・crop"], "edge"),
    ])

    s.box(780, 120, 890, 130, "規則ベースとの比較", ["規則ベースは頸部・肩・腹部を掴むことがある",
                                                   "破綻 23.8% → U-Net で 0.5%"],
          kind="ops", title_size=T - 2, line_size=L)

    s.box(780, 275, 890, 130, "判定の結果", ["肺野 2% 未満 → 却下（手動確認へ）",
                                            "片肺なし・左右差 → 続行（信頼度 0.5）"],
          kind="judge", title_size=T - 2, line_size=L)
    s.arrow([(730, 526), (752, 526), (752, 340), (778, 340)], color=BAD, width=3)

    s.zone(780, 430, 890, 382, "胸部野 mask の作り方", color="#d27a0a", dashed=False, fill="#fffaf2",
           label_size=24)
    frames = [("肺・心臓", {}), ("凸包", {"hull": True}), ("下へ 10%", {"band": True}),
              ("全周 2%", {"band": True, "margin": True})]
    for i, (label, opts) in enumerate(frames):
        fx = 795 + i * 222
        _lungs(s, fx, 515, 1.08, **opts)
        s.text(fx + 102, 778, label, size=22, weight="bold", color="#7c4a03", anchor="middle")
        if i < 3:
            s.text(fx + 212, 630, "→", size=30, color="#9ca3af", anchor="middle")
    link(s, 730, 778, 646)

    s.note(30, 840, 1640, ["縦隔・心臓（心拡大）と横隔膜（胸水）を枠に入れる。脊柱は縦隔と重なるので除けない"],
           size=22)
    return s.render()


DIAGRAMS = {"preprocess-normalize": normalize_flow, "preprocess-roi": roi_flow}


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
