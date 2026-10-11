"""画像データの取り込みと、画像検索・結果表示の処理フロー図（docs/diagrams/flow-*-simple.png / .svg）
を作り直す。

    python -B docs/diagrams/build_flows.py

Playwright（headless Chromium）が要る:
    pip install playwright && python -m playwright install chromium

内容は backend-api（routers/ingestions・searches、services/search）・ingest-worker
（processing.py）・frontend（app.js）と DESIGN.md §3 に合わせてある。
スライドで読めるよう、文字は大きく・少なくしてある（build_preprocess.py と同じ寸法）。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE
sys.path.insert(0, str(HERE))

from svgkit import PALETTE, Svg  # noqa: E402

T, L = 27, 21
STEP = "#1f9e7a"
BAD = "#c0392b"


def page(width, height, title, subtitle) -> Svg:
    return Svg(width, height, title, subtitle, title_size=38, subtitle_size=22)


def steps(s: Svg, x, w, items) -> None:
    for i, (y, h, title, lines, kind) in enumerate(items):
        s.box(x, y, w, h, title, lines, kind=kind, title_size=T, line_size=L)
        if i < len(items) - 1:
            s.arrow([(x + w / 2, y + h + 2), (x + w / 2, items[i + 1][0] - 3)], color=STEP, width=3)


def link(s: Svg, x1, x2, y, color=STEP) -> None:
    s.arrow([(x1, y), (x2, y)], color=color, width=3)


def pill(s: Svg, x, y, text, color, size=18) -> float:
    w = size * len(text) * 0.98 + 24
    s.raw(f'<rect x="{x}" y="{y}" width="{w:.0f}" height="{size + 14}" rx="{(size + 14) / 2}" '
          f'fill="{color}" fill-opacity="0.13" stroke="{color}" stroke-width="1.5"/>')
    s.text(x + w / 2, y + size + 2, text, size=size, color=color, anchor="middle", weight="bold")
    return w


# ============================================================================ 取り込み


def ingest_flow() -> str:
    s = page(1700, 905, "画像データの取り込み", "画像とラベルを登録し、検索できる状態にする")
    X, W = 30, 700
    steps(s, X, W, [
        (120, 92, "① 画面で指定", ["画像フォルダとラベル CSV"], "edge"),
        (240, 92, "② 検証", ["CSV と画像を突き合わせる"], "edge"),
        (360, 92, "③ ジョブ登録", ["1 枚 = 1 件をキューへ"], "worker"),
        (480, 92, "④ 胸部野の抽出", ["roi-service（U-Net）"], "ai"),
        (600, 92, "⑤ ベクトル化", ["BiomedCLIP・所見プロファイル"], "ai"),
        (720, 92, "⑥ 登録", ["メタデータ・ベクトル・所見の推定"], "data"),
    ])

    s.box(780, 206, 890, 160, "受付で確かめる", ["件数・欠損・重複を先に表示",
                                                 "同じ画像の再送は二重に入らない"],
          kind="edge", title_size=T - 2, line_size=L)
    link(s, 730, 778, 286)

    s.box(780, 390, 890, 186, "1 枚ずつ処理（非同期）", ["複数の ingest-worker が並行して処理",
                                                        "失敗しても止めない（理由を記録）",
                                                        "肺野が小さい画像 → 手動確認へ"],
          kind="worker", title_size=T - 2, line_size=L)
    link(s, 730, 778, 406)

    s.raw('<rect x="780" y="600" width="890" height="212" rx="10" fill="#f6f4fe" stroke="#d6cdf5"/>')
    s.cylinder(805, 618, 410, 176, "metadata-db", ["画像・ラベル", "胸部野・所見の推定"], title_size=T - 3,
               line_size=L - 1)
    s.cylinder(1240, 618, 410, 176, "ベクトル用クラスタ", ["1 画像 1 行", "512・15・16 次元"],
               title_size=T - 3, line_size=L - 1)
    link(s, 730, 778, 766)

    s.note(30, 840, 1640, ["実績: EC2（GPU）で 112,066 枚を 231 分。画面は進捗を 1.5 秒ごとに更新"], size=22)
    return s.render()


# ============================================================================ 検索と結果表示


def search_flow() -> str:
    s = page(1700, 990, "画像検索と結果表示", "検索元に似た症例を探し、所見の推定と一緒に表示する")
    X, W = 30, 700
    steps(s, X, W, [
        (120, 88, "① 検索条件を入力", ["画像・文・件数・絞り込み"], "edge"),
        (234, 88, "② 文を英語に", ["承認済みの辞書だけ（外部 API なし）"], "edge"),
        (348, 88, "③ 胸部野の抽出", ["検索元画像を roi-service へ"], "ai"),
        (462, 88, "④ ベクトル化", ["BiomedCLIP・所見プロファイル"], "ai"),
        (576, 88, "⑤ ベクトル検索", ["画像: 全件比較 ／ 文: BiomedCLIP"], "worker"),
        (690, 88, "⑥ 結果を組み立てる", ["ラベル・所見の推定・絞り込み"], "data"),
        (804, 88, "⑦ 画面に表示", ["似た症例＋胸郭の形が似た症例"], "edge"),
    ])

    s.box(780, 150, 890, 196, "検索元の扱い", ["辞書に無い自由文は翻訳しない",
                                              "胸部野の信頼度が低いと結果を抑制",
                                              "検索元画像は 60 分で削除"],
          kind="judge", title_size=T - 2, line_size=L)
    link(s, 730, 778, 278)

    s.box(780, 378, 890, 204, "⑤ の進め方", ["画像だけ → 所見プロファイルで",
                                            "　全画像と厳密に比較（p95 24 ms）",
                                            "文を含む → BiomedCLIP の HNSW で検索"],
          kind="worker", title_size=T - 2, line_size=L, align="left")
    s.arrow([(730, 620), (754, 620), (754, 480), (778, 480)], color=STEP, width=3)

    # 結果カードの見本
    s.raw('<rect x="780" y="610" width="890" height="282" rx="12" fill="#ffffff" stroke="#3b6fd8" '
          'stroke-width="2"/>')
    s.text(800, 642, "結果カード（見本）", size=19, weight="bold", color="#3b6fd8")
    s.raw('<rect x="800" y="660" width="190" height="212" rx="8" fill="#374151"/>')
    s.raw('<path d="M828,690 C840,668 950,668 962,690 L968,820 C940,850 850,850 822,820 Z" '
          'fill="none" stroke="#f59e0b" stroke-width="3"/>')
    s.raw('<ellipse cx="866" cy="760" rx="26" ry="62" fill="#9ca3af" fill-opacity="0.7"/>')
    s.raw('<ellipse cx="926" cy="760" rx="25" ry="60" fill="#9ca3af" fill-opacity="0.7"/>')
    s.text(1015, 690, "#1　類似度 0.968", size=24, weight="bold", color="#111827")
    pill(s, 1015, 706, "所見プロファイルで全件検索", "#2f6fdb", size=17)
    s.text(1015, 778, "データセットのラベル: 結節影", size=20, color="#374151")
    s.text(1015, 816, "所見の推定（研究用・診断ではない）", size=18, color="#6b7280")
    x = 1015
    for label in ("結節 0.71", "腫瘤 0.42", "浸潤影 0.30"):
        x += pill(s, x, 830, label, "#c2418a", size=18) + 10
    s.arrow([(730, 848), (778, 848)], color=STEP, width=3)

    s.note(30, 920, 1640, ["非同期実行を選ぶと search-worker が処理し、画面は結果を待って表示"], size=22)
    return s.render()


DIAGRAMS = {"flow-ingest-simple": ingest_flow, "flow-search-simple": search_flow}


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
