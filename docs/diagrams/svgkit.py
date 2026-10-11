"""構成図・フロー図を SVG で組み立てる小さな部品集（docs/diagrams/build_*.py から使う）。

座標は手で決める。自動配置に任せると、文書の図として読みにくい交差が出やすいため。
"""

from __future__ import annotations

from html import escape

FONT = "'Yu Gothic UI','Yu Gothic','Meiryo','Noto Sans JP',sans-serif"

#: 役割ごとの色（塗り, 枠, 見出しの文字色）
PALETTE = {
    "user": ("#f3f4f6", "#6b7280", "#111827"),
    "edge": ("#e0ecff", "#3b6fd8", "#1e3a8a"),
    "app": ("#ddf4ee", "#1f9e7a", "#065f46"),
    "worker": ("#fff1dc", "#d98a1c", "#7c4a03"),
    "ai": ("#fde7f3", "#c2418a", "#831843"),
    "data": ("#ece8fb", "#6d55c9", "#3b2a86"),
    "file": ("#f5f5f4", "#8a8580", "#3f3a36"),
    "ops": ("#f1f5f9", "#94a3b8", "#334155"),
    "host": ("#fffbe6", "#b59b2b", "#5f4b00"),
    "judge": ("#fff7d6", "#c99a06", "#6b4f00"),
}

NET_COLORS = {
    "edge": "#3b6fd8",
    "app": "#1f9e7a",
    "data": "#6d55c9",
    "cluster-vector": "#c2418a",
}


def t(text: str) -> str:
    return escape(text, quote=False)


class Svg:
    def __init__(self, width: int, height: int, title: str, subtitle: str = "", *,
                 title_size: int = 26, subtitle_size: int = 14) -> None:
        self.width, self.height = width, height
        self.parts: list[str] = []
        self.title, self.subtitle = title, subtitle
        self.title_size, self.subtitle_size = title_size, subtitle_size

    # ------------------------------------------------------------------ 基本形
    def text(self, x, y, s, *, size=14, weight="normal", color="#1f2937", anchor="start",
             italic=False) -> None:
        style = "font-style:italic;" if italic else ""
        self.parts.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}" style="{style}">{t(s)}</text>'
        )

    def box(self, x, y, w, h, title, lines=(), *, kind="app", title_size=16, line_size=12.5,
            dashed=False, align="center", badge: str | None = None) -> tuple:
        fill, stroke, head = PALETTE[kind]
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.8"{dash}/>'
        )
        n = len(lines)
        block = title_size + 6 + n * (line_size + 5)
        top = y + (h - block) / 2 + title_size
        if align == "center":
            cx, anchor = x + w / 2, "middle"
        else:
            cx, anchor = x + 12, "start"
        self.text(cx, top, title, size=title_size, weight="bold", color=head, anchor=anchor)
        for i, line in enumerate(lines):
            self.text(cx, top + 8 + (i + 1) * (line_size + 5) - 3, line, size=line_size,
                      color="#374151", anchor=anchor)
        if badge:
            bw = 9 * len(badge) + 14
            self.parts.append(
                f'<rect x="{x + w - bw - 6}" y="{y - 11}" width="{bw}" height="22" rx="11" '
                f'fill="{stroke}"/>'
            )
            self.text(x + w - bw / 2 - 6, y + 5, badge, size=12, weight="bold", color="#fff",
                      anchor="middle")
        return (x, y, w, h)

    def cylinder(self, x, y, w, h, title, lines=(), *, kind="data", title_size=15,
                 line_size=12) -> tuple:
        fill, stroke, head = PALETTE[kind]
        ry = 10
        self.parts.append(
            f'<path d="M{x},{y + ry} a{w / 2},{ry} 0 0,0 {w},0 v{h - 2 * ry} '
            f'a{w / 2},{ry} 0 0,1 {-w},0 z" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>'
            f'<ellipse cx="{x + w / 2}" cy="{y + ry}" rx="{w / 2}" ry="{ry}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.8"/>'
        )
        n = len(lines)
        block = title_size + 6 + n * (line_size + 5)
        top = y + ry + (h - ry - block) / 2 + title_size
        self.text(x + w / 2, top, title, size=title_size, weight="bold", color=head,
                  anchor="middle")
        for i, line in enumerate(lines):
            self.text(x + w / 2, top + 8 + (i + 1) * (line_size + 5) - 3, line, size=line_size,
                      color="#374151", anchor="middle")
        return (x, y, w, h)

    def zone(self, x, y, w, h, label, *, color="#94a3b8", fill="none", dashed=True,
             label_size=14, sub: str = "") -> None:
        dash = ' stroke-dasharray="9 6"' if dashed else ""
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" '
            f'stroke="{color}" stroke-width="1.6"{dash}/>'
        )
        self.text(x + 14, y + 22, label, size=label_size, weight="bold", color=color)
        if sub:
            self.text(x + 14, y + 40, sub, size=12, color="#6b7280")

    def arrow(self, points, label="", *, color="#4b5563", dashed=False, both=False,
              label_pos=0.5, label_dx=0, label_dy=-6, size=12.5, width=1.8,
              label_anchor="middle", label_bg=True) -> None:
        d = "M" + " L".join(f"{px},{py}" for px, py in points)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        start = ' marker-start="url(#arrow-start)"' if both else ""
        self.parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash}'
            f' marker-end="url(#arrow)"{start} style="color:{color}"/>'
        )
        if label:
            # 折れ線の全長に対する位置へ置く
            segs = list(zip(points, points[1:]))
            lengths = [((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5 for a, b in segs]
            target = sum(lengths) * label_pos
            for (a, b), seg in zip(segs, lengths):
                if target <= seg or (a, b) == segs[-1]:
                    r = target / seg if seg else 0
                    lx = a[0] + (b[0] - a[0]) * r + label_dx
                    ly = a[1] + (b[1] - a[1]) * r + label_dy
                    break
                target -= seg
            for i, line in enumerate(label.split("\n")):
                yy = ly + i * (size + 4)
                if label_bg:
                    wbg = max(len(line), 1) * size * 0.95 + 8
                    xbg = lx - wbg / 2 if label_anchor == "middle" else (lx - 4 if label_anchor == "start" else lx - wbg + 4)
                    self.parts.append(
                        f'<rect x="{xbg}" y="{yy - size}" width="{wbg}" height="{size + 5}" '
                        f'rx="4" fill="#ffffff" fill-opacity="0.88"/>'
                    )
                self.text(lx, yy, line, size=size, color=color, anchor=label_anchor)

    def note(self, x, y, w, lines, *, size=12.5, color="#6b4f00", fill="#fff8db",
             stroke="#e2c044") -> int:
        h = 14 + len(lines) * (size + 6)
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="1.2"/>'
        )
        for i, line in enumerate(lines):
            self.text(x + 10, y + 8 + (i + 1) * (size + 6) - 4, line, size=size, color=color)
        return h

    def chip(self, x, y, label, color) -> int:
        w = 8.2 * len(label) + 14
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="18" rx="9" fill="{color}" '
            f'fill-opacity="0.14" stroke="{color}" stroke-width="1"/>'
        )
        self.text(x + w / 2, y + 13, label, size=11, weight="bold", color=color, anchor="middle")
        return int(w)

    def legend(self, x, y, items) -> None:
        for i, (label, kind) in enumerate(items):
            fill, stroke, _ = PALETTE[kind]
            yy = y + i * 24
            self.parts.append(
                f'<rect x="{x}" y="{yy}" width="22" height="16" rx="4" fill="{fill}" '
                f'stroke="{stroke}" stroke-width="1.5"/>'
            )
            self.text(x + 30, yy + 13, label, size=12.5)

    def raw(self, s: str) -> None:
        self.parts.append(s)

    # ------------------------------------------------------------------ 出力
    def render(self) -> str:
        ty = 18 + self.title_size
        head = (
            f'<text x="32" y="{ty}" font-size="{self.title_size}" font-weight="bold" '
            f'fill="#111827">{t(self.title)}</text>'
        )
        if self.subtitle:
            head += (
                f'<text x="32" y="{ty + self.subtitle_size + 12}" font-size="{self.subtitle_size}" '
                f'fill="#6b7280">{t(self.subtitle)}</text>'
            )
        defs = (
            '<defs>'
            '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
            'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            'fill="context-stroke"/></marker>'
            '<marker id="arrow-start" viewBox="0 0 10 10" refX="1" refY="5" markerWidth="8" '
            'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
            'fill="context-stroke"/></marker>'
            '</defs>'
        )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" font-family="{FONT}">'
            f'<rect width="100%" height="100%" fill="#ffffff"/>{defs}{head}'
            + "".join(self.parts)
            + "</svg>"
        )


# ---------------------------------------------------------------------------- シーケンス図


class Sequence:
    """参加者を横に並べ、メッセージを上から順に描くシーケンス図。"""

    def __init__(self, title, subtitle, participants, *, width=1800, col_w=None, top=110,
                 head_h=64) -> None:
        self.title, self.subtitle = title, subtitle
        self.participants = participants  # [(key, label, sub, kind)]
        self.width = width
        n = len(participants)
        self.margin = 40
        self.col_w = col_w or (width - 2 * self.margin) / n
        self.x = {p[0]: self.margin + self.col_w * (i + 0.5) for i, p in enumerate(participants)}
        self.top, self.head_h = top, head_h
        self.y = top + head_h + 30
        self.items: list = []

    def msg(self, a, b, label, *, dashed=False, color="#374151", gap=None):
        lines = label.split("\n")
        h = gap or (16 + 17 * len(lines))
        self.items.append(("msg", a, b, lines, dashed, color, self.y + h - 10))
        self.y += h

    def self_msg(self, a, label, *, color="#374151"):
        lines = label.split("\n")
        h = 30 + 17 * len(lines)
        self.items.append(("self", a, lines, color, self.y + 8, h - 14))
        self.y += h

    def note(self, a, b, lines, *, kind="note"):
        h = 14 + len(lines) * 18.5
        self.items.append(("note", a, b, lines, self.y + 4, h, kind))
        self.y += h + 14

    def section(self, label):
        self.items.append(("section", label, self.y + 10))
        self.y += 40

    def alt(self, label, start_y, end_y):
        self.items.append(("alt", label, start_y, end_y))

    def render(self) -> str:
        height = int(self.y + 40)
        svg = Svg(self.width, height, self.title, self.subtitle)
        w = self.col_w - 26
        for key, label, sub, kind in self.participants:
            cx = self.x[key]
            fill, stroke, _ = PALETTE[kind]
            svg.raw(
                f'<line x1="{cx}" y1="{self.top + self.head_h}" x2="{cx}" y2="{height - 20}" '
                f'stroke="{stroke}" stroke-width="1.4" stroke-dasharray="5 5"/>'
            )
            svg.box(cx - w / 2, self.top, w, self.head_h, label, [sub] if sub else [],
                    kind=kind, title_size=15, line_size=11.5)
        for item in self.items:
            if item[0] == "section":
                _, label, yy = item
                svg.raw(
                    f'<line x1="{self.margin - 10}" y1="{yy}" x2="{self.width - self.margin + 10}" '
                    f'y2="{yy}" stroke="#cbd5e1" stroke-width="1"/>'
                )
                wbg = len(label) * 15 + 24
                svg.raw(
                    f'<rect x="{self.margin}" y="{yy - 13}" width="{wbg}" height="26" rx="13" '
                    f'fill="#334155"/>'
                )
                svg.text(self.margin + wbg / 2, yy + 5, label, size=14, weight="bold",
                         color="#fff", anchor="middle")
            elif item[0] == "msg":
                _, a, b, lines, dashed, color, yy = item
                x1, x2 = self.x[a], self.x[b]
                sign = 1 if x2 > x1 else -1
                svg.arrow([(x1 + sign * 4, yy), (x2 - sign * 6, yy)], "", color=color,
                          dashed=dashed)
                lx = (x1 + x2) / 2
                for i, line in enumerate(lines):
                    ly = yy - 7 - (len(lines) - 1 - i) * 17
                    wbg = len(line) * 12.4 + 8
                    svg.raw(
                        f'<rect x="{lx - wbg / 2}" y="{ly - 13}" width="{wbg}" height="17" rx="3" '
                        f'fill="#fff" fill-opacity="0.9"/>'
                    )
                    svg.text(lx, ly, line, size=12.5, color=color, anchor="middle")
            elif item[0] == "self":
                _, a, lines, color, yy, h = item
                x = self.x[a]
                svg.arrow([(x + 4, yy), (x + 34, yy), (x + 34, yy + h), (x + 6, yy + h)], "",
                          color=color)
                for i, line in enumerate(lines):
                    svg.text(x + 44, yy + 14 + i * 17, line, size=12.5, color=color)
            elif item[0] == "note":
                _, a, b, lines, yy, h, kind = item
                xa, xb = sorted((self.x[a], self.x[b]))
                x0, x1 = xa - self.col_w * 0.42, xb + self.col_w * 0.42
                fill, stroke = ("#fff8db", "#e2c044") if kind == "note" else ("#fdecec", "#e06666")
                color = "#6b4f00" if kind == "note" else "#8a1c1c"
                svg.note(x0, yy, x1 - x0, lines, fill=fill, stroke=stroke, color=color)
            elif item[0] == "alt":
                pass
        return svg.render()
