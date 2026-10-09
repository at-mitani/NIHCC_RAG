"""胸郭（肺野全体）の**形状**を表す低次元ベクトル（見た目ではなく輪郭と配置）。

なぜ見た目と別に要るのか
------------------------
気胸は肺が虚脱して肺野そのものが小さくなり、心拡大は心陰影が広がって左右の肺が
外側へ押される。どちらも「画素がどう見えるか」ではなく**形と位置関係**に出る所見
である。

実測でもこの区別が必要なことが確認できている。

- 正解 box の 49.3% しか 1 部位に収まらず、心拡大と胸水は肺野の外にある。
  見た目の部位埋め込みでは原理的に捉えられない
- 画像レベル AUC が 0.810 の見た目の表現でも、部位レベル正答率は 46.6% にとどまる。
  見た目の表現は部位の所在を持っていない

形状は ROI の副産物としてすでに得られており、**追加の推論を要しない**。

なぜ部位単位ではないのか
------------------------
部位は胸郭を幾何規則（正規化高さ u と側方位置 w の固定境界）で切り分けたもの
なので、**部位の形は胸郭の形から一意に決まる従属変数**である。独立した情報を
持たないため、検索結果に出しても胸郭の形の順位を粗く写すだけになる。そのため
部位ごとの形状ベクトルは持たない。

設計
----
すべての特徴を作りの段階で概ね [0, 1] か [-1, 1] に収める。学習した統計を持たずに
済み、成果物の版管理が要らない。最後に L2 正規化して cosine で比較する。

**尺度不変にはしない。** Hu モーメントのような完全不変量は心拡大の「大きさ」を
捨ててしまう。面積比と位置は残す。
"""

from __future__ import annotations

import numpy as np

def _moments(mask: np.ndarray) -> dict[str, float]:
    """二値 mask の幾何モーメント。cv2 に依存しないよう numpy で計算する。

    `np.nonzero` は座標の配列を確保するため、1024 四方の mask を 6 回まわすと
    1 枚 180ms かかり取り込みの律速になる。周辺和（行和・列和）と行列ベクトル積で
    求めると、確保が要らず桁違いに速い。交差モーメント mu11 だけは周辺和から
    復元できないため、行ごとの x の和を経由して求める。
    """
    data = np.asarray(mask)
    if data.dtype != np.bool_:
        data = data > 0
    rows = data.any(axis=1)
    if not rows.any():
        return {"area": 0.0}
    cols = data.any(axis=0)
    y0, y1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))

    # 外接矩形へ切ってから計算する。部位は画面の数 % なので効果が大きい。
    crop = data[y0:y1, x0:x1].astype(np.float32)
    xs = np.arange(crop.shape[1], dtype=np.float32)
    ys = np.arange(crop.shape[0], dtype=np.float32)
    per_row = crop.sum(axis=1)
    per_col = crop.sum(axis=0)
    area = float(per_row.sum())
    if area <= 0:
        return {"area": 0.0}
    cx = float(per_col @ xs) / area
    cy = float(per_row @ ys) / area
    mu20 = float(per_col @ (xs - cx) ** 2) / area
    mu02 = float(per_row @ (ys - cy) ** 2) / area
    # sum(x*y) = sum_y y * (行 y の x の和)
    sum_xy = float(ys @ (crop @ xs))
    mu11 = sum_xy / area - cx * cy
    return {
        "area": area, "cx": cx + x0, "cy": cy + y0,
        "mu20": mu20, "mu02": mu02, "mu11": mu11,
        "x0": float(x0), "x1": float(x1),
        "y0": float(y0), "y1": float(y1),
    }


def _squash(value: float, scale: float = 1.0) -> float:
    """外れ値で 1 次元が支配しないよう [-1, 1] へ収める。"""
    if not np.isfinite(value):
        return 0.0
    return float(np.tanh(value / scale))


def _hu(mu20: float, mu02: float, mu11: float, area: float) -> tuple[float, float, float]:
    """Hu モーメントのうち 2 次までで決まる 3 つ。対数で圧縮する。

    3 次以上は mask の粗さに敏感で、ROI の境界のゆらぎを拾いやすいため使わない。
    """
    if area <= 0:
        return 0.0, 0.0, 0.0
    n = area  # すでに面積で割った中心モーメントを使っている
    del n
    i1 = mu20 + mu02
    i2 = (mu20 - mu02) ** 2 + 4.0 * mu11**2
    i3 = (mu20 - mu02) ** 2 * (mu20 + mu02)
    def log_signed(v: float) -> float:
        return float(np.sign(v) * np.log1p(abs(v)))
    return _squash(log_signed(i1), 8.0), _squash(log_signed(i2), 8.0), _squash(log_signed(i3), 12.0)


def _mediastinum_width(lung: np.ndarray, stats: dict[str, float] | None = None) -> float:
    """左右肺の内側縁の間隔（画像幅で正規化）。

    心拡大では心陰影が左右へ広がり、肺野の間が空く。行ごとに肺野の穴（左右の
    間の非肺野区間）の幅を測り、心臓の高さに当たる下半分で中央値を取る。

    行ごとの Python ループにすると 1 枚あたり数百回まわり、取り込みの律速になる。
    numpy で一括に処理する。
    """
    height, width = lung.shape
    if height == 0 or width == 0:
        return 0.0
    # 肺野の外接矩形だけを見る。画像の余白の広さに結果を左右させないため。
    s = stats or _moments(lung)
    y0, y1 = int(s.get("y0", 0)), int(s.get("y1", height))
    x0, x1 = int(s.get("x0", 0)), int(s.get("x1", width))
    box = lung[y0:y1, x0:x1]
    box_h, box_w = box.shape
    if box_h == 0 or box_w == 0:
        return 0.0
    lower = box[box_h // 2 :]
    width = box_w
    if lower.size == 0:
        return 0.0
    count = lower.sum(axis=1)
    usable = count >= 2
    if not usable.any():
        return 0.0
    rows = lower[usable]
    first = np.argmax(rows, axis=1)
    last = rows.shape[1] - 1 - np.argmax(rows[:, ::-1], axis=1)
    inner = (last - first + 1) - count[usable]
    return float(np.median(inner) / width)


def lung_stats(lung_mask: np.ndarray) -> dict[str, float]:
    """肺野全体から決まる値。1 枚につき 1 回だけ計算する。"""
    lung = np.asarray(lung_mask) > 0
    stats = _moments(lung)
    stats["mediastinum"] = _mediastinum_width(lung, stats)
    return stats


# =============================================================================
# 肺野全体の形状
#
# **これは所見の判別器ではない。「胸郭の形が似た症例を探す」道具である。**
# 判別能は高くないので、所見の有無を当てる用途には使わない。画面でもその旨を
# 明示する。
#
# それでも別軸として持つのは、見た目のベクトルや部位ごとの量には**存在しない量**が
# あるためである。左右肺の重心距離（心臓が肺を左右へ押す量、片肺の虚脱で重心がずれる量）は
# 単一特徴としてこれまで測った中で最も強く（気胸 0.648 / 胸水 0.667）、部位に分けて
# しまうと作れない。
# =============================================================================

LUNG_DIMENSION = 16

LUNG_FEATURE_NAMES: tuple[str, ...] = (
    "compactness",         # 肺野面積 / 外接矩形（虚脱すると下がる）
    "aspect",              # 外接矩形の縦横比
    "mediastinum_median",  # 左右肺の内側縁の間隔（中央値）
    "mediastinum_max",     # 同・最大。心胸郭比の代理
    "mediastinum_lower",   # 下 1/3 での間隔。心臓の高さ
    "lr_area_ratio",       # 左右肺の面積比（小さい方 / 大きい方）
    "lr_height_ratio",     # 左右肺の高さ比
    "lr_centroid_dx",      # 左右肺の重心の水平距離 / 肺野幅
    "lr_centroid_dy",      # 同・垂直距離（片側の虚脱で片方が上がる）
    "band_q1",             # 上から 1/4 の帯での肺野の詰まり具合
    "band_q2",
    "band_q3",
    "band_q4",
    "elongation",          # 肺野全体の伸び
    "hu1",                 # 肺野全体の形の不変量
    "hu2",
)
assert len(LUNG_FEATURE_NAMES) == LUNG_DIMENSION


def _gap_profile(lung: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    """行ごとの「左右肺の内側縁の間隔 / 肺野幅」。心陰影の広がりを表す。"""
    y0, y1 = int(stats.get("y0", 0)), int(stats.get("y1", lung.shape[0]))
    x0, x1 = int(stats.get("x0", 0)), int(stats.get("x1", lung.shape[1]))
    box = lung[y0:y1, x0:x1]
    if box.size == 0:
        return np.zeros(1, dtype=np.float32)
    count = box.sum(axis=1)
    usable = count >= 2
    if not usable.any():
        return np.zeros(1, dtype=np.float32)
    rows = box[usable]
    first = np.argmax(rows, axis=1)
    last = rows.shape[1] - 1 - np.argmax(rows[:, ::-1], axis=1)
    return ((last - first + 1) - count[usable]).astype(np.float32) / max(box.shape[1], 1)


def _band_profile(lung: np.ndarray, stats: dict[str, float], bands: int = 4) -> list[float]:
    """高さを bands 等分し、各帯での肺野の詰まり具合（面積 / 帯の面積）を返す。"""
    y0, y1 = int(stats.get("y0", 0)), int(stats.get("y1", lung.shape[0]))
    x0, x1 = int(stats.get("x0", 0)), int(stats.get("x1", lung.shape[1]))
    box = lung[y0:y1, x0:x1]
    if box.size == 0:
        return [0.0] * bands
    height, width = box.shape
    out: list[float] = []
    for index in range(bands):
        band = box[height * index // bands : height * (index + 1) // bands]
        out.append(float(band.sum()) / max(band.shape[0] * width, 1))
    return out


def describe_lung(
    region_masks: dict[str, np.ndarray], lung_mask: np.ndarray | None = None
) -> np.ndarray:
    """肺野全体の形状ベクトル（L2 正規化済み、LUNG_DIMENSION 次元）。

    部位ベクトルと同じく**画像の縦横には依存させない**（肺野の外接矩形を基準に
    正規化する）。取り込み時と backfill 時で同じ値になる。

    左右は部位コードの頭文字（R / L）で分ける。ROI で片側が丸ごと落ちた画像では
    左右比が定義できないため中立（1.0）にする。0 にすると「極端な非対称」に見える。
    """
    if not region_masks:
        return np.zeros(LUNG_DIMENSION, dtype=np.float32)
    first = np.asarray(next(iter(region_masks.values())))
    if lung_mask is None:
        lung = np.zeros(first.shape[:2], dtype=bool)
        for mask in region_masks.values():
            lung |= np.asarray(mask) > 0
    else:
        lung = np.asarray(lung_mask) > 0

    stats = _moments(lung)
    if stats.get("area", 0.0) <= 0:
        return np.zeros(LUNG_DIMENSION, dtype=np.float32)
    lw = max(stats["x1"] - stats["x0"], 1.0)
    lh = max(stats["y1"] - stats["y0"], 1.0)

    gaps = _gap_profile(lung, stats)
    lower = gaps[len(gaps) * 2 // 3 :] if len(gaps) >= 3 else gaps

    right = np.zeros(first.shape[:2], dtype=bool)
    left = np.zeros(first.shape[:2], dtype=bool)
    for code, mask in region_masks.items():
        target = right if code.startswith("R") else left
        target |= np.asarray(mask) > 0
    rs, ls = _moments(right), _moments(left)
    r_area, l_area = rs.get("area", 0.0), ls.get("area", 0.0)
    both = r_area > 0 and l_area > 0
    lr_area = min(r_area, l_area) / max(r_area, l_area) if both else 1.0
    r_h = rs.get("y1", 0.0) - rs.get("y0", 0.0)
    l_h = ls.get("y1", 0.0) - ls.get("y0", 0.0)
    lr_height = min(r_h, l_h) / max(r_h, l_h, 1.0) if both else 1.0
    dx = abs(rs.get("cx", 0.0) - ls.get("cx", 0.0)) / lw if both else 0.0
    dy = abs(rs.get("cy", 0.0) - ls.get("cy", 0.0)) / lh if both else 0.0

    mu20, mu02, mu11 = stats["mu20"], stats["mu02"], stats["mu11"]
    common = mu20 + mu02
    diff = np.sqrt(max((mu20 - mu02) ** 2 + 4.0 * mu11**2, 0.0))
    major, minor = 0.5 * (common + diff), max(0.5 * (common - diff), 1e-9)
    hu1, hu2, _ = _hu(mu20, mu02, mu11, stats["area"])

    values = [
        stats["area"] / (lw * lh),
        _squash(lw / lh, 2.0),
        float(np.median(gaps)),
        float(np.max(gaps)),
        float(np.median(lower)),
        lr_area,
        lr_height,
        _squash(dx, 0.5),
        _squash(dy, 0.2),
        *_band_profile(lung, stats),
        _squash(np.sqrt(major / minor), 3.0),
        hu1,
        hu2,
    ]
    vector = np.asarray(values, dtype=np.float64)
    vector[~np.isfinite(vector)] = 0.0
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros(LUNG_DIMENSION, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def lung_from_masks(right: np.ndarray, left: np.ndarray) -> np.ndarray:
    """左右の肺 mask（画像基準、同じ寸法）から肺野全体の形状ベクトルを作る。

    胸部野 1 単位の取り込み（roi-service `/roi/chest`）が使う。U-Net が出す左右の
    肺 mask をそのまま使う。
    """
    return describe_lung({"R": np.asarray(right) > 0, "L": np.asarray(left) > 0})
