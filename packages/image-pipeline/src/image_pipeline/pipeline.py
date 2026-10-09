"""古典的画像処理による肺野・脊柱・5部位 ROI 抽出（設計書 6 章）。

処理段階:
  Step 0  入力・正規化（percentile clip、極性統一）
  Step 1  body/thorax 推定と黒い肺野候補の抽出（Otsu + local threshold）
  Step 2  脊柱 score map と中心線推定、患者左右への分割
  Step 3  肺 mask 内の正規化高さ u による 5 部位分割と crop 生成
  QC      設計書 7.1 の自動品質検査

肺野の取り方は profile の `lung_segmentation.backend` で選ぶ。

  classical : Otsu + local threshold の規則ベース（既定）
  unet      : CheXmask を教師に学習した U-Net（`segmenter.py`、onnxruntime）

規則ベースは corpus の約 4 分の 1 で破綻する。
どちらを使っても、脊柱推定から先の部位分割は共通である。

出力の全件に algorithm version と parameter hash を付与し、再現可能にする。
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .config import Profile, load_profile
from . import segmenter as segmentation
from .quality import QualityReport, evaluate_quality

REGION_CODES = ("RUL", "RML", "RLL", "LUL", "LLL")


class ExtractionError(ValueError):
    """抽出不能（対象外画像を含む）。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


#: 表示用 crop で mask 外に残す明るさの割合。
#: 0 にすると向きが分からなくなるため、輪郭が薄く見える程度に残す。
#: nihcc_rag.imaging.REGION_DIM_FACTOR と同値にすること（検索元と候補で見た目を揃える）。
DISPLAY_DIM_FACTOR = 0.12


@dataclass
class RegionOutput:
    region_code: str
    bbox: list[int]
    crop_bbox: list[int]
    area_px: int
    centroid: list[float]
    confidence: float
    quality_status: str
    mask: np.ndarray
    #: 埋め込みに使う crop（profile の crop.embedding_crop に従う）。
    #: crop.output_side を設定した場合は encoder の入力解像度へ縮小済みのため、
    #: mask / display_crop とは shape が一致しないことがある（mask は表示合成に
    #: 使うので作業解像度のまま保持する）。
    crop: np.ndarray
    #: 画面表示用 crop。mask 内はそのまま、mask 外を減光した部位単位の画像。
    display_crop: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionOutput:
    profile: Profile
    normalized: np.ndarray
    scale: float
    lung_mask: np.ndarray
    right_mask: np.ndarray
    left_mask: np.ndarray
    spine_x: np.ndarray
    regions: dict[str, RegionOutput]
    quality: QualityReport
    metrics: dict[str, Any]

    @property
    def overall_status(self) -> str:
        return self.quality.overall_status


# --- Step 0: 入力・正規化 ----------------------------------------------------


def decode_image(data: bytes) -> np.ndarray:
    """PNG/JPEG バイト列を単チャネル float32（0..1 未正規化の生値）へ変換する。"""
    with Image.open(io.BytesIO(data)) as image:
        if image.mode in ("I;16", "I", "F"):
            array = np.asarray(image).astype(np.float32)
        else:
            array = np.asarray(image.convert("L")).astype(np.float32)
    if array.ndim != 2:
        raise ExtractionError("REJECTED_VIEW", "grayscale frontal image is required")
    return array


def normalize(array: np.ndarray, profile: Profile) -> tuple[np.ndarray, float, bool]:
    """percentile clip → [0,255] uint8、必要なら極性反転、作業解像度へ縮小。"""
    norm_cfg = profile.section("normalization")
    low_p = float(norm_cfg["low_percentile"])
    high_p = float(norm_cfg["high_percentile"])
    low, high = np.percentile(array, [low_p, high_p])
    if high <= low:
        raise ExtractionError("IMAGE_FLAT", "image has no dynamic range")
    clipped = np.clip(array, low, high)
    gray = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)

    inverted = _needs_polarity_inversion(gray)
    if inverted:
        gray = 255 - gray

    work_side = int(norm_cfg.get("work_side", 1024))
    height, width = gray.shape
    scale = 1.0
    if max(height, width) > work_side:
        scale = work_side / float(max(height, width))
        gray = cv2.resize(
            gray, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
        )
    return gray, scale, inverted


def _needs_polarity_inversion(gray: np.ndarray) -> bool:
    """正常な胸部X線では側方（肺野）が中央（脊柱・縦隔）より暗い。"""
    height, width = gray.shape
    band = gray[int(height * 0.30) : int(height * 0.60), :]
    if band.size == 0:
        return False
    central = band[:, int(width * 0.42) : int(width * 0.58)]
    lateral = np.concatenate(
        [band[:, int(width * 0.10) : int(width * 0.28)], band[:, int(width * 0.72) : int(width * 0.90)]],
        axis=1,
    )
    return float(lateral.mean()) > float(central.mean())


# --- Step 1: body / thorax / 肺野候補 ---------------------------------------


def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def body_mask(gray: np.ndarray) -> np.ndarray:
    """人体（軟部組織）領域を推定する（設計書 6.3.1）。

    背景・体外空気は人体より暗いため、Otsu 閾値以上の明部を人体候補とし、
    最大連結成分を取ってから内部の穴（肺野・気管）を埋める。これにより
    画像外周に連結する暗部（頸部周囲・肩の外側）を肺野候補から除外する。
    """
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(1.0, min(height, width) * 0.006))
    threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (blurred >= threshold).astype(np.uint8) * 255

    radius = max(3, round(min(height, width) * 0.01))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    if count <= 1:
        return np.full(gray.shape, 255, np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = np.where(labels == largest, 255, 0).astype(np.uint8)

    # 肺野は頸部・体外の暗部と連結し得るため、穴埋めでは体内領域を確定できない。
    # 軟部組織成分の凸包を人体シルエットとし、その内側を探索対象とする。
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full(gray.shape, 255, np.uint8)
    hull = cv2.convexHull(np.vstack(contours))
    body = np.zeros_like(gray)
    cv2.fillConvexPoly(body, hull, 255)

    if float((body > 0).sum()) / float(height * width) < 0.25:
        # 人体候補が極端に小さい場合は画像全体を対象にし、後段の QC で検出する
        return np.full(gray.shape, 255, np.uint8)
    return body


def thorax_mask(gray: np.ndarray, body: np.ndarray, profile: Profile) -> tuple[np.ndarray, list[int]]:
    """body bbox の上下左右を除外した胸郭探索領域（設計書 6.3.1）。"""
    ys, xs = np.nonzero(body)
    if xs.size == 0:
        raise ExtractionError("BODY_NOT_FOUND", "body region could not be estimated")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    bw, bh = x1 - x0, y1 - y0

    lung_cfg = profile.section("lung_candidate")
    ax0, ax1 = lung_cfg["allowed_x_ratio"]
    ay0, ay1 = lung_cfg["allowed_y_ratio"]
    tx0 = x0 + int(bw * float(ax0))
    tx1 = x0 + int(bw * float(ax1))
    ty0 = y0 + int(bh * float(ay0))
    ty1 = y0 + int(bh * float(ay1))

    thorax = np.zeros_like(gray, np.uint8)
    thorax[ty0:ty1, tx0:tx1] = 255
    thorax = cv2.bitwise_and(thorax, body)
    return thorax, [x0, y0, x1, y1]


def lung_candidate(gray: np.ndarray, thorax: np.ndarray, profile: Profile) -> np.ndarray:
    """thorax 内の低輝度画素から肺野候補二値画像を作る（設計書 6.3.2）。"""
    cfg = profile.section("lung_candidate")
    pixels = gray[thorax > 0]
    if pixels.size < 1000:
        raise ExtractionError("THORAX_TOO_SMALL", "thorax region is too small")

    t_global = float(
        cv2.threshold(pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    )
    delta = float(cfg["threshold_delta"]) * 255.0
    mode = cfg["intensity_mode"]

    # 肋骨・血管の細かいテクスチャで local threshold が肺野を分断しないよう、
    # 二値化の入力にだけ平滑化を適用する（正規化画像そのものは置換しない）。
    sigma = float(cfg.get("smooth_sigma_ratio", 0.0)) * min(gray.shape)
    source = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma) if sigma > 0.5 else gray

    global_mask = (source < (t_global + delta)).astype(np.uint8) * 255
    if mode == "fixed":
        low, high = cfg["fixed_preview_range"]
        candidate = cv2.inRange(source, np.uint8(low), np.uint8(high))
    elif mode == "otsu":
        candidate = global_mask
    else:  # otsu_and_local
        block = _odd(round(int(cfg["local_block_size_at_1024"]) * max(gray.shape) / 1024))
        offset = float(cfg.get("local_offset", 0.02)) * 255.0
        local = cv2.adaptiveThreshold(
            source,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            offset,
        )
        candidate = cv2.bitwise_and(global_mask, local)

    candidate = cv2.bitwise_and(candidate, thorax)

    morph = profile.section("morphology")
    radius = max(1, round(min(gray.shape) * float(morph["kernel_short_side_ratio"])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, kernel, iterations=int(morph["close_iterations"])
    )
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_OPEN, kernel, iterations=int(morph["open_iterations"])
    )
    return candidate


def select_lung_components(
    candidate: np.ndarray, profile: Profile
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """面積比・縦横比・境界接触で連結成分を選別する（設計書 6.3.3）。"""
    cfg = profile.section("component_filter")
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        candidate, connectivity=int(cfg["connectivity"])
    )
    height, width = candidate.shape
    image_area = float(height * width)

    selected: list[dict[str, Any]] = []
    mask = np.zeros_like(candidate)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        ratio = area / image_area
        if not float(cfg["min_area_ratio"]) <= ratio <= float(cfg["max_area_ratio"]):
            continue
        aspect = w / max(h, 1)
        if not float(cfg["min_aspect_ratio"]) <= aspect <= float(cfg["max_aspect_ratio"]):
            continue
        component = (labels == label)
        border = _border_touch_ratio(component)
        if border > float(cfg.get("max_border_touch_ratio", 0.4)):
            continue
        selected.append(
            {
                "label": int(label),
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "area_px": int(area),
                "centroid": [float(centroids[label][0]), float(centroids[label][1])],
                "border_touch_ratio": border,
            }
        )
        mask[component] = 255

    if not selected:
        raise ExtractionError("LUNG_NOT_FOUND", "no lung field candidate satisfied the filters")
    # 肋骨・血管による小穴を埋める（縦隔を跨ぐ過剰連結は closing 半径で抑制）
    mask = _fill_holes(mask)
    return mask, selected


def _border_touch_ratio(component: np.ndarray) -> float:
    height, width = component.shape
    perimeter = component.sum()
    if perimeter == 0:
        return 1.0
    touching = (
        component[0, :].sum()
        + component[height - 1, :].sum()
        + component[:, 0].sum()
        + component[:, width - 1].sum()
    )
    edge_px = 2 * (height + width)
    return float(touching) / float(edge_px)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """連結成分ごとに外郭輪郭を塗り潰し、内部の穴を埋める（設計書 6.3.3）。

    背景 floodFill 方式は mask が境界で開いていると穴埋めが漏れるため、
    成分単位の外郭輪郭塗り潰しを用いる。これにより肋骨・血管・病変による
    白い領域を肺外にしない。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


# --- Step 2: 脊柱検出と左右判定 ---------------------------------------------


def spine_curve(
    gray: np.ndarray, lung: np.ndarray, body_bbox: list[int], profile: Profile
) -> tuple[np.ndarray, float, str]:
    """spine score map から中心線 x_spine(y) を推定する（設計書 6.4.1）。

    戻り値: (x_spine 配列, 連続性比率, 推定方法)
    """
    cfg = profile.section("spine")
    height, width = gray.shape
    bx0, _by0, bx1, _by1 = body_bbox
    sx0 = bx0 + int((bx1 - bx0) * float(cfg["search_x_ratio"][0]))
    sx1 = bx0 + int((bx1 - bx0) * float(cfg["search_x_ratio"][1]))
    sx1 = max(sx1, sx0 + 3)

    sigma_y = max(1.0, height * float(cfg.get("vertical_smooth_ratio", 0.03)))
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=sigma_y)

    band = smooth[:, sx0:sx1]
    brightness = _minmax(band)

    # 縦方向連続性: 長い縦カーネルとの一致度
    continuity = _minmax(
        cv2.blur(band, (1, max(3, _odd(int(height * 0.15)))))
    )

    lung_bool = lung[:, sx0:sx1] > 0
    lung_full = lung > 0
    left_has = np.cumsum(lung_full, axis=1)[:, sx0:sx1] > 0
    right_total = np.cumsum(lung_full[:, ::-1], axis=1)[:, ::-1]
    right_has = right_total[:, sx0:sx1] > 0
    gap_score = (left_has & right_has & ~lung_bool).astype(np.float32)

    score = 0.50 * brightness + 0.25 * continuity + 0.25 * gap_score

    candidate_x = np.argmax(score, axis=1) + sx0
    candidate_v = score.max(axis=1)
    valid = candidate_v > 0.35

    method = "spine_score"
    if valid.sum() < height * 0.5:
        # fallback: 左右肺 mask の中点曲線（設計書 6.4.1）
        midpoints = _lung_midpoints(lung_full)
        if midpoints is None:
            raise ExtractionError("SPINE_NOT_FOUND", "spine centre line could not be estimated")
        candidate_x, valid = midpoints
        method = "lung_midpoint"

    ys = np.nonzero(valid)[0]
    xs = candidate_x[ys].astype(np.float64)
    degree = int(cfg.get("polynomial_degree", 2))
    sigma = float(cfg.get("outlier_sigma", 2.0))
    for _ in range(3):
        if ys.size <= degree + 1:
            break
        coeffs = np.polyfit(ys, xs, degree)
        residual = np.abs(np.polyval(coeffs, ys) - xs)
        threshold = residual.mean() + sigma * residual.std()
        keep = residual <= max(threshold, 1.0)
        if keep.all():
            break
        ys, xs = ys[keep], xs[keep]
    if ys.size <= degree + 1:
        raise ExtractionError("SPINE_NOT_FOUND", "spine centre line has too few inliers")

    coeffs = np.polyfit(ys, xs, degree)
    spine = np.clip(np.polyval(coeffs, np.arange(height)), sx0, sx1 - 1)
    continuity_ratio = float(ys.size) / float(height)
    return spine, continuity_ratio, method


def _minmax(array: np.ndarray) -> np.ndarray:
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - low) / (high - low)).astype(np.float32)


def _lung_midpoints(lung_bool: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = lung_bool.shape
    xs = np.zeros(height, dtype=np.float64)
    valid = np.zeros(height, dtype=bool)
    half = width // 2
    for y in range(height):
        row = np.nonzero(lung_bool[y])[0]
        if row.size < 2:
            continue
        left = row[row < half]
        right = row[row >= half]
        if left.size == 0 or right.size == 0:
            continue
        xs[y] = (left.max() + right.min()) / 2.0
        valid[y] = True
    if valid.sum() < height * 0.2:
        return None
    return xs, valid


def refine_lung_outline(
    mask: np.ndarray, gray: np.ndarray, spine: np.ndarray, side: str, profile: Profile
) -> np.ndarray:
    """片肺の外周を、解剖学的にありえる滑らかな曲線へ整える（設計書 6.4.2 の補正）。

    二値化だけで取った肺野は、**肋骨が明るいために肋間の暗い帯だけが残り**、外周が
    鋸歯状になる。参考図（`NIHCC/sample/lung.jpg` の (2)）の肺野外周は連続した滑らかな
    曲線であり、肋骨に沿った凹凸を持たない。

    2 段階で整える。

      1. 閉処理で肋骨を橋渡しし、肋間で分断された肺野を 1 つの面にする
      2. 外周を弧長方向に低域通過させ、肋骨由来の高周波を落とす

    **左右に分けた後で行う。** 分ける前に強い閉処理をかけると、上縦隔の暗部を通じて
    左右の肺が繋がり、縦隔・気管まで肺野に入る（実測で確認）。

    平滑化した輪郭が脊柱を越えないよう、最後に元の側へ戻す。
    """
    cfg = profile.section("outline")
    if int(mask.sum()) == 0:
        return mask

    mask = _keep_dark(
        gray, mask, float(cfg["apex_percentile"]), float(cfg["apex_ratio"])
    )
    if int(mask.sum()) == 0:
        return mask

    radius = max(1, round(min(mask.shape) * float(cfg["close_ratio"])))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    closed = _fill_holes(closed)

    smoothed = _smooth_outline(closed, int(cfg["harmonics"]))

    # 脊柱を越えた分を戻す。越えたまま返すと左右の肺が重なる。
    xs = np.arange(mask.shape[1])[None, :]
    spine_col = spine[:, None]
    half = (xs < spine_col) if side == "RIGHT" else (xs > spine_col)
    smoothed = np.where(half, smoothed, 0).astype(np.uint8)
    return _largest_component(smoothed)


def _keep_dark(
    gray: np.ndarray, mask: np.ndarray, percentile: float, upper_ratio: float = 1.0
) -> np.ndarray:
    """肺野を「暗い範囲」に限定する（鎖骨上・頸部の軟部組織を外す）。

    肺は空気を含むため暗く、鎖骨上・頸部の軟部組織はそれより明るい。二値化は
    thorax 全体の分布で閾値を決めるため、この差を捉えきれず軟部組織まで肺野に入る。

    **肺野自身の輝度を基準にする。** 中央 40%（確実に肺である範囲）の輝度の
    `percentile` を上限とし、それより暗い画素のうち中央部と繋がっている範囲だけを残す。
    連結を要求するのは、肺と関係のない暗部（腋窩・体外）を拾わないためである。

    上限を上げると肺尖の明るい部分まで残るが軟部組織も入り、下げると軟部組織は
    落ちるが肺尖まで削れる。

    **`upper_ratio` で上部だけに適用する。** 肺野全体へかけると、下部・側方の本来の
    肺まで約 30% 削れる（実測）。軟部組織がはみ出すのは肺野上部なので、そこだけで
    足りる。
    """
    arr = mask > 0
    if not arr.any():
        return mask
    rows = np.nonzero(arr.any(axis=1))[0]
    top, bottom = int(rows.min()), int(rows.max())
    core = np.zeros_like(arr)
    lo = top + int((bottom - top) * 0.30)
    hi = top + int((bottom - top) * 0.70)
    core[lo:hi] = arr[lo:hi]
    if not core.any():
        return mask

    ceiling = float(np.percentile(gray[core], percentile))
    dark = (gray <= ceiling) & arr
    seed = core & dark
    if not seed.any():
        return mask

    count, labels = cv2.connectedComponents((dark * 255).astype(np.uint8), connectivity=8)
    keep = np.unique(labels[seed])
    keep = keep[keep > 0]
    limited = np.isin(labels, keep) & arr
    if upper_ratio >= 1.0:
        return limited.astype(np.uint8) * 255
    # 上部だけを置き換え、下部は元のまま残す
    out = arr.copy()
    cut = top + int((bottom - top) * upper_ratio)
    out[:cut] = limited[:cut]
    return out.astype(np.uint8) * 255


def _smooth_outline(mask: np.ndarray, harmonics: int) -> np.ndarray:
    """外周を弧長方向の低域通過で滑らかにする（楕円フーリエ記述子と同じ考え方）。

    輪郭を複素数列 z = x + iy として FFT し、低次の調和成分だけを残す。
    harmonics を大きくすると形は保たれるが凹凸が残り、小さくすると滑らかになるが
    肋骨横隔膜角のような鋭角まで丸まる。
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return mask
    points = max(contours, key=len)[:, 0, :].astype(np.float64)
    if len(points) < 8 * harmonics:
        # 輪郭点が少ないと低域通過が形を壊すため、そのまま返す
        return mask
    spectrum = np.fft.fft(points[:, 0] + 1j * points[:, 1])
    spectrum[harmonics + 1 : -harmonics] = 0
    curve = np.fft.ifft(spectrum)
    polygon = np.stack([curve.real, curve.imag], axis=1).round().astype(np.int32)
    out = np.zeros_like(mask)
    cv2.fillPoly(out, [polygon], 255)
    return out


def split_lungs(lung: np.ndarray, spine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """画像左 = 患者右、画像右 = 患者左（標準表示方向、設計書 6.4.2）。"""
    height, width = lung.shape
    xs = np.arange(width)[None, :]
    spine_col = spine[:, None]
    right = np.where((lung > 0) & (xs < spine_col), 255, 0).astype(np.uint8)
    left = np.where((lung > 0) & (xs > spine_col), 255, 0).astype(np.uint8)
    return _largest_component(right), _largest_component(left)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


# --- Step 3: 部位分割 --------------------------------------------------------


def normalized_height_map(mask: np.ndarray, profile: Profile) -> np.ndarray:
    """各 x 列の肺 mask 上端・下端から u(x,y) を計算する（設計書 6.5.1）。"""
    height, width = mask.shape
    binary = mask > 0
    tops = np.full(width, np.nan)
    bottoms = np.full(width, np.nan)
    columns = np.nonzero(binary.any(axis=0))[0]
    for x in columns:
        ys = np.nonzero(binary[:, x])[0]
        tops[x] = ys[0]
        bottoms[x] = ys[-1]

    if columns.size == 0:
        return np.full(mask.shape, np.nan, dtype=np.float32)

    # 欠損列は隣接列から補間し、平滑化する
    all_x = np.arange(width)
    tops = np.interp(all_x, columns, tops[columns])
    bottoms = np.interp(all_x, columns, bottoms[columns])
    window = max(3, _odd(int(width * float(profile.section("partition")["column_smooth_ratio"]))))
    kernel = np.ones(window) / window
    tops = np.convolve(np.pad(tops, (window, window), mode="edge"), kernel, mode="same")[
        window:-window
    ]
    bottoms = np.convolve(np.pad(bottoms, (window, window), mode="edge"), kernel, mode="same")[
        window:-window
    ]

    span = np.maximum(bottoms - tops, 1.0)
    ys = np.arange(height)[:, None]
    u = (ys - tops[None, :]) / span[None, :]
    u = np.where(binary, u, np.nan)
    return u.astype(np.float32)


def _dim_outside_mask(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mask 外を減光した表示用画像を作る。"""
    dimmed = (crop.astype(np.float32) * DISPLAY_DIM_FACTOR).astype(np.uint8)
    return np.where(mask > 0, crop, dimmed)


def lateral_position_map(mask: np.ndarray, laterality: str, profile: Profile) -> np.ndarray:
    """各行の肺幅で正規化した水平位置 w を返す（内側 0、外側 1）。

    標準表示方向では画像左が患者右肺であり、その内側（縦隔側）は右端になる。
    肺葉の内外配置（中葉＝内側、下葉＝外側）を表すためにこの座標を使う。
    """
    height, width = mask.shape
    binary = mask > 0
    rows = np.nonzero(binary.any(axis=1))[0]
    if rows.size == 0:
        return np.full(mask.shape, np.nan, dtype=np.float32)

    lefts = np.zeros(height)
    rights = np.zeros(height)
    for y in rows:
        xs = np.nonzero(binary[y])[0]
        lefts[y] = xs[0]
        rights[y] = xs[-1]

    all_y = np.arange(height)
    lefts = np.interp(all_y, rows, lefts[rows])
    rights = np.interp(all_y, rows, rights[rows])

    # 行ごとの端点は肋骨・血管で揺れるため平滑化する
    window = max(3, _odd(int(height * float(profile.section("partition")["row_smooth_ratio"]))))
    kernel = np.ones(window) / window
    lefts = np.convolve(np.pad(lefts, (window, window), mode="edge"), kernel, mode="same")[
        window:-window
    ]
    rights = np.convolve(np.pad(rights, (window, window), mode="edge"), kernel, mode="same")[
        window:-window
    ]

    span = np.maximum(rights - lefts, 1.0)
    xs_grid = np.arange(width)[None, :]
    if laterality == "RIGHT":
        # 画像左の肺。内側（縦隔側）は右端
        w = (rights[:, None] - xs_grid) / span[:, None]
    else:
        # 画像右の肺。内側は左端
        w = (xs_grid - lefts[:, None]) / span[:, None]
    return np.where(binary, np.clip(w, 0.0, 1.0), np.nan).astype(np.float32)


def absolute_height_map(mask: np.ndarray) -> np.ndarray:
    """肺 bbox 内で正規化した絶対高さ v（上端 0、下端 1）。

    裂は正面投影でほぼ直線として写るため、列ごとの輪郭正規化（u）ではなく
    画像座標に対する比率で境界を定義する。
    """
    binary = mask > 0
    ys = np.nonzero(binary.any(axis=1))[0]
    if ys.size == 0:
        return np.full(mask.shape, np.nan, dtype=np.float32)
    top, bottom = float(ys[0]), float(ys[-1])
    span = max(bottom - top, 1.0)
    rows = (np.arange(mask.shape[0], dtype=np.float32) - top) / span
    v = np.repeat(rows[:, None], mask.shape[1], axis=1)
    return np.where(binary, np.clip(v, 0.0, 1.0), np.nan).astype(np.float32)


def _fissure_boundary(w: np.ndarray, medial_u: float, lateral_u: float) -> np.ndarray:
    """裂の投影位置。内側で medial_u、外側で lateral_u になる線形境界。"""
    return medial_u + (lateral_u - medial_u) * w


def region_masks(
    *,
    right_mask: np.ndarray,
    left_mask: np.ndarray,
    right_u: np.ndarray,
    left_u: np.ndarray,
    right_v: np.ndarray,
    left_v: np.ndarray,
    right_w: np.ndarray,
    left_w: np.ndarray,
    profile: Profile,
) -> dict[str, np.ndarray]:
    """5 部位の bool mask を作る。partition.mode で分割方式を切り替える。"""
    partition = profile.section("partition")
    mode = partition["mode"]
    right = right_mask > 0
    left = left_mask > 0

    with np.errstate(invalid="ignore"):
        if mode == "ANATOMICAL_LOBES_V1":
            right_cfg = partition["right"]
            left_cfg = partition["left"]
            horizontal = float(right_cfg["horizontal_fissure_u"])
            right_oblique = _fissure_boundary(
                right_w,
                float(right_cfg["oblique_medial_u"]),
                float(right_cfg["oblique_lateral_u"]),
            )
            left_oblique = _fissure_boundary(
                left_w,
                float(left_cfg["oblique_medial_u"]),
                float(left_cfg["oblique_lateral_u"]),
            )
            # 裂の位置は肺 bbox 内の絶対高さ v で判定する
            below_horizontal = right & (right_v >= horizontal)
            return {
                # 水平裂より上は全幅が上葉
                "RUL": right & (right_v < horizontal),
                # 水平裂より下のうち、斜裂より上（内側寄り）が中葉
                "RML": below_horizontal & (right_v < right_oblique),
                # 斜裂より下（外側・下方）が下葉
                "RLL": below_horizontal & (right_v >= right_oblique),
                # 左は斜裂のみ。内側は舌区として下方まで上葉が伸びる
                "LUL": left & (left_v < left_oblique),
                "LLL": left & (left_v >= left_oblique),
            }

        if mode == "EQUAL_THIRDS":
            first, second = (float(v) for v in partition["right_boundaries"])
            left_boundary = float(partition["left_boundary"])
            return {
                "RUL": right & (right_u >= 0.0) & (right_u < first),
                "RML": right & (right_u >= first) & (right_u < second),
                "RLL": right & (right_u >= second) & (right_u <= 1.0),
                "LUL": left & (left_u >= 0.0) & (left_u < left_boundary),
                "LLL": left & (left_u >= left_boundary) & (left_u <= 1.0),
            }

        # OVERLAP_TWO_THIRDS（要求文どおりの重複方式。利用者承認が必要）
        return {
            "RUL": right & (right_u >= 0.0) & (right_u <= 2 / 3),
            "RML": right & (right_u >= 1 / 3) & (right_u <= 1.0),
            "RLL": right & (right_u >= 2 / 3) & (right_u <= 1.0),
            "LUL": left & (left_u >= 0.0) & (left_u <= 2 / 3),
            "LLL": left & (left_u >= 1 / 3) & (left_u <= 1.0),
        }


def _make_region(
    code: str,
    gray: np.ndarray,
    region_bool: np.ndarray,
    profile: Profile,
) -> RegionOutput | None:
    ys, xs = np.nonzero(region_bool)
    if xs.size == 0:
        return None

    crop_cfg = profile.section("crop")
    min_side = int(crop_cfg.get("min_side_px", 24))
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    if (x1 - x0) < min_side or (y1 - y0) < min_side:
        return None

    height, width = gray.shape
    region_mask = (region_bool[y0:y1, x0:x1] * 255).astype(np.uint8)

    margin = int(round(max(x1 - x0, y1 - y0) * float(crop_cfg["margin_ratio"])))
    cx0, cy0 = max(0, x0 - margin), max(0, y0 - margin)
    cx1, cy1 = min(width, x1 + margin), min(height, y1 + margin)

    if crop_cfg["embedding_crop"] == "masked":
        crop = cv2.bitwise_and(gray[y0:y1, x0:x1], gray[y0:y1, x0:x1], mask=region_mask)
        crop_bbox = [x0, y0, x1, y1]
    else:  # context crop: mask 外も残す
        crop = gray[cy0:cy1, cx0:cx1].copy()
        crop_bbox = [cx0, cy0, cx1, cy1]

    # 埋め込みモデルの入力解像度へ先に落とす（設定時のみ）。
    # 元の crop は短辺 300px 前後あり、PNG 符号化・転送・埋め込み側の再デコードが
    # 1 枚あたり数百 ms かかる。encoder はどのみち短辺 224 へ縮小するため、
    # ここで縮小しておくと画質を落とさずに往復コストだけ削れる。
    output_side = int(crop_cfg.get("output_side", 0))
    if output_side > 0 and min(crop.shape[:2]) > output_side:
        scale = output_side / min(crop.shape[:2])
        crop = cv2.resize(
            crop,
            (max(1, int(round(crop.shape[1] * scale))), max(1, int(round(crop.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )

    # 表示用: mask 外を減光し、その部位だけを見せる。
    # 部位 mask は肺輪郭に沿った帯状で、外接矩形は肺のほぼ全体を覆うため、
    # 矩形で切り出しただけでは部位を判別できない。
    display_crop = _dim_outside_mask(gray[y0:y1, x0:x1], region_mask)

    area_px = int(region_bool.sum())
    centroid = [float(xs.mean()), float(ys.mean())]
    border_touch = _border_touch_ratio(region_bool)
    return RegionOutput(
        region_code=code,
        bbox=[x0, y0, x1, y1],
        crop_bbox=crop_bbox,
        area_px=area_px,
        centroid=centroid,
        confidence=0.0,  # 品質検査で決定する
        quality_status="OK",
        mask=region_mask,
        crop=crop,
        display_crop=display_crop,
        metrics={
            "border_touch_ratio": border_touch,
            "bbox_width": x1 - x0,
            "bbox_height": y1 - y0,
        },
    )


# --- パイプライン ------------------------------------------------------------


def extract(
    data: bytes,
    *,
    profile_name: str | None = None,
    profile_dir: str | None = None,
    regions: tuple[str, ...] = REGION_CODES,
) -> ExtractionOutput:
    profile = load_profile(profile_name or "nihcc-classical-v1", profile_dir)

    raw = decode_image(data)
    gray, scale, inverted = normalize(raw, profile)
    body = body_mask(gray)
    thorax, bbox = thorax_mask(gray, body, profile)

    segmenter = segmentation.from_profile(profile)
    heart_mask = None
    if segmenter is None:
        # 規則ベース（nihcc-classical-v1）。corpus の約 4 分の 1 で破綻する
        # ので、取り込みでは U-Net の profile を使う。
        candidate = lung_candidate(gray, thorax, profile)
        lung, components = select_lung_components(candidate, profile)
        spine, continuity, spine_method = spine_curve(gray, lung, bbox, profile)
        right_mask, left_mask = split_lungs(lung, spine)
        # 肋骨で分断された外周を、解剖学的にありえる滑らかな曲線へ整える。
        # 左右に分けた後で行う（前に行うと上縦隔で左右が繋がる）。
        right_mask = refine_lung_outline(right_mask, gray, spine, "RIGHT", profile)
        left_mask = refine_lung_outline(left_mask, gray, spine, "LEFT", profile)
        lung = cv2.bitwise_or(right_mask, left_mask)
    else:
        # 学習した分割。左右は模型が直接出すので split_lungs は通さない。
        # 外周の整形も不要（模型の出力はもともと滑らかで、肋骨で分断されない）。
        result = segmenter.predict(gray)
        right_mask, left_mask, heart_mask = result.right, result.left, result.heart
        lung = cv2.bitwise_or(right_mask, left_mask)
        if lung.sum() == 0:
            raise ExtractionError("NO_LUNG_FIELD", "segmentation model produced an empty mask")
        components = [m for m in (right_mask, left_mask) if m.sum() > 0]
        # 脊柱は部位分割の幾何に使うので、規則ベースと同じ関数で求める。
        # 入力の肺 mask が綺麗になるぶん、こちらの方が安定する。
        spine, continuity, spine_method = spine_curve(gray, lung, bbox, profile)

    right_u = normalized_height_map(right_mask, profile)
    left_u = normalized_height_map(left_mask, profile)
    right_v = absolute_height_map(right_mask)
    left_v = absolute_height_map(left_mask)
    right_w = lateral_position_map(right_mask, "RIGHT", profile)
    left_w = lateral_position_map(left_mask, "LEFT", profile)
    masks = region_masks(
        right_mask=right_mask,
        left_mask=left_mask,
        right_u=right_u,
        left_u=left_u,
        right_v=right_v,
        left_v=left_v,
        right_w=right_w,
        left_w=left_w,
        profile=profile,
    )

    outputs: dict[str, RegionOutput] = {}
    for code in regions:
        region = _make_region(code, gray, masks[code], profile)
        if region is not None:
            outputs[code] = region

    metrics = {
        "scale": scale,
        "polarity_inverted": inverted,
        "work_size": [int(gray.shape[1]), int(gray.shape[0])],
        "body_bbox": bbox,
        "component_count": len(components),
        "spine_method": spine_method,
        "spine_continuity_ratio": continuity,
        "spine_mean_x": float(spine.mean()),
        "lung_backend": "classical" if segmenter is None else "unet",
    }
    if heart_mask is not None:
        # 心胸郭比のような指標に使えるよう、面積だけ残す（mask 自体は保存しない）
        metrics["heart_area_px"] = int((heart_mask > 0).sum())
    quality = evaluate_quality(
        gray=gray,
        body_bbox=bbox,
        lung_mask=lung,
        right_mask=right_mask,
        left_mask=left_mask,
        spine=spine,
        spine_continuity=continuity,
        regions=outputs,
        profile=profile,
        polarity_inverted=inverted,
    )
    for code, region in outputs.items():
        region.confidence = quality.region_confidence.get(code, 0.0)
        region.quality_status = quality.region_status.get(code, "ROI_UNCERTAIN")

    return ExtractionOutput(
        profile=profile,
        normalized=gray,
        scale=scale,
        lung_mask=lung,
        right_mask=right_mask,
        left_mask=left_mask,
        spine_x=spine,
        regions=outputs,
        quality=quality,
        metrics=metrics,
    )


def encode_png(array: np.ndarray) -> bytes:
    """lossless PNG へ符号化する。書き込み結果を必ず確認する（設計書 6.6.2）。"""
    ok, buffer = cv2.imencode(".png", array, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise ExtractionError("PNG_ENCODE_FAILED", "failed to encode PNG")
    return buffer.tobytes()
