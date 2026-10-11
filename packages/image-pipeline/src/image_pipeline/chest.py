"""肺野＋縦隔＋横隔膜を 1 つにした「胸部野」mask。

肺野だけの枠では、心拡大・胸水・横隔膜ヘルニアが表現できない。心拡大は左右の肺の
「間」にあり、胸水は肋骨横隔膜角にたまって肺野の下端より外へ出る。画像 1 枚を
1 単位にして検索するので、検索対象の枠もこれらを含む範囲へ広げる。

作り方
------
1. U-Net の右肺・左肺・心臓の和を取る
2. その**凸包**で埋める。左右の肺の間（縦隔・心陰影・気管）と、両側の
   肋骨横隔膜角を結んだ線より上（横隔膜ドーム）が入る
3. **下方向へだけ**帯を足す。胸水で鈍化した肋骨横隔膜角と横隔膜そのものは、
   肺野 mask の下端より下にある
4. 全周へ薄く広げる。胸膜（気胸の外縁、胸膜肥厚）は肺野 mask の外縁に接している

正面像では脊柱が縦隔と重なって写るため、**脊柱だけを除くことはできない**。
縦隔を残す以上、その背後の脊柱も枠に入る。

評価と本番で同じ関数を使う
--------------------------
胸部野の分類器は `scripts/chest_image_eval.py` が作った特徴で学習している。
本番の取り込み（roi-service）が 1 画素でも違う前処理をすると、分類器は**黙って
壊れる**（例外は出ず、確率だけがずれる）。そのため正規化 → 分割 → mask → 縮小 →
PNG 化までを `extract_chest` と `chest_model_inputs` に閉じ込め、評価スクリプトと
roi-service の両方がこれを呼ぶ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import segmenter as segmentation
from .config import Profile, load_profile
from .pipeline import ExtractionError, decode_image, encode_png, normalize

#: 下方向へ足す帯の高さ（肺野の高さに対する比）。肋骨横隔膜角の鈍化と横隔膜を含める。
DIAPHRAGM_RATIO = 0.10
#: 全周へ広げる幅（肺野の幅に対する比）。胸膜を含める。
MARGIN_RATIO = 0.02
#: 肺野が画像に占める割合の下限。これ未満は分割の失敗とみなす（profile の
#: quality.lung_area_ratio の下限と同じ）。片肺の虚脱があり得るので両肺は要求しない。
MIN_LUNG_AREA = 0.02
#: pooling に渡す画像と mask の 1 辺。RAD-DINO の入力（518）に合わせる。
#: 評価（vec_pool-chest_*）はこの値で作ってあり、分類器はこれを前提にしている。
POOL_SIDE = 518
#: 胸部野 crop の 1 辺（1 段目 BiomedCLIP 用。encoder 側で 224 へ縮小される）。
CROP_SIDE = 518


def chest_mask(
    right: np.ndarray,
    left: np.ndarray,
    heart: np.ndarray | None = None,
    *,
    diaphragm_ratio: float = DIAPHRAGM_RATIO,
    margin_ratio: float = MARGIN_RATIO,
) -> np.ndarray:
    """胸部野の 2 値 mask（0 / 255、uint8）を返す。入力と同じ寸法。

    `heart` は無くてよい（規則ベースの profile は心臓を出さない）。その場合も
    凸包が左右の肺の間を埋めるので、心陰影の大半は入る。
    """
    lungs = (right > 0) | (left > 0)
    if not lungs.any():
        raise ValueError("肺野 mask が空なので胸部野を作れない")
    base = lungs if heart is None else lungs | (heart > 0)

    out = np.zeros(lungs.shape, dtype=np.uint8)
    hull = cv2.convexHull(cv2.findNonZero(base.astype(np.uint8)))
    cv2.fillConvexPoly(out, hull, 255)

    ys, xs = np.nonzero(lungs)
    down = int(round(diaphragm_ratio * (ys.max() - ys.min() + 1)))
    if down > 0:
        # anchor を kernel の最下端に置くと、各画素は自分より上 down 画素の最大を取る。
        # つまり mask が下へ down 画素だけ伸び、上へは伸びない。
        kernel = np.ones((down + 1, 1), dtype=np.uint8)
        out = cv2.dilate(out, kernel, anchor=(0, down))

    margin = int(round(margin_ratio * (xs.max() - xs.min() + 1)))
    if margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
        out = cv2.dilate(out, kernel)
    return out


# --- 抽出（評価と本番で共有する） ---------------------------------------------


@dataclass
class ChestExtraction:
    """胸部野の抽出結果。座標はすべて作業解像度（`normalized` と同じ枠）。"""

    profile: Profile
    #: 正規化済みの濃淡画像（pipeline.normalize の出力）
    normalized: np.ndarray
    #: 原画像 → 作業解像度の縮小率（原画像座標 = 作業座標 / scale）
    scale: float
    right: np.ndarray
    left: np.ndarray
    heart: np.ndarray
    lung: np.ndarray
    chest: np.ndarray
    #: 肺野・胸部野・心臓が画像に占める割合
    areas: dict[str, float]
    #: OK / ROI_UNCERTAIN（REJECTED は ExtractionError で表す）
    quality_status: str
    #: 品質の目安 [0, 1]。flags が無ければ 1.0、あれば 0.5
    confidence: float
    flags: list[str] = field(default_factory=list)

    @property
    def chest_bbox(self) -> list[int]:
        """胸部野の外接矩形 [x0, y0, x1, y1]（作業解像度）。"""
        ys, xs = np.nonzero(self.chest)
        return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]

    @property
    def source_bbox(self) -> list[int]:
        """胸部野の外接矩形を原画像の座標へ戻したもの。"""
        return [int(round(v / (self.scale or 1.0))) for v in self.chest_bbox]


def _assess(right: np.ndarray, left: np.ndarray, lung_area: float, profile: Profile
            ) -> list[str]:
    """分割の怪しさを旗で返す。旗が立っても取り込みは止めない（ROI_UNCERTAIN）。"""
    quality = profile.data.get("quality", {})
    _, area_max = quality.get("lung_area_ratio", [MIN_LUNG_AREA, 0.40])
    sym_min, sym_max = quality.get("lung_symmetry_ratio", [0.45, 2.20])
    flags: list[str] = []
    r_area, l_area = int((right > 0).sum()), int((left > 0).sum())
    if r_area == 0 or l_area == 0:
        # 気胸・大量胸水による片肺の虚脱でも起きる。枠は凸包で作れるので落とさない
        flags.append("ONE_LUNG_MISSING")
    elif not sym_min <= r_area / l_area <= sym_max:
        flags.append("LUNG_ASYMMETRY")
    if lung_area > area_max:
        flags.append("LUNG_AREA_OUT_OF_RANGE")
    return flags


def extract_chest(
    data: bytes, *, profile_name: str, profile_dir: str | None = None
) -> ChestExtraction:
    """画像バイト列から胸部野を抽出する。

    5 部位の分割（脊柱推定・葉の分割）は通さない。部位を使わないので不要であり、
    脊柱推定の失敗で画像を落とすこともない。

    失敗は ExtractionError（`PROFILE_NOT_UNET` / `LUNG_TOO_SMALL` / 画像の不正）。
    """
    profile = load_profile(profile_name, profile_dir)
    segmenter = segmentation.from_profile(profile)
    if segmenter is None:
        raise ExtractionError(
            "PROFILE_NOT_UNET", f"profile {profile.name} does not use a U-Net, so there is no heart mask"
        )
    gray, scale, _ = normalize(decode_image(data), profile)
    seg = segmenter.predict(gray)
    lung = cv2.bitwise_or(seg.right, seg.left)
    lung_area = float((lung > 0).mean())
    if lung_area < MIN_LUNG_AREA:
        raise ExtractionError("LUNG_TOO_SMALL", f"the lung fields cover only {lung_area:.1%} of the image")
    chest = chest_mask(seg.right, seg.left, seg.heart)
    flags = _assess(seg.right, seg.left, lung_area, profile)
    return ChestExtraction(
        profile=profile,
        normalized=gray,
        scale=scale,
        right=seg.right,
        left=seg.left,
        heart=seg.heart,
        lung=lung,
        chest=chest,
        areas={
            "lung": lung_area,
            "chest": float((chest > 0).mean()),
            "heart": float((seg.heart > 0).mean()),
        },
        quality_status="ROI_UNCERTAIN" if flags else "OK",
        confidence=0.5 if flags else 1.0,
        flags=flags,
    )


def masked_square_crop(gray: np.ndarray, mask: np.ndarray, side: int) -> np.ndarray:
    """mask の外を黒くして外接矩形で切り、**正方形へ余白を足して**から縮小する。

    縦長のまま渡すと、encoder の前処理（短辺合わせ + 中央切り出し）が肺尖か
    横隔膜を切り落とす。
    """
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    roi = np.where(mask[y0:y1, x0:x1] > 0, gray[y0:y1, x0:x1], 0).astype(np.uint8)
    h, w = roi.shape
    n = max(h, w)
    canvas = np.zeros((n, n), dtype=np.uint8)
    canvas[(n - h) // 2 : (n - h) // 2 + h, (n - w) // 2 : (n - w) // 2 + w] = roi
    return cv2.resize(canvas, (side, side), interpolation=cv2.INTER_AREA)


@dataclass(frozen=True)
class ChestModelInputs:
    """埋め込みモデルへ渡す PNG。評価と本番でこの 3 つが一致していなければならない。"""

    #: pooling 用の全体画像（POOL_SIDE 四方）
    pool_image: bytes
    #: pooling 用の mask（POOL_SIDE 四方）。名前 -> PNG。CHEST は必ず含む
    pool_masks: dict[str, bytes]
    #: 胸部野の外を黒塗りした正方形 crop（1 段目用）
    crop: bytes


def chest_model_inputs(
    extraction: ChestExtraction,
    *,
    pool_side: int = POOL_SIDE,
    crop_side: int = CROP_SIDE,
    masks: tuple[str, ...] = ("CHEST",),
) -> ChestModelInputs:
    """抽出結果から、埋め込みモデルへ渡す PNG を作る。

    画像と mask を**同じ寸法**へ縮小しておくのが要点で、モデル側の入力解像度や
    patch grid が何であっても、両者を同じ枠で縮約でき部位の位置がずれない。
    縮小は面積平均（INTER_AREA）。mask は 0/255 の中間値を持つ被覆率になる。
    """
    available = {
        "LUNG": extraction.lung,
        "CHEST": extraction.chest,
        "WHOLE": np.full_like(extraction.lung, 255),
    }
    unknown = sorted(set(masks) - set(available))
    if unknown:
        raise ValueError(f"未知の mask 名: {unknown}")
    side = (pool_side, pool_side)
    return ChestModelInputs(
        pool_image=encode_png(cv2.resize(extraction.normalized, side, interpolation=cv2.INTER_AREA)),
        pool_masks={
            name: encode_png(cv2.resize(available[name], side, interpolation=cv2.INTER_AREA))
            for name in masks
        },
        crop=encode_png(masked_square_crop(extraction.normalized, extraction.chest, crop_side)),
    )
