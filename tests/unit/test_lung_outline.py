"""肺野外周の描画を固定する。

**これは確認のための表示である。** 胸郭ランキング（`shape.describe_lung`）が何を
「肺野」として扱ったかを目で確かめられないと、脊柱・心陰影・横隔膜が正しく抜けて
いるかを判断できない。肺尖部が水平に切られる、肋骨横隔膜角が欠けるといった
ROI の欠陥は、この表示で見つかる。
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from nihcc_rag import imaging
from PIL import Image

SIDE = 240


def png_of(array: np.ndarray, mode: str = "L") -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def parent(side: int = SIDE) -> bytes:
    rng = np.random.default_rng(0)
    grey = (rng.random((side, side)) * 80 + 80).astype(np.uint8)
    return png_of(np.dstack([grey] * 3), mode="RGB")


def mask(w: int, h: int) -> bytes:
    return png_of(np.full((h, w), 255))


def pixels(png: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png)) as image:
        return np.asarray(image.convert("RGB"))


def outline_pixels(png: bytes, color=(255, 214, 0)) -> np.ndarray:
    out = pixels(png).astype(int)
    return (
        (abs(out[:, :, 0] - color[0]) < 30)
        & (abs(out[:, :, 1] - color[1]) < 30)
        & (abs(out[:, :, 2] - color[2]) < 30)
    )


TWO_LUNGS = [(mask(60, 120), [30, 40, 90, 160]), (mask(60, 120), [150, 40, 210, 160])]


# --- 描けていること ----------------------------------------------------------


def test_outline_is_drawn() -> None:
    edge = outline_pixels(imaging.lung_outline(parent(), TWO_LUNGS))
    assert int(edge.sum()) > 0


def test_output_is_a_png_of_the_expected_size() -> None:
    out = pixels(imaging.lung_outline(parent(), TWO_LUNGS, max_size=128))
    assert max(out.shape[:2]) <= 128


def test_empty_masks_are_rejected() -> None:
    """肺野が空のまま描くと、外周の無い画像を「確認済み」と誤認させる。"""
    with pytest.raises(ValueError, match="肺野"):
        imaging.lung_outline(parent(), [(png_of(np.zeros((40, 40))), [0, 0, 40, 40])])


# --- 外周であること（内側の継ぎ目を描かない）--------------------------------


def test_touching_regions_do_not_draw_their_shared_border() -> None:
    """**和集合の外周だけを描くこと。**

    部位ごとの輪郭を重ねると部位間の継ぎ目まで描かれ、どこが肺野の外縁なのかが
    判らなくなる。上下に接する 2 部位で、境目に線が出ないことを確かめる。
    """
    upper = (mask(80, 60), [80, 40, 160, 100])
    lower = (mask(80, 60), [80, 100, 160, 160])
    edge = outline_pixels(imaging.lung_outline(parent(), [upper, lower], max_size=SIDE))
    # 継ぎ目（y=100 付近）の内側に線が無いこと
    seam = edge[97:103, 95:145]
    assert int(seam.sum()) == 0, "部位の継ぎ目に線が描かれている"


def test_separate_lungs_each_get_an_outline() -> None:
    """左右が離れていれば、それぞれの外周が描かれること。"""
    edge = outline_pixels(imaging.lung_outline(parent(), TWO_LUNGS, max_size=SIDE))
    left = int(edge[:, : SIDE // 2].sum())
    right = int(edge[:, SIDE // 2 :].sum())
    assert left > 0 and right > 0


# --- 原画像が見えること ------------------------------------------------------


def test_the_image_stays_visible_under_the_tint() -> None:
    """肺野の内側を塗り潰さないこと。中身が見えないと所見を確認できない。"""
    plain = pixels(imaging.lung_outline(parent(), TWO_LUNGS, fill_alpha=0.0, max_size=SIDE))
    tinted = pixels(imaging.lung_outline(parent(), TWO_LUNGS, fill_alpha=0.12, max_size=SIDE))
    inside = (slice(60, 140), slice(40, 80))
    assert not np.array_equal(plain[inside], tinted[inside]), "着色が効いていない"
    # 元の濃淡が残っていること（分散がつぶれていない）
    assert float(tinted[inside].std()) > 5.0


def test_thickness_widens_the_line() -> None:
    """縮小表示で線が消えないよう、太さを調整できること。"""
    thin = int(outline_pixels(imaging.lung_outline(parent(), TWO_LUNGS, thickness=1, max_size=SIDE)).sum())
    thick = int(outline_pixels(imaging.lung_outline(parent(), TWO_LUNGS, thickness=5, max_size=SIDE)).sum())
    assert thick > thin


def test_mask_is_placed_at_its_bbox() -> None:
    """mask を bbox の位置へ戻すこと。ずれると別の場所の外周を描く。"""
    edge = outline_pixels(
        imaging.lung_outline(parent(), [(mask(40, 40), [20, 20, 60, 60])], max_size=SIDE)
    )
    ys, xs = np.nonzero(edge)
    assert 15 <= int(xs.min()) <= 25 and 15 <= int(ys.min()) <= 25
    assert 55 <= int(xs.max()) <= 65 and 55 <= int(ys.max()) <= 65
