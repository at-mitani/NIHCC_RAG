"""ROI 抽出パイプラインの単体テスト（設計書 12.1 / 15.1）。"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from image_pipeline import ExtractionError, config, extract, pipeline


def synthetic_chest(width: int = 1024, height: int = 1024, *, invert: bool = False) -> bytes:
    """左右 2 つの暗い肺野と明るい脊柱を持つ合成画像。"""
    canvas = np.full((height, width), 205, dtype=np.uint8)
    # 体外背景を暗くする（body mask の凸包が体内に収まるようにする）
    canvas[:, : int(width * 0.06)] = 20
    canvas[:, int(width * 0.94) :] = 20
    canvas[: int(height * 0.04), :] = 20
    for cx in (int(width * 0.32), int(width * 0.68)):
        cv2.ellipse(
            canvas,
            (cx, int(height * 0.50)),
            (int(width * 0.12), int(height * 0.28)),
            0, 0, 360, 25, -1,
        )
    cv2.rectangle(
        canvas,
        (int(width * 0.47), int(height * 0.10)),
        (int(width * 0.53), int(height * 0.90)),
        245, -1,
    )
    if invert:
        canvas = 255 - canvas
    ok, buffer = cv2.imencode(".png", canvas)
    assert ok
    return buffer.tobytes()


def test_extract_returns_five_regions() -> None:
    result = extract(synthetic_chest())
    assert set(result.regions) == {"RUL", "RML", "RLL", "LUL", "LLL"}
    for region in result.regions.values():
        assert region.crop.size > 0
        assert region.mask.size > 0
        assert 0.0 <= region.confidence <= 1.0


def test_regions_do_not_overlap_in_equal_thirds() -> None:
    """境界画素の二重所属を防ぐ（設計書 6.5.2）。"""
    result = extract(synthetic_chest())
    canvas = np.zeros(result.normalized.shape, dtype=np.int32)
    for region in result.regions.values():
        x0, y0, x1, y1 = region.bbox
        canvas[y0:y1, x0:x1][region.mask > 0] += 1
    assert canvas.max() <= 1


def test_right_lung_is_left_of_spine() -> None:
    """画像左 = 患者右（設計書 6.4.2）。"""
    result = extract(synthetic_chest())
    right_x = np.nonzero(result.right_mask)[1].mean()
    left_x = np.nonzero(result.left_mask)[1].mean()
    assert right_x < result.spine_x.mean() < left_x


def test_polarity_is_normalised_for_inverted_input() -> None:
    """極性反転画像でも肺野が暗い前提へ揃える（設計書 6.2）。"""
    result = extract(synthetic_chest(invert=True))
    assert result.metrics["polarity_inverted"] is True
    assert len(result.regions) == 5


def _region_full_masks(result) -> dict:
    """部位 mask を画像全体の座標へ戻す。"""
    masks = {}
    for code, region in result.regions.items():
        x0, y0, x1, y1 = region.bbox
        full = np.zeros(result.normalized.shape, dtype=bool)
        full[y0:y1, x0:x1] = region.mask > 0
        masks[code] = full
    return masks


def test_region_areas_match_expected_share() -> None:
    """各部位の面積比が profile の期待値に概ね一致する（参照図の肺葉配置）。"""
    profile = config.load_profile()
    expected = profile.section("quality")["region_expected_share"]
    result = extract(synthetic_chest())
    right_total = sum(result.regions[code].area_px for code in ("RUL", "RML", "RLL"))
    left_total = sum(result.regions[code].area_px for code in ("LUL", "LLL"))
    for code in ("RUL", "RML", "RLL", "LUL", "LLL"):
        total = right_total if code.startswith("R") else left_total
        share = result.regions[code].area_px / total
        assert abs(share - float(expected[code])) < 0.20, f"{code} share={share:.3f}"


def test_middle_lobe_is_medial_and_lower_lobe_is_lateral() -> None:
    """参照図どおり、右中葉は内側寄り・右下葉は外側寄りに置かれる。

    水平3分割では中葉と下葉が上下に並んでしまい、この配置にならない。
    """
    profile = config.load_profile()
    result = extract(synthetic_chest())
    masks = _region_full_masks(result)
    w = pipeline.lateral_position_map(result.right_mask, "RIGHT", profile)
    v = pipeline.absolute_height_map(result.right_mask)

    lower_zone = v > 0.55
    middle_w = np.nanmean(w[masks["RML"] & lower_zone])
    lower_w = np.nanmean(w[masks["RLL"] & lower_zone])
    # w は 0 が内側、1 が外側
    assert middle_w < lower_w - 0.15, f"RML={middle_w:.2f} RLL={lower_w:.2f}"


def test_left_upper_lobe_keeps_a_medial_lingula() -> None:
    """左上葉は舌区として内側を下方へ伸びる（参照図の S4・S5）。"""
    profile = config.load_profile()
    result = extract(synthetic_chest())
    masks = _region_full_masks(result)
    w = pipeline.lateral_position_map(result.left_mask, "LEFT", profile)
    v = pipeline.absolute_height_map(result.left_mask)

    lower_zone = v > 0.45
    upper_w = np.nanmean(w[masks["LUL"] & lower_zone])
    lower_w = np.nanmean(w[masks["LLL"] & lower_zone])
    assert upper_w < lower_w, f"LUL={upper_w:.2f} LLL={lower_w:.2f}"


def test_lung_base_belongs_to_the_lower_lobe() -> None:
    """肺底部は左右とも下葉のみ（参照図で最下部は下葉）。"""
    result = extract(synthetic_chest())
    masks = _region_full_masks(result)
    for side, lower in (("right_mask", "RLL"), ("left_mask", "LLL")):
        v = pipeline.absolute_height_map(getattr(result, side))
        base = v > 0.9
        for code, mask in masks.items():
            if code[0] != lower[0]:
                continue
            overlap = (mask & base).sum()
            if code != lower:
                assert overlap == 0, f"{code} が肺底部に現れている"
        assert (masks[lower] & base).sum() > 0


def test_partition_modes_are_switchable() -> None:
    """比較用の EQUAL_THIRDS も引き続き利用できる（version 付きで区別する）。"""
    profile = config.load_profile()
    assert profile.data["partition"]["mode"] == "ANATOMICAL_LOBES_V1"
    assert profile.partition_rule_version == "ANATOMICAL_LOBES_V1-v1"
    assert "EQUAL_THIRDS" in config.PARTITION_MODES


def test_flat_image_is_rejected() -> None:
    flat = np.full((1024, 1024), 128, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", flat)
    assert ok
    with pytest.raises(ExtractionError):
        extract(buffer.tobytes())


def test_profile_is_validated_and_hashed() -> None:
    profile = config.load_profile()
    assert profile.data["partition"]["mode"] in config.PARTITION_MODES
    assert len(profile.parameter_hash) == 64
    # roi_model_version は algorithm version と parameter hash から決まる
    assert profile.algorithm_version in profile.roi_model_version
    assert profile.parameter_hash[:12] in profile.roi_model_version


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(config.ProfileError):
        config.load_profile("does-not-exist")


def test_profile_name_traversal_is_rejected() -> None:
    with pytest.raises(config.ProfileError):
        config.load_profile("../../etc/passwd")


def test_normalized_height_map_is_within_unit_range() -> None:
    mask = np.zeros((200, 200), np.uint8)
    cv2.rectangle(mask, (50, 40), (150, 160), 255, -1)
    u = pipeline.normalized_height_map(mask, config.load_profile())
    values = u[~np.isnan(u)]
    assert values.min() >= -0.05
    assert values.max() <= 1.05


# --- 部位単位の画像（表示・埋め込み） -----------------------------------------


def test_display_crop_isolates_the_region() -> None:
    """表示用 crop は mask 内だけが明るく、mask 外は減光されている。

    部位 mask は肺輪郭に沿った帯状のため、外接矩形で切り出しただけでは
    肺のほぼ全体が写り、どの部位か判別できない（実データで確認した不具合）。
    """
    result = extract(synthetic_chest())
    for region in result.regions.values():
        x0, y0, x1, y1 = region.bbox
        original = result.normalized[y0:y1, x0:x1]
        assert region.display_crop.shape == original.shape

        # mask 内は原画像のまま
        inside = region.mask > 0
        assert np.array_equal(region.display_crop[inside], original[inside])

        # mask 外は減光されている（肺野が暗い画像でも成り立つよう、
        # 同じ画素の原画像値との比較で判定する）
        outside = region.mask == 0
        if outside.any() and original[outside].mean() > 5:
            assert region.display_crop[outside].mean() < original[outside].mean()


def test_region_crops_differ_between_regions() -> None:
    """同側の各部位で埋め込み用 crop の内容が異なること。

    矩形だけで切り出すと RUL / RML / RLL がほぼ同一画像になり、部位別検索が
    成立しない。mask 適用により内容が分かれることを保証する。
    """
    result = extract(synthetic_chest())
    right = [result.regions[code] for code in ("RUL", "RML", "RLL")]
    size = (128, 128)
    resized = [cv2.resize(region.crop, size).astype(float) for region in right]
    for first in range(len(resized)):
        for second in range(first + 1, len(resized)):
            difference = np.abs(resized[first] - resized[second]).mean()
            assert difference > 1.0, (
                "同側の部位 crop がほぼ同一。mask が適用されていない可能性がある"
            )


def test_embedding_crop_is_masked_by_default() -> None:
    """既定 profile では mask 外が黒（埋め込みが部位内容だけを見る）。"""
    profile = config.load_profile()
    assert profile.data["crop"]["embedding_crop"] == "masked"
    result = extract(synthetic_chest())
    for region in result.regions.values():
        mask = region.mask
        if region.crop.shape != mask.shape:
            # crop.output_side で縮小された場合は mask を crop 側へ合わせる。
            # 縮小時は境界画素が混ざるため、内側へ 1px 収縮してから判定する。
            mask = cv2.resize(
                mask, (region.crop.shape[1], region.crop.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        outside = region.crop[mask == 0]
        assert outside.size == 0 or outside.max() == 0


def test_embedding_crop_is_downscaled_to_output_side() -> None:
    """埋め込み crop は encoder の入力解像度へ縮小する（設計書 7.3 / 取り込み性能）。

    PNG 符号化・転送・埋め込み側の再デコードが 1 枚あたり数百 ms かかるため、
    ROI 側で短辺を落としておく。mask は表示合成に使うので縮小しない。
    """
    side = int(config.load_profile().data["crop"]["output_side"])
    assert side > 0
    result = extract(synthetic_chest())
    resized = 0
    for region in result.regions.values():
        assert min(region.crop.shape[:2]) <= side
        if min(region.mask.shape[:2]) > side:
            resized += 1
            assert min(region.mask.shape[:2]) > min(region.crop.shape[:2])
            # 縦横比は保つ（encoder の center crop が部位を切り落とさないように）
            source_ratio = region.mask.shape[1] / region.mask.shape[0]
            crop_ratio = region.crop.shape[1] / region.crop.shape[0]
            assert abs(source_ratio - crop_ratio) < 0.02
    assert resized > 0, "縮小が働く部位が 1 つも無い（テスト画像が小さすぎる）"
