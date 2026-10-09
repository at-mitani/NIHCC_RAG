"""病変の位置の地図の評価（scripts/chest_lesion_map_eval.py）の部品を固定する。

この評価の結果で「地図が病変の位置・大きさを示すか」を判定する。
座標の取り違えや合否判定の誤りがあると、結論がそのまま逆になる。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _services import load  # noqa: E402

ev = load("chest_lesion_map_eval_script", "scripts/chest_lesion_map_eval.py")


def test_patch_center_maps_back_to_the_original() -> None:
    """区画は作業画像を grid 等分したもの。中心を原画像の座標へ戻す（縮小率で割る）。"""
    assert ev.patch_center(0, 0, 4, (400, 800), 0.5) == pytest.approx((100.0, 200.0))
    assert ev.patch_center(3, 3, 4, (400, 800), 1.0) == pytest.approx((350.0, 700.0))


def test_inside_any_box() -> None:
    boxes = [(10, 10, 20, 20), (100, 100, 5, 5)]
    assert ev.inside_any((15, 25), boxes)
    assert ev.inside_any((102, 104), boxes)
    assert not ev.inside_any((50, 50), boxes)


def test_box_mask_uses_the_work_scale() -> None:
    mask = ev.box_mask([(100, 100, 100, 100)], (100, 100), 0.5)
    assert mask[50:100, 50:100].all() and not mask[:50].any()


def test_bright_point_finds_a_small_white_spot_in_the_lung() -> None:
    """肺野の中の小さい白い点を拾い、肺野の外のもっと白いものは拾わないこと。"""
    gray = np.full((1024, 1024), 60, dtype=np.uint8)
    lung = np.zeros_like(gray)
    lung[200:800, 100:450] = 255
    gray[400:430, 300:330] = 200          # 肺野の中の結節（小さい白い点）
    gray[50:80, 700:730] = 255            # 肺野の外（鎖骨のつもり）
    x, y = ev.bright_point(gray, lung, 1.0)
    assert 300 <= x <= 330 and 400 <= y <= 430


def test_bright_point_ignores_large_bright_regions() -> None:
    """肋骨の帯のように大きい明るい構造は top-hat で引かれ、小さい点が勝つこと。"""
    gray = np.full((1024, 1024), 60, dtype=np.uint8)
    lung = np.full_like(gray, 255)
    gray[100:400, 100:900] = 180          # 大きく明るい帯
    gray[600:625, 500:525] = 170          # 小さい点（帯より暗い）
    x, y = ev.bright_point(gray, lung, 1.0)
    assert 500 <= x <= 525 and 600 <= y <= 625


def test_blob_area_counts_the_connected_half_max_region() -> None:
    values = np.zeros((5, 5))
    values[1:3, 1:3] = [[1.0, 0.6], [0.7, 0.2]]
    values[4, 4] = 0.9                    # 離れた島は数えない
    assert ev.blob_area(values, np.ones((5, 5), bool), (1, 1)) == 3


def test_spearman() -> None:
    assert ev.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ev.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_score_map_ignores_patches_outside_the_chest() -> None:
    """胸部野の外の区画の値が大きくても、点は胸部野の中に置くこと。"""
    grid = 4
    values = np.zeros((grid, grid))
    values[0, 0] = 10.0                   # 胸部野の外
    values[2, 2] = 1.0
    chest = np.zeros((grid, grid))
    chest[1:, 1:] = 1.0
    box = np.zeros((grid, grid))
    box[2, 2] = 1.0
    record = {"work": (400, 400), "scale": 1.0}
    point, enrichment, area = ev.score_map(record, values, chest, box, grid)
    assert point == pytest.approx((250.0, 250.0))
    assert enrichment == pytest.approx(9.0)   # 正の値がすべて box（胸部野の 1/9）に入る


def record(finding, hit_map, hit_bright, random_hit, i, spread=True):
    """spread=True なら box を胸部野のあちこちに置く（平均位置では当たらない）。"""
    x = 50.0 + (i % 5) * 190.0 if spread else 100.0 + i
    y = 50.0 + (i // 5 % 5) * 190.0 if spread else 100.0
    box = (x, y, 50.0, 50.0)
    inside, outside = (x + 20.0, y + 20.0), (990.0, 990.0)
    return {
        "finding": finding, "boxes": [box], "random_hit": random_hit, "box_area": 2500.0,
        "chest_bbox": [0.0, 0.0, 1000.0, 1000.0],
        "bright": inside if hit_bright else outside,
        "map_point": inside if hit_map else outside,
        "enrichment": 2.0, "map_area": 1000.0 + i, "probability": 0.7,
    }


def test_a_map_that_finds_the_lesions_passes() -> None:
    rows = [record("Nodule", True, False, 0.05, i) for i in range(30)]
    report = ev.evaluate(rows, 37, SimpleNamespace(bootstrap=200, seed=1))
    assert report["Nodule"]["passed"] is True
    assert report["Nodule"]["hit"]["map"] == 1.0


def test_a_map_no_better_than_random_fails() -> None:
    rows = [record("Mass", i % 10 == 0, True, 0.1, i) for i in range(30)]
    report = ev.evaluate(rows, 37, SimpleNamespace(bootstrap=200, seed=1))
    assert report["Mass"]["passed"] is False


def test_a_map_that_only_matches_the_typical_position_fails() -> None:
    """所見がいつも同じ場所に出るなら、平均位置（prior）でも当たる。それを超えなければ不合格。"""
    rows = [record("Pneumonia", True, False, 0.05, i, spread=False) for i in range(30)]
    report = ev.evaluate(rows, 37, SimpleNamespace(bootstrap=200, seed=1))
    # box がどれもほぼ同じ位置なので prior も 100% 当たり、map は prior を上回れない
    assert report["Pneumonia"]["hit"]["prior"] == 1.0
    assert report["Pneumonia"]["passed"] is False
