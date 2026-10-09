"""形状ベクトルの共有部品を固定する。

**部位単位の形状ベクトルは持たない。** 部位は胸郭を幾何規則で切り分けたものなので、
その形は胸郭の形の従属変数でしかない。

ここで固定するのは胸郭の記述子（`test_lung_shape.py`）と共用する部品である。

  `_moments`      二値 mask の幾何モーメント。取り込みの律速になるので周辺和で計算する
  `lung_stats`    肺野全体から決まる値（縦隔幅を含む）
"""

from __future__ import annotations

import numpy as np
import pytest
from nihcc_rag import shape

SIDE = 200


def rect(x0: int, y0: int, x1: int, y1: int, side: int = SIDE) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def lungs(inner_gap: int = 40, top: int = 30, bottom: int = 170) -> dict[str, np.ndarray]:
    """左右 2 つの肺野を模した 5 部位（RML は右の一部）。"""
    half = SIDE // 2
    left_x1 = half - inner_gap // 2
    right_x0 = half + inner_gap // 2
    mid = (top + bottom) // 2
    return {
        "RUL": rect(20, top, left_x1, mid),
        "RLL": rect(20, mid, left_x1, bottom),
        "RML": rect(30, mid - 20, left_x1, mid + 10),
        "LUL": rect(right_x0, top, SIDE - 20, mid),
        "LLL": rect(right_x0, mid, SIDE - 20, bottom),
    }


def union_of(masks: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros((SIDE, SIDE), bool)
    for mask in masks.values():
        out |= mask
    return out


def test_moments_match_a_naive_implementation() -> None:
    """周辺和を使った高速版が、素朴な実装と一致すること。

    取り込みの律速になるため np.nonzero を使わない実装にしている。ここが違うと形が変わる。
    """
    rng = np.random.default_rng(3)
    mask = rng.random((80, 120)) > 0.6
    ys, xs = np.nonzero(mask)
    got = shape._moments(mask)
    assert got["area"] == pytest.approx(float(xs.size))
    assert got["cx"] == pytest.approx(float(xs.mean()), abs=1e-3)
    assert got["cy"] == pytest.approx(float(ys.mean()), abs=1e-3)
    assert got["mu20"] == pytest.approx(float(((xs - xs.mean()) ** 2).mean()), rel=1e-3)
    assert got["mu02"] == pytest.approx(float(((ys - ys.mean()) ** 2).mean()), rel=1e-3)
    assert got["mu11"] == pytest.approx(
        float(((xs - xs.mean()) * (ys - ys.mean())).mean()), abs=1e-2
    )


def test_moments_of_an_empty_mask_do_not_raise() -> None:
    """ROI が取れなかったときに例外にしない。1 部位のために 1 枚を落とさない。"""
    stats = shape._moments(np.zeros((10, 10), bool))
    assert stats["area"] == 0.0


def test_mediastinum_widens_when_lungs_are_pushed_apart() -> None:
    """左右肺の間隔が広がると縦隔幅が増えること（心拡大の代理）。"""
    narrow = shape.lung_stats(union_of(lungs(inner_gap=20)))
    wide = shape.lung_stats(union_of(lungs(inner_gap=70)))
    assert wide["mediastinum"] > narrow["mediastinum"]


def test_lung_stats_reports_a_positive_mediastinum() -> None:
    """左右に分かれた肺野なら縦隔幅が正になること。"""
    assert shape.lung_stats(union_of(lungs()))["mediastinum"] > 0
