"""肺野全体の形状ベクトルを固定する。

**これは所見の判別器ではない。** 判別能は部位ごとの形状に劣ることを測定済みで
（4 所見すべてで部位ごとが上回る）、「胸郭の形が似た
症例を探す」道具として提供している。

それでも別軸として持つのは、部位に分けると作れない量があるためである。左右肺の
重心距離は単一特徴としてこれまで測った中で最も強い（気胸 0.648 / 胸水 0.667）。
その性質を壊さないよう、**左右を比べていること**と**枠に依存しないこと**を守る。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from nihcc_rag import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _services import aggregator  # noqa: E402

agg = aggregator()

from nihcc_rag.contract import LungShapeRequest  # noqa: E402

SIDE = 200


def rect(x0: int, y0: int, x1: int, y1: int, side: int = SIDE) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def lungs(gap: int = 40, side: int = SIDE, pad: int = 0) -> dict[str, np.ndarray]:
    total = side + pad * 2
    half = total // 2
    lx, rx = half - gap // 2, half + gap // 2
    top, mid, bottom = pad + 30, pad + 100, pad + 170
    return {
        "RUL": rect(pad + 20, top, lx, mid, total),
        "RLL": rect(pad + 20, mid, lx, bottom, total),
        "RML": rect(pad + 30, mid - 20, lx, mid + 10, total),
        "LUL": rect(rx, top, total - pad - 20, mid, total),
        "LLL": rect(rx, mid, total - pad - 20, bottom, total),
    }


# --- 基本 --------------------------------------------------------------------


def test_dimension_and_normalisation() -> None:
    v = shape.describe_lung(lungs())
    assert v.shape == (shape.LUNG_DIMENSION,)
    assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)
    assert np.isfinite(v).all()


def test_feature_names_match_dimension() -> None:
    assert len(shape.LUNG_FEATURE_NAMES) == shape.LUNG_DIMENSION


def test_empty_input_is_zero_not_an_error() -> None:
    assert not shape.describe_lung({}).any()


def test_does_not_depend_on_the_frame_size() -> None:
    """余白が違っても同じベクトルになること。

    取り込み時と backfill 時で基準がずれると、同じ画像から違うベクトルが出る。
    """
    a = shape.describe_lung(lungs(pad=0))
    b = shape.describe_lung(lungs(pad=50))
    assert np.allclose(a, b, atol=1e-6)


# --- 心拡大・気胸に対応する量 ------------------------------------------------


def test_mediastinum_widens_when_lungs_are_pushed_apart() -> None:
    """左右肺の間隔が広がると mediastinum_max が増えること（心拡大の代理）。"""
    i = shape.LUNG_FEATURE_NAMES.index("mediastinum_max")
    assert shape.describe_lung(lungs(gap=80))[i] > shape.describe_lung(lungs(gap=20))[i]


def test_left_right_centroid_distance_responds_to_widening() -> None:
    """左右肺の重心距離が広がりに反応すること。

    単一特徴としてこれまで測った中で最も強く、部位に分けると作れない量である。
    """
    i = shape.LUNG_FEATURE_NAMES.index("lr_centroid_dx")
    assert shape.describe_lung(lungs(gap=80))[i] > shape.describe_lung(lungs(gap=20))[i]


def test_area_ratio_drops_when_one_lung_collapses() -> None:
    """片肺が縮むと左右の面積比が下がること（気胸は片側性）。"""
    i = shape.LUNG_FEATURE_NAMES.index("lr_area_ratio")
    healthy = lungs()
    sick = lungs()
    sick["RUL"] = rect(20, 30, 60, 70)
    sick["RLL"] = rect(20, 70, 60, 110)
    sick["RML"] = rect(25, 60, 60, 80)
    assert shape.describe_lung(sick)[i] < shape.describe_lung(healthy)[i]


def test_missing_side_is_neutral_not_extreme() -> None:
    """片側の ROI が丸ごと落ちた画像を「極端な非対称」にしないこと。"""
    i = shape.LUNG_FEATURE_NAMES.index("lr_area_ratio")
    only_right = {k: v for k, v in lungs().items() if k.startswith("R")}
    assert shape.describe_lung(only_right)[i] > 0.0


# --- 部位ごとの記述子と別物であること ----------------------------------------


def test_the_region_descriptor_is_gone() -> None:
    """**部位単位の形状は持たない。**

    部位は胸郭を幾何規則で切り分けたものなので、その形は胸郭の形の従属変数で
    しかなく独立した情報を持たない。加えると、検索結果に胸郭の順位を粗く
    写した欄がもう 1 つ増えるだけになる。
    """
    for name in ("DIMENSION", "FEATURE_NAMES", "MIRROR", "describe", "describe_all",
                 "from_roi_regions"):
        assert not hasattr(shape, name), f"部位単位の形状が戻っている: {name}"


def test_left_right_masks_give_the_same_vector_as_the_regions() -> None:
    """左右の肺 mask から作るベクトル（胸部野 1 単位の取り込み）が、部位を左右で
    束ねて作ったものと一致すること。describe_lung は部位コードの頭文字で左右を
    分けるので、U-Net の左右 mask を直接渡しても同じ値になる。"""
    regions = lungs()
    right = regions["RUL"] | regions["RML"] | regions["RLL"]
    left = regions["LUL"] | regions["LLL"]
    assert np.allclose(shape.lung_from_masks(right, left), shape.describe_lung(regions))


def test_the_region_mask_recovery_is_gone() -> None:
    """部位の mask から肺野を組み立て直す経路は持たない（取り込みは左右の肺 mask を直接使う）。"""
    for name in ("lung_from_roi_regions", "_roi_masks"):
        assert not hasattr(shape, name), f"部位の mask の経路が残っている: {name}"


# --- 集約器の配線 ------------------------------------------------------------


def request_for(vector=None):
    return LungShapeRequest(
        preprocess_version="prep-v1.0.0",
        query={"query_vector": list(vector or [0.1] * shape.LUNG_DIMENSION), "candidate_k": 5},
    )


def test_duplicate_rows_of_one_image_are_merged(monkeypatch) -> None:
    """**同じ画像を 2 回返さないこと。**

    モデル版を変えて再取り込みすると、同じ画像の行が版ごとに残る。重複を除かないと
    結果が同じ画像で埋まる。
    """
    row = {"image_id": "11111111-1111-1111-1111-111111111111", "distance": 0.1, "roi_confidence": 0.9}
    monkeypatch.setattr(agg, "_query_lung", lambda *a, **k: [dict(row), dict(row, distance=0.2)])
    response = agg.lung_shape_search(request_for())
    assert len(response.candidates) == 1
    assert response.candidates[0].distance == pytest.approx(0.1)


def test_failure_is_unavailable(monkeypatch) -> None:
    def fail(*a, **k):
        raise RuntimeError("OperationalError")

    monkeypatch.setattr(agg, "_query_lung", fail)
    with pytest.raises(Exception) as exc:
        agg.lung_shape_search(request_for())
    assert exc.value.status == 503


def test_timeout_is_a_gateway_timeout(monkeypatch) -> None:
    import psycopg

    def slow(*a, **k):
        raise psycopg.errors.QueryCanceled("canceling statement due to statement timeout")

    monkeypatch.setattr(agg, "_query_lung", slow)
    with pytest.raises(Exception) as exc:
        agg.lung_shape_search(request_for())
    assert exc.value.status == 504


def test_no_rows_is_reported_as_empty(monkeypatch) -> None:
    """形状ベクトルが 1 件も無いのは失敗ではなく「未登録」。画面が理由を出し分ける。"""
    monkeypatch.setattr(agg, "_query_lung", lambda *a, **k: [])
    response = agg.lung_shape_search(request_for())
    assert response.empty_space is True
    assert response.candidates == []


def test_sql_skips_unfilled_rows() -> None:
    assert "embedding_lung IS NOT NULL" in agg._LUNG_SQL
