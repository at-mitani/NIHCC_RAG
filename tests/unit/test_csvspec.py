"""CSV 解析の単体テスト（設計書 15.1）。"""

from __future__ import annotations

import io

import pytest
from nihcc_rag import csvspec

HEADER = (
    "Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Sex,"
    "View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y]\n"
)
ROW = "00000001_000.png,Cardiomegaly|Emphysema,0,1,57,M,PA,2682,2749,0.143,0.143\n"


def parse(text: str) -> list[csvspec.CsvRow]:
    return list(csvspec.iter_rows_from_stream(io.StringIO(text)))


def test_multi_value_columns_are_expanded() -> None:
    """header 1 列に対し data 2 列となる項目を正しく展開する。"""
    row = parse(HEADER + ROW)[0]
    assert row.source_image_name == "00000001_000.png"
    assert row.raw["original_width"] == "2682"
    assert row.raw["original_height"] == "2749"
    assert row.raw["pixel_spacing_x"] == "0.143"
    assert row.raw["pixel_spacing_y"] == "0.143"


def test_labels_are_normalised_to_allowed_vocabulary() -> None:
    row = parse(HEADER + ROW)[0]
    assert row.finding_labels == ["Cardiomegaly", "Emphysema"]
    assert row.unknown_labels == []


def test_unknown_labels_are_reported_not_dropped_silently() -> None:
    row = parse(HEADER + "00000001_000.png,Cardiomegaly|Zebra,0,1,57,M,PA,1,1,0.1,0.1\n")[0]
    assert row.finding_labels == ["Cardiomegaly"]
    assert row.unknown_labels == ["Zebra"]


def test_label_normalisation_handles_case_and_separator() -> None:
    assert csvspec.normalise_finding("pleural thickening") == "Pleural_Thickening"
    assert csvspec.normalise_finding(" no finding ") == "No Finding"
    assert csvspec.normalise_finding("unknown-label") is None


def test_patient_key_is_hashed_not_stored_raw() -> None:
    row = parse(HEADER + ROW)[0]
    assert row.patient_key == "1"
    assert len(row.patient_key_hash) == 64
    assert "1" != row.patient_key_hash


def test_unexpected_header_is_rejected() -> None:
    with pytest.raises(csvspec.CsvSchemaError):
        parse("A,B,C\n1,2,3\n")


def test_empty_csv_is_rejected() -> None:
    with pytest.raises(csvspec.CsvSchemaError):
        parse("")


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("00000001_000.png", True),
        ("00012345_017.PNG", True),
        ("1_0.png", False),
        ("../etc/passwd", False),
        ("00000001_000.jpg", False),
    ],
)
def test_image_name_validation(name: str, valid: bool) -> None:
    assert csvspec.is_valid_image_name(name) is valid


# --- 部位の所見状態の導出 ----------------------------------------------------


def test_no_finding_confirms_every_region_is_normal() -> None:
    """`No Finding` は 5 部位すべてが正常であることを論理的に確定させる。

    これが部位レベルの唯一の確定情報であり、表示でも評価でも同じ規則を使う。
    """
    assert csvspec.region_finding_state(["No Finding"]) == csvspec.REGION_STATE_NORMAL


def test_a_finding_leaves_each_region_unknown() -> None:
    """所見があっても、どの部位にあるかはデータセットに記録されていない。

    気胸画像の右上葉は正常かもしれないし、そうでないかもしれない。
    断定してはならない（断定すると部位別ランキングの正誤が判別できなくなる）。
    """
    assert csvspec.region_finding_state(["Pneumothorax"]) == csvspec.REGION_STATE_UNKNOWN
    assert (
        csvspec.region_finding_state(["Pneumothorax", "Effusion"])
        == csvspec.REGION_STATE_UNKNOWN
    )


def test_empty_labels_are_unknown_not_normal() -> None:
    """ラベルが付いていないことと「所見なし」は別である。"""
    assert csvspec.region_finding_state([]) == csvspec.REGION_STATE_UNKNOWN


def test_state_accepts_any_iterable() -> None:
    assert csvspec.region_finding_state({"No Finding"}) == csvspec.REGION_STATE_NORMAL
    assert csvspec.region_finding_state(iter(["No Finding"])) == csvspec.REGION_STATE_NORMAL
