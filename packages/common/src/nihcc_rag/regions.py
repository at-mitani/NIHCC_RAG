"""肺部位の定義。設計書 3.2 の 5 論理部位。"""

from __future__ import annotations

from dataclasses import dataclass

REGION_CODES: tuple[str, ...] = ("RUL", "RML", "RLL", "LUL", "LLL")


@dataclass(frozen=True)
class Region:
    code: str
    display_name: str
    laterality: str  # RIGHT | LEFT  （患者側）
    segments: tuple[str, ...]
    #: 肺 mask 内の正規化高さ u の範囲（EQUAL_THIRDS）
    u_min: float
    u_max: float
    #: 最下部位のみ u_max を含む（境界画素の二重所属防止）
    include_upper_bound: bool
    #: 画面・overlay で部位を識別するための表示色（R, G, B）。
    #: グレースケール画像上で区別できるよう彩度の高い色を選ぶ。
    color: tuple[int, int, int]

    @property
    def database(self) -> str:
        return f"vector_{self.code.lower()}"

    @property
    def coordinator_host(self) -> str:
        return f"{self.code.lower()}-coordinator"


_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0

REGIONS: dict[str, Region] = {
    "RUL": Region(
        "RUL", "右肺上葉", "RIGHT", ("S1", "S2", "S3"), 0.0, _THIRD, False, (0, 122, 255)
    ),
    "RML": Region(
        "RML", "右肺中葉", "RIGHT", ("S4", "S5"), _THIRD, _TWO_THIRDS, False, (255, 176, 0)
    ),
    # 参照図に合わせ、正面像で心陰影に重なる S7（内側肺底区）は表記しない
    "RLL": Region(
        "RLL", "右肺下葉", "RIGHT", ("S6", "S8", "S9", "S10"), _TWO_THIRDS, 1.0, True,
        (255, 87, 34),
    ),
    # 左肺は S1 と S2 が癒合するため S1+2。S4・S5 は舌区で上葉に属する
    "LUL": Region(
        "LUL", "左肺上葉", "LEFT", ("S1+2", "S3", "S4", "S5"), 0.0, _TWO_THIRDS, False,
        (0, 176, 80),
    ),
    "LLL": Region(
        "LLL", "左肺下葉", "LEFT", ("S6", "S8", "S9", "S10"), _TWO_THIRDS, 1.0, True,
        (170, 70, 220),
    ),
}


def region(code: str) -> Region:
    try:
        return REGIONS[code]
    except KeyError as exc:  # pragma: no cover - 呼び出し側で 422 に変換する
        raise ValueError(f"unknown region_code: {code}") from exc


def validate_region_codes(codes: list[str]) -> list[str]:
    """重複を除き、REGION_CODES の並び順に正規化する。"""
    unknown = [c for c in codes if c not in REGIONS]
    if unknown:
        raise ValueError(f"unknown region_code: {','.join(unknown)}")
    return [c for c in REGION_CODES if c in set(codes)]
