"""自動品質検査 QC-01〜QC-10 と ROI 信頼度（設計書 7 章）。

自動検査は「技術的失敗の検出」に限定する。閾値外の画像は削除せず
`MANUAL_REVIEW_REQUIRED` として人の確認へ回す。病的画像を異常値として
除外しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

STATUS_AUTO_ACCEPT = "AUTO_ACCEPT_CANDIDATE"
STATUS_MANUAL_REVIEW = "MANUAL_REVIEW_REQUIRED"
STATUS_REJECTED = "REJECTED"

#: 部位ごとの期待面積比（同側肺に対する割合）の既定値。
#: partition.mode により配置が変わるため、profile の
#: quality.region_expected_share があればそちらを優先する。
_DEFAULT_EXPECTED_SHARE = {
    "RUL": 1 / 3,
    "RML": 1 / 3,
    "RLL": 1 / 3,
    "LUL": 2 / 3,
    "LLL": 1 / 3,
}

REGION_CODES = ("RUL", "RML", "RLL", "LUL", "LLL")


def _expected_share(profile) -> dict[str, float]:
    configured = profile.section("quality").get("region_expected_share")
    if not configured:
        return dict(_DEFAULT_EXPECTED_SHARE)
    return {code: float(configured[code]) for code in REGION_CODES}


@dataclass
class QualityCheck:
    id: str
    name: str
    passed: bool
    code: str | None
    value: Any = None


@dataclass
class QualityReport:
    checks: list[QualityCheck] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    lung_metrics: dict[str, Any] = field(default_factory=dict)
    region_confidence: dict[str, float] = field(default_factory=dict)
    region_status: dict[str, str] = field(default_factory=dict)
    overall_status: str = STATUS_MANUAL_REVIEW
    base_confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "flags": self.flags,
            "base_confidence": self.base_confidence,
            "checks": [
                {"id": c.id, "name": c.name, "passed": c.passed, "code": c.code, "value": c.value}
                for c in self.checks
            ],
            "lung_metrics": self.lung_metrics,
        }


def _range_score(value: float, low: float, high: float, *, softness: float = 0.35) -> float:
    """範囲内なら 1.0、外れるほど滑らかに 0 へ近づけたスコア。"""
    if low <= value <= high:
        return 1.0
    span = max(high - low, 1e-6)
    distance = (low - value) if value < low else (value - high)
    return float(max(0.0, 1.0 - (distance / (span * softness + 1e-6)) * 0.5))


def evaluate_quality(
    *,
    gray: np.ndarray,
    body_bbox: list[int],
    lung_mask: np.ndarray,
    right_mask: np.ndarray,
    left_mask: np.ndarray,
    spine: np.ndarray,
    spine_continuity: float,
    regions: dict[str, Any],
    profile,
    polarity_inverted: bool,
) -> QualityReport:
    cfg = profile.section("quality")
    expected_share = _expected_share(profile)
    report = QualityReport()
    height, width = gray.shape
    image_area = float(height * width)

    right_area = float((right_mask > 0).sum())
    left_area = float((left_mask > 0).sum())
    right_ratio = right_area / image_area
    left_ratio = left_area / image_area

    # QC-01 画像極性
    band = gray[int(height * 0.30) : int(height * 0.60), :]
    central = float(band[:, int(width * 0.42) : int(width * 0.58)].mean()) if band.size else 0.0
    lateral_pixels = gray[(lung_mask > 0)]
    lateral = float(lateral_pixels.mean()) if lateral_pixels.size else 0.0
    polarity_ok = lateral < central
    report.checks.append(
        QualityCheck(
            "QC-01",
            "画像極性",
            polarity_ok,
            None if polarity_ok else "POLARITY_UNCERTAIN",
            {"lung_mean": round(lateral, 2), "central_mean": round(central, 2),
             "auto_inverted": polarity_inverted},
        )
    )

    # QC-02 肺数（左右に主要連結成分が各 1 個）
    lungs_present = right_area > 0 and left_area > 0
    report.checks.append(
        QualityCheck(
            "QC-02",
            "肺数",
            lungs_present,
            None if lungs_present else "LUNG_COMPONENT_INVALID",
            {"right_px": int(right_area), "left_px": int(left_area)},
        )
    )

    # QC-03 肺面積
    area_low, area_high = (float(v) for v in cfg["lung_area_ratio"])
    area_ok = area_low <= right_ratio <= area_high and area_low <= left_ratio <= area_high
    report.checks.append(
        QualityCheck(
            "QC-03",
            "肺面積",
            area_ok,
            None if area_ok else "LUNG_AREA_OUTLIER",
            {"right_ratio": round(right_ratio, 4), "left_ratio": round(left_ratio, 4)},
        )
    )

    # QC-04 左右面積比
    sym_low, sym_high = (float(v) for v in cfg["lung_symmetry_ratio"])
    symmetry = right_area / left_area if left_area > 0 else 0.0
    symmetry_ok = sym_low <= symmetry <= sym_high
    report.checks.append(
        QualityCheck(
            "QC-04",
            "左右面積比",
            symmetry_ok,
            None if symmetry_ok else "LUNG_ASYMMETRY",
            round(symmetry, 3),
        )
    )

    # QC-05 脊柱位置（body 幅に対する相対位置）
    bx0, _, bx1, _ = body_bbox
    body_width = max(bx1 - bx0, 1)
    spine_rel = float((spine.mean() - bx0) / body_width)
    center_low, center_high = (float(v) for v in cfg["spine_center_ratio"])
    spine_ok = center_low <= spine_rel <= center_high
    report.checks.append(
        QualityCheck(
            "QC-05",
            "脊柱位置",
            spine_ok,
            None if spine_ok else "SPINE_OFF_CENTER",
            round(spine_rel, 3),
        )
    )

    # QC-06 脊柱連続性
    continuity_min = float(profile.section("spine")["min_continuity_ratio"])
    continuity_ok = spine_continuity >= continuity_min
    report.checks.append(
        QualityCheck(
            "QC-06",
            "脊柱連続性",
            continuity_ok,
            None if continuity_ok else "SPINE_LOW_CONTINUITY",
            round(float(spine_continuity), 3),
        )
    )

    # QC-07 左右分離（centroid が中心線の反対側にあるか）
    right_centroid = _centroid_x(right_mask)
    left_centroid = _centroid_x(left_mask)
    spine_mean = float(spine.mean())
    laterality_ok = (
        right_centroid is not None
        and left_centroid is not None
        and right_centroid < spine_mean < left_centroid
    )
    report.checks.append(
        QualityCheck(
            "QC-07",
            "左右分離",
            laterality_ok,
            None if laterality_ok else "LATERALITY_INVALID",
            {"right_x": right_centroid, "left_x": left_centroid, "spine_x": round(spine_mean, 1)},
        )
    )

    # QC-08 部位面積
    tolerance = float(cfg["region_area_tolerance"])
    region_area_ok = True
    region_share: dict[str, float] = {}
    for code, region in regions.items():
        side_area = right_area if code.startswith("R") else left_area
        share = region.area_px / side_area if side_area > 0 else 0.0
        region_share[code] = round(share, 4)
        expected = expected_share[code]
        if abs(share - expected) > expected * tolerance + 0.05:
            region_area_ok = False
    report.checks.append(
        QualityCheck(
            "QC-08",
            "部位面積",
            region_area_ok,
            None if region_area_ok else "REGION_AREA_OUTLIER",
            region_share,
        )
    )

    # QC-09 Mask 境界接触
    border_limit = float(cfg["mask_border_touch_ratio"])
    border_values = {
        code: round(float(region.metrics.get("border_touch_ratio", 0.0)), 4)
        for code, region in regions.items()
    }
    border_ok = all(value <= border_limit for value in border_values.values())
    report.checks.append(
        QualityCheck(
            "QC-09",
            "Mask境界",
            border_ok,
            None if border_ok else "MASK_TOUCHES_BORDER",
            border_values,
        )
    )

    # QC-10 Orientation
    # 左右 marker 検出は本 baseline の対象外。位置整合のみを評価し、
    # marker / metadata との突合は上位（取り込み時の view_position 照合）で行う。
    report.checks.append(
        QualityCheck("QC-10", "Orientation", laterality_ok, None if laterality_ok else "ORIENTATION_CONFLICT",
                     {"marker_detection": "NOT_EVALUATED"})
    )

    missing = [code for code in expected_share if code not in regions]
    if missing:
        report.flags.append("REGION_MISSING")

    report.flags.extend(check.code for check in report.checks if check.code)
    report.flags = list(dict.fromkeys(report.flags))

    # --- 信頼度 -------------------------------------------------------------
    base = (
        _range_score(right_ratio, area_low, area_high)
        * _range_score(left_ratio, area_low, area_high)
        * _range_score(symmetry, sym_low, sym_high)
        * _range_score(spine_rel, center_low, center_high)
        * min(1.0, float(spine_continuity) / max(continuity_min, 1e-6))
    )
    if not lungs_present:
        base = 0.0
    if not laterality_ok:
        base *= 0.5
    if not polarity_ok:
        base *= 0.7
    base = float(max(0.0, min(1.0, base)))
    report.base_confidence = round(base, 4)

    min_conf = float(cfg["min_region_confidence"])
    for code, region in regions.items():
        expected = expected_share[code]
        share = region_share.get(code, 0.0)
        share_score = _range_score(
            share, expected * (1 - tolerance) - 0.05, expected * (1 + tolerance) + 0.05
        )
        border_penalty = 1.0 - min(1.0, border_values.get(code, 0.0) / max(border_limit, 1e-6)) * 0.3
        confidence = float(max(0.0, min(1.0, base * share_score * border_penalty)))
        report.region_confidence[code] = round(confidence, 4)
        report.region_status[code] = "OK" if confidence >= min_conf else "ROI_UNCERTAIN"

    for code in missing:
        report.region_confidence[code] = 0.0
        report.region_status[code] = STATUS_REJECTED

    failed = [check for check in report.checks if not check.passed]
    if not lungs_present or len(regions) == 0:
        report.overall_status = STATUS_REJECTED
    elif failed or missing:
        report.overall_status = STATUS_MANUAL_REVIEW
    else:
        report.overall_status = STATUS_AUTO_ACCEPT

    report.lung_metrics = {
        "right_area_ratio": round(right_ratio, 4),
        "left_area_ratio": round(left_ratio, 4),
        "symmetry_ratio": round(symmetry, 3),
        "spine_relative_x": round(spine_rel, 3),
        "spine_continuity_ratio": round(float(spine_continuity), 3),
        "region_share": region_share,
    }
    return report


def _centroid_x(mask: np.ndarray) -> float | None:
    xs = np.nonzero(mask > 0)[1]
    if xs.size == 0:
        return None
    return round(float(xs.mean()), 1)
