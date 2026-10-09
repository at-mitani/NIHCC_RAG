"""抽出プロファイルの読み込みと検証（設計書 6.8）。

- API 入力で任意値を直接本番適用させない（許可済み profile 名から選択する）
- 起動時に型・範囲・相互制約を検証する
- YAML 全体の SHA-256 を parameter_hash として保存する
"""

from __future__ import annotations

import functools
import hashlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PROFILE = "nihcc-classical-v1"
PARTITION_MODES = ("ANATOMICAL_LOBES_V1", "EQUAL_THIRDS", "OVERLAP_TWO_THIRDS")


class ProfileError(ValueError):
    """プロファイル定義が不正。"""


@dataclass(frozen=True)
class Profile:
    name: str
    algorithm_version: str
    parameter_hash: str
    data: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        return self.data[name]

    @property
    def partition_rule_version(self) -> str:
        return f"{self.data['partition']['mode']}-v1"

    @property
    def roi_model_version(self) -> str:
        """ROI 生成物の再現キー。algorithm version と parameter hash を含める。"""
        return f"{self.algorithm_version}+{self.parameter_hash[:12]}"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProfileError(message)


def _ratio(value: Any, name: str) -> float:
    _require(isinstance(value, (int, float)), f"{name} must be numeric")
    _require(0.0 <= float(value) <= 1.0, f"{name} must be within 0..1")
    return float(value)


def _validate(data: dict[str, Any]) -> None:
    _require(data.get("schema_version") == 1, "schema_version must be 1")
    for key in (
        "profile_name",
        "algorithm_version",
        "input",
        "normalization",
        "lung_candidate",
        "component_filter",
        "morphology",
        "outline",
        "spine",
        "partition",
        "crop",
        "quality",
    ):
        _require(key in data, f"missing section: {key}")

    lung = data["lung_candidate"]
    _require(
        lung["intensity_mode"] in ("otsu_and_local", "otsu", "fixed"),
        "lung_candidate.intensity_mode is invalid",
    )
    block = int(lung["local_block_size_at_1024"])
    _require(block % 2 == 1 and 31 <= block <= 127, "local_block_size_at_1024 must be odd, 31..127")
    for name in ("allowed_x_ratio", "allowed_y_ratio"):
        low, high = lung[name]
        _ratio(low, name)
        _ratio(high, name)
        _require(low < high, f"{name} must be ascending")

    comp = data["component_filter"]
    _require(comp["connectivity"] in (4, 8), "connectivity must be 4 or 8")
    _require(
        0.0 < float(comp["min_area_ratio"]) < float(comp["max_area_ratio"]) <= 1.0,
        "component area ratios must be ascending within 0..1",
    )

    partition = data["partition"]
    _require(partition["mode"] in PARTITION_MODES, f"partition.mode must be one of {PARTITION_MODES}")
    boundaries = [float(v) for v in partition["right_boundaries"]]
    _require(len(boundaries) == 2, "partition.right_boundaries must have 2 values")
    _require(
        0.0 < boundaries[0] < boundaries[1] < 1.0, "partition.right_boundaries must be ascending"
    )
    _ratio(partition["left_boundary"], "partition.left_boundary")

    if partition["mode"] == "ANATOMICAL_LOBES_V1":
        for side in ("right", "left"):
            _require(side in partition, f"partition.{side} is required for ANATOMICAL_LOBES_V1")
        right = partition["right"]
        horizontal = _ratio(right["horizontal_fissure_u"], "partition.right.horizontal_fissure_u")
        medial = _ratio(right["oblique_medial_u"], "partition.right.oblique_medial_u")
        lateral = _ratio(right["oblique_lateral_u"], "partition.right.oblique_lateral_u")
        # 斜裂は内側ほど下方（値が大きい）。水平裂より下でなければ中葉が消える。
        _require(lateral <= medial, "partition.right.oblique_lateral_u must not exceed oblique_medial_u")
        _require(
            horizontal <= medial,
            "partition.right.horizontal_fissure_u must not exceed oblique_medial_u",
        )
        left = partition["left"]
        left_medial = _ratio(left["oblique_medial_u"], "partition.left.oblique_medial_u")
        left_lateral = _ratio(left["oblique_lateral_u"], "partition.left.oblique_lateral_u")
        _require(
            left_lateral <= left_medial,
            "partition.left.oblique_lateral_u must not exceed oblique_medial_u",
        )
    _ratio(partition.get("row_smooth_ratio", 0.02), "partition.row_smooth_ratio")

    # 省略可。無い場合は規則ベース。
    segmentation = data.get("lung_segmentation")
    if segmentation is not None:
        backend = segmentation.get("backend", "classical")
        _require(backend in ("classical", "unet"),
                 "lung_segmentation.backend must be classical or unet")
        if backend == "unet":
            for key in ("model_path", "input_size"):
                _require(key in segmentation, f"lung_segmentation.{key} is required for unet")
            size = int(segmentation["input_size"])
            # U-Net は 4 段の pooling を通すので 16 の倍数でないと形が合わない
            _require(64 <= size <= 1024 and size % 16 == 0,
                     "lung_segmentation.input_size must be a multiple of 16 within 64..1024")
            _ratio(segmentation.get("threshold", 0.5), "lung_segmentation.threshold")

    crop = data["crop"]
    _require(crop["embedding_crop"] in ("context", "masked"), "crop.embedding_crop is invalid")
    _ratio(crop["margin_ratio"], "crop.margin_ratio")

    quality = data["quality"]
    for name in ("lung_area_ratio", "spine_center_ratio"):
        low, high = quality[name]
        _require(0.0 <= float(low) < float(high) <= 1.0, f"quality.{name} must be ascending 0..1")
    low, high = quality["lung_symmetry_ratio"]
    _require(0.0 < float(low) < 1.0 < float(high), "quality.lung_symmetry_ratio must bracket 1.0")


def _load_text(profile_name: str, search_dir: Path | None) -> str:
    if search_dir is not None:
        candidate = search_dir / f"{profile_name}.yaml"
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    package_file = resources.files("image_pipeline.profiles").joinpath(f"{profile_name}.yaml")
    if not package_file.is_file():
        raise ProfileError(f"unknown extraction profile: {profile_name}")
    return package_file.read_text(encoding="utf-8")


@functools.lru_cache(maxsize=8)
def load_profile(profile_name: str = DEFAULT_PROFILE, profile_dir: str | None = None) -> Profile:
    if not profile_name.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise ProfileError("profile name contains unsupported characters")
    text = _load_text(profile_name, Path(profile_dir) if profile_dir else None)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ProfileError("profile must be a mapping")
    _validate(data)
    if data["profile_name"] != profile_name:
        raise ProfileError("profile_name inside the file does not match the requested name")
    return Profile(
        name=profile_name,
        algorithm_version=str(data["algorithm_version"]),
        parameter_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        data=data,
    )


def available_profiles(profile_dir: str | None = None) -> list[str]:
    names = {
        path.name.removesuffix(".yaml")
        for path in resources.files("image_pipeline.profiles").iterdir()
        if path.name.endswith(".yaml")
    }
    if profile_dir:
        directory = Path(profile_dir)
        if directory.is_dir():
            names.update(p.stem for p in directory.glob("*.yaml"))
    return sorted(names)
