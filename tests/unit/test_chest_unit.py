"""胸部野 1 単位の検索の部品を固定する。

いちばん守りたいのは、**評価で学習した分類器の確率が、本番でも同じ計算で出ること**
である。特徴の作り方や計算が 1 つでも違うと、分類器は例外を出さずに確率だけがずれる。

  - ChestHead の推論が、評価スクリプトの predict / profile と一致する
  - 成果物は特徴の作り方を持ち、違う構成では読み込みを拒む（未固定の版も拒む）
  - 書き出しスクリプトが評価の head をそのまま運ぶ
  - 取り込みワーカーが roi-service と分類器の申告の食い違いを見つける
  - 胸部野の品質判定とモデル入力の作り方
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from image_pipeline import chest as chest_spec
from image_pipeline.chest import ChestExtraction, _assess, chest_model_inputs
from image_pipeline.config import load_profile
from nihcc_rag import mil
from nihcc_rag.chest_head import UNPINNED, ChestHead, ChestHeadMismatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _services import REPO, load  # noqa: E402

FINDINGS = ("Atelectasis", "Cardiomegaly", "Effusion")
REVISION = "a" * 40


def head(revision: str = REVISION, d: int = 6, seed: int = 0) -> ChestHead:
    rng = np.random.default_rng(seed)
    return ChestHead(
        mean=rng.normal(size=d), std=rng.uniform(0.5, 2.0, size=d),
        weights=rng.normal(size=(d, len(FINDINGS))), bias=rng.normal(size=len(FINDINGS)),
        findings=FINDINGS, kappa=1.0,
        backbone_model="microsoft/rad-dino", backbone_revision=revision,
        pool_side=chest_spec.POOL_SIDE, diaphragm_ratio=chest_spec.DIAPHRAGM_RATIO,
        margin_ratio=chest_spec.MARGIN_RATIO, roi_profile="nihcc-unet-local-v1",
        train_images=100, split_seed=1, test_auc_mean=0.8,
    )


# --- 分類器の推論は評価と同じ計算 ---------------------------------------------


def test_probabilities_match_the_evaluation_script() -> None:
    """評価（chest_image_eval.py）の standardise -> predict と同じ値を出すこと。"""
    cie = load("chest_image_eval_for_head", "scripts/chest_image_eval.py")
    h = head()
    features = np.random.default_rng(1).normal(size=(5, 6)).astype(np.float32)
    x = ((features - h.mean) / h.std).astype(np.float32)  # 評価の standardise
    expected = cie.predict(x, h.weights, h.bias)
    assert np.allclose(h.probabilities(features), expected, atol=1e-6)


def test_profile_is_the_evaluation_profile() -> None:
    h = head()
    features = np.random.default_rng(2).normal(size=(4, 6))
    profiles, probs = h.encode(features)
    assert np.allclose(profiles, mil.profile_vectors(probs.astype(np.float64), 1.0), atol=1e-6)
    assert profiles.shape == (4, len(FINDINGS) + 1)
    assert np.allclose(np.linalg.norm(profiles, axis=1), 1.0, atol=1e-5)


def test_wrong_feature_width_is_rejected() -> None:
    with pytest.raises(ValueError):
        head().probabilities(np.zeros((2, 5)))


def test_save_and_load_roundtrip(tmp_path) -> None:
    h = head()
    digest = h.save(tmp_path / "h.npz")
    back = ChestHead.load(tmp_path / "h.npz")
    assert len(digest) == 64
    assert back.findings == FINDINGS
    assert back.backbone_revision == REVISION
    features = np.random.default_rng(3).normal(size=(3, 6))
    assert np.allclose(back.probabilities(features), h.probabilities(features))


# --- 成果物と構成の照合 -------------------------------------------------------


def compatible(h: ChestHead, **overrides) -> None:
    args = dict(backbone_model="microsoft/rad-dino", backbone_revision=REVISION,
                pool_side=chest_spec.POOL_SIDE, diaphragm_ratio=chest_spec.DIAPHRAGM_RATIO,
                margin_ratio=chest_spec.MARGIN_RATIO)
    args.update(overrides)
    h.check_compatible(**args)


def test_matching_configuration_is_accepted() -> None:
    compatible(head())


@pytest.mark.parametrize("override", [
    {"backbone_revision": "b" * 40},
    {"backbone_model": "microsoft/other"},
    {"pool_side": 224},
    {"diaphragm_ratio": 0.2},
    {"margin_ratio": 0.0},
])
def test_any_difference_in_the_features_is_rejected(override) -> None:
    """特徴の作り方が 1 つでも違えば拒む（確率が黙ってずれるのを防ぐ）。"""
    with pytest.raises(ChestHeadMismatch):
        compatible(head(), **override)


def test_unpinned_backbone_is_rejected_by_default() -> None:
    """版が固定されていない構成は、重みが変わっても気づけないので拒む。"""
    with pytest.raises(ChestHeadMismatch, match="固定されていない"):
        compatible(head(), backbone_revision=UNPINNED)
    with pytest.raises(ChestHeadMismatch):
        compatible(head(revision=UNPINNED))


def test_unpinned_can_be_allowed_for_local_trials() -> None:
    compatible(head(revision=UNPINNED), backbone_revision=UNPINNED, allow_unpinned=True)


def test_mask_settings_can_be_left_to_the_ingest_worker() -> None:
    """embedding-service は mask を作らないので、mask の照合は省略できる。"""
    compatible(head(), diaphragm_ratio=None, margin_ratio=None)


# --- 書き出しスクリプト -------------------------------------------------------


def write_eval_outputs(tmp_path: Path, d: int = 6) -> tuple[Path, Path, dict]:
    rng = np.random.default_rng(4)
    blob = {"mean": rng.normal(size=d), "std": rng.uniform(0.5, 2, size=d),
            "w": rng.normal(size=(d, len(FINDINGS))), "b": rng.normal(size=len(FINDINGS))}
    head_path = tmp_path / "head.npz"
    np.savez(head_path, **blob, findings=np.array(FINDINGS, dtype=object), kappa=1.0)
    train = {"findings": list(FINDINGS), "split": {"train": 70, "val": 10, "test": 20},
             "variants": {"pool-chest": {"dimension": d, "image_auc_mean": 0.833}}}
    train_path = tmp_path / "train.json"
    train_path.write_text(json.dumps(train), encoding="utf-8")
    return head_path, train_path, blob


def run_export(monkeypatch, *argv) -> int:
    module = load("export_chest_head_script", "scripts/export_chest_head.py")
    monkeypatch.setattr(sys, "argv", ["export_chest_head.py", *argv])
    return module.main()


def test_export_carries_the_evaluation_head(tmp_path, monkeypatch) -> None:
    head_path, train_path, blob = write_eval_outputs(tmp_path)
    out = tmp_path / "out.npz"
    assert run_export(monkeypatch, "--head", str(head_path), "--train-json", str(train_path),
                      "--backbone-revision", REVISION, "--output", str(out)) == 0
    exported = ChestHead.load(out)
    assert np.allclose(exported.weights, blob["w"]) and np.allclose(exported.mean, blob["mean"])
    assert exported.backbone_revision == REVISION
    assert exported.pool_side == chest_spec.POOL_SIDE
    assert exported.train_images == 70
    assert exported.test_auc_mean == pytest.approx(0.833)


def test_export_refuses_an_unpinned_backbone(tmp_path, monkeypatch) -> None:
    head_path, train_path, _ = write_eval_outputs(tmp_path)
    with pytest.raises(SystemExit, match="固定されていません"):
        run_export(monkeypatch, "--head", str(head_path), "--train-json", str(train_path),
                   "--backbone-revision", "unpinned", "--output", str(tmp_path / "o.npz"))


def test_export_refuses_a_short_revision(tmp_path, monkeypatch) -> None:
    head_path, train_path, _ = write_eval_outputs(tmp_path)
    with pytest.raises(SystemExit, match="40 桁"):
        run_export(monkeypatch, "--head", str(head_path), "--train-json", str(train_path),
                   "--backbone-revision", "abc123", "--output", str(tmp_path / "o.npz"))


# --- 取り込みワーカーの起動時照合 ---------------------------------------------


def problems_of(roi, emb):
    from nihcc_rag.chest_head import pipeline_problems

    return pipeline_problems(roi, emb)


def roi_model(**overrides) -> dict:
    spec = {"diaphragm_ratio": chest_spec.DIAPHRAGM_RATIO, "margin_ratio": chest_spec.MARGIN_RATIO,
            "pool_side": chest_spec.POOL_SIDE}
    spec.update(overrides)
    return {"profile_name": "nihcc-unet-local-v1", "chest_mask": spec}


def embedding_model(**overrides) -> dict:
    described = {"diaphragm_ratio": chest_spec.DIAPHRAGM_RATIO,
                 "margin_ratio": chest_spec.MARGIN_RATIO, "pool_side": chest_spec.POOL_SIDE,
                 "roi_profile": "nihcc-unet-local-v1"}
    described.update(overrides)
    return {"chest_head": described}


def test_matching_pipeline_has_no_problems() -> None:
    assert problems_of(roi_model(), embedding_model()) == []


def test_mask_difference_is_found() -> None:
    problems = problems_of(roi_model(diaphragm_ratio=0.2), embedding_model())
    assert any("diaphragm_ratio" in p for p in problems)


def test_profile_difference_is_found() -> None:
    problems = problems_of(
        roi_model(), embedding_model(roi_profile="nihcc-unet-v1"))
    assert any("profile" in p for p in problems)


def test_old_roi_service_is_found() -> None:
    """chest_mask を申告しない roi-service（古い image）は照合できないので問題にする。"""
    problems = problems_of({"profile_name": "x"}, embedding_model())
    assert problems


def test_single_stage_configuration_is_not_checked() -> None:
    """2 段目が分類器でない構成では、照合するものが無い。"""
    assert problems_of(roi_model(), {"chest_head": None}) == []


# --- 胸部野の品質判定とモデル入力 ---------------------------------------------


def masks(right_area: int, left_area: int, side: int = 100):
    right = np.zeros((side, side), dtype=np.uint8)
    left = np.zeros((side, side), dtype=np.uint8)
    if right_area:
        right[10:10 + right_area // 20, 10:30] = 255
    if left_area:
        left[10:10 + left_area // 20, 60:80] = 255
    return right, left


def test_balanced_lungs_have_no_flags() -> None:
    right, left = masks(800, 800)
    assert _assess(right, left, 0.16, load_profile("nihcc-unet-local-v1")) == []


def test_one_missing_lung_is_flagged_not_rejected() -> None:
    """片肺の虚脱（気胸・大量胸水）でも取り込みは止めない。旗を立てて ROI_UNCERTAIN。"""
    right, left = masks(800, 0)
    assert _assess(right, left, 0.08, load_profile("nihcc-unet-local-v1")) == ["ONE_LUNG_MISSING"]


def test_strong_asymmetry_is_flagged() -> None:
    right, left = masks(1200, 200)
    assert "LUNG_ASYMMETRY" in _assess(right, left, 0.14, load_profile("nihcc-unet-local-v1"))


def extraction() -> ChestExtraction:
    gray = np.full((120, 100), 128, dtype=np.uint8)
    right, left = masks(800, 800, side=100)
    right = np.vstack([right, np.zeros((20, 100), np.uint8)])
    left = np.vstack([left, np.zeros((20, 100), np.uint8)])
    lung = right | left
    chest = chest_spec.chest_mask(right, left)
    return ChestExtraction(
        profile=load_profile("nihcc-unet-local-v1"), normalized=gray, scale=0.5,
        right=right, left=left, heart=np.zeros_like(lung), lung=lung, chest=chest,
        areas={"lung": 0.1, "chest": 0.2, "heart": 0.0}, quality_status="OK", confidence=1.0,
    )


def test_model_inputs_have_matching_sizes() -> None:
    """pooling の画像と mask は同じ寸法（ずれると部位の位置がずれたまま pooling する）。"""
    from PIL import Image
    import io

    inputs = chest_model_inputs(extraction(), pool_side=64, crop_side=32)
    with Image.open(io.BytesIO(inputs.pool_image)) as image, \
         Image.open(io.BytesIO(inputs.pool_masks["CHEST"])) as mask, \
         Image.open(io.BytesIO(inputs.crop)) as crop:
        assert image.size == mask.size == (64, 64)
        assert crop.size == (32, 32)


def test_only_known_mask_names_are_accepted() -> None:
    with pytest.raises(ValueError):
        chest_model_inputs(extraction(), masks=("HEART",))


def test_source_bbox_is_scaled_back_to_the_original() -> None:
    ex = extraction()
    assert ex.source_bbox == [round(v / 0.5) for v in ex.chest_bbox]
