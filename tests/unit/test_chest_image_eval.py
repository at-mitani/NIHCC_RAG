"""胸部野 mask と、画像単位の教師あり評価（scripts/chest_image_eval.py）を固定する。

この評価の数字が、胸部野 1 単位の検索の精度の根拠になる。
ハーネスが信号を拾えること（陽性対照）と、信号の無いところで拾わないこと
（陰性対照）の両方を確かめる。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from image_pipeline import chest_mask
from nihcc_rag import mil

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "chest_image_eval", ROOT / "scripts" / "chest_image_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["chest_image_eval"] = module
    spec.loader.exec_module(module)
    return module


cie = _load()


# --- 胸部野 mask ---------------------------------------------------------------


def _lungs(side: int = 200):
    """左右の肺を縦長の矩形で置く。間（x=80..120）が縦隔にあたる。"""
    right = np.zeros((side, side), dtype=np.uint8)
    left = np.zeros((side, side), dtype=np.uint8)
    right[40:140, 30:80] = 255
    left[40:140, 120:170] = 255
    return right, left


def test_mediastinum_between_the_lungs_is_included() -> None:
    right, left = _lungs()
    mask = chest_mask(right, left, margin_ratio=0.0, diaphragm_ratio=0.0)
    assert (mask[40:140, 80:120] > 0).all()


def test_band_is_added_below_but_not_above() -> None:
    right, left = _lungs()
    mask = chest_mask(right, left, margin_ratio=0.0, diaphragm_ratio=0.10)
    # 肺野の高さ 100 の 10% = 10 画素だけ下へ伸びる
    assert (mask[140:150, 30:170] > 0).all()
    assert not (mask[151:, :] > 0).any()
    assert not (mask[:40, :] > 0).any()


def test_heart_below_the_lungs_is_included() -> None:
    right, left = _lungs()
    heart = np.zeros_like(right)
    heart[120:160, 90:150] = 255
    mask = chest_mask(right, left, heart, margin_ratio=0.0, diaphragm_ratio=0.0)
    assert (mask[155, 95:145] > 0).all()


def test_margin_widens_laterally() -> None:
    right, left = _lungs()
    mask = chest_mask(right, left, margin_ratio=0.05, diaphragm_ratio=0.0)
    # 肺野の幅 140 の 5% = 7 画素
    assert (mask[90, 24:30] > 0).all()
    assert not (mask[90, :20] > 0).any()


def test_single_lung_still_gives_a_mask() -> None:
    """片肺の虚脱でも落とさない（気胸・大量胸水で起きる）。"""
    right, left = _lungs()
    mask = chest_mask(right, np.zeros_like(left))
    assert (mask > 0).sum() >= (right > 0).sum()


def test_empty_lungs_raise() -> None:
    empty = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError):
        chest_mask(empty, empty)


def test_masked_square_crop_blanks_outside_and_is_square() -> None:
    gray = np.full((100, 60), 200, dtype=np.uint8)
    mask = np.zeros_like(gray)
    mask[10:90, 10:50] = 255
    mask[10:20, 10:20] = 0
    crop = cie.masked_square_crop(gray, mask, 64)
    assert crop.shape == (64, 64)
    assert crop[0, 0] == 0             # 正方形に足した余白
    assert crop[32, 32] == 200         # mask の中


# --- 学習と評価 ----------------------------------------------------------------


def _args(**overrides) -> SimpleNamespace:
    base = dict(
        seed=1, epochs=40, batch=256, lr=0.01, l2=1e-4, class_weight_power=0.5,
        class_weight_max=10.0, patience=5, log_every=100, top_k=10, min_queries=5,
        max_queries=200, bootstrap=20,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _synthetic(n: int, d: int, signal: float, seed: int = 0):
    """所見 f の陽性で特徴 f が signal だけずれるデータ。signal=0 なら無相関。"""
    rng = np.random.default_rng(seed)
    k = len(cie.FINDINGS)
    y = (rng.random((n, k)) < 0.15).astype(np.float64)
    x = rng.normal(size=(n, d)).astype(np.float32)
    x[:, :k] += signal * y.astype(np.float32)
    rows = np.arange(n)
    idx = {"train": rows[: n * 7 // 10], "val": rows[n * 7 // 10 : n * 8 // 10],
           "test": rows[n * 8 // 10 :]}
    return x, y, idx


def _test_auc(x, y, idx, args) -> float:
    w, b, _ = cie.train_head(x, y, idx, args, log=lambda *_: None)
    probs = cie.predict(x[idx["test"]], w, b)
    return float(np.nanmean([mil.roc_auc(probs[:, f], y[idx["test"], f])
                             for f in range(y.shape[1])]))


def test_head_learns_a_planted_signal() -> None:
    x, y, idx = _synthetic(3000, 32, signal=2.0)
    assert _test_auc(x, y, idx, _args()) > 0.85


def test_head_finds_nothing_in_noise() -> None:
    x, y, idx = _synthetic(3000, 32, signal=0.0)
    assert abs(_test_auc(x, y, idx, _args()) - 0.5) < 0.06


def test_class_weights_favour_rare_findings() -> None:
    y = np.zeros((100, 2))
    y[:50, 0] = 1
    y[:5, 1] = 1
    w = cie.class_weights(y, power=0.5, cap=10.0)
    assert w[1] > w[0]
    assert w.max() <= 10.0


def _retrieval(signal: float) -> dict:
    x, y, idx = _synthetic(2000, 32, signal=signal, seed=3)
    embed = cie.l2(x)
    index = np.concatenate([idx["train"], idx["val"]])
    test = idx["test"]
    patients = np.array([f"p{i}" for i in range(len(x))])
    return cie.retrieval_test_b(embed[test], embed[index], y[test], y[index], patients[test],
                                _args())


def test_retrieval_lift_detects_signal() -> None:
    result = _retrieval(signal=4.0)
    lifts = [v["lift"] for k, v in result.items() if k != "No Finding"]
    assert lifts and np.mean(lifts) > 2.0


def test_retrieval_lift_is_near_one_without_signal() -> None:
    result = _retrieval(signal=0.0)
    lifts = [v["lift"] for k, v in result.items() if k != "No Finding"]
    assert lifts and abs(np.mean(lifts) - 1.0) < 0.3


class _FlakyEncoder:
    """最初の `failures` 回は接続を切られる encoder。"""

    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def encode(self, crops):
        import httpx

        self.calls += 1
        if self.calls <= self.failures:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return np.ones((len(crops), 4), dtype=np.float32)


def test_crop_encoding_survives_a_service_restart(monkeypatch) -> None:
    """全量の途中で embedding-service が切断しても、投げ直して続けること。"""
    monkeypatch.setattr(cie.rse, "POOL_RETRY_WAIT", 0.0)
    encoder = _FlakyEncoder(failures=2)
    out = cie.encode_with_retry(encoder, [b"a", b"b"], "crop-chest")
    assert out.shape == (2, 4)
    assert encoder.calls == 3


def test_crop_encoding_gives_up_with_a_resume_hint(monkeypatch) -> None:
    monkeypatch.setattr(cie.rse, "POOL_RETRY_WAIT", 0.0)
    encoder = _FlakyEncoder(failures=99)
    with pytest.raises(RuntimeError, match="続きから"):
        cie.encode_with_retry(encoder, [b"a"], "crop-chest")
    assert encoder.calls == cie.rse.POOL_RETRIES


def test_every_variant_is_known_to_the_contract() -> None:
    """pooling で送る mask 名が契約で許されていること（422 で全量が止まらないように）。"""
    from typing import get_args

    from nihcc_rag.contract import PoolMaskCode

    assert set(cie.POOL_VARIANTS.values()) <= set(get_args(PoolMaskCode))


# --- verify-ingest（本番の取り込み経路と評価の一致） --------------------------


def _inputs(pool=b"P", mask=b"M", crop=b"C"):
    from image_pipeline import ChestModelInputs

    return ChestModelInputs(pool_image=pool, pool_masks={"CHEST": mask}, crop=crop)


def _roi(pool=b"P", mask=b"M", crop=b"C"):
    import base64

    enc = lambda b: base64.b64encode(b).decode()  # noqa: E731
    return {"pool_image_base64": enc(pool), "pool_mask_base64": enc(mask), "crop_base64": enc(crop)}


def test_identical_inputs_have_no_mismatch() -> None:
    assert cie.compare_inputs(_inputs(), _roi()) == []


def test_a_single_byte_difference_is_reported() -> None:
    """1 バイトでも違えば不一致（分類器は特徴のずれを黙って受け入れるため）。"""
    assert cie.compare_inputs(_inputs(), _roi(mask=b"N")) == ["pool_mask"]


def test_probability_agreement_is_measured() -> None:
    p = np.array([[0.1, 0.8, 0.3], [0.6, 0.2, 0.05]])
    diff, cos = cie.compare_probabilities(p, p)
    assert diff == 0.0 and cos == pytest.approx(1.0)
    diff, cos = cie.compare_probabilities(p, p + np.array([[0.0, 0.05, 0.0], [0.0, 0.0, 0.0]]))
    assert diff == pytest.approx(0.05)
    assert cos < 1.0
