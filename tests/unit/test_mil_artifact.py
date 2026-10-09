"""MIL 成果物（学習 → 本番推論の受け渡し）の単体テスト。

**最重要は「学習時と本番推論が同じ値を出すこと」**である。ここがずれると、
評価で測った部位レベル正答率（region_mil_stage_a.py）がそのまま本番へ
乗らない。しかも壊れ方が静かで、検索結果を見ても気づけない。

あわせて、部位コードを必要とする設計が守られているかを固定する。部位ごとの
標準化とバイアスは部位を知らないと適用できず、crop の並びだけでは推論できない。
"""

from __future__ import annotations

import numpy as np
import pytest
from nihcc_rag import mil

REGIONS = ("RUL", "RML", "RLL", "LUL", "LLL")
FINDINGS = ("Atelectasis", "Effusion", "Nodule")


def _corpus(seed: int = 0, n: int = 40, d: int = 12):
    """部位ごとに分布がずれた特徴（本番と同じく L2 正規化済み）。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, len(REGIONS), d))
    # 部位ごとに平均をずらす。実データの RUL 偏りを模す
    for r in range(len(REGIONS)):
        x[:, r] += (r + 1) * 0.7
    x /= np.linalg.norm(x, axis=2, keepdims=True)
    present = np.ones((n, len(REGIONS)))
    present[0, 1] = 0.0
    return x, present


def _artifact(x, present, whitened: bool, seed: int = 1) -> tuple[mil.MilArtifact, mil.MilHead]:
    train_idx = np.arange(x.shape[0])
    mean, std = (
        mil.region_stats(x, present, train_idx)
        if whitened
        else (np.zeros((len(REGIONS), x.shape[2])), np.ones((len(REGIONS), x.shape[2])))
    )
    head = mil.MilHead(
        dim=x.shape[2], n_regions=len(REGIONS), n_findings=len(FINDINGS), tau=4.0, seed=seed
    )
    head.W = np.random.default_rng(seed).normal(0, 0.3, head.W.shape)
    head.b = np.random.default_rng(seed + 1).normal(0, 0.2, head.b.shape)
    art = mil.MilArtifact(
        weights=head.W,
        bias=head.b,
        whiten_mean=mean,
        whiten_std=std,
        regions=REGIONS,
        findings=FINDINGS,
        kappa=1.0,
        tau=4.0,
        backbone_version="test-backbone",
        whitened=whitened,
    )
    return art, head


# --- 学習側と本番側の一致 ----------------------------------------------------


@pytest.mark.parametrize("whitened", [False, True])
def test_artifact_matches_training_time_inference(whitened: bool) -> None:
    """成果物経由の推論が、学習時と同じ確率・同じベクトルを出すこと。

    学習側は [n, r, d] の一括計算、本番側は [n, d] + 部位コードの列で計算する。
    経路が違うので、一致は自明ではない。
    """
    x, present = _corpus()
    art, head = _artifact(x, present, whitened)

    # 学習側の経路
    x_train = x.copy()
    if whitened:
        for r in range(len(REGIONS)):
            rows = np.nonzero(present[:, r] > 0)[0]
            x_train[rows, r] = mil.apply_whiten(
                x[rows, r], np.zeros(len(rows), dtype=int),
                art.whiten_mean[r : r + 1], art.whiten_std[r : r + 1],
            )
    expected = mil.profile_vectors(head.region_probs(x_train), art.kappa)

    # 本番側の経路（部位ごとに平坦化して渡す）
    for r, code in enumerate(REGIONS):
        rows = np.nonzero(present[:, r] > 0)[0]
        got = art.encode(x[rows, r], [code] * len(rows))
        assert np.allclose(got, expected[rows, r], atol=1e-10), f"{code} で不一致"


def test_whitening_changes_the_result() -> None:
    """標準化の有無で結果が変わること。

    成果物に統計を同梱していないと、本番で標準化が抜けても気づけない。
    実データではこの差が部位レベル正答率 34.1% と 51.0% の差になっている。
    """
    x, present = _corpus()
    with_w, _ = _artifact(x, present, whitened=True)
    without_w, _ = _artifact(x, present, whitened=False)
    a = with_w.encode(x[:, 0], ["RUL"] * x.shape[0])
    b = without_w.encode(x[:, 0], ["RUL"] * x.shape[0])
    assert not np.allclose(a, b)


# --- 保存と読み込み ----------------------------------------------------------


def test_save_and_load_roundtrip(tmp_path) -> None:
    x, present = _corpus()
    art, _ = _artifact(x, present, whitened=True)
    path = tmp_path / "mil.npz"
    digest = art.save(path)
    loaded = mil.MilArtifact.load(path)

    assert len(digest) == 64
    assert loaded.regions == REGIONS
    assert loaded.findings == FINDINGS
    assert loaded.whitened is True
    assert loaded.backbone_version == "test-backbone"
    codes = ["RUL", "RLL", "LLL"]
    assert np.allclose(loaded.encode(x[:3, 0], codes), art.encode(x[:3, 0], codes))


def test_sha256_is_stable_for_the_same_file(tmp_path) -> None:
    x, present = _corpus()
    art, _ = _artifact(x, present, whitened=True)
    path = tmp_path / "mil.npz"
    first = art.save(path)
    assert art.sha256(path) == first


# --- 出力ベクトルの性質 ------------------------------------------------------


def test_output_dimension_is_findings_plus_normal() -> None:
    x, present = _corpus()
    art, _ = _artifact(x, present, whitened=True)
    assert art.dimension == len(FINDINGS) + 1
    assert art.encode(x[:5, 0], ["RUL"] * 5).shape == (5, len(FINDINGS) + 1)


def test_output_is_l2_normalised() -> None:
    x, present = _corpus()
    art, _ = _artifact(x, present, whitened=True)
    v = art.encode(x[:, 2], ["RLL"] * x.shape[0])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_normal_regions_are_similar_to_each_other() -> None:
    """所見確率が全て低い部位どうしは高い cosine になること。

    これが「正常らしさ」の次元を足した理由である。確率だけだと正常部位は
    零ベクトルに潰れ、正常どうしの類似度が定義できない。
    """
    probs_normal = np.array([[0.01, 0.02, 0.01], [0.02, 0.01, 0.02]])
    probs_finding = np.array([[0.9, 0.02, 0.01]])
    v_norm = mil.profile_vectors(probs_normal, kappa=1.0)
    v_find = mil.profile_vectors(probs_finding, kappa=1.0)
    assert float(v_norm[0] @ v_norm[1]) > 0.99
    assert float(v_norm[0] @ v_find[0]) < 0.5


# --- 部位コードの扱い --------------------------------------------------------


def test_region_codes_are_required_and_validated() -> None:
    x, present = _corpus()
    art, _ = _artifact(x, present, whitened=True)
    with pytest.raises(ValueError, match="一致しない"):
        art.encode(x[:3, 0], ["RUL", "RLL"])
    with pytest.raises(ValueError, match="未知の部位コード"):
        art.encode(x[:1, 0], ["NOPE"])


def test_same_feature_gives_different_result_per_region() -> None:
    """同じ特徴でも部位が違えば結果が変わること。

    部位ごとの標準化とバイアスが効いている証拠。ここが同じになるなら、
    部位コードを渡す意味が無くなっている。
    """
    x, present = _corpus()
    art, _ = _artifact(x, present, whitened=True)
    one = x[:1, 0]
    assert not np.allclose(art.encode(one, ["RUL"]), art.encode(one, ["LLL"]))
