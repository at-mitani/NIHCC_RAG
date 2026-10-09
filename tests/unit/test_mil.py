"""MIL ヘッド（nihcc_rag.mil）の単体テスト。

勾配は手で導出しているため、**数値微分との一致が最重要**である。ここが狂うと
学習は静かに失敗し、結果を見ても気づけない（損失は下がるが意味の無い解へ行く）。

あわせて、この実装が満たすべき性質を固定する:
  - LSE pooling が tau で平均と max を補間すること
  - 欠損部位が pooling にも損失にも寄与しないこと
  - 確定負例（No Finding）の損失が全部位へ効くこと
"""

from __future__ import annotations

import numpy as np
import pytest
from nihcc_rag import mil


def _batch(seed: int = 0, n: int = 6, r: int = 5, d: int = 7, f: int = 3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, r, d))
    x /= np.linalg.norm(x, axis=2, keepdims=True)  # 本番同様に L2 正規化
    present = np.ones((n, r))
    present[0, 1] = 0.0  # 欠損部位を混ぜる
    present[3, 4] = 0.0
    y = (rng.random((n, f)) < 0.3).astype(float)
    normal = np.zeros(n)
    normal[:2] = 1.0  # No Finding 画像
    y[:2] = 0.0
    return x, present, y, normal


# --- 数値微分との一致 --------------------------------------------------------


@pytest.mark.parametrize("tau", [1.0, 4.0, 12.0])
def test_gradients_match_finite_differences(tau: float) -> None:
    """手で導出した勾配が数値微分と一致すること。

    LSE pooling の softmax 重み、確定負例項、L2 のすべてを含んだ状態で確認する。
    """
    x, present, y, normal = _batch()
    head = mil.MilHead(dim=x.shape[2], n_regions=x.shape[1], n_findings=y.shape[1], tau=tau, seed=1)
    cw = np.array([1.0, 2.5, 0.4])
    kwargs = dict(class_weight=cw, neg_weight=0.7, l2=1e-3)

    loss, dW, db = head.loss_and_grads(x, present, y, normal, **kwargs)

    eps = 1e-6
    for name, param, grad in (("W", head.W, dW), ("b", head.b, db)):
        num = np.zeros_like(param)
        it = np.ndindex(param.shape)
        for idx in it:
            original = param[idx]
            param[idx] = original + eps
            plus, _, _ = head.loss_and_grads(x, present, y, normal, **kwargs)
            param[idx] = original - eps
            minus, _, _ = head.loss_and_grads(x, present, y, normal, **kwargs)
            param[idx] = original
            num[idx] = (plus - minus) / (2 * eps)
        assert np.allclose(grad, num, rtol=1e-4, atol=1e-7), (
            f"{name} の勾配が数値微分と一致しない (tau={tau})"
        )
    assert np.isfinite(loss)


# --- LSE pooling の性質 ------------------------------------------------------


def test_lse_interpolates_between_mean_and_max() -> None:
    """tau -> 0 で平均、tau -> 大 で max に近づくこと。

    平均への収束誤差は O(tau * 分散) なので、tau は分散に見合うだけ小さく取る
    （このデータの分散は約 2.94 で、tau=0.01 だと誤差 0.015 が残る）。
    """
    z = np.array([[[0.0], [1.0], [3.0], [-2.0], [0.5]]])  # [1, 5, 1]
    present = np.ones((1, 5))
    low, _ = mil.lse_pool(z, present, tau=1e-3)
    high, _ = mil.lse_pool(z, present, tau=200.0)
    assert low[0, 0] == pytest.approx(z.mean(), abs=1e-2)
    assert high[0, 0] == pytest.approx(z.max(), abs=1e-2)
    # 単調性: tau を上げると max へ近づく
    mids = [mil.lse_pool(z, present, tau=t)[0][0, 0] for t in (0.5, 2.0, 8.0)]
    assert mids[0] < mids[1] < mids[2] <= z.max()


def test_pooling_weights_sum_to_one_over_present_regions() -> None:
    x, present, _, _ = _batch()
    head = mil.MilHead(dim=x.shape[2], n_regions=5, n_findings=3, seed=2)
    z = head.region_logits(x)
    _, weights = mil.lse_pool(z, present, tau=4.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    # 欠損部位の寄与はゼロ
    assert np.allclose(weights[0, 1, :], 0.0)
    assert np.allclose(weights[3, 4, :], 0.0)


def test_missing_regions_do_not_change_pooled_value() -> None:
    """欠損部位のロジットを動かしても pooling 結果が変わらないこと。"""
    x, present, _, _ = _batch()
    head = mil.MilHead(dim=x.shape[2], n_regions=5, n_findings=3, seed=3)
    z = head.region_logits(x)
    before, _ = mil.lse_pool(z, present, tau=4.0)
    z[0, 1, :] += 50.0  # 欠損部位を極端に動かす
    after, _ = mil.lse_pool(z, present, tau=4.0)
    assert np.allclose(before, after)


# --- 確定負例 ---------------------------------------------------------------


def test_confirmed_negative_loss_applies_to_every_region() -> None:
    """No Finding 画像では全部位が負例として学習されること。

    「No Finding は 5 部位すべてが正常と確定する」という確定情報を
    実装が使えているかの確認。バッグ pooling だけだと勾配が最大部位へ偏る。
    """
    x, present, y, normal = _batch()
    head = mil.MilHead(dim=x.shape[2], n_regions=5, n_findings=3, seed=4)
    cw = np.ones(3)
    _, _, db_without = head.loss_and_grads(x, present, y, normal, cw, neg_weight=0.0, l2=0.0)
    _, _, db_with = head.loss_and_grads(x, present, y, normal, cw, neg_weight=1.0, l2=0.0)
    delta = np.abs(db_with - db_without)
    # 5 部位すべてのバイアスに勾配が届いている
    assert (delta.sum(axis=1) > 1e-9).all()


def test_training_reduces_loss_on_separable_data() -> None:
    """分離可能な合成データで損失が下がること（最適化そのものの健全性）。"""
    rng = np.random.default_rng(5)
    n, r, d, f = 200, 5, 8, 2
    x = rng.normal(size=(n, r, d))
    signal = np.zeros(d)
    signal[0] = 1.0
    y = np.zeros((n, f))
    # 半数の画像で、ある 1 部位だけに信号を入れる
    for i in range(0, n, 2):
        x[i, i % r] += 3.0 * signal
        y[i, 0] = 1.0
    x /= np.linalg.norm(x, axis=2, keepdims=True)
    present = np.ones((n, r))
    normal = (y.sum(axis=1) == 0).astype(float)

    head = mil.MilHead(dim=d, n_regions=r, n_findings=f, tau=4.0, seed=6)
    opt = mil.Adam(lr=0.1)
    cw = np.ones(f)
    first, _, _ = head.loss_and_grads(x, present, y, normal, cw, 0.5, 1e-4)
    for _ in range(300):
        _, dW, db = head.loss_and_grads(x, present, y, normal, cw, 0.5, 1e-4)
        opt.step("W", head.W, dW)
        opt.step("b", head.b, db)
    last, _, _ = head.loss_and_grads(x, present, y, normal, cw, 0.5, 1e-4)
    assert last < first * 0.6, (first, last)

    # 信号を入れた部位が最も高い確率になること（局在が学習されている）
    probs = head.region_probs(x)[:, :, 0]
    hits = sum(int(probs[i].argmax() == i % r) for i in range(0, n, 2))
    assert hits / (n / 2) > 0.8, hits


# --- AUC --------------------------------------------------------------------


def test_roc_auc_matches_known_values() -> None:
    assert mil.roc_auc(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])) == 1.0
    assert mil.roc_auc(np.array([4.0, 3.0, 2.0, 1.0]), np.array([0, 0, 1, 1])) == 0.0
    assert mil.roc_auc(np.array([1.0, 1.0, 1.0, 1.0]), np.array([0, 0, 1, 1])) == 0.5


def test_roc_auc_is_nan_without_both_classes() -> None:
    assert np.isnan(mil.roc_auc(np.array([1.0, 2.0]), np.array([1, 1])))
