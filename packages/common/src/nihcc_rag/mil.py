"""部位レベル所見推定の MIL ヘッド（評価スクリプト用）。

凍結した backbone が出す部位特徴（画像あたり 5 部位 x D 次元）に対し、線形ヘッド
だけを学習する。検索の本番経路（胸部野 1 単位）では使わない。

なぜ numpy か
-------------
モデルは「D -> 14 の線形写像 + 部位ごとのバイアス」でしかない。評価環境に torch が
無くても動くよう、2GB の依存は足さない。勾配は手で導出し、数値微分で検証する
（tests/unit/test_mil.py）。

定式化
------
部位 r・所見 f のロジット:

    z[n,r,f] = x[n,r] . W[:,f] + b[r,f]

W は全部位で共有し、バイアスだけ部位ごとに持つ。共有しないと「どの部位か」を
当てる方向へ逃げやすい（特徴は部位の形に支配される）。

バッグ（画像）レベルは部位方向の LSE pooling:

    Z[n,f] = (1/tau) * log( mean_r exp(tau * z[n,r,f]) )

tau -> 0 で平均、tau -> ∞ で max に一致する。max は勾配が 1 部位にしか流れず
不安定、平均は「少なくとも 1 部位」という MIL の仮定と合わないため、間を取る。

損失は 2 項:

    L_bag = sum_f w[f] * BCE(Z[n,f], y[n,f])          画像単位ラベル
    L_neg = lam * sum_{r,f} BCE(z[n,r,f], 0)          No Finding 画像のみ

L_neg が要点である。`No Finding` は全体の 53.8% を占め、その 5 部位すべてが
全所見について負例と**確定**する。この確定教師を
pooling を通さずインスタンス単位で直接与える。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: 欠損部位（ROI 失敗など）を表す重み。pooling からも損失からも外す。
MISSING = 0.0


def bce_with_logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """数値的に安定な BCE。要素ごとの値を返す。"""
    return np.maximum(logits, 0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def lse_pool(
    z: np.ndarray, present: np.ndarray, tau: float
) -> tuple[np.ndarray, np.ndarray]:
    """部位方向の LSE pooling。(Z[n,f], 各部位の寄与 softmax[n,r,f]) を返す。

    present[n,r] が 0 の部位は除外する（ROI 失敗で部位が無い画像がある）。
    """
    mask = present[:, :, None] > MISSING
    scaled = np.where(mask, tau * z, -np.inf)
    peak = np.max(scaled, axis=1, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    exp = np.where(mask, np.exp(scaled - peak), 0.0)
    total = exp.sum(axis=1, keepdims=True)
    count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
    pooled = (peak + np.log(np.maximum(total, 1e-300)) - np.log(count)) / tau
    # dZ/dz = softmax_r（欠損部位は 0）
    weights = np.where(total > 0, exp / np.maximum(total, 1e-300), 0.0)
    return pooled[:, 0, :], weights


@dataclass
class MilHead:
    """部位共有の線形ヘッド + 部位ごとのバイアス。"""

    dim: int
    n_regions: int
    n_findings: int
    tau: float = 4.0
    seed: int = 0
    W: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        # 入力は L2 正規化済みなので、小さめの初期値で十分
        self.W = rng.normal(0.0, 0.01, (self.dim, self.n_findings))
        self.b = np.zeros((self.n_regions, self.n_findings))

    # --- 前向き ------------------------------------------------------------

    def region_logits(self, x: np.ndarray) -> np.ndarray:
        """z[n,r,f]。x は [n, r, d]。"""
        return x @ self.W + self.b[None, :, :]

    def bag_logits(self, x: np.ndarray, present: np.ndarray) -> np.ndarray:
        z = self.region_logits(x)
        pooled, _ = lse_pool(z, present, self.tau)
        return pooled

    def region_probs(self, x: np.ndarray) -> np.ndarray:
        return sigmoid(self.region_logits(x))

    # --- 損失と勾配 --------------------------------------------------------

    def loss_and_grads(
        self,
        x: np.ndarray,
        present: np.ndarray,
        y: np.ndarray,
        normal: np.ndarray,
        class_weight: np.ndarray,
        neg_weight: float,
        l2: float,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """損失と (dW, db) を返す。

        x            [n, r, d]  部位特徴（L2 正規化済み）
        present      [n, r]     部位の有無（1/0）
        y            [n, f]     画像単位ラベル
        normal       [n]        `No Finding` なら 1
        class_weight [f]        所見ごとの重み（頻度の偏りが 88 倍あるため必須）
        """
        n = x.shape[0]
        z = self.region_logits(x)
        pooled, weights = lse_pool(z, present, self.tau)

        # --- バッグ損失 ---
        bag = bce_with_logits(pooled, y) * class_weight[None, :]
        loss_bag = float(bag.sum() / n)
        # dL/dZ
        d_pooled = (sigmoid(pooled) - y) * class_weight[None, :] / n
        dz = weights * d_pooled[:, None, :]

        # --- 確定負例のインスタンス損失 ---
        loss_neg = 0.0
        n_normal = float(normal.sum())
        if neg_weight > 0 and n_normal > 0:
            live = (present > MISSING).astype(np.float64) * normal[:, None]
            inst = bce_with_logits(z, np.zeros_like(z)) * live[:, :, None]
            denom = max(1.0, float(live.sum()) * self.n_findings)
            loss_neg = float(neg_weight * inst.sum() / denom)
            dz = dz + neg_weight * (sigmoid(z) * live[:, :, None]) / denom

        # --- 勾配 ---
        dW = np.einsum("nrd,nrf->df", x, dz) + 2.0 * l2 * self.W
        db = dz.sum(axis=0)
        loss = loss_bag + loss_neg + l2 * float((self.W**2).sum())
        return loss, dW, db


@dataclass
class Adam:
    """最小限の Adam。パラメータは 2 つだけなので汎用実装は要らない。"""

    lr: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    _state: dict = field(default_factory=dict)

    def step(self, name: str, param: np.ndarray, grad: np.ndarray) -> None:
        m, v, t = self._state.get(name, (np.zeros_like(param), np.zeros_like(param), 0))
        t += 1
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad**2)
        param -= self.lr * (m / (1 - self.beta1**t)) / (np.sqrt(v / (1 - self.beta2**t)) + self.eps)
        self._state[name] = (m, v, t)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U による AUC。同順位は平均順位で扱う。"""
    labels = labels.astype(bool)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# 推論用の成果物（学習側と本番側で共有する）
# --------------------------------------------------------------------------


def region_stats(
    x: np.ndarray, present: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """部位ごとの平均と標準偏差を train の行だけから求める。

    部位ごとに別々に符号化した特徴は、部位の輪郭・面積・黒塗りの形がそのまま
    分布差になる。共有ヘッド z = x . W は「どの部位の特徴が W と揃いやすいか」に
    引きずられ、特定の部位（RUL など）を過剰に予測する。部位ごとに分布を揃えると
    部位レベル正答率が上がる（実測 34.1% -> 51.0%）。

    部位バイアス b[r,f] では代替できない。b はロジットの定数オフセットしか
    動かせず、特徴分布そのものの違いには効かない。
    """
    n_regions, dim = x.shape[1], x.shape[2]
    mean = np.zeros((n_regions, dim), dtype=np.float64)
    std = np.ones((n_regions, dim), dtype=np.float64)
    for r in range(n_regions):
        live = train_idx[present[train_idx, r] > MISSING]
        if len(live) < 2:
            continue
        block = x[live, r].astype(np.float64)
        mean[r] = block.mean(axis=0)
        sd = block.std(axis=0)
        sd[sd < 1e-6] = 1.0
        std[r] = sd
    return mean, std


def apply_whiten(
    features: np.ndarray, region_index: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """特徴を部位ごとの統計で標準化し、L2 正規化し直す。

    features    [n, d]  部位特徴（L2 正規化済み）
    region_index[n]     各行がどの部位か（mean / std の行番号）
    """
    out = (features.astype(np.float64) - mean[region_index]) / std[region_index]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    return out / norm


def profile_vectors(probs: np.ndarray, kappa: float) -> np.ndarray:
    """所見確率 -> 検索用ベクトル（末尾に「正常らしさ」を足して L2 正規化）。

    確率をそのまま使うと、正常な部位はほぼ零ベクトルになり cosine が退化する
    （正常どうしの類似度が定義できない）。「正常らしさ」を明示的な次元として
    持たせることで、正常部位どうし・同一所見どうしがそれぞれ高い類似度になる。

    probs は [..., f]。戻り値は [..., f+1]。
    """
    normal = kappa * (1.0 - probs.max(axis=-1, keepdims=True))
    v = np.concatenate([probs, normal], axis=-1)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-9, 1.0, norm)
    return v / norm


@dataclass
class MilArtifact:
    """学習済み MIL ヘッドと前処理統計。本番の推論はこれ 1 つで完結する。

    入力は backbone が出した部位特徴（L2 正規化済み）と、その部位コード。
    **部位コードが要る**のが要点である。部位ごとの標準化とバイアスの両方が
    部位を知っている必要があり、crop のリストだけでは推論できない。
    """

    weights: np.ndarray  # [d, f]
    bias: np.ndarray  # [r, f]
    whiten_mean: np.ndarray  # [r, d]
    whiten_std: np.ndarray  # [r, d]
    regions: tuple[str, ...]
    findings: tuple[str, ...]
    kappa: float
    tau: float
    backbone_version: str
    whitened: bool

    @property
    def dimension(self) -> int:
        """出力次元（所見数 + 「正常らしさ」1）。"""
        return len(self.findings) + 1

    @property
    def input_dimension(self) -> int:
        return int(self.weights.shape[0])

    def save(self, path) -> str:
        """npz へ保存し、内容の sha256 を返す（model_version の追跡用）。"""
        import hashlib
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights,
            bias=self.bias,
            whiten_mean=self.whiten_mean,
            whiten_std=self.whiten_std,
            regions=np.array(self.regions, dtype=object),
            findings=np.array(self.findings, dtype=object),
            kappa=np.array(self.kappa),
            tau=np.array(self.tau),
            backbone_version=np.array(self.backbone_version, dtype=object),
            whitened=np.array(self.whitened),
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path) -> "MilArtifact":
        from pathlib import Path

        blob = np.load(Path(path), allow_pickle=True)
        return cls(
            weights=blob["weights"],
            bias=blob["bias"],
            whiten_mean=blob["whiten_mean"],
            whiten_std=blob["whiten_std"],
            regions=tuple(str(v) for v in blob["regions"]),
            findings=tuple(str(v) for v in blob["findings"]),
            kappa=float(blob["kappa"]),
            tau=float(blob["tau"]),
            backbone_version=str(blob["backbone_version"]),
            whitened=bool(blob["whitened"]),
        )

    def sha256(self, path) -> str:
        import hashlib
        from pathlib import Path

        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def encode(self, features: np.ndarray, region_codes: list[str]) -> np.ndarray:
        """部位特徴 -> 検索用プロファイル [n, findings+1]。

        features     [n, d]  backbone の出力（L2 正規化済み）
        region_codes [n]     各行の部位コード
        """
        if len(region_codes) != features.shape[0]:
            raise ValueError(
                f"region_codes の数 {len(region_codes)} が特徴数 {features.shape[0]} と一致しない"
            )
        lookup = {code: i for i, code in enumerate(self.regions)}
        unknown = sorted({c for c in region_codes if c not in lookup})
        if unknown:
            raise ValueError(f"未知の部位コード: {unknown}; 既知: {self.regions}")
        index = np.array([lookup[c] for c in region_codes], dtype=int)

        x = features.astype(np.float64)
        if self.whitened:
            x = apply_whiten(x, index, self.whiten_mean, self.whiten_std)
        logits = x @ self.weights + self.bias[index]
        return profile_vectors(sigmoid(logits), self.kappa)
