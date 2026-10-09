"""胸部野の分類器（学習済みの線形ヘッド）と、その成果物。

画像 1 枚の胸部野を RAD-DINO で pooling した特徴（768 次元、L2 正規化済み）を、
14 所見の確率へ写す。確率は 2 段検索の 2 段目（プロファイル）と、画面の
「所見の推定」の両方に使う。

学習は `scripts/chest_image_eval.py train`、成果物の書き出しは
`scripts/export_chest_head.py`。本番では embedding-service が読み込む。

**特徴の作り方が学習時と 1 つでも違うと、分類器は黙って壊れる**（例外は出ず、
確率だけがずれる）。そのため成果物は特徴の作り方を記録して持ち、読み込む側が
自分の構成と照合する（`check_compatible`）。照合する項目:

  - backbone のモデル名と revision（**未固定 `unpinned` は受け付けない**。
    重みが更新されると同じ名前で別の特徴になる）
  - pooling の解像度（518）
  - 胸部野 mask の作り方（DIAPHRAGM_RATIO / MARGIN_RATIO）

計算は評価スクリプトと同じである。

    x = (pooled - mean) / std            # train の統計で標準化
    p = sigmoid(x @ weights + bias)      # 所見ごとの確率 [14]
    profile = profile_vectors(p, kappa)  # 14 + 「正常らしさ」1 = 15 次元、L2 正規化
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mil import profile_vectors, sigmoid

#: revision が固定されていないことを表す値。この成果物は本番で読み込まない。
UNPINNED = "unpinned"


class ChestHeadMismatch(RuntimeError):
    """成果物が前提とする特徴の作り方と、読み込む側の構成が違う。"""


@dataclass(frozen=True)
class ChestHead:
    mean: np.ndarray  # [d]
    std: np.ndarray  # [d]
    weights: np.ndarray  # [d, f]
    bias: np.ndarray  # [f]
    findings: tuple[str, ...]
    kappa: float
    #: 特徴の作り方（照合用）
    backbone_model: str
    backbone_revision: str
    pool_side: int
    diaphragm_ratio: float
    margin_ratio: float
    roi_profile: str
    #: 学習の出どころ（追跡用。照合には使わない）
    train_images: int
    split_seed: int
    test_auc_mean: float

    @property
    def input_dimension(self) -> int:
        return int(self.weights.shape[0])

    @property
    def dimension(self) -> int:
        """プロファイルの次元（所見数 + 「正常らしさ」1）。"""
        return len(self.findings) + 1

    # --- 推論 -----------------------------------------------------------------

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """pooling 特徴 [n, d] -> 所見ごとの確率 [n, f]。"""
        if features.ndim != 2 or features.shape[1] != self.input_dimension:
            raise ValueError(
                f"特徴の形 {features.shape} が成果物の想定 [n, {self.input_dimension}] と違う"
            )
        x = (features.astype(np.float64) - self.mean) / self.std
        return sigmoid(x @ self.weights + self.bias)

    def encode(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(検索用プロファイル [n, f+1], 確率 [n, f]) を返す。"""
        probs = self.probabilities(features)
        return profile_vectors(probs, self.kappa).astype(np.float32), probs.astype(np.float32)

    # --- 照合 -----------------------------------------------------------------

    def check_compatible(
        self,
        *,
        backbone_model: str,
        backbone_revision: str,
        pool_side: int,
        diaphragm_ratio: float | None = None,
        margin_ratio: float | None = None,
        allow_unpinned: bool = False,
    ) -> None:
        """読み込む側の構成が、学習時の特徴の作り方と一致することを確かめる。

        mask の作り方（diaphragm_ratio / margin_ratio）は None なら照合しない。
        embedding-service は mask を作らない（roi-service が作る）ので、そちらの
        照合は ingest-worker が roi-service と embedding-service の申告を突き合わせて
        行う（`verify_chest_pipeline`）。
        """
        problems: list[str] = []
        if not allow_unpinned and UNPINNED in (self.backbone_revision, backbone_revision):
            problems.append(
                "backbone の revision が固定されていない（成果物: "
                f"{self.backbone_revision}、構成: {backbone_revision or UNPINNED}）。"
                "重みが更新されると同じ名前で別の特徴になる"
            )
        if backbone_model != self.backbone_model:
            problems.append(f"backbone のモデル: 成果物 {self.backbone_model} / 構成 {backbone_model}")
        if (
            backbone_revision
            and self.backbone_revision != UNPINNED
            and backbone_revision != self.backbone_revision
        ):
            problems.append(
                f"backbone の revision: 成果物 {self.backbone_revision} / 構成 {backbone_revision}"
            )
        if pool_side != self.pool_side:
            problems.append(f"pooling の解像度: 成果物 {self.pool_side} / 構成 {pool_side}")
        if diaphragm_ratio is not None and not np.isclose(diaphragm_ratio, self.diaphragm_ratio):
            problems.append(
                f"DIAPHRAGM_RATIO: 成果物 {self.diaphragm_ratio} / 構成 {diaphragm_ratio}"
            )
        if margin_ratio is not None and not np.isclose(margin_ratio, self.margin_ratio):
            problems.append(f"MARGIN_RATIO: 成果物 {self.margin_ratio} / 構成 {margin_ratio}")
        if problems:
            raise ChestHeadMismatch(
                "胸部野の分類器が前提とする特徴の作り方と、いまの構成が違います。"
                "このまま使うと確率が黙ってずれます:\n  - " + "\n  - ".join(problems)
            )

    # --- 保存と読み込み -------------------------------------------------------

    def save(self, path) -> str:
        """npz へ保存し、内容の sha256 を返す（model_version の追跡用）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            std=self.std,
            weights=self.weights,
            bias=self.bias,
            findings=np.array(self.findings, dtype=object),
            kappa=np.array(self.kappa),
            backbone_model=np.array(self.backbone_model, dtype=object),
            backbone_revision=np.array(self.backbone_revision, dtype=object),
            pool_side=np.array(self.pool_side),
            diaphragm_ratio=np.array(self.diaphragm_ratio),
            margin_ratio=np.array(self.margin_ratio),
            roi_profile=np.array(self.roi_profile, dtype=object),
            train_images=np.array(self.train_images),
            split_seed=np.array(self.split_seed),
            test_auc_mean=np.array(self.test_auc_mean),
        )
        return sha256(path)

    @classmethod
    def load(cls, path) -> "ChestHead":
        blob = np.load(Path(path), allow_pickle=True)
        return cls(
            mean=blob["mean"].astype(np.float64),
            std=blob["std"].astype(np.float64),
            weights=blob["weights"].astype(np.float64),
            bias=blob["bias"].astype(np.float64),
            findings=tuple(str(v) for v in blob["findings"]),
            kappa=float(blob["kappa"]),
            backbone_model=str(blob["backbone_model"]),
            backbone_revision=str(blob["backbone_revision"]),
            pool_side=int(blob["pool_side"]),
            diaphragm_ratio=float(blob["diaphragm_ratio"]),
            margin_ratio=float(blob["margin_ratio"]),
            roi_profile=str(blob["roi_profile"]),
            train_images=int(blob["train_images"]),
            split_seed=int(blob["split_seed"]),
            test_auc_mean=float(blob["test_auc_mean"]),
        )


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pipeline_problems(roi_model: dict, embedding_model: dict) -> list[str]:
    """roi-service と embedding-service の `/api/v1/model` の申告を突き合わせる。

    分類器は、学習時と同じ mask・同じ解像度・同じ U-Net の profile で作った特徴を
    前提にしている。mask は roi-service が、分類器は embedding-service が持つので、
    両方を見られる取り込みワーカーが起動時にこれを呼ぶ（`verify_chest_pipeline`）。
    食い違いの説明を返す。空なら一致（または 2 段目が分類器でない構成）。
    """
    head = embedding_model.get("chest_head")
    if not head:
        return []  # 2 段目が chest-profile でない構成（1 段検索）
    spec = roi_model.get("chest_mask") or {}
    problems = []
    for key in ("diaphragm_ratio", "margin_ratio", "pool_side"):
        if key not in spec:
            problems.append(f"roi-service が {key} を申告していない（古い image の可能性）")
        elif abs(float(spec[key]) - float(head[key])) > 1e-9:
            problems.append(f"{key}: roi-service {spec[key]} / 分類器 {head[key]}")
    if roi_model.get("profile_name") != head.get("roi_profile"):
        problems.append(
            f"U-Net の profile: roi-service {roi_model.get('profile_name')} / "
            f"分類器 {head.get('roi_profile')}"
        )
    return problems
