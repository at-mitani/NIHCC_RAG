"""学習した肺野分割モデル（U-Net）の推論。

規則ベースの抽出は corpus の約 4 分の 1 で破綻するため、CheXmask を教師にした
U-Net を使う。学習は `scripts/train_lung_unet.py`、運用手順は `docs/OPERATIONS.md`。

**推論は onnxruntime で行い、torch は持ち込まない。** roi-service は OpenCV と
NumPy だけの軽い image で、torch を入れると 2 GB を超える。onnxruntime なら 50 MB
ほどで CPU でも速い。

前処理は学習時と揃える必要がある。`pipeline.normalize` を通した gray を受け取り、
学習解像度へ縮小して 0-1 にするところまでがこの module の責務である。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

#: 出力 channel の並び。train_lung_unet.py の CHANNELS と揃える。
CHANNELS = ("right_lung", "left_lung", "heart")

#: profile の `model_path` を上書きする環境変数。
#:
#: profile が持つのはコンテナ内のパス（`/models/lung-unet/...`）である。同じ profile
#: をホストの script（`eval_roi_dice.py` など）から使うときはそこに無いので、
#: **profile を書き換えずに置き場所だけ差し替えられる**ようにしておく。
MODEL_PATH_ENV = "LUNG_MODEL_PATH"


class SegmenterError(RuntimeError):
    """分割モデルを読めない、または推論できない。"""


@dataclass(frozen=True)
class Segmentation:
    """画像基準の mask。`right` は画像の左半分（患者の右肺）。"""

    right: np.ndarray
    left: np.ndarray
    heart: np.ndarray


def model_input_size(session: Any) -> int | None:
    """ONNX が宣言している入力の一辺を返す。動的なら None。

    **モデルは自分の入力の大きさを知っている。** profile の `input_size` と食い違う
    と onnxruntime が分かりにくい例外を投げるので、先にこちらで見て名前付きの
    誤りとして返す。
    """
    shape = session.get_inputs()[0].shape          # [batch, 1, H, W]
    if len(shape) != 4:
        return None
    sides = [d for d in shape[2:] if isinstance(d, int)]
    return sides[0] if len(sides) == 2 and sides[0] == sides[1] else None


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """最大の連結成分だけを残す。

    模型の出力は概ね綺麗だが、稀に小さな島が出る。肺野は左右それぞれ 1 つの
    かたまりなので、ここで落としても解剖学的に失うものは無い。
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return ((labels == largest).astype(np.uint8)) * 255


class LungSegmenter:
    """ONNX の U-Net を読み、gray から左右肺と心臓の mask を作る。

    session は生成に時間がかかるので使い回す。onnxruntime の session 自体は
    thread-safe だが、生成の競合を避けるため lock で包む。
    """

    def __init__(self, model_path: str | Path, input_size: int, threshold: float = 0.5,
                 providers: tuple[str, ...] = ("CPUExecutionProvider",)) -> None:
        self.model_path = Path(model_path)
        self.input_size = int(input_size)
        self.threshold = float(threshold)
        self._providers = list(providers)
        self._session: Any | None = None
        self._lock = threading.Lock()

    def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session
        with self._lock:
            if self._session is not None:
                return self._session
            try:
                import onnxruntime
            except ImportError as exc:                       # noqa: BLE001
                raise SegmenterError(
                    "onnxruntime が無い。roi-service の requirements に追加する") from exc
            if not self.model_path.is_file():
                raise SegmenterError(f"分割モデルが無い: {self.model_path}")
            available = set(onnxruntime.get_available_providers())
            providers = [p for p in self._providers if p in available] or ["CPUExecutionProvider"]
            session = onnxruntime.InferenceSession(
                str(self.model_path), providers=providers)
            declared = model_input_size(session)
            if declared is not None and declared != self.input_size:
                raise SegmenterError(
                    f"profile の input_size ({self.input_size}) とモデルの入力"
                    f" ({declared}) が違う: {self.model_path}. "
                    f"解像度が違うモデルは互換ではない。profile を合わせるか、"
                    f"その解像度で学習したモデルを置く"
                )
            self._session = session
            return self._session

    def predict(self, gray: np.ndarray) -> Segmentation:
        """`pipeline.normalize` 済みの gray から mask を返す。"""
        session = self._ensure_session()
        size = self.input_size
        small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
        batch = (small.astype(np.float32) / 255.0)[None, None]
        name = session.get_inputs()[0].name
        try:
            logits = session.run(None, {name: batch})[0]
        except Exception as exc:                             # noqa: BLE001
            raise SegmenterError(f"分割モデルの推論に失敗した: {exc}") from exc

        probabilities = 1.0 / (1.0 + np.exp(-logits[0].astype(np.float32)))
        masks = []
        for index in range(len(CHANNELS)):
            # 元の解像度へ戻してから閾値を掛ける。先に閾値を掛けて拡大すると
            # 境界が階段状になり、形状の特徴量が荒れる。
            full = cv2.resize(probabilities[index], (gray.shape[1], gray.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            masks.append(((full > self.threshold).astype(np.uint8)) * 255)
        return Segmentation(
            right=_largest_component(masks[0]),
            left=_largest_component(masks[1]),
            heart=_largest_component(masks[2]),
        )


_CACHE: dict[tuple[str, int, float], LungSegmenter] = {}
_CACHE_LOCK = threading.Lock()


def from_profile(profile) -> LungSegmenter | None:
    """profile が U-Net を指していれば segmenter を返す。そうでなければ None。

    同じ設定なら session を共有する（読み込みが数百 ms かかるため）。
    """
    section = profile.data.get("lung_segmentation")
    if not section or section.get("backend", "classical") != "unet":
        return None
    model_path = os.environ.get(MODEL_PATH_ENV) or str(section["model_path"])
    key = (model_path, int(section["input_size"]),
           float(section.get("threshold", 0.5)))
    with _CACHE_LOCK:
        if key not in _CACHE:
            _CACHE[key] = LungSegmenter(key[0], key[1], key[2],
                                        tuple(section.get("providers",
                                                          ("CPUExecutionProvider",))))
        return _CACHE[key]
