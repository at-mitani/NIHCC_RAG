"""Embedding backend 実装（設計書 7.3）。

biomedclip   : Microsoft BiomedCLIP を OpenCLIP でロードする採用モデル。
               画像 encoder(ViT-B/16) と英語医学テキスト encoder(PubMedBERT) が
               共通 512 次元空間へ射影されるため、IMAGE_TO_IMAGE / TEXT_TO_IMAGE /
               HYBRID を単一 vector 型で実現できる。
rad-dino     : Microsoft RAD-DINO。胸部 X 線に特化した DINOv2 系 image encoder。
               **text tower を持たない**ため supports_text = False。2 段検索の fine
               空間として使うと TEXT_TO_IMAGE / HYBRID では 2 段目が自動的に無効に
               なり、画像モードのみ 2 段になる（縮退は search.py が扱う）。
               ライセンスは MIT だが公開元は「研究目的のみ・臨床実践での使用を
               意図しない」と明記している。
chest-profile: 2 段目の採用構成。RAD-DINO で胸部野を pooling し、学習済みの分類器で
               14 所見の確率と検索用プロファイル（15 次元）を出す。
deterministic: モデル artifact を取得できない閉域・検証環境用の代替 backend。
               画像統計量から決定論的に L2 正規化 vector を作る（次元は指定可能）。
               医学的意味を持たないため embedding_model_version に `stub-` を付け、
               検索品質評価には使用しない。

2 段検索（coarse -> fine）のための 2 空間構成
----------------------------------------------
本サービスは backend を 2 つ持てる。

  coarse（既定・必須）: EMBEDDING_BACKEND。HNSW index を張る低次元空間。
  fine（任意）        : EMBEDDING_FINE_BACKEND。index を張らず、1 段目が返した
                        候補の厳密な再スコアリングにのみ使う高次元空間。

fine backend の既定は `none`（無効）である。理由は 2 つある。

  1. 低次元ベクトルから決定論的に生成した高次元ベクトルには情報が増えない。
     512 次元を 1024 次元へ写しても値は元の 512 次元が張る部分空間に収まるため、
     順位は変わらず計算量だけが増える。精度向上は「fine 側が本当に強い別モデル
     である」ことからのみ得られる。
  2. 別モデルを足すと 1 回の検索で推論が 2 回になる。CPU 環境では BiomedCLIP の
     推論が検索時間の支配項であり、単純に倍増する。

したがって fine モデルの採用は、評価データで精度向上を確認し、かつ GPU 推論を
前提にできる段階で行う判断事項である。
"""

from __future__ import annotations

import abc
import contextlib
import hashlib
import io
import logging
import os
import queue

import numpy as np
from PIL import Image

from pathlib import Path

_LOG = logging.getLogger("embedding-service")


class EmbeddingBackend(abc.ABC):
    name: str
    model_name: str
    model_revision: str
    dimension: int
    #: text encoder を共有空間に持つか。False の backend を fine として使うと
    #: TEXT_TO_IMAGE / HYBRID では再ランキングができない（画像モードのみ有効）。
    supports_text: bool = True
    #: ログ・エラーメッセージ用の役割名（"coarse" / "fine"）
    role: str = "coarse"

    @property
    def model_version(self) -> str:
        raise NotImplementedError

    #: 推論に部位コードが要るか。True の backend は region_codes 無しでは呼べない。
    needs_region_codes: bool = False
    #: 画像 1 回の順伝播 + mask pooling に対応するか。
    #: patch token を取り出せる backend だけが True になる。
    supports_region_pool: bool = False
    #: `/api/v1/embeddings/chest` の 2 段目（胸部野 mask + 分類器）に使えるか。
    supports_chest: bool = False

    def encode_region_pool(
        self, images: list[bytes], masks: list[dict[str, bytes]]
    ) -> list[tuple[int, str, np.ndarray]]:
        """画像を 1 回順伝播し、部位 mask で patch token を pooling する。

        (画像 index, 部位コード, ベクトル) を返す。mask が patch grid 上で空に
        なった部位は返さないため、件数は要求より少なくなりうる。
        """
        raise RuntimeError(
            f"{self.name} does not support region pooling (supports_region_pool=False)"
        )

    @property
    def patch_grid(self) -> int:
        """patch grid の一辺。supports_region_pool の backend だけが意味を持つ。"""
        raise RuntimeError(f"{self.name} has no patch grid")

    @abc.abstractmethod
    def encode_images(
        self, images: list[bytes], region_codes: list[str] | None = None
    ) -> np.ndarray: ...

    @abc.abstractmethod
    def encode_texts(self, texts: list[str]) -> np.ndarray: ...

    def sample_region_code(self) -> str:
        """self-test / warmup で使う部位コード。needs_region_codes の backend が上書きする。"""
        return "RUL"

    def warmup(self, batches: int) -> None:  # pragma: no cover - 起動時のみ
        return None

    def self_test(self) -> None:
        """起動時 self-test（設計書 7.3.4）: shape / 有限値 / L2 norm ≒ 1。

        次元の不一致はここで必ず落とす。DB 側の列幅（embedding / embedding_fine）は
        migration 時に EMBEDDING_DIMENSION / EMBEDDING_FINE_DIMENSION で確定して
        いるため、モデルの実出力がそれと違うと取り込みが全件失敗する。
        """
        dimension_env = (
            "EMBEDDING_FINE_DIMENSION" if self.role == "fine" else "EMBEDDING_DIMENSION"
        )
        canvas = Image.new("L", (256, 256), 128)
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        sample_regions = [self.sample_region_code()] if self.needs_region_codes else None
        checks: list[tuple[str, np.ndarray]] = [
            ("image", self.encode_images([buffer.getvalue()], sample_regions))
        ]
        if self.supports_text:
            checks.append(
                ("text", self.encode_texts(["pulmonary nodule in the right upper lung field"]))
            )
        for label, vec in checks:
            if vec.shape != (1, self.dimension):
                raise RuntimeError(
                    f"[{self.role}] {label} vector shape {vec.shape} != (1, {self.dimension}); "
                    f"model {self.model_name} actually returns {vec.shape[-1]} dimensions. "
                    f"set {dimension_env} and the DB column width to match the model"
                )
            if not np.isfinite(vec).all():
                raise RuntimeError(f"[{self.role}] {label} vector contains non-finite values")
            norm = float(np.linalg.norm(vec[0]))
            if abs(norm - 1.0) > 1e-3:
                raise RuntimeError(f"[{self.role}] {label} vector L2 norm {norm:.5f} is not 1.0")
        _LOG.info(
            "embedding self-test passed",
            extra={
                "fields": {
                    "backend": self.name,
                    "role": self.role,
                    "dimension": self.dimension,
                    "supports_text": self.supports_text,
                }
            },
        )


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class DeterministicBackend(EmbeddingBackend):
    """モデル非依存の決定論的 encoder（閉域検証・CI 用）。"""

    name = "deterministic"

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension
        self.model_name = "deterministic-image-statistics"
        self.model_revision = "v1"
        self._grid = 8
        self._bins = 32

    @property
    def model_version(self) -> str:
        return f"stub-{self.model_name}-{self.model_revision}-d{self.dimension}"

    def encode_images(
        self, images: list[bytes], region_codes: list[str] | None = None
    ) -> np.ndarray:
        vectors = [self._encode_image(data) for data in images]
        return _l2_normalise(np.vstack(vectors))

    def _encode_image(self, data: bytes) -> np.ndarray:
        with Image.open(io.BytesIO(data)) as image:
            array = np.asarray(image.convert("L").resize((64, 64), Image.BILINEAR), dtype=np.float32)
        array /= 255.0
        features: list[float] = []
        # 8x8 ブロックの平均・標準偏差（128 次元）
        blocks = array.reshape(self._grid, 8, self._grid, 8).mean(axis=(1, 3))
        stds = array.reshape(self._grid, 8, self._grid, 8).std(axis=(1, 3))
        features.extend(blocks.ravel().tolist())
        features.extend(stds.ravel().tolist())
        # 輝度ヒストグラム（32 次元）
        hist, _ = np.histogram(array, bins=self._bins, range=(0.0, 1.0), density=True)
        features.extend((hist / max(hist.max(), 1e-6)).tolist())
        # 勾配統計（32 次元）
        gy, gx = np.gradient(array)
        magnitude = np.hypot(gx, gy)
        gm = magnitude.reshape(4, 16, 4, 16).mean(axis=(1, 3))
        features.extend(gm.ravel().tolist())
        vector = np.zeros(self.dimension, dtype=np.float32)
        raw = np.asarray(features, dtype=np.float32)
        vector[: min(len(raw), self.dimension)] = raw[: self.dimension]
        # 残りは内容ハッシュ由来の決定論的成分で埋める（同一画像で常に同一）
        if len(raw) < self.dimension:
            digest = hashlib.sha256(array.tobytes()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            vector[len(raw) :] = rng.normal(0.0, 0.05, self.dimension - len(raw)).astype(np.float32)
        return vector[None, :]

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            for token in text.lower().split():
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            if not vector.any():
                vector[0] = 1.0
            vectors.append(vector[None, :])
        return _l2_normalise(np.vstack(vectors))


class BiomedClipBackend(EmbeddingBackend):
    """OpenCLIP 経由の BiomedCLIP（設計書 7.3.1 / 7.3.4）。"""

    name = "biomedclip"

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        dimension: int,
        device: str,
        dtype: str,
        batch_size: int,
        max_text_tokens: int,
        replicas: int = 1,
    ) -> None:
        import open_clip  # 遅延 import: deterministic backend では torch を要求しない
        import torch

        # CPU 推論では既定のスレッド数が過剰購読になりやすい（1 ホストに多数の
        # コンテナが同居するため）。明示指定できるようにする。
        threads = int(os.environ.get("EMBEDDING_TORCH_THREADS", "0"))
        if threads > 0:
            torch.set_num_threads(threads)

        self._torch = torch
        self.model_name = model_id.removeprefix("hf-hub:")
        self.model_revision = revision or "unpinned"
        self.dimension = dimension
        self.batch_size = batch_size
        self.max_text_tokens = max_text_tokens

        if not revision:
            _LOG.warning(
                "embedding model revision is not pinned; set EMBEDDING_MODEL_REVISION "
                "to a verified commit hash before production use"
            )

        resolved_device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            _LOG.warning("cuda requested but unavailable; falling back to cpu")
            resolved_device = "cpu"
        self.device = resolved_device
        self._dtype = torch.float16 if (dtype == "float16" and resolved_device != "cpu") else torch.float32

        # 推論はモデルインスタンスの貸し出しで並行化する（設計書 7.3.4 の
        # スループット要件）。1 個のモデルを lock で共有すると、システム全体の
        # 埋め込みが 1 本の直列ストリームになり、取り込みの支配的な待ちになる。
        # replicas × EMBEDDING_TORCH_THREADS が vCPU を超えないよう設定する。
        self.replicas = max(1, replicas)
        self._pool: queue.Queue = queue.Queue()
        preprocess = None
        for _ in range(self.replicas):
            model, _, preprocess = open_clip.create_model_and_transforms(model_id)
            model = model.to(self.device).eval()
            if self._dtype == torch.float16:
                model = model.half()
            self._pool.put(model)
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_id)

    @contextlib.contextmanager
    def _borrow(self):
        """プールからモデルを 1 個借りる。返却は必ず行う。"""
        model = self._pool.get()
        try:
            yield model
        finally:
            self._pool.put(model)

    @property
    def model_version(self) -> str:
        short = self.model_revision[:8] if self.model_revision != "unpinned" else "unpinned"
        return f"biomedclip-{short}-d{self.dimension}"

    def _prepare(self, data: bytes):
        with Image.open(io.BytesIO(data)) as image:
            return self._preprocess(image.convert("RGB"))

    def encode_images(
        self, images: list[bytes], region_codes: list[str] | None = None
    ) -> np.ndarray:
        torch = self._torch
        # 復号とリサイズはモデルを借りる前に済ませる（PIL は GIL を解放するため
        # 他スレッドの推論と重ねられる）。
        tensors = [self._prepare(item) for item in images]
        outputs: list[np.ndarray] = []
        with self._borrow() as model, torch.inference_mode():
            for start in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[start : start + self.batch_size]).to(self.device)
                if self._dtype == torch.float16:
                    batch = batch.half()
                vectors = model.encode_image(batch)
                vectors = vectors / vectors.norm(dim=-1, keepdim=True)
                outputs.append(vectors.float().cpu().numpy())
        return np.vstack(outputs).astype(np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        tokens = self._tokenizer(texts, context_length=self.max_text_tokens)
        with self._borrow() as model, torch.inference_mode():
            vectors = model.encode_text(tokens.to(self.device))
            vectors = vectors / vectors.norm(dim=-1, keepdim=True)
            return vectors.float().cpu().numpy().astype(np.float32)

    def warmup(self, batches: int) -> None:  # pragma: no cover - 起動時のみ
        canvas = Image.new("RGB", (224, 224), (128, 128, 128))
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        payload = [buffer.getvalue()] * min(self.batch_size, 4)
        for _ in range(max(0, batches)):
            self.encode_images(payload)
            self.encode_texts(["chest radiograph"])


#: fine backend を無効にする指定。
_FINE_DISABLED = {"", "none", "off", "disabled", "0", "false"}

_DEFAULT_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

#: backend 種別ごとの既定リポジトリ。rad-dino は open_clip ではなく transformers
#: 経由でロードするため `hf-hub:` 接頭辞を付けない。
_DEFAULT_FINE_MODEL_ID = {
    "rad-dino": "microsoft/rad-dino",
    "biomedclip": _DEFAULT_MODEL_ID,
}


def _build_one(
    *,
    role: str,
    requested: str,
    dimension: int,
    model_id: str,
    revision: str,
    replicas: int,
) -> EmbeddingBackend:
    """1 つの backend を構築する。device / dtype / batch / token 長は共通設定を使う。"""
    requested = requested.strip().lower()
    if requested == "deterministic":
        backend: EmbeddingBackend = DeterministicBackend(dimension=dimension)
    elif requested == "rad-dino":
        backend = RadDinoBackend(
            model_id=model_id,
            revision=revision,
            dimension=dimension,
            device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
            dtype=os.environ.get("EMBEDDING_DTYPE", "float32"),
            batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")),
            replicas=replicas,
        )
    elif requested == "chest-profile":
        # backbone は分類器を学習したときと同じものを指定する（成果物と照合する）。
        # 既定は RAD-DINO（EMBEDDING_FINE_BACKBONE で変更できる）。revision は
        # EMBEDDING_FINE_MODEL_REVISION で固定する。
        inner_name = os.environ.get("EMBEDDING_FINE_BACKBONE", "rad-dino").strip().lower()
        inner = _build_one(
            role=role,
            requested=inner_name,
            dimension=int(os.environ.get("EMBEDDING_FINE_BACKBONE_DIMENSION", "768")),
            model_id=os.environ.get("EMBEDDING_FINE_MODEL_ID", "").strip()
            or _DEFAULT_FINE_MODEL_ID.get(inner_name, _DEFAULT_MODEL_ID),
            revision=revision,
            replicas=replicas,
        )
        backend = ChestProfileBackend(
            artifact_path=os.environ.get("CHEST_HEAD_PATH", "/models/chest/head-pool-chest.npz"),
            backbone=inner,
            dimension=dimension,
            allow_unpinned=os.environ.get("CHEST_HEAD_ALLOW_UNPINNED", "false").strip().lower()
            in {"1", "true", "yes", "on"},
        )
    elif requested == "biomedclip":
        backend = BiomedClipBackend(
            model_id=model_id,
            revision=revision,
            dimension=dimension,
            device=os.environ.get("EMBEDDING_DEVICE", "cpu"),
            dtype=os.environ.get("EMBEDDING_DTYPE", "float32"),
            batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")),
            max_text_tokens=int(os.environ.get("MAX_TEXT_TOKENS", "256")),
            replicas=replicas,
        )
    else:
        env_name = "EMBEDDING_FINE_BACKEND" if role == "fine" else "EMBEDDING_BACKEND"
        raise RuntimeError(f"unsupported {env_name}: {requested}")
    backend.role = role
    return backend


def mask_weights(png: bytes, grid: int) -> np.ndarray:
    """mask PNG を patch grid 上の被覆率 [0, 1] へ縮約する。

    面積平均（Image.BOX）を使う。最近傍だと 1 patch 未満の細い部位が消え、
    RML（最小部位、正解 box の 4.4%）が落ちやすい。
    """
    with Image.open(io.BytesIO(png)) as raw:
        reduced = raw.convert("L").resize((grid, grid), Image.BOX)
        return np.asarray(reduced, dtype=np.float32) / 255.0


def pool_patches(patches: np.ndarray, weights: np.ndarray) -> np.ndarray | None:
    """[g, g, d] の patch token を [g, g] の被覆率で加重平均する。

    被覆率で割るのが要点である。単純な総和だと**部位の面積がそのままベクトルの
    大きさになる**ため、L2 正規化しても大きい部位と小さい部位で向きが揃わない。

    被覆が皆無なら None を返す。これは失敗ではなく、その部位が patch grid の
    分解能を下回ったという事実であり、呼び出し側は結果から落とす。
    """
    total = float(weights.sum())
    if total <= 0.0:
        return None
    return ((patches * weights[:, :, None]).sum((0, 1)) / total).astype(np.float32)


class RadDinoBackend(EmbeddingBackend):
    """RAD-DINO（Microsoft）: 胸部 X 線特化の DINOv2 系 image encoder。

    BiomedCLIP（PubMed 図版で学習した汎用生物医学モデル）に対し、胸部 X 線の
    局所所見に対する感度が高いことを狙う。patch token の空間解像度が 37x37 と
    細かく（BiomedCLIP ViT-B/16 @224 は 14x14）、部位 mask による pooling にも
    向く。本 backend は CLS（pooler_output）だけを返す。

    text tower を持たないため、テキストは符号化できない。
    """

    name = "rad-dino"
    supports_text = False
    supports_region_pool = True

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        dimension: int,
        device: str,
        dtype: str,
        batch_size: int,
        replicas: int = 1,
    ) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        threads = int(os.environ.get("EMBEDDING_TORCH_THREADS", "0"))
        if threads > 0:
            torch.set_num_threads(threads)

        self._torch = torch
        self.model_name = model_id
        self.model_revision = revision or "unpinned"
        self.dimension = dimension
        self.batch_size = batch_size

        if not revision:
            _LOG.warning(
                "fine embedding model revision is not pinned; set "
                "EMBEDDING_FINE_MODEL_REVISION to a verified commit hash before production use"
            )

        resolved_device = device
        if device.startswith("cuda") and not torch.cuda.is_available():
            _LOG.warning("cuda requested but unavailable; falling back to cpu")
            resolved_device = "cpu"
        self.device = resolved_device
        self._dtype = (
            torch.float16 if (dtype == "float16" and resolved_device != "cpu") else torch.float32
        )

        kwargs = {"revision": revision} if revision else {}
        self._processor = AutoImageProcessor.from_pretrained(model_id, **kwargs)
        self.replicas = max(1, replicas)
        self._pool: queue.Queue = queue.Queue()
        for _ in range(self.replicas):
            model = AutoModel.from_pretrained(model_id, **kwargs).to(self.device).eval()
            if self._dtype == torch.float16:
                model = model.half()
            self._pool.put(model)

        # 部位 mask pooling に必要な入力解像度と patch grid を、processor と
        # model config の実値から取る。定数で持つと、モデルを差し替えたときに
        # patch の数え方だけが古いまま残り、pooling が静かにずれる。
        # patch pooling は CLS だけを取る推論より遥かに重い。518 四方では token が
        # 1,369 個あり、224 四方（196 個）の約 7 倍で、attention は token 数の 2 乗に
        # 効く。EMBEDDING_BATCH_SIZE は 224 四方向けに調整された値なので流用しない。
        self.pool_batch_size = max(
            1, int(os.environ.get("EMBEDDING_POOL_BATCH_SIZE", "8"))
        )
        self._input_side = self._resolve_input_side()
        patch_size = int(getattr(self._peek_config(), "patch_size", 14) or 14)
        self._patch_size = patch_size
        self._grid = self._input_side // patch_size
        _LOG.info(
            "rad-dino region pooling ready",
            extra={
                "fields": {
                    "input_side": self._input_side,
                    "patch_size": patch_size,
                    "patch_grid": self._grid,
                }
            },
        )

    def _peek_config(self):
        with self._borrow() as model:
            return model.config

    def _resolve_input_side(self) -> int:
        """processor が最終的に出す 1 辺を求める。

        crop_size があればそれが最終寸法。無ければ size を見る。どちらも無い形の
        processor に当たったときは RAD-DINO の既定である 518 へ倒す。
        """
        for attr in ("crop_size", "size"):
            value = getattr(self._processor, attr, None)
            if isinstance(value, dict):
                side = value.get("height") or value.get("width") or value.get("shortest_edge")
                if side:
                    return int(side)
        return 518

    @property
    def patch_grid(self) -> int:
        return self._grid

    @contextlib.contextmanager
    def _borrow(self):
        model = self._pool.get()
        try:
            yield model
        finally:
            self._pool.put(model)

    @property
    def model_version(self) -> str:
        short = self.model_revision[:8] if self.model_revision != "unpinned" else "unpinned"
        return f"rad-dino-{short}-d{self.dimension}"

    def encode_images(
        self, images: list[bytes], region_codes: list[str] | None = None
    ) -> np.ndarray:
        torch = self._torch
        frames = [Image.open(io.BytesIO(item)).convert("RGB") for item in images]
        outputs: list[np.ndarray] = []
        with self._borrow() as model, torch.inference_mode():
            for start in range(0, len(frames), self.batch_size):
                batch = self._processor(
                    images=frames[start : start + self.batch_size], return_tensors="pt"
                ).to(self.device)
                if self._dtype == torch.float16:
                    batch = batch.to(torch.float16)
                pooled = model(**batch).pooler_output
                outputs.append(pooled.float().cpu().numpy())
        for frame in frames:
            frame.close()
        # DINOv2 の出力は正規化されていないため、cosine 検索のためここで L2 正規化する
        # （BiomedCLIP は encode_image 側で正規化済み）。
        return _l2_normalise(np.vstack(outputs))

    def encode_region_pool(
        self, images: list[bytes], masks: list[dict[str, bytes]]
    ) -> list[tuple[int, str, np.ndarray]]:
        """画像を 1 回順伝播し、patch token を部位 mask の被覆率で加重平均する。

        画像を先に入力解像度の正方形へ揃えてから processor へ渡す。processor の
        resize / center crop がいずれも恒等になり、**mask と画像の枠が一致する**。
        processor に任せて縦横比を保ったまま中央切り出しさせると、mask だけが
        元の枠のままになり、部位が上下にずれたまま pooling される。

        mask は patch grid へ面積平均（Image.BOX）で縮約する。最近傍だと 1 patch
        未満の細い部位が消え、RML（最小部位、正解の 4.4%）が落ちやすい。
        """
        if len(images) != len(masks):
            raise ValueError(f"images と masks の長さが違う ({len(images)} != {len(masks)})")

        grid = self._grid
        results: list[tuple[int, str, np.ndarray]] = []
        for start, patches in self.iter_patch_tokens(images):
            for offset in range(patches.shape[0]):
                index = start + offset
                for code, png in masks[index].items():
                    pooled = pool_patches(patches[offset], mask_weights(png, grid))
                    if pooled is None:
                        continue
                    results.append((index, code, pooled))

        if not results:
            return []
        vectors = _l2_normalise(np.vstack([v for _, _, v in results]))
        return [(i, c, vectors[n]) for n, (i, c, _) in enumerate(results)]

    def iter_patch_tokens(self, images: list[bytes]):
        """画像を 1 回ずつ順伝播し、(開始 index, patch token [n, g, g, d]) を順に返す。

        pooling（encode_region_pool）と胸部野の地図（ChestProfileBackend.chest_map）が
        同じ patch を使う。**計算は 1 か所にしかない**ので、取り込みの特徴と地図の
        特徴が食い違わない。
        """
        torch = self._torch
        side, grid = self._input_side, self._grid
        n_patch = grid * grid

        # **チャンクごとに復号し、その場で捨てる。** 全件を先に 518 四方へ展開すると、
        # 1 枚 0.8MB として 64 枚で 51MB を無駄に抱える。pooling は 1 チャンクぶんの
        # patch しか要らない。
        with self._borrow() as model, torch.inference_mode():
            for start in range(0, len(images), self.pool_batch_size):
                window = images[start : start + self.pool_batch_size]
                frames = []
                try:
                    for item in window:
                        with Image.open(io.BytesIO(item)) as raw:
                            frames.append(
                                raw.convert("RGB").resize((side, side), Image.BILINEAR)
                            )
                    batch = self._processor(images=frames, return_tensors="pt").to(self.device)
                    if self._dtype == torch.float16:
                        batch = batch.to(torch.float16)
                    hidden = model(**batch).last_hidden_state
                    if hidden.shape[1] < n_patch:
                        raise RuntimeError(
                            f"patch token が足りない (期待 {n_patch}, 実際 {hidden.shape[1]}); "
                            f"input_side={side} patch_size={self._patch_size}"
                        )
                    # CLS と register token は先頭に並ぶ。patch は末尾 n_patch 個なので、
                    # 先頭から数えず末尾から取る（register の有無に左右されない）。
                    patches = hidden[:, -n_patch:, :].float().cpu().numpy()
                    patches = patches.reshape(len(frames), grid, grid, -1)
                finally:
                    for frame in frames:
                        frame.close()
                # GPU 側の中間を明示的に手放す。caching allocator が確保済みブロックを
                # 抱えたまま次のチャンクを積むと、長時間の連続処理で断片化しやすい。
                del batch, hidden
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                yield start, patches
                del patches

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError(
            "rad-dino has no text encoder; it can only be used as an image-only "
            "embedding space (supports_text=False)"
        )

    def warmup(self, batches: int) -> None:  # pragma: no cover - 起動時のみ
        canvas = Image.new("RGB", (518, 518), (128, 128, 128))
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
        payload = [buffer.getvalue()] * min(self.batch_size, 4)
        for _ in range(max(0, batches)):
            self.encode_images(payload)


class ChestProfileBackend(EmbeddingBackend):
    """胸部野の分類器が出す**所見プロファイル**を返す 2 段目の backend。

    backbone（RAD-DINO）で画像を 1 回順伝播し、patch token を胸部野 mask で
    pooling する（評価の `pool-chest` と同じ計算）。それを学習済みの線形ヘッド
    （`nihcc_rag.chest_head.ChestHead`）へ通して、14 所見の確率と、検索用の
    プロファイル（14 + 「正常らしさ」1 = 15 次元、L2 正規化）を出す。

    確率は画面の「所見の推定」にも使う。

    **起動時に成果物と構成を照合する。** 分類器は特徴の作り方が 1 つでも違うと、
    例外を出さずに確率だけがずれる。backbone のモデル名・revision・pooling の
    解像度が成果物と違えば起動を止める。revision が固定されていない構成は
    `CHEST_HEAD_ALLOW_UNPINNED=true` のときだけ（警告つきで）通す。手元の
    動作確認用で、本番では使わない。

    crop を受け取る `encode_images` は持たない（mask が要る）。入口は
    `/api/v1/embeddings/chest` の `encode_chest` だけである。
    """

    name = "chest-profile"
    supports_text = False
    #: `/embeddings/chest` が 2 段目に使えるか
    supports_chest = True

    def __init__(
        self,
        *,
        artifact_path: str,
        backbone: EmbeddingBackend,
        dimension: int,
        allow_unpinned: bool = False,
    ) -> None:
        from nihcc_rag.chest_head import ChestHead, sha256

        path = Path(artifact_path)
        if not path.exists():
            raise RuntimeError(
                f"胸部野の分類器がありません: {path}. "
                "scripts/export_chest_head.py で作成し、CHEST_HEAD_PATH へ "
                "read-only で mount してください"
            )
        if not backbone.supports_region_pool:
            raise RuntimeError(
                f"backbone {backbone.name} は patch token の pooling に対応していない。"
                "EMBEDDING_FINE_BACKBONE=rad-dino を指定すること"
            )
        self.head = ChestHead.load(path)
        self._backbone = backbone
        self.artifact_sha256 = sha256(path)
        self.model_revision = self.artifact_sha256[:12]
        self.model_name = (
            f"chest-profile/{self.head.backbone_model}@{self.head.backbone_revision[:12]}"
        )
        self.dimension = dimension

        if backbone.dimension != self.head.input_dimension:
            raise RuntimeError(
                f"backbone の出力 {backbone.dimension} 次元が分類器の想定 "
                f"{self.head.input_dimension} 次元と一致しない"
            )
        if dimension != self.head.dimension:
            raise RuntimeError(
                f"EMBEDDING_FINE_DIMENSION={dimension} が分類器の出力 "
                f"{self.head.dimension} 次元と一致しない"
            )
        self.head.check_compatible(
            backbone_model=backbone.model_name,
            backbone_revision=backbone.model_revision,
            pool_side=backbone._input_side,
            allow_unpinned=allow_unpinned,
        )
        if allow_unpinned and "unpinned" in (self.head.backbone_revision, backbone.model_revision):
            _LOG.warning(
                "chest head accepted with an unpinned backbone revision; "
                "probabilities may silently drift if the backbone weights change. "
                "do not use this in production (CHEST_HEAD_ALLOW_UNPINNED=true)"
            )
        _LOG.info(
            "chest head loaded",
            extra={
                "fields": {
                    "artifact": str(path),
                    "sha256": self.model_revision,
                    "backbone": f"{self.head.backbone_model}@{self.head.backbone_revision}",
                    "pool_side": self.head.pool_side,
                    "findings": len(self.head.findings),
                    "train_images": self.head.train_images,
                    "test_auc_mean": round(self.head.test_auc_mean, 4),
                }
            },
        )

    @property
    def model_version(self) -> str:
        return f"chest-{self.model_revision}-d{self.dimension}"

    def describe(self) -> dict[str, object]:
        """`/api/v1/model` で申告する特徴の作り方。ingest-worker が roi-service と突き合わせる。"""
        head = self.head
        return {
            "artifact_sha256": self.artifact_sha256,
            "backbone_model": head.backbone_model,
            "backbone_revision": head.backbone_revision,
            "pool_side": head.pool_side,
            "diaphragm_ratio": head.diaphragm_ratio,
            "margin_ratio": head.margin_ratio,
            "roi_profile": head.roi_profile,
            "findings": list(head.findings),
            "train_images": head.train_images,
            "test_auc_mean": head.test_auc_mean,
        }

    def encode_chest(
        self, images: list[bytes], masks: list[bytes]
    ) -> tuple[np.ndarray, np.ndarray]:
        """(プロファイル [n, f+1], 確率 [n, f]) を返す。並びは入力と同じ。"""
        if len(images) != len(masks):
            raise ValueError(f"images と masks の長さが違う ({len(images)} != {len(masks)})")
        pooled = self._backbone.encode_region_pool(images, [{"CHEST": m} for m in masks])
        features = np.zeros((len(images), self._backbone.dimension), dtype=np.float32)
        seen = set()
        for index, _, vector in pooled:
            features[index] = vector
            seen.add(index)
        missing = sorted(set(range(len(images))) - seen)
        if missing:
            # 胸部野は画像の 3〜5 割を占めるので、patch grid 上で消えることはない。
            # 起きたら mask の受け渡しが壊れている
            raise RuntimeError(f"胸部野 mask が patch grid 上で空になった画像: {missing}")
        return self.head.encode(features)

    def chest_map(
        self, images: list[bytes], masks: list[bytes], findings: list[str] | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """区画ごとの所見の寄与の地図。(地図 [n, k, g, g], 確率 [n, k], 所見名 [k]) を返す。

        分類器は「胸部野で加重平均した patch → L2 正規化 → 標準化 → 線形」なので、
        ロジットは区画ごとの寄与の加重平均に**正確に**分解できる（CAM と同じ考え方）。

            logit_f = Σ_ij (w_ij / Σw) · m_ij,f  −  (mean / std) · W_f  +  b_f
            m_ij,f  = (P_ij / (‖pooled‖ · std)) · W_f

        地図は m（加重前）を返す。胸部野の外の区画にも値は出るが意味は無いので、
        呼び出し側が mask で切る。地図が「病変の位置」を示すかは測るまで分からない
        （ViT の patch は画像全体の情報を持つ）。
        """
        if not hasattr(self._backbone, "iter_patch_tokens"):
            raise RuntimeError(f"backbone {self._backbone.name} は patch token を返せない")
        head = self.head
        if findings:
            unknown = sorted(set(findings) - set(head.findings))
            if unknown:
                raise ValueError(f"未知の所見: {unknown}")
            index = [head.findings.index(f) for f in findings]
        else:
            index = list(range(len(head.findings)))
        weights = head.weights[:, index]
        grid = self._backbone.patch_grid
        maps: list[np.ndarray] = []
        probs: list[np.ndarray] = []
        for start, patches in self._backbone.iter_patch_tokens(images):
            for offset in range(patches.shape[0]):
                tokens = patches[offset].astype(np.float64)
                coverage = mask_weights(masks[start + offset], grid).astype(np.float64)
                total = float(coverage.sum())
                if total <= 0.0:
                    raise RuntimeError(f"胸部野 mask が patch grid 上で空になった画像: {start + offset}")
                pooled = (tokens * coverage[:, :, None]).sum((0, 1)) / total
                scale = float(np.linalg.norm(pooled))
                contribution = (tokens / (scale * head.std)) @ weights  # [g, g, k]
                maps.append(np.moveaxis(contribution, -1, 0))
                probs.append(head.probabilities((pooled / scale)[None, :])[0][index])
        return (
            np.stack(maps).astype(np.float32),
            np.stack(probs).astype(np.float32),
            [head.findings[i] for i in index],
        )

    def encode_images(
        self, images: list[bytes], region_codes: list[str] | None = None
    ) -> np.ndarray:
        raise RuntimeError(
            "chest-profile は胸部野 mask を必要とする。/api/v1/embeddings/chest を使うこと"
        )

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        raise RuntimeError("chest-profile は所見プロファイルであり、テキストを同じ空間へ写せない")

    def self_test(self) -> None:
        """全面 mask の灰色画像で、出力の形・有限値・L2 norm を確かめる。"""
        side = self.head.pool_side
        buffer = io.BytesIO()
        Image.new("L", (side, side), 128).save(buffer, format="PNG")
        image = buffer.getvalue()
        buffer = io.BytesIO()
        Image.new("L", (side, side), 255).save(buffer, format="PNG")
        mask = buffer.getvalue()
        profiles, probs = self.encode_chest([image], [mask])
        if profiles.shape != (1, self.dimension) or probs.shape != (1, len(self.head.findings)):
            raise RuntimeError(
                f"[fine] chest-profile の出力の形が違う: {profiles.shape}, {probs.shape}"
            )
        if not (np.isfinite(profiles).all() and np.isfinite(probs).all()):
            raise RuntimeError("[fine] chest-profile の出力に非有限値がある")
        if abs(float(np.linalg.norm(profiles[0])) - 1.0) > 1e-3:
            raise RuntimeError("[fine] chest-profile のプロファイルが L2 正規化されていない")
        _LOG.info(
            "embedding self-test passed",
            extra={"fields": {"backend": self.name, "role": self.role, "dimension": self.dimension}},
        )

    def warmup(self, batches: int) -> None:  # pragma: no cover - 起動時のみ
        self._backbone.warmup(batches)


def build_backend() -> EmbeddingBackend:
    """1 段目（coarse）の backend。HNSW index を張る低次元空間を生成する。"""
    return _build_one(
        role="coarse",
        requested=os.environ.get("EMBEDDING_BACKEND", "biomedclip"),
        dimension=int(os.environ.get("EMBEDDING_DIMENSION", "512")),
        model_id=os.environ.get("EMBEDDING_MODEL_ID", _DEFAULT_MODEL_ID),
        revision=os.environ.get("EMBEDDING_MODEL_REVISION", "").strip(),
        replicas=int(os.environ.get("EMBEDDING_MODEL_REPLICAS", "1")),
    )


def build_fine_backend() -> EmbeddingBackend | None:
    """2 段目（fine）の backend。EMBEDDING_FINE_BACKEND が未設定なら None。

    fine モデルはプロセス内にもう 1 組のモデルを常駐させる。replicas の既定は 1
    とする（coarse と同じ 2 にすると ViT が 4 つ常駐し、CPU 機ではメモリと
    スレッド競合で coarse 側の推論まで遅くなる）。
    """
    requested = os.environ.get("EMBEDDING_FINE_BACKEND", "none").strip().lower()
    if requested in _FINE_DISABLED:
        return None
    # backend ごとに既定のリポジトリが違う。fine は「別の、より強いモデル」で
    # なければ意味が無いため、明示指定を推奨する（既定は取り違え防止の保険）。
    default_id = _DEFAULT_FINE_MODEL_ID.get(requested, _DEFAULT_MODEL_ID)
    return _build_one(
        role="fine",
        requested=requested,
        dimension=int(os.environ.get("EMBEDDING_FINE_DIMENSION", "1024")),
        model_id=os.environ.get("EMBEDDING_FINE_MODEL_ID", "").strip() or default_id,
        revision=os.environ.get("EMBEDDING_FINE_MODEL_REVISION", "").strip(),
        replicas=int(os.environ.get("EMBEDDING_FINE_MODEL_REPLICAS", "1")),
    )
