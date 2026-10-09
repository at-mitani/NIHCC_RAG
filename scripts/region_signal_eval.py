"""部位ベクトルが「部位レベルの所見」を符号化しているかを測る。

背景
----
検索精度の定義は「部位レベルでの所見一致」である。気胸画像を検索元としたとき、
病変のある部位では気胸の画像が、病変のない部位では所見なしの画像が上位に来るのが
理想である。しかし NIHCC のラベルは**画像単位**であり、部位単位の正解は存在しない。

そこで、追加アノテーションなしで測れる**基準率（base rate）との比較**を行う。
論理的に確定する事実を使う。

  `No Finding` ⇒ 5 部位すべてが正常   … 無償で得られる大量の負例
  所見 F あり  ⇒ 少なくとも 1 部位が F … どの部位かは不明（MIL の正例バッグ）

部位ベクトルに病変シグナルが無ければ、上位 K の所見分布は index 集合の基準率に
一致する。有意に上回るかどうかが、そのベクトルが使えるかの最初の判定になる。

実施する 3 つの検定
------------------
Test A  正常部位の enrichment
        `No Finding` をクエリにしたとき、上位 K に占める `No Finding` の割合が
        基準率（約 55%）を上回るか。上回らなければ正常・異常を区別できていない。

Test B  所見の enrichment
        単独所見 F の画像をクエリにしたとき、**いずれかの部位**の上位 K で F の
        出現率が基準率を上回るか。MIL の正例バッグに対応する弱い検定。

Test C  部位同定の対照
        全部位を横断して検索したとき、上位 K が同一部位で占められる割合。
        100% に近ければ、ベクトルは主に「部位の形」を符号化しており、病理所見に
        割ける表現力が残っていないことを示す。Test A/B の解釈に必要。

統計
----
- **患者単位分割**。ChestX-ray14 は 1 患者あたり複数回の撮影を含むため、患者を
  跨いで分割しないと同一患者の別撮影が近傍に来て数字が不当に良くなる。
- 主統計量は「クエリ 1 件あたりの上位 K 適合率」の平均。上位 K の 1 件 1 件を
  独立試行として扱うと、同一クエリ内の相関を無視して有意性を過大評価する。
- 信頼区間は**患者を単位とした cluster bootstrap**。同一患者のクエリは相関する。

使い方
------
    # 帰無分布の確認（torch 不要。医学的意味を持たない stub encoder）
    python scripts/region_signal_eval.py --encoder deterministic

    # 複数モデルを 1 回の ROI パスで比較する（**全量ではこの形で実行すること**）
    python scripts/region_signal_eval.py --encoder deterministic,biomedclip,rad-dino

    # ホストに torch を入れず、GPU の embedding-service へ投げる（EC2 推奨）
    # EMBEDDING_FINE_BACKEND=rad-dino を設定しておけば、1 回の POST で
    # 1 段目（BiomedCLIP）と 2 段目（RAD-DINO）の両方が得られる
    python scripts/region_signal_eval.py \
        --encoder deterministic,service,service-fine \
        --service-url http://localhost:8002

ROI 抽出が全体の律速である（112,120 枚で数時間）。encoder を 1 つずつ別々に
実行すると、そのたびに ROI をやり直すことになる。カンマ区切りで複数指定すれば
**ROI は 1 回だけ**行い、各 chunk をすべての encoder へ渡す。

ベクトルは tmp/region-signal/ へ encoder ごとにキャッシュする。全 encoder の
キャッシュが揃っていれば ROI は実行せず、検定だけをやり直す。
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import uuid
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# 本番と同じ実装を使う。評価用に書き直すと、測っているものが本番と食い違う。
for _path in ("packages/common/src", "packages/image-pipeline/src"):
    sys.path.insert(0, str(ROOT / _path))

from image_pipeline import ExtractionError, encode_png, extract  # noqa: E402
from nihcc_rag import csvspec  # noqa: E402
from nihcc_rag.regions import REGION_CODES  # noqa: E402

CACHE_DIR = ROOT / "tmp" / "region-signal"
DEFAULT_PROFILE = os.environ.get("ROI_PROFILE", "nihcc-classical-v1")


# --------------------------------------------------------------------------
# データ読み込み
# --------------------------------------------------------------------------


@dataclass
class Sample:
    name: str
    patient: str
    labels: frozenset[str]
    view: str | None

    @property
    def normal(self) -> bool:
        """5 部位すべてが正常と確定するか。判定規則は API 表示と共有する。"""
        return csvspec.region_finding_state(self.labels) == csvspec.REGION_STATE_NORMAL

    @property
    def single_finding(self) -> str | None:
        """単独所見。複数所見の画像は Test B のクエリから外すために使う。"""
        if len(self.labels) == 1:
            only = next(iter(self.labels))
            return None if only == "No Finding" else only
        return None


def _env_file_value(key: str) -> str | None:
    """`.env` から 1 項目を読む。

    シェルは `.env` を読まない（scripts/fetch_model.sh に同じ前提の注記がある）。
    `NIHCC_DATA_DIR` は `.env` にしか無いことが多いため、
    `${NIHCC_DATA_DIR}/images` と書いて空に展開され `/images` を見に行く事故が起きる。
    スクリプト側で拾っておけば、シェルへ export しなくても既定で動く。
    """
    path = ROOT / ".env"
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == key:
            return value.strip().strip('"').strip("'") or None
    return None


def _under_root(raw: str) -> Path:
    """相対パスはリポジトリ基準、絶対パスはそのまま。"""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (ROOT / path)


def resolve_data_dir(explicit: str) -> Path:
    """NIHCC データの置き場。明示指定 > 環境変数 > .env > ./NIHCC の順で決める。"""
    raw = (
        explicit
        or os.environ.get("NIHCC_DATA_DIR")
        or _env_file_value("NIHCC_DATA_DIR")
        or "NIHCC"
    )
    return _under_root(raw)


def resolve_input(explicit: str, data_dir: Path, leaf: str, what: str) -> Path:
    """入力パスを決め、存在しなければ原因の分かるメッセージで止める。"""
    path = _under_root(explicit) if explicit else (data_dir / leaf)
    if not path.exists():
        hint = ""
        if explicit and str(path) in ("/" + leaf, "\\" + leaf):
            # "${NIHCC_DATA_DIR}/images" が空に展開されて "/images" になった形
            hint = (
                "\n  シェル変数が空に展開されています。シェルは .env を読みません。"
                "\n  指定を省略すれば .env の NIHCC_DATA_DIR を自動で使います。"
            )
        raise SystemExit(
            f"{what} が見つかりません: {path}{hint}"
            f"\n  データ位置: {data_dir}"
            f"\n  --data-dir / --{what.lower()} で明示するか、.env の NIHCC_DATA_DIR を確認してください"
        )
    return path


def load_samples(csv_path: Path, image_dir: Path, view: str | None) -> list[Sample]:
    available = {p.name for p in image_dir.iterdir() if p.is_file()}
    samples: list[Sample] = []
    for row in csvspec.iter_rows(csv_path):
        if row.source_image_name not in available:
            continue
        if view and (row.view_position or "").upper() != view.upper():
            continue
        samples.append(
            Sample(
                name=row.source_image_name,
                patient=row.patient_key,
                labels=frozenset(row.finding_labels),
                view=row.view_position,
            )
        )
    return samples


# --------------------------------------------------------------------------
# ROI 抽出（worker プロセスで実行する）
# --------------------------------------------------------------------------


def _init_worker() -> None:
    """worker プロセスごとの初期化。

    OpenCV は既定でコア数ぶんのスレッドを立てる。ProcessPoolExecutor で
    N プロセス動かすと N x コア数のスレッドが競合し、1 枚あたりの所要時間が
    ホスト単体の実測（p50 約 210ms）より大きく伸びる。
    プロセス側で並列化しているので、OpenCV 内部は 1 スレッドに固定する。
    """
    import cv2

    cv2.setNumThreads(1)


#: (画像 PNG, {部位: mask PNG}) — 部位 mask pooling（ServicePoolEncoder）用
PoolPayload = tuple[bytes, dict[str, bytes]]


def _roi_worker(
    task: tuple[str, str, str, int],
) -> tuple[str, dict[str, bytes] | None, PoolPayload | None, str]:
    """1 枚の ROI 抽出。(画像名, {部位: crop PNG}, pool 用, エラー) を返す。

    REJECTED と crop 無しは本番検索でも除外されるため、ここでも落とす。

    `pool_side` が 0 でないとき、mask pooling 用に**全体画像と部位 mask**も返す。
    両者を同じ画素寸法へ揃えておくのが要点で、モデル側の入力解像度や patch grid
    がいくつでも同じ枠で縮約でき、部位の位置がずれない。

    **crop と mask で採用する部位を揃える。** ここを別条件にすると crop 方式と
    mask pooling 方式で対象部位が変わり、比較が成り立たなくなる。
    """
    name, path, profile, pool_side = task
    try:
        result = extract(
            Path(path).read_bytes(), profile_name=profile, regions=tuple(REGION_CODES)
        )
    except ExtractionError as exc:
        return name, None, None, exc.code
    except Exception as exc:  # noqa: BLE001 - 1 枚の失敗で全体を止めない
        return name, None, None, type(exc).__name__

    crops: dict[str, bytes] = {}
    usable: list[str] = []
    for code, region in result.regions.items():
        if region.quality_status == "REJECTED" or region.crop is None:
            continue
        crops[code] = encode_png(region.crop)
        usable.append(code)

    pool: PoolPayload | None = None
    if pool_side and usable:
        import cv2

        normalized = result.normalized
        height, width = normalized.shape[:2]
        side = (pool_side, pool_side)
        masks: dict[str, bytes] = {}
        for code in usable:
            region = result.regions[code]
            x0, y0, x1, y1 = region.bbox
            # bbox 内にしか無い mask を、原画像と同じ枠の 2 値画像へ戻す
            full = np.zeros((height, width), dtype=np.uint8)
            full[y0:y1, x0:x1] = (region.mask > 0).astype(np.uint8) * 255
            masks[code] = encode_png(cv2.resize(full, side, interpolation=cv2.INTER_AREA))
        pool = (
            encode_png(cv2.resize(normalized, side, interpolation=cv2.INTER_AREA)),
            masks,
        )
    return name, crops, pool, ""


def iter_roi(
    samples: Sequence[Sample],
    image_dir: Path,
    profile: str,
    workers: int,
    chunk: int,
    pool_side: int = 0,
) -> Iterable[list[tuple[str, dict[str, bytes], PoolPayload | None]]]:
    """ROI を並列抽出し、chunk 単位で返す。

    crop 全件をメモリに載せると 25,000 枚 x 約 30KB で 750MB になるため、
    chunk ごとに符号化まで済ませて捨てる。

    `pool_side` を指定すると mask pooling 用の全体画像と mask も載る。1 枚あたり
    約 100KB 増えるので、必要な encoder を指定したときだけ有効にする。
    """
    tasks = [(s.name, str(image_dir / s.name), profile, pool_side) for s in samples]
    failures: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as executor:
        batch: list[tuple[str, dict[str, bytes], PoolPayload | None]] = []
        for name, crops, payload, error in executor.map(_roi_worker, tasks, chunksize=8):
            if crops is None or not crops:
                failures[error or "NO_USABLE_REGION"] += 1
                continue
            batch.append((name, crops, payload))
            if len(batch) >= chunk:
                yield batch
                batch = []
        if batch:
            yield batch
    if failures:
        print(f"  ROI 失敗: {sum(failures.values())} 枚 {dict(failures)}", flush=True)


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------


def _load_production_backends():
    """本番の backends.py をパス指定で読み込む（apps/*/app との名前衝突を避ける）。"""
    import importlib.util

    path = ROOT / "services" / "embedding-service" / "app" / "backends.py"
    spec = importlib.util.spec_from_file_location("_prod_backends", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: 1 リクエストで送れる crop の上限。contract.py の
#: EmbedImagesRequest.images_base64 が max_length=256 のため、これを超えると 422。
SERVICE_MAX_BATCH = 256
#: 部位 mask pooling は全体画像を送るため 1 件が重い。contract の上限は 64 だが、
#: 1 リクエストの本文が 64 枚で 20MB 近くになるため既定は小さくしておく。
POOL_MAX_BATCH = 16
#: サービスが落ちても実行を継続するための再試行。embedding-service は
#: restart: unless-stopped で自動復帰するが、モデルの再ロードに 1〜2 分かかる。
POOL_RETRIES = 4
POOL_RETRY_WAIT = 45.0
#: pooling へ送る画像と mask の 1 辺。RAD-DINO の入力（518）に合わせる。
#: 画像と mask を同じ寸法にしておけば、モデル側の patch grid が変わっても
#: 同じ枠で縮約でき、部位の位置がずれない。
POOL_SIDE = 518


class ServiceEncoder:
    """起動中の embedding-service を叩く encoder。

    ホストに torch を入れずに GPU 推論を使える。ROI 抽出は呼び出し側（ホスト）で
    行い、埋め込みだけを HTTP で投げる。

    `field="fine_vectors"` にすると、同じ応答の 2 段目（fine）空間を読む。
    EMBEDDING_FINE_BACKEND を設定しておけば **1 回の POST で 2 つのモデルの
    ベクトルが得られる**ため、`service,service-fine` を同時に指定しても
    GPU 推論とネットワーク往復は 1 回で済む。
    """

    #: (base_url, crops の digest) -> 応答 JSON。`service` と `service-fine` が
    #: 同じ chunk を続けて要求するため、2 回 POST しないよう直近の応答を共有する。
    _shared: dict[tuple[str, str], dict] = {}

    def __init__(self, url: str, field: str = "vectors", batch: int = 0) -> None:
        import httpx

        self._url = url.rstrip("/")
        self._field = field
        self._batch = batch or SERVICE_MAX_BATCH
        self._client = httpx.Client(base_url=self._url, timeout=600.0)
        info = self._client.get("/api/v1/model").json()
        if field == "fine_vectors":
            if not info.get("rerank_available"):
                raise RuntimeError(
                    "embedding-service に 2 段目モデルが構成されていません。"
                    "EMBEDDING_FINE_BACKEND（例: rad-dino）と EMBEDDING_FINE_DIMENSION を"
                    "設定して再起動してください"
                )
            self.model_version = info["fine_embedding_model_version"]
            self.dimension = int(info["fine_dimension"])
        else:
            self.model_version = info["embedding_model_version"]
            self.dimension = int(info["embedding_dimension"])

    def _fetch(self, crops: list[bytes]) -> dict:
        """crops 全体ぶんの応答を取得する。応答は 1 段目/2 段目で共有する。

        1 リクエストあたりの枚数には契約上の上限があるため（contract.py の
        EmbedImagesRequest.images_base64 は max_length=256）、上限ごとに分割して
        投げ、結果を連結する。ROI の chunk は 1 枚あたり最大 5 部位ぶんの crop を
        生むので、200 枚の chunk では約 1,000 件になり 1 回では送れない。
        """
        import hashlib

        digest = hashlib.sha256(b"".join(crops)).hexdigest()
        key = (self._url, digest)
        cached = ServiceEncoder._shared.get(key)
        if cached is not None:
            return cached

        merged: dict[str, list | None] = {"vectors": [], "fine_vectors": []}
        for start in range(0, len(crops), self._batch):
            sub = crops[start : start + self._batch]
            response = self._client.post(
                "/api/v1/embeddings/images",
                json={"images_base64": [base64.b64encode(c).decode() for c in sub]},
            )
            response.raise_for_status()
            payload = response.json()
            merged["vectors"].extend(payload["vectors"])
            fine = payload.get("fine_vectors")
            if fine is None:
                merged["fine_vectors"] = None
            elif merged["fine_vectors"] is not None:
                merged["fine_vectors"].extend(fine)

        # 直近 1 件だけ保持する（chunk 単位で呼ばれるため、それ以上は不要）
        ServiceEncoder._shared.clear()
        ServiceEncoder._shared[key] = merged
        return merged

    def encode(self, crops: list[bytes]) -> np.ndarray:
        payload = self._fetch(crops)
        vectors = payload.get(self._field)
        if vectors is None:
            raise RuntimeError(
                f"embedding-service の応答に {self._field} がありません"
                "（2 段目モデルが無効か、テキスト専用の応答です）"
            )
        return np.asarray(vectors, dtype=np.float32)


class ServicePoolEncoder:
    """画像 1 回の順伝播 + 部位 mask pooling で部位ベクトルを得る。

    他の encoder が「crop を渡してベクトルを受け取る」のに対し、これは
    「全体画像と部位 mask を渡し、(部位, ベクトル) の並びを受け取る」。入出力の
    形が違うので `needs_region_masks` で区別し、呼び出し側が渡す物を変える。

    部位ごとに 5 回別々に符号化する crop 方式が部位固有の偏り（RUL の過剰予測）の
    発生源かどうかを確かめるための対照である。そうであれば、同一の順伝播から
    5 部位を出すこの作りで偏りが小さくなるはずである。
    """

    needs_region_masks = True

    def __init__(self, url: str, batch: int = 0) -> None:
        import httpx

        self._url = url.rstrip("/")
        #: 全体画像を送るため 1 件が重い。contract の上限（64）を超えない範囲で分割する。
        self._batch = max(1, min(batch or POOL_MAX_BATCH, POOL_MAX_BATCH))
        self._client = httpx.Client(base_url=self._url, timeout=600.0)
        info = self._client.get("/api/v1/model").json()
        if not info.get("region_pool_backend"):
            raise RuntimeError(
                "embedding-service が部位 mask pooling に対応していません。"
                "EMBEDDING_FINE_BACKEND=rad-dino を設定して再起動してください"
            )
        self.model_version = info["fine_embedding_model_version"] or info[
            "embedding_model_version"
        ]
        self.dimension = int(info["fine_dimension"] or info["embedding_dimension"])
        self.patch_grid = int(info["region_pool_patch_grid"])

    def encode_pool(
        self, payloads: list[PoolPayload]
    ) -> list[tuple[int, str, np.ndarray]]:
        """(画像 index, 部位コード, ベクトル) を返す。

        mask が patch grid 上で空になった部位はサービス側が返さないため、
        結果の件数は要求より少なくなりうる。index で突き合わせる。
        """
        out: list[tuple[int, str, np.ndarray]] = []
        for start in range(0, len(payloads), self._batch):
            sub = payloads[start : start + self._batch]
            body = {
                "images_base64": [base64.b64encode(image).decode() for image, _ in sub],
                "masks_base64": [
                    {code: base64.b64encode(png).decode() for code, png in masks.items()}
                    for _, masks in sub
                ],
            }
            response = self._post_with_retry(body, start)
            for item in response.json()["items"]:
                out.append(
                    (
                        start + int(item["image_index"]),
                        item["region_code"],
                        np.asarray(item["vector"], dtype=np.float32),
                    )
                )
        return out

    def _post_with_retry(self, body: dict, start: int):
        """落ちたサービスの復帰を待って再試行する。

        全量は 3 時間かかる。1 回の切断で最初からやり直すのは現実的でない。
        embedding-service は restart: unless-stopped で自動復帰するが、モデルの
        再ロードに 1〜2 分かかるため、待ってから投げ直す。
        """
        import httpx

        last: Exception | None = None
        for attempt in range(1, POOL_RETRIES + 1):
            try:
                response = self._client.post("/api/v1/embeddings/region-pool", json=body)
                response.raise_for_status()
                return response
            except (httpx.HTTPError, httpx.RemoteProtocolError) as exc:
                last = exc
                if attempt == POOL_RETRIES:
                    break
                wait = POOL_RETRY_WAIT * attempt
                print(
                    f"  !! embedding-service への送信に失敗 "
                    f"(画像 {start} から {len(body['images_base64'])} 枚, "
                    f"{type(exc).__name__}). {wait:.0f} 秒待って再試行 "
                    f"({attempt}/{POOL_RETRIES - 1})",
                    flush=True,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"embedding-service への送信が {POOL_RETRIES} 回失敗しました "
            f"(画像 {start} から). サービス側のログを確認してください: "
            f"docker compose logs --tail 100 embedding-service"
        ) from last


class BackendEncoder:
    """本番 backend をそのままプロセス内で使う encoder。"""

    def __init__(self, name: str, dimension: int, model_id: str, revision: str) -> None:
        backends = _load_production_backends()
        self._backend = backends._build_one(
            role="coarse",
            requested=name,
            dimension=dimension,
            model_id=model_id,
            revision=revision,
            replicas=1,
        )
        self._backend.self_test()
        self.model_version = self._backend.model_version
        self.dimension = self._backend.dimension

    def encode(self, crops: list[bytes]) -> np.ndarray:
        return self._backend.encode_images(crops)


def build_encoder(name: str, args) -> object:
    if name == "service":
        return ServiceEncoder(args.service_url, field="vectors", batch=args.service_batch)
    if name == "service-fine":
        return ServiceEncoder(args.service_url, field="fine_vectors", batch=args.service_batch)
    if name == "service-pool":
        return ServicePoolEncoder(args.service_url, batch=args.pool_batch)
    defaults = {
        "deterministic": ("", "", 512),
        "biomedclip": (
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
            os.environ.get("EMBEDDING_MODEL_REVISION", ""),
            512,
        ),
        "rad-dino": ("microsoft/rad-dino", os.environ.get("EMBEDDING_FINE_MODEL_REVISION", ""), 768),
    }
    model_id, revision, dimension = defaults[name]
    return BackendEncoder(
        name, args.dimension or dimension, args.model_id or model_id, args.revision or revision
    )


# --------------------------------------------------------------------------
# ベクトル生成（キャッシュ付き）
# --------------------------------------------------------------------------


@dataclass
class VectorStore:
    """部位ごとの (画像名, ベクトル)。すべて L2 正規化済み。"""

    names: dict[str, list[str]] = field(default_factory=lambda: {c: [] for c in REGION_CODES})
    vectors: dict[str, list[np.ndarray]] = field(
        default_factory=lambda: {c: [] for c in REGION_CODES}
    )

    def add(self, code: str, name: str, vector: np.ndarray) -> None:
        self.names[code].append(name)
        self.vectors[code].append(vector)

    def finalise(self) -> dict[str, tuple[list[str], np.ndarray]]:
        out = {}
        for code in REGION_CODES:
            if self.vectors[code]:
                out[code] = (self.names[code], np.vstack(self.vectors[code]).astype(np.float32))
        return out


def _cache_path(encoder_name: str, args, total: int) -> Path:
    return CACHE_DIR / f"vec_{encoder_name}_{args.profile}_{total}.npz"


def _load_cache(path: Path) -> dict[str, tuple[list[str], np.ndarray]]:
    blob = np.load(path, allow_pickle=True)
    return {
        code: (list(blob[f"{code}_names"]), blob[f"{code}_vecs"])
        for code in REGION_CODES
        if f"{code}_vecs" in blob
    }


def compute_vectors(
    samples: Sequence[Sample], image_dir: Path, encoders: dict[str, object], args
) -> dict[str, dict[str, tuple[list[str], np.ndarray]]]:
    """ROI を 1 回だけ実行し、各 chunk をすべての encoder へ渡す。

    ROI 抽出が律速（112,120 枚で数時間）なので、encoder ごとにやり直してはならない。
    キャッシュが揃っている encoder は ROI パスの対象から外す。
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, tuple[list[str], np.ndarray]]] = {}
    pending: dict[str, object] = {}
    for name, encoder in encoders.items():
        path = _cache_path(name, args, len(samples))
        if path.exists() and not args.refresh:
            print(f"キャッシュを使用 [{name}]: {path.relative_to(ROOT)}", flush=True)
            results[name] = _load_cache(path)
        else:
            pending[name] = encoder
    if not pending:
        return results

    stores = {name: VectorStore() for name in pending}
    # 部位 mask pooling を使う encoder があるときだけ、ROI に全体画像と mask を
    # 載せる（1 枚あたり約 100KB 増えるため）。
    wants_pool = any(getattr(e, "needs_region_masks", False) for e in pending.values())
    started = time.perf_counter()
    done = 0
    print(f"ROI 抽出を開始（対象 encoder: {', '.join(pending)}）", flush=True)
    for batch in iter_roi(
        samples,
        image_dir,
        args.profile,
        args.workers,
        args.chunk,
        pool_side=args.pool_side if wants_pool else 0,
    ):
        # chunk 内の全部位 crop を 1 度に符号化する（バッチ効率のため）
        flat_crops: list[bytes] = []
        index: list[tuple[str, str]] = []
        for name, crops, _ in batch:
            for code, png in crops.items():
                flat_crops.append(png)
                index.append((code, name))
        pool_names = [name for name, _, payload in batch if payload is not None]
        pool_payloads = [payload for _, _, payload in batch if payload is not None]
        for enc_name, encoder in pending.items():
            store = stores[enc_name]
            if getattr(encoder, "needs_region_masks", False):
                if not pool_payloads:
                    continue
                for image_index, code, vector in encoder.encode_pool(pool_payloads):
                    store.add(code, pool_names[image_index], np.asarray(vector, dtype=np.float32))
                continue
            vectors = encoder.encode(flat_crops)
            for (code, name), vector in zip(index, vectors, strict=True):
                store.add(code, name, np.asarray(vector, dtype=np.float32))
        done += len(batch)
        elapsed = time.perf_counter() - started
        rate = done / max(1e-9, elapsed)
        eta = (len(samples) - done) / max(1e-9, rate)
        print(
            f"  {done}/{len(samples)} 枚 ({rate:.1f} 枚/秒, 残り {eta / 60:.0f} 分)",
            flush=True,
        )

    for enc_name, store in stores.items():
        result = store.finalise()
        path = _cache_path(enc_name, args, len(samples))
        np.savez_compressed(
            path,
            **{f"{c}_names": np.array(n, dtype=object) for c, (n, _) in result.items()},
            **{f"{c}_vecs": v for c, (_, v) in result.items()},
        )
        print(f"キャッシュへ保存 [{enc_name}]: {path.relative_to(ROOT)}", flush=True)
        results[enc_name] = result
    return results


# --------------------------------------------------------------------------
# 統計
# --------------------------------------------------------------------------


def test_a_ceiling(samples, abnormal_regions_per_image: float) -> tuple[float, float]:
    """Test A が原理的に到達しうる上限を返す (正常部位の No Finding 率, lift)。

    **Test A の lift には低い天井がある。** クエリは `No Finding` 画像の部位なので
    真に正常であり、理想的な検索はやはり真に正常な部位を返す。しかし所見のある画像
    でも大半の部位は正常なので、返ってくる正常部位のうち `No Finding` 画像由来は
    一部にすぎない。

        上限 = (NF 画像数 x 5) / (NF 画像数 x 5 + 異常画像数 x (5 - k))  ÷ 基準率

    k は異常画像 1 枚あたりの異常部位数。本データ（NF 53.8%、所見あり画像の平均
    所見数 1.57）では k=1 で lift 1.10、k=2 で 1.23 にしかならない。

    この天井を知らずに「帰無比 1.3 以上」のような目標を置くと、**達成不可能な条件**
    になる。また、天井を超える
    値が出た encoder は部位レベルの正常判定ではなく**画像レベルの nuisance**
    （撮影条件・体格など、No Finding であることと相関する見た目）を拾っている疑いがある。
    """
    total = len(samples)
    if not total:
        return (0.0, float("nan"))
    n_normal = sum(1 for s in samples if s.normal)
    n_abnormal = total - n_normal
    base = n_normal / total
    k = max(0.0, min(5.0, abnormal_regions_per_image))
    numerator = n_normal * len(REGION_CODES)
    denominator = numerator + n_abnormal * (len(REGION_CODES) - k)
    share = numerator / denominator if denominator else 0.0
    return (share, share / base if base else float("nan"))


def wilson(successes: float, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% 信頼区間。比率が 0/1 に近くても破綻しない。"""
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    margin = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def cluster_bootstrap(
    per_query: list[float],
    patients: list[str],
    iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """患者を単位とした bootstrap 信頼区間（両側 1 - alpha）。

    同一患者の複数クエリは相関するため、クエリ単位の resample では区間が狭すぎる。

    alpha は多重比較の補正込みで呼び出し側が決める。5 部位ぶん検定を繰り返すため、
    部位ごとに alpha=0.05 を使うと帰無でも高い確率でどこかの部位が有意になる。
    """
    if not per_query:
        return (0.0, 0.0)
    by_patient: dict[str, list[float]] = defaultdict(list)
    for value, patient in zip(per_query, patients, strict=True):
        by_patient[patient].append(value)
    keys = list(by_patient)
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        picked = [by_patient[keys[rng.randrange(len(keys))]] for _ in range(len(keys))]
        flat = [v for group in picked for v in group]
        means.append(sum(flat) / len(flat))
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return (lo, hi)


#: 類似度行列 1 回ぶんのメモリ上限。クエリをこの範囲に収まる数ずつ処理する。
SIMILARITY_BUDGET_BYTES = 256 * 1024 * 1024


def top_k_indices(
    queries: np.ndarray, index: np.ndarray, k: int, budget: int = SIMILARITY_BUDGET_BYTES
) -> np.ndarray:
    """cosine 類似度の上位 K。ベクトルは L2 正規化済みなので内積でよい。

    クエリ x index の類似度行列を一括で確保しない。全量 112,120 枚では
    1 部位あたり 12,000 x 89,700 x 4 バイト = 約 4.3GB になり確保に失敗する。
    メモリ上限から逆算したバッチ数ずつ処理する。
    """
    k = min(k, index.shape[0])
    if queries.shape[0] == 0:
        return np.empty((0, k), dtype=np.int64)
    per_query = max(1, index.shape[0] * 4)
    batch = max(1, min(queries.shape[0], budget // per_query))
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], batch):
        chunk = queries[start : start + batch]
        sims = chunk @ index.T
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        out[start : start + chunk.shape[0]] = np.take_along_axis(
            part, np.argsort(-np.take_along_axis(sims, part, 1), axis=1), 1
        )
    return out


# --------------------------------------------------------------------------
# 検定
# --------------------------------------------------------------------------


#: 「シグナルあり」と呼ぶために必要な最小 lift（基準率からの相対変化）。
#:
#: 95% 信頼区間が基準率を外れることだけを条件にすると、5 部位 x 複数所見ぶん検定を
#: 繰り返すため偽陽性が頻発する（帰無コーパス 60 回で 8.3% が誤判定。95%CI の
#: 期待どおりの挙動であり、統計側の不具合ではない）。加えて、統計的に有意でも
#: lift 1.05 のような差はモデル採否の根拠にならない。そこで統計的有意性と
#: 実用的有意性の両方を要求する。帰無での lift の標準偏差は約 0.05 なので、
#: 0.10 はおよそ 2 SD に当たる。
DEFAULT_MIN_LIFT = 0.10

#: 判定に必要な最小クエリ数。これを下回ると区間推定が意味を持たない。
MIN_QUERIES = 20


@dataclass
class Row:
    label: str
    queries: int
    observed: float
    base: float
    ci: tuple[float, float]
    min_lift: float = DEFAULT_MIN_LIFT

    @property
    def lift(self) -> float:
        return self.observed / self.base if self.base > 0 else float("nan")

    @property
    def verdict(self) -> str:
        """統計的有意性（CI が基準率を外れる）と実用的有意性（lift 差）の両方を要求する。"""
        if self.queries < MIN_QUERIES:
            return "サンプル不足"
        lift = self.lift
        if not np.isfinite(lift):
            return "判定不能"
        if abs(lift - 1.0) < self.min_lift:
            # 統計的に有意でも、この幅の差はモデル採否の根拠にならない
            return "シグナルなし"
        if self.ci[0] > self.base:
            return "シグナルあり"
        if self.ci[1] < self.base:
            return "逆相関"
        return "シグナルなし"


def render(title: str, note: str, rows: list[Row]) -> str:
    out = [f"\n=== {title} ===", note, ""]
    if rows:
        out.insert(
            2,
            f"判定: 信頼区間が帰無を外れ、かつ lift の差が {rows[0].min_lift:.0%} 以上"
            f"（クエリ {MIN_QUERIES} 件未満は判定しない。区間は 5 部位ぶん Bonferroni 補正）",
        )
    out.append(f"{'':<8}{'クエリ':>7}{'観測':>9}{'帰無':>9}{'lift':>7}  {'95%CI':<18}判定")
    for row in rows:
        ci = f"[{row.ci[0]:.1%}, {row.ci[1]:.1%}]"
        out.append(
            f"{row.label:<8}{row.queries:>7}{row.observed:>9.1%}{row.base:>9.1%}"
            f"{row.lift:>7.2f}  {ci:<18}{row.verdict}"
        )
    return "\n".join(out)


def run_tests(
    vectors: dict[str, tuple[list[str], np.ndarray]],
    samples: Sequence[Sample],
    args,
) -> tuple[str, dict]:
    by_name = {s.name: s for s in samples}
    rng = random.Random(args.seed)
    # 各検定は 5 部位ぶん繰り返すため、族単位で Bonferroni 補正する。
    # 補正しないと帰無でも「どこかの部位が有意」になる確率が 20% を超える。
    alpha = args.alpha / len(REGION_CODES)

    # --- 患者単位分割 -----------------------------------------------------
    patients = sorted({s.patient for s in samples})
    rng.shuffle(patients)
    n_query = max(1, int(len(patients) * args.query_fraction))
    query_patients = set(patients[:n_query])

    report: dict = {
        "encoder": args.encoder,
        "top_k": args.top_k,
        "min_lift": args.min_lift,
        "alpha": args.alpha,
        "alpha_corrected": alpha,
        "images": len(samples),
        "patients": len(patients),
        "query_patients": len(query_patients),
    }
    lines: list[str] = []

    # 部位ごとに index / query を分ける
    split: dict[str, dict] = {}
    for code, (names, vecs) in vectors.items():
        q_idx = [i for i, n in enumerate(names) if by_name[n].patient in query_patients]
        i_idx = [i for i, n in enumerate(names) if by_name[n].patient not in query_patients]
        split[code] = {
            "q_names": [names[i] for i in q_idx],
            "q_vecs": vecs[q_idx],
            "i_names": [names[i] for i in i_idx],
            "i_vecs": vecs[i_idx],
        }

    # --- Test A: 正常部位の enrichment ------------------------------------
    rows_a: list[Row] = []
    for code in REGION_CODES:
        if code not in split or len(split[code]["i_names"]) < args.top_k:
            continue
        i_names = split[code]["i_names"]
        base = sum(by_name[n].normal for n in i_names) / len(i_names)
        q_pick = [
            (j, n) for j, n in enumerate(split[code]["q_names"]) if by_name[n].normal
        ]
        if not q_pick:
            continue
        rows = np.array([j for j, _ in q_pick])
        neigh = top_k_indices(split[code]["q_vecs"][rows], split[code]["i_vecs"], args.top_k)
        per_query = [
            float(np.mean([by_name[i_names[c]].normal for c in row])) for row in neigh
        ]
        ci = cluster_bootstrap(
            per_query, [by_name[n].patient for _, n in q_pick], args.bootstrap, args.seed, alpha
        )
        rows_a.append(Row(code, len(per_query), float(np.mean(per_query)), base, ci, args.min_lift))
    # Test A の天井を併記する。これを知らずに目標値を置くと達成不可能な条件になる。
    ceilings = [
        (k, test_a_ceiling(samples, k)[1]) for k in (1.0, 1.5, 2.0)
    ]
    ceiling_note = "  ".join(f"異常部位 {k:.1f}/画像 → lift {c:.2f}" for k, c in ceilings)
    lines.append(
        render(
            "Test A: 正常部位の enrichment",
            "クエリ = No Finding 画像（5 部位すべて正常と確定）。"
            "帰無 = index 集合の No Finding 率。\n"
            f"**理論上限**（部位レベルの正常判定が完璧な場合）: {ceiling_note}\n"
            "上限を超える値は、部位ではなく画像レベルの nuisance を拾っている疑いがある。",
            rows_a,
        )
    )
    report["test_a_ceiling"] = {str(k): c for k, c in ceilings}
    report["test_a"] = [vars(r) | {"lift": r.lift, "verdict": r.verdict} for r in rows_a]

    # --- Test B: 所見の enrichment ---------------------------------------
    findings = Counter(
        f for s in samples if (f := s.single_finding) and s.patient in query_patients
    )
    report["test_b"] = {}
    for finding, _count in findings.most_common(args.findings):
        rows_b: list[Row] = []
        for code in REGION_CODES:
            if code not in split or len(split[code]["i_names"]) < args.top_k:
                continue
            i_names = split[code]["i_names"]
            base = sum(finding in by_name[n].labels for n in i_names) / len(i_names)
            q_pick = [
                (j, n)
                for j, n in enumerate(split[code]["q_names"])
                if by_name[n].single_finding == finding
            ]
            if not q_pick:
                continue
            rows = np.array([j for j, _ in q_pick])
            neigh = top_k_indices(split[code]["q_vecs"][rows], split[code]["i_vecs"], args.top_k)
            per_query = [
                float(np.mean([finding in by_name[i_names[c]].labels for c in row]))
                for row in neigh
            ]
            ci = cluster_bootstrap(
                per_query, [by_name[n].patient for _, n in q_pick], args.bootstrap, args.seed, alpha
            )
            rows_b.append(Row(code, len(per_query), float(np.mean(per_query)), base, ci, args.min_lift))
        if rows_b:
            lines.append(
                render(
                    f"Test B: {finding} の enrichment",
                    f"クエリ = {finding} 単独の画像。帰無 = index 集合の {finding} 出現率。"
                    "【注意】ヒット判定は**画像単位のラベル**なので、この値は"
                    "画像レベルの検出力を測る。部位レベルの寄与を見るには、部位情報を"
                    "捨てた対照（region_mil_profile.py --output broadcast）と比べること。",
                    rows_b,
                )
            )
            report["test_b"][finding] = [
                vars(r) | {"lift": r.lift, "verdict": r.verdict} for r in rows_b
            ]

    # --- Test C: 部位同定の対照 -------------------------------------------
    all_names = [n for code in REGION_CODES if code in split for n in split[code]["i_names"]]
    all_codes = [code for code in REGION_CODES if code in split for _ in split[code]["i_names"]]
    all_vecs = np.vstack([split[code]["i_vecs"] for code in REGION_CODES if code in split])
    rows_c: list[Row] = []
    for code in REGION_CODES:
        if code not in split or not len(split[code]["q_names"]):
            continue
        take = min(args.control_queries, len(split[code]["q_names"]))
        picked = list(range(len(split[code]["q_names"])))
        rng.shuffle(picked)
        picked = picked[:take]
        neigh = top_k_indices(split[code]["q_vecs"][np.array(picked)], all_vecs, args.top_k)
        per_query = [float(np.mean([all_codes[c] == code for c in row])) for row in neigh]
        names = [split[code]["q_names"][p] for p in picked]
        ci = cluster_bootstrap(
            per_query, [by_name[n].patient for n in names], args.bootstrap, args.seed, alpha
        )
        # 帰無 = index 集合に占める当該部位の割合（5 部位ならおよそ 20%）
        base = all_codes.count(code) / len(all_codes)
        rows_c.append(Row(code, len(per_query), float(np.mean(per_query)), base, ci, args.min_lift))
    lines.append(
        render(
            "Test C: 部位同定の対照",
            "全部位を横断検索したときに上位 K が同一部位で占められる割合。"
            "100% に近いほど、ベクトルは病理ではなく部位の形を符号化している。",
            rows_c,
        )
    )
    report["test_c"] = [vars(r) | {"lift": r.lift, "verdict": r.verdict} for r in rows_c]
    _ = all_names

    return "\n".join(lines), report


# --------------------------------------------------------------------------


def _compare(reports: dict[str, dict], names: list[str]) -> str:
    """Test A の lift を encoder 間で並べる。

    判定の基準は基準率（lift=1.0）ではなく **deterministic の lift** である。
    決定論的 stub は医学的意味を持たないが、撮影条件・体格・画質と `No Finding` の
    相関（nuisance correlation）を拾うため lift が 1 を超える。実モデルがこれを
    超えられなければ、病変ではなく同じ nuisance を見ているだけと解釈する。
    """
    out = ["", "=" * 70, " encoder 比較: Test A（正常部位の enrichment）の lift", "=" * 70, ""]
    null_by_region: dict[str, float] = {}
    if "deterministic" in reports:
        null_by_region = {r["label"]: r["lift"] for r in reports["deterministic"]["test_a"]}

    header = f"{'部位':<6}" + "".join(f"{n:>16}" for n in names)
    if null_by_region:
        header += f"{'帰無比':>10}"
    out.append(header)
    for code in REGION_CODES:
        cells = []
        for name in names:
            row = next((r for r in reports[name]["test_a"] if r["label"] == code), None)
            cells.append(f"{row['lift']:>16.3f}" if row else f"{'-':>16}")
        line = f"{code:<6}" + "".join(cells)
        if null_by_region and code in null_by_region:
            best = max(
                (next((r["lift"] for r in reports[n]["test_a"] if r["label"] == code), 0.0))
                for n in names
                if n != "deterministic"
            ) if len(names) > 1 else 0.0
            line += f"{best / null_by_region[code]:>10.2f}" if null_by_region[code] else f"{'-':>10}"
        out.append(line)
    out.append("")
    out.append(
        "帰無比 = （deterministic 以外の最大 lift）/（deterministic の lift）。"
        "1.0 前後なら、実モデルは stub と同じ nuisance を見ているだけである。"
    )
    return "\n".join(out)


def _record(reports: dict, names: list[str], args, sample_size: int) -> None:
    """評価結果を DB へ残す（docs/OPERATIONS.md §6）。

    **encoder ごとに 1 実行**として記録する。まとめて 1 実行にすると、どの encoder の
    数字かを指標名へ埋め込むことになり、あとから条件で絞れなくなる。

    `deterministic` は帰無（null）である。他の encoder の指標にはその値を
    `baseline` として添える。**帰無を測らずに「効いた」と言わないため**の欄で、
    これが無いと画像レベルの性能を部位レベルの性能と取り違えやすい。
    """
    from nihcc_rag import evaluation

    null_report = reports.get("deterministic", {})

    def null_lift(section: str, label: str, finding: str | None = None) -> float | None:
        rows = null_report.get(section)
        if finding is not None:
            rows = (rows or {}).get(finding)
        if not rows:
            return None
        row = next((r for r in rows if r["label"] == label), None)
        return None if row is None else row.get("lift")

    for name in names:
        report = reports.get(name) or {}
        run = evaluation.Run(run_id=uuid.uuid4())

        for row in report.get("test_a", []):
            run.metric("test_a_lift", row["lift"], region_code=row["label"],
                       baseline=None if name == "deterministic" else null_lift("test_a", row["label"]))
        for finding, rows in (report.get("test_b") or {}).items():
            for row in rows:
                run.metric("test_b_lift", row["lift"], region_code=row["label"], finding=finding,
                           baseline=None if name == "deterministic"
                           else null_lift("test_b", row["label"], finding))
        for row in report.get("test_c", []):
            run.metric("test_c_lift", row["lift"], region_code=row["label"],
                       baseline=None if name == "deterministic" else null_lift("test_c", row["label"]))

        evaluation.persist(
            run,
            kind="region-signal",
            target_version=report.get("model_version") or name,
            params={"encoder": name, "top_k": getattr(args, "top_k", None),
                    "findings": getattr(args, "findings", None),
                    "dimension": report.get("dimension"),
                    "is_null": name == "deterministic"},
            sample_size=sample_size,
            note=getattr(args, "note", "") or None,
        )
        if not run.persisted:
            print(f"（{name}: DB へ記録できなかった。評価そのものは成功している）")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--encoder",
        default="deterministic",
        help="カンマ区切りで複数指定できる（ROI は 1 回だけ実行される）。"
        "deterministic / biomedclip / rad-dino / service / service-fine。"
        "service と service-fine を同時に指定すると、1 回の POST で 1 段目と"
        "2 段目の両モデルのベクトルを取得する",
    )
    parser.add_argument("--service-url", default="http://localhost:8002")
    parser.add_argument(
        "--service-batch",
        type=int,
        default=SERVICE_MAX_BATCH,
        help=f"1 リクエストで送る crop 数（契約上の上限 {SERVICE_MAX_BATCH}）",
    )
    parser.add_argument(
        "--pool-batch",
        type=int,
        default=POOL_MAX_BATCH,
        help=f"service-pool が 1 リクエストで送る画像数（契約上の上限 {POOL_MAX_BATCH}）",
    )
    parser.add_argument(
        "--pool-side",
        type=int,
        default=POOL_SIDE,
        help=f"service-pool へ送る画像と mask の 1 辺（既定 {POOL_SIDE}）",
    )
    parser.add_argument("--model-id", default="")
    parser.add_argument("--revision", default="")
    parser.add_argument("--dimension", type=int, default=0)
    # 既定は .env の NIHCC_DATA_DIR から導く。シェルへ export しなくても動く。
    parser.add_argument(
        "--data-dir", default="", help="NIHCC データの置き場（既定: .env の NIHCC_DATA_DIR）"
    )
    parser.add_argument("--csv", default="", help="既定: <data-dir>/Data_Entry_2017_v2020.csv")
    parser.add_argument("--images", default="", help="既定: <data-dir>/images")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--limit", type=int, default=0, help="使用する画像数の上限（0 で全件）")
    parser.add_argument("--view", default="", help="撮影方向で絞る（PA / AP）")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--query-fraction", type=float, default=0.2)
    parser.add_argument("--findings", type=int, default=4, help="Test B で扱う所見の数")
    parser.add_argument("--control-queries", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="有意水準（5 部位ぶん Bonferroni 補正する）"
    )
    parser.add_argument(
        "--min-lift",
        type=float,
        default=DEFAULT_MIN_LIFT,
        help="「シグナルあり」と呼ぶために必要な基準率からの相対変化",
    )
    # ROI は PNG 読み込みと復号の待ちを含むため、コア数より多めの worker が速い
    # （実測: コア数 -1 に対しコア数 x2 で 1.29 倍）。
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) * 2))
    parser.add_argument("--chunk", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--refresh", action="store_true", help="ベクトルキャッシュを作り直す")
    parser.add_argument("--note", default="", help="DB に残す覚書")
    parser.add_argument("--json", default="", help="結果を JSON で書き出すパス")
    args = parser.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    csv_path = resolve_input(args.csv, data_dir, "Data_Entry_2017_v2020.csv", "CSV")
    image_dir = resolve_input(args.images, data_dir, "images", "images")
    samples = load_samples(csv_path, image_dir, args.view or None)
    if not samples:
        raise SystemExit(
            f"CSV と画像の突合結果が 0 件です: csv={csv_path} images={image_dir}"
            + (f" view={args.view}" if args.view else "")
        )
    if args.limit:
        rng = random.Random(args.seed)
        # 患者ごとにまとめて選ぶ。画像単位で切ると同一患者が分割を跨ぐ
        by_patient: dict[str, list[Sample]] = defaultdict(list)
        for s in samples:
            by_patient[s.patient].append(s)
        keys = sorted(by_patient)
        rng.shuffle(keys)
        picked: list[Sample] = []
        for key in keys:
            if len(picked) >= args.limit:
                break
            picked.extend(by_patient[key])
        samples = picked[: args.limit]

    names = [n.strip() for n in args.encoder.split(",") if n.strip()]
    # 既知の encoder に加え、キャッシュが既にある名前は受け付ける。
    # 他のスクリプト（region_mil_profile.py など）が書き出したベクトルを、
    # 同じ土俵で比較できるようにするため。
    allowed = {
        "deterministic",
        "biomedclip",
        "rad-dino",
        "service",
        "service-fine",
        "service-pool",
    }
    unknown = [
        n for n in names
        if n not in allowed and not _cache_path(n, args, len(samples)).exists()
    ]
    if unknown:
        parser.error(
            f"unknown encoder(s): {unknown}; "
            f"choose from {sorted(allowed)} またはキャッシュ済みの名前"
        )

    print(f"画像 {len(samples)} 枚 / 患者 {len({s.patient for s in samples})} 人 "
          f"/ encoder={','.join(names)} / profile={args.profile}", flush=True)
    print(f"データ: {image_dir}", flush=True)

    encoders: dict[str, object] = {}
    for name in names:
        # キャッシュ済みの encoder はモデルをロードしない（GPU も torch も要らない）
        if _cache_path(name, args, len(samples)).exists() and not args.refresh:
            continue
        encoders[name] = build_encoder(name, args)
        print(
            f"encoder ready [{name}]: {encoders[name].model_version} "
            f"({encoders[name].dimension} 次元)",
            flush=True,
        )
    # キャッシュ済みぶんも結果の読み出し対象に含める
    vectors_by_encoder = compute_vectors(
        samples,
        image_dir,
        {n: encoders.get(n) for n in names if n in encoders} or {},
        args,
    )
    for name in names:
        if name in vectors_by_encoder:
            continue
        path = _cache_path(name, args, len(samples))
        if not path.exists():
            parser.error(f"encoder {name} のベクトルがありません: {path}")
        print(f"キャッシュを使用 [{name}]: {path.relative_to(ROOT)}", flush=True)
        vectors_by_encoder[name] = _load_cache(path)

    reports: dict[str, dict] = {}
    for name in names:
        vectors = vectors_by_encoder[name]
        covered = {c: len(v[0]) for c, v in vectors.items()}
        print(f"\n{'=' * 70}\n encoder: {name}   部位ベクトル: {covered}\n{'=' * 70}",
              flush=True)
        text, report = run_tests(vectors, samples, args)
        if name in encoders:
            report["model_version"] = encoders[name].model_version
            report["dimension"] = encoders[name].dimension
        reports[name] = report
        print(text)

    if len(names) > 1:
        print(_compare(reports, names))

    _record(reports, names, args, len(samples))

    if args.json:
        Path(args.json).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nJSON を書き出しました: {args.json}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
