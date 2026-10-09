"""部位 mask pooling（region_signal_eval.py の ServicePoolEncoder）の演算を固定する。

mask pooling は「画像 1 回の順伝播 + 部位 mask pooling にすれば、部位ごとに
5 回別々に符号化することで生じる偏り（RUL の過剰予測）が構造的に小さくなるか」を
確かめるための方式である。その検証は pooling が
**部位の面積に依存しない**ことに依存している。総和で pooling すると面積が
そのままベクトルの大きさになり、大きい部位と小さい部位で向きが揃わない。

小さい部位が patch grid で消えないことも固定する。RML は正解 box の 4.4% しか
占めず、crop 方式の特徴では一度も予測されないことがある。
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "packages" / "common" / "src"))
_spec = importlib.util.spec_from_file_location(
    "eb_backends", REPO / "services" / "embedding-service" / "app" / "backends.py"
)
backends = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backends)

GRID = 37


def mask_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def solid(side: int, box: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = box
    canvas = np.zeros((side, side), dtype=np.uint8)
    canvas[y0:y1, x0:x1] = 255
    return canvas


# --- 被覆率 ------------------------------------------------------------------


def test_weights_are_coverage_fractions() -> None:
    weights = backends.mask_weights(mask_png(solid(518, (0, 0, 518, 518))), GRID)
    assert weights.shape == (GRID, GRID)
    assert np.allclose(weights, 1.0)


def test_empty_mask_gives_zero_weights() -> None:
    weights = backends.mask_weights(mask_png(np.zeros((518, 518))), GRID)
    assert float(weights.sum()) == 0.0


def test_partial_coverage_is_between_zero_and_one() -> None:
    """面積平均なので、半分だけかかる patch は中間の重みになる。

    最近傍だと 0 か 1 に振れ、境界の patch が丸ごと入るか丸ごと落ちるかになる。
    """
    weights = backends.mask_weights(mask_png(solid(518, (0, 0, 518, 259))), GRID)
    assert 0.45 < float(weights.mean()) < 0.55
    assert weights.max() == pytest.approx(1.0, abs=0.02)


def test_small_region_survives_the_patch_grid() -> None:
    """RML 相当の小さな部位が patch grid で消えないこと。

    RML は正解の 4.4% しか占めず、crop 方式の特徴では一度も予測されないことがある。
    pooling 側でさらに落とすと、mask pooling でも RML が評価できない。
    """
    # 518 の 1/10 角 ≒ 面積 1%。実画像の RML（全体の約 2.2%）より小さい
    weights = backends.mask_weights(mask_png(solid(518, (200, 200, 252, 252))), GRID)
    assert float(weights.sum()) > 0.0
    assert int((weights > 0).sum()) >= 9


# --- pooling -----------------------------------------------------------------


def test_uniform_weights_give_the_plain_mean() -> None:
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(GRID, GRID, 8))
    pooled = backends.pool_patches(patches, np.ones((GRID, GRID), dtype=np.float32))
    assert np.allclose(pooled, patches.mean((0, 1)), atol=1e-5)


def test_pooling_selects_only_the_masked_patches() -> None:
    patches = np.zeros((GRID, GRID, 4), dtype=np.float32)
    patches[3, 5] = [1.0, 2.0, 3.0, 4.0]
    weights = np.zeros((GRID, GRID), dtype=np.float32)
    weights[3, 5] = 1.0
    assert np.allclose(backends.pool_patches(patches, weights), [1.0, 2.0, 3.0, 4.0])


def test_pooling_is_independent_of_region_area() -> None:
    """同じ内容の部位なら、面積が違っても同じベクトルになること。

    **mask pooling の比較が成立する前提**である。総和で pooling すると面積が
    そのままベクトルの大きさになり、部位の大小が向きの違いとして残る。
    """
    patches = np.zeros((GRID, GRID, 3), dtype=np.float32)
    patches[:] = [0.2, -0.5, 0.9]
    small = np.zeros((GRID, GRID), dtype=np.float32)
    small[0, 0] = 1.0
    large = np.ones((GRID, GRID), dtype=np.float32)
    # 許容差は float32 で 1,369 要素を総和したときの累積誤差に合わせる。
    # モデル側が fp16 で推論するため、この差は元から乗っている誤差より小さい。
    assert np.allclose(
        backends.pool_patches(patches, small),
        backends.pool_patches(patches, large),
        atol=1e-4,
    )


def test_partial_coverage_weights_the_average() -> None:
    patches = np.zeros((GRID, GRID, 1), dtype=np.float32)
    patches[0, 0] = 10.0
    patches[0, 1] = 0.0
    weights = np.zeros((GRID, GRID), dtype=np.float32)
    weights[0, 0] = 0.25
    weights[0, 1] = 0.75
    assert backends.pool_patches(patches, weights)[0] == pytest.approx(2.5)


def test_empty_mask_returns_none() -> None:
    """被覆が皆無の部位は失敗ではなく「返さない」こと。

    例外にすると 1 部位のために画像 1 枚ぶんが落ちる。
    """
    patches = np.ones((GRID, GRID, 2), dtype=np.float32)
    assert backends.pool_patches(patches, np.zeros((GRID, GRID), dtype=np.float32)) is None


def test_pooling_keeps_float32() -> None:
    """float64 へ昇格させないこと。キャッシュと転送量が倍になる。"""
    patches = np.ones((GRID, GRID, 2), dtype=np.float32)
    weights = np.ones((GRID, GRID), dtype=np.float32)
    assert backends.pool_patches(patches, weights).dtype == np.float32


# --- 実際の mask 形状での振る舞い --------------------------------------------


def test_disjoint_regions_do_not_share_patches_after_reduction() -> None:
    """排他な部位は縮約後もほぼ排他であること。

    重なると、ある部位のベクトルに隣の部位の patch が混ざる。
    """
    left = backends.mask_weights(mask_png(solid(518, (0, 0, 259, 518))), GRID)
    right = backends.mask_weights(mask_png(solid(518, (259, 0, 518, 518))), GRID)
    overlap = np.minimum(left, right)
    assert float(overlap.sum()) / float(left.sum()) < 0.05


@pytest.mark.parametrize("grid", [14, 37, 64])
def test_grid_size_is_honoured(grid: int) -> None:
    """patch grid はモデル由来で変わる。決め打ちにしないこと。"""
    weights = backends.mask_weights(mask_png(solid(518, (0, 0, 518, 518))), grid)
    assert weights.shape == (grid, grid)


# --- エンドポイントの配線 ----------------------------------------------------


class _StubPoolBackend:
    """pooling に対応する最小の backend。モデルを読まずに配線だけを確かめる。"""

    name = "stub-pool"
    model_name = "stub"
    model_revision = "v0"
    model_version = "stub-pool-v0-d4"
    dimension = 4
    supports_text = False
    supports_region_pool = True
    needs_region_codes = False
    role = "fine"
    patch_grid = GRID

    def __init__(self) -> None:
        self.seen: list[tuple[int, list[str]]] = []

    def encode_region_pool(self, images, masks):
        self.seen.append((len(images), sorted(masks[0])))
        out = []
        for index, item in enumerate(masks):
            for order, code in enumerate(sorted(item)):
                out.append((index, code, np.full(4, order + 1, dtype=np.float32)))
        return out


def _client(monkeypatch, fine=None, coarse=None):
    import importlib

    main = importlib.import_module("app.main")
    monkeypatch.setattr(main, "_fine_backend", fine, raising=False)
    monkeypatch.setattr(main, "_backend", coarse, raising=False)
    from fastapi.testclient import TestClient

    return TestClient(main.app, raise_server_exceptions=False), main


@pytest.fixture(autouse=True)
def _service_on_path():
    path = str(REPO / "services" / "embedding-service")
    if path not in sys.path:
        sys.path.insert(0, path)
    yield


def test_endpoint_returns_one_vector_per_region(monkeypatch) -> None:
    stub = _StubPoolBackend()
    client, _ = _client(monkeypatch, fine=stub)
    png = mask_png(solid(64, (0, 0, 64, 64)))
    import base64

    encoded = base64.b64encode(png).decode()
    response = client.post(
        "/api/v1/embeddings/region-pool",
        json={
            "images_base64": [encoded],
            "masks_base64": [{"RUL": encoded, "RLL": encoded}],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["patch_grid"] == GRID
    assert body["dimension"] == 4
    assert {item["region_code"] for item in body["items"]} == {"RUL", "RLL"}
    assert all(item["image_index"] == 0 for item in body["items"])


def test_endpoint_is_explicit_when_no_backend_supports_pooling(monkeypatch) -> None:
    """黙って 1 段目へ落とさず、設定の誤りとして伝えること。"""
    client, _ = _client(monkeypatch, fine=None, coarse=None)
    encoded = "aGk="
    response = client.post(
        "/api/v1/embeddings/region-pool",
        json={"images_base64": [encoded], "masks_base64": [{"RUL": encoded}]},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "REGION_POOL_UNAVAILABLE"


def test_endpoint_rejects_misaligned_masks(monkeypatch) -> None:
    client, _ = _client(monkeypatch, fine=_StubPoolBackend())
    encoded = "aGk="
    response = client.post(
        "/api/v1/embeddings/region-pool",
        json={"images_base64": [encoded, encoded], "masks_base64": [{"RUL": encoded}]},
    )
    assert response.status_code == 422


def test_endpoint_rejects_invalid_base64(monkeypatch) -> None:
    client, _ = _client(monkeypatch, fine=_StubPoolBackend())
    response = client.post(
        "/api/v1/embeddings/region-pool",
        json={"images_base64": ["aGk="], "masks_base64": [{"RUL": "not base64!!"}]},
    )
    assert response.status_code == 422
    assert response.json()["code"] == "IMAGE_BASE64_INVALID"


def test_endpoint_accepts_image_level_masks(monkeypatch) -> None:
    """画像単位の評価（胸部野・肺野など）の mask 名が通り、名前のまま返ること。"""
    import base64

    client, _ = _client(monkeypatch, fine=_StubPoolBackend())
    encoded = base64.b64encode(mask_png(solid(64, (0, 0, 64, 64)))).decode()
    response = client.post(
        "/api/v1/embeddings/region-pool",
        json={
            "images_base64": [encoded],
            "masks_base64": [{"LUNG": encoded, "CHEST": encoded, "WHOLE": encoded}],
        },
    )
    assert response.status_code == 200, response.text
    assert {item["region_code"] for item in response.json()["items"]} == {"LUNG", "CHEST", "WHOLE"}


def test_endpoint_rejects_unknown_mask_name(monkeypatch) -> None:
    import base64

    client, _ = _client(monkeypatch, fine=_StubPoolBackend())
    encoded = base64.b64encode(mask_png(solid(64, (0, 0, 64, 64)))).decode()
    response = client.post(
        "/api/v1/embeddings/region-pool",
        json={"images_base64": [encoded], "masks_base64": [{"HEART": encoded}]},
    )
    assert response.status_code == 422


# --- 胸部野 1 単位（/embeddings/chest と chest-profile） ----------------------


class _StubCoarse:
    """1 段目の最小 backend。crop の枚数だけ固定のベクトルを返す。"""

    name = "stub-coarse"
    model_name = "stub"
    model_revision = "v0"
    model_version = "stub-coarse-v0-d3"
    dimension = 3
    supports_text = True
    supports_region_pool = False
    supports_chest = False
    needs_region_codes = False
    role = "coarse"

    def encode_images(self, images, region_codes=None):
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(images), 1))


class _StubBackbone:
    """chest-profile の backbone の代わり。mask ごとに順番が分かるベクトルを返す。"""

    name = "rad-dino"
    model_name = "microsoft/rad-dino"
    model_revision = "a" * 40
    dimension = 4
    supports_region_pool = True
    _input_side = 518

    def encode_region_pool(self, images, masks):
        # 返す順番をわざと逆にする。呼び出し側が index で並べ直すことを確かめるため
        out = []
        for index in reversed(range(len(images))):
            vector = np.zeros(4, dtype=np.float32)
            vector[index % 4] = 1.0
            out.append((index, "CHEST", vector))
        return out


def _chest_head_file(tmp_path, revision="a" * 40):
    from nihcc_rag.chest_head import ChestHead

    head = ChestHead(
        mean=np.zeros(4), std=np.ones(4), weights=np.eye(4)[:, :3] * 5.0, bias=np.zeros(3),
        findings=("Atelectasis", "Cardiomegaly", "Effusion"), kappa=1.0,
        backbone_model="microsoft/rad-dino", backbone_revision=revision, pool_side=518,
        diaphragm_ratio=0.10, margin_ratio=0.02, roi_profile="nihcc-unet-local-v1",
        train_images=10, split_seed=1, test_auc_mean=0.8,
    )
    path = tmp_path / "head.npz"
    head.save(path)
    return path


def test_chest_backend_keeps_the_input_order(tmp_path) -> None:
    """backbone が順不同で返しても、確率は入力の並びで返ること。"""
    backend = backends.ChestProfileBackend(
        artifact_path=str(_chest_head_file(tmp_path)), backbone=_StubBackbone(), dimension=4
    )
    profiles, probs = backend.encode_chest([b"i0", b"i1", b"i2"], [b"m0", b"m1", b"m2"])
    # 画像 i は特徴の i 番目だけが立っている -> 所見 i の確率が最大
    assert list(np.argmax(probs, axis=1)) == [0, 1, 2]
    assert profiles.shape == (3, 4)


def test_chest_backend_refuses_a_different_backbone_revision(tmp_path) -> None:
    from nihcc_rag.chest_head import ChestHeadMismatch

    with pytest.raises(ChestHeadMismatch):
        backends.ChestProfileBackend(
            artifact_path=str(_chest_head_file(tmp_path, revision="b" * 40)),
            backbone=_StubBackbone(), dimension=4,
        )


def test_chest_backend_refuses_a_wrong_profile_width(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="EMBEDDING_FINE_DIMENSION"):
        backends.ChestProfileBackend(
            artifact_path=str(_chest_head_file(tmp_path)), backbone=_StubBackbone(), dimension=15
        )


def test_chest_backend_needs_the_artifact(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="分類器がありません"):
        backends.ChestProfileBackend(
            artifact_path=str(tmp_path / "missing.npz"), backbone=_StubBackbone(), dimension=4
        )


def test_chest_endpoint_returns_profiles_and_probabilities(monkeypatch, tmp_path) -> None:
    import base64

    fine = backends.ChestProfileBackend(
        artifact_path=str(_chest_head_file(tmp_path)), backbone=_StubBackbone(), dimension=4
    )
    client, _ = _client(monkeypatch, fine=fine, coarse=_StubCoarse())
    encoded = base64.b64encode(b"x").decode()
    response = client.post("/api/v1/embeddings/chest", json={
        "crops_base64": [encoded, encoded], "images_base64": [encoded, encoded],
        "masks_base64": [encoded, encoded],
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["vectors"]) == 2 and len(body["fine_vectors"]) == 2
    assert body["findings"] == ["Atelectasis", "Cardiomegaly", "Effusion"]
    assert len(body["probabilities"][0]) == 3
    assert body["fine_embedding_model_version"].startswith("chest-")


def test_chest_endpoint_without_classifier_is_single_stage(monkeypatch) -> None:
    import base64

    client, _ = _client(monkeypatch, fine=None, coarse=_StubCoarse())
    encoded = base64.b64encode(b"x").decode()
    response = client.post("/api/v1/embeddings/chest", json={
        "crops_base64": [encoded], "images_base64": [encoded], "masks_base64": [encoded],
    })
    assert response.status_code == 200, response.text
    assert response.json()["fine_vectors"] is None
    assert response.json()["probabilities"] is None


def test_images_endpoint_does_not_ask_the_chest_classifier(monkeypatch, tmp_path) -> None:
    """crop だけでは分類器は推論できない（mask が要る）。1 段目だけ返して落ちないこと。"""
    import base64

    fine = backends.ChestProfileBackend(
        artifact_path=str(_chest_head_file(tmp_path)), backbone=_StubBackbone(), dimension=4
    )
    client, _ = _client(monkeypatch, fine=fine, coarse=_StubCoarse())
    encoded = base64.b64encode(b"x").decode()
    response = client.post("/api/v1/embeddings/images", json={"images_base64": [encoded]})
    assert response.status_code == 200, response.text
    assert response.json()["fine_vectors"] is None


# --- 区画ごとの地図（/embeddings/chest-map） ----------------------------------


class _StubPatchBackbone(_StubBackbone):
    """patch token を返せる backbone の代わり。grid 3 × 3、4 次元。"""

    patch_grid = 3

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def iter_patch_tokens(self, images):
        tokens = self._rng.normal(size=(len(images), 3, 3, 4)).astype(np.float32)
        self.last = tokens
        yield 0, tokens


def test_chest_map_decomposes_the_logit_exactly(tmp_path) -> None:
    """地図を mask で加重平均して定数を足すと、分類器のロジットに一致すること。"""
    from nihcc_rag.mil import sigmoid

    backbone = _StubPatchBackbone()
    backend = backends.ChestProfileBackend(
        artifact_path=str(_chest_head_file(tmp_path)), backbone=backbone, dimension=4
    )
    mask = mask_png(solid(3, (0, 0, 2, 3)))  # 左の 2 列だけが胸部野
    maps, probs, findings = backend.chest_map([b"img"], [mask])
    head = backend.head
    weights = backends.mask_weights(mask, 3).astype(np.float64)
    offset = -(head.mean / head.std) @ head.weights + head.bias
    logits = (maps[0] * weights[None]).sum((1, 2)) / weights.sum() + offset
    assert np.allclose(sigmoid(logits), probs[0], atol=1e-5)
    assert maps.shape == (1, 3, 3, 3) and findings == list(head.findings)


def test_chest_map_probabilities_match_encode_chest(tmp_path) -> None:
    """地図と一緒に返す確率は、取り込みの確率（encode_chest）と同じ計算であること。"""

    class Pooling(_StubPatchBackbone):
        def encode_region_pool(self, images, masks):
            tokens = self.last
            out = []
            for i, item in enumerate(masks):
                w = backends.mask_weights(item["CHEST"], 3)
                pooled = backends.pool_patches(tokens[i], w)
                out.append((i, "CHEST", pooled / np.linalg.norm(pooled)))
            return out

    backbone = Pooling(seed=3)
    backend = backends.ChestProfileBackend(
        artifact_path=str(_chest_head_file(tmp_path)), backbone=backbone, dimension=4
    )
    mask = mask_png(solid(3, (0, 0, 3, 3)))
    _, probs, _ = backend.chest_map([b"img"], [mask])
    _, encoded = backend.encode_chest([b"img"], [mask])
    assert np.allclose(probs, encoded, atol=1e-5)


def test_chest_map_can_select_findings(tmp_path) -> None:
    backend = backends.ChestProfileBackend(
        artifact_path=str(_chest_head_file(tmp_path)), backbone=_StubPatchBackbone(), dimension=4
    )
    maps, probs, findings = backend.chest_map([b"i"], [mask_png(solid(3, (0, 0, 3, 3)))],
                                              ["Effusion"])
    assert findings == ["Effusion"] and maps.shape == (1, 1, 3, 3) and probs.shape == (1, 1)
    with pytest.raises(ValueError):
        backend.chest_map([b"i"], [mask_png(solid(3, (0, 0, 3, 3)))], ["Fibrosis"])


def test_chest_map_endpoint(monkeypatch, tmp_path) -> None:
    import base64

    fine = backends.ChestProfileBackend(
        artifact_path=str(_chest_head_file(tmp_path)), backbone=_StubPatchBackbone(), dimension=4
    )
    client, _ = _client(monkeypatch, fine=fine, coarse=_StubCoarse())
    image = base64.b64encode(b"x").decode()
    mask = base64.b64encode(mask_png(solid(3, (0, 0, 3, 3)))).decode()
    response = client.post("/api/v1/embeddings/chest-map", json={
        "images_base64": [image], "masks_base64": [mask], "findings": ["Cardiomegaly"],
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["grid"] == 3 and body["findings"] == ["Cardiomegaly"]
    assert len(body["maps"][0][0]) == 9


def test_chest_map_endpoint_needs_the_classifier(monkeypatch) -> None:
    import base64

    client, _ = _client(monkeypatch, fine=None, coarse=_StubCoarse())
    encoded = base64.b64encode(b"x").decode()
    response = client.post("/api/v1/embeddings/chest-map",
                           json={"images_base64": [encoded], "masks_base64": [encoded]})
    assert response.status_code == 422
    assert response.json()["code"] == "CHEST_MAP_UNAVAILABLE"
