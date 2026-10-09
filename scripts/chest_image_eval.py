"""胸部野（肺野＋縦隔＋横隔膜）を 1 単位にした検索の教師データと精度を測る。

背景
----
ChestX-ray14 のラベルは画像単位であり、部位単位の正解は無い。**検索の単位を
画像 1 枚の胸部野へ揃えれば、そのラベルがそのまま教師データになる。** MIL は要らない。

枠は肺野だけでなく縦隔と横隔膜まで含める。心拡大は左右の肺の間にあり、胸水は
肋骨横隔膜角にたまる。肺野だけに絞ると、これらの根拠が枠の外へ出る（`coverage`
の正解 box 集計で、心拡大の box が肺野へ 50% 以上入るのは 2.1%）。

比べる表現（すべて RAD-DINO、1 回の ROI パスで同時に作る）
----------------------------------------------------------
pool-chest     胸部野 mask で patch token を pooling する（**本命**）
pool-lung      肺野 mask だけで pooling する（心臓を外すと何を失うかの対照）
pool-whole     画像全体で pooling する（枠を絞る意味があるかの対照）
crop-chest     胸部野の外を黒塗りした crop の CLS（枠外を物理的に消す場合）
deterministic  胸部野 crop の画像統計量（医学的意味なし。**帰無**）

ViT の patch token は自己注意で画像全体の情報を持つので、mask pooling は枠外を
完全には消さない。crop-chest はその比較のためにある。

サブコマンド
------------
coverage  正解 box（BBox_List_2017）が肺野・胸部野へどれだけ入るか。880 枚、数分
sheet     無作為抽出した画像に mask の輪郭を重ねた一覧。良否は手選びで見ない
vectors   全量のベクトル生成（数時間）。途中から再開できる
train     教師データを組み、画像単位の分類器を学習して AUC と検索品質を測る

    python scripts/chest_image_eval.py coverage
    python scripts/chest_image_eval.py sheet --count 48
    python scripts/chest_image_eval.py vectors --service-url http://127.0.0.1:8002
    python scripts/chest_image_eval.py train --json tmp/chest/train.json

U-Net の profile が必須である（規則ベースは心臓を出さず、約 4 分の 1 で破綻する）。
ホストから使うときは `LUNG_MODEL_PATH` でモデルの場所を指定する。

評価の読み方
------------
単位が画像なので、**画像単位のラベルから作った指標がそのまま正規の指標になる**。
部位レベルの主張では「画像レベルへ退化する」ため証拠にならない AUC と Test B も、
ここでは主指標である。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in ("packages/common/src", "packages/image-pipeline/src"):
    sys.path.insert(0, str(ROOT / _path))

from image_pipeline import (  # noqa: E402
    ExtractionError,
    chest_model_inputs,
    extract_chest,
    load_profile,
    masked_square_crop,
)
from image_pipeline import segmenter as segmentation  # noqa: E402
from nihcc_rag import mil  # noqa: E402
from nihcc_rag.csvspec import ALLOWED_FINDINGS  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# region_mil_stage_a が region_signal_eval を同じ名前で読み込むので、そちらを使う。
# 別に読み込むと sys.modules の登録が上書きされ、worker へ渡す関数を pickle できない。
stage_a = _load("_region_mil_stage_a", ROOT / "scripts" / "region_mil_stage_a.py")
rse = stage_a.rse


def _init_worker() -> None:
    """OpenCV の内部スレッドを 1 本にする（プロセス側で並列化しているため）。"""
    import cv2

    cv2.setNumThreads(1)


OUT_DIR = ROOT / "tmp" / "chest"
#: 教師データの置き場。公開リポジトリで追跡する場所に合わせ、手順書・chest_search_eval.py もここを指す
TEACHER_DIR = OUT_DIR / "ec2-full"
DEFAULT_PROFILE = os.environ.get("CHEST_PROFILE", "nihcc-unet-local-v1")
#: 学習・評価の対象（`No Finding` を除く 14 所見）
FINDINGS = tuple(f for f in ALLOWED_FINDINGS if f != "No Finding")
#: pooling 系の表現と、サービスへ送る mask 名（contract.PoolMaskCode）
POOL_VARIANTS = {"pool-lung": "LUNG", "pool-chest": "CHEST", "pool-whole": "WHOLE"}
CROP_VARIANTS = ("crop-chest", "deterministic")
VARIANTS = (*POOL_VARIANTS, *CROP_VARIANTS)
#: 再開の単位。この枚数ごとに shard を書き出す。
CHECKPOINT = 2000
#: 正解 box の所見名をラベルの綴りへ揃える（region_gold_eval.py と同じ）
GOLD_LABEL = {"Infiltrate": "Infiltration"}


# --------------------------------------------------------------------------
# mask（本番と同じ実装を使う。image_pipeline.chest）
# --------------------------------------------------------------------------


def segment(data: bytes, profile) -> tuple[np.ndarray, dict[str, np.ndarray], dict, float]:
    """(正規化画像, {LUNG, CHEST, WHOLE: mask}, 面積の記録, 縮小率) を返す。

    中身は本番の roi-service と同じ `image_pipeline.extract_chest`。評価用に
    書き直すと、分類器が学習した特徴と本番の特徴が食い違う。
    """
    extraction = extract_chest(data, profile_name=profile.name)
    masks = {
        "LUNG": extraction.lung,
        "CHEST": extraction.chest,
        "WHOLE": np.full_like(extraction.lung, 255),
    }
    return extraction.normalized, masks, extraction.areas, extraction.scale


# --------------------------------------------------------------------------
# 入力
# --------------------------------------------------------------------------


def load_samples(args) -> list:
    data_dir = rse.resolve_data_dir(args.data_dir)
    csv_path = rse.resolve_input(args.csv, data_dir, "Data_Entry_2017_v2020.csv", "CSV")
    image_dir = rse.resolve_input(args.images, data_dir, "images", "images")
    samples = rse.load_samples(csv_path, image_dir, args.view)
    if args.limit:
        random.Random(args.seed).shuffle(samples)
        samples = sorted(samples[: args.limit], key=lambda s: s.name)
    return samples, image_dir


def check_profile(name: str) -> None:
    """全量の数時間に入る前に、U-Net が読めることを確かめる。"""
    profile = load_profile(name)
    segmenter = segmentation.from_profile(profile)
    if segmenter is None:
        raise SystemExit(f"{name} は U-Net の profile ではありません（--profile を確認）")
    try:
        segmenter.predict(np.full((512, 512), 128, dtype=np.uint8))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"U-Net を読めません: {exc}\n"
            f"  ホストから使うときは LUNG_MODEL_PATH にモデルの場所を指定してください"
            f"（例: LUNG_MODEL_PATH=infra/lung-unet/lung-unet.onnx）"
        ) from exc


# --------------------------------------------------------------------------
# coverage: 正解 box が枠へどれだけ入るか
# --------------------------------------------------------------------------


def _coverage_worker(task):
    name, path, profile_name, boxes = task
    try:
        _, masks, _, scale = segment(Path(path).read_bytes(), load_profile(profile_name))
    except ExtractionError as exc:
        return name, None, exc.code
    except Exception as exc:  # noqa: BLE001
        return name, None, type(exc).__name__
    out = []
    for finding, (x, y, w, h) in boxes:
        x0, y0 = int(round(x * scale)), int(round(y * scale))
        x1, y1 = int(round((x + w) * scale)), int(round((y + h) * scale))
        shares = {}
        for code in ("LUNG", "CHEST"):
            window = masks[code][max(0, y0) : max(0, y1), max(0, x0) : max(0, x1)]
            shares[code] = float((window > 0).mean()) if window.size else 0.0
        out.append((finding, shares))
    return name, out, ""


def cmd_coverage(args) -> int:
    check_profile(args.profile)
    data_dir = rse.resolve_data_dir(args.data_dir)
    image_dir = rse.resolve_input(args.images, data_dir, "images", "images")
    gold = ROOT / args.gold_csv
    if not gold.exists():
        raise SystemExit(
            f"正解 box がありません: {gold}\n  取得方法は docs/runbooks/chest-classifier.md §1.2"
        )
    boxes: dict[str, list] = defaultdict(list)
    with gold.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            # CSV は "Infiltrate"、ラベルは "Infiltration"。region_gold_eval.py と揃える
            finding = GOLD_LABEL.get(row[1], row[1])
            boxes[row[0]].append((finding, tuple(float(v) for v in row[2:6])))
    available = {p.name for p in image_dir.iterdir()}
    tasks = [
        (name, str(image_dir / name), args.profile, items)
        for name, items in sorted(boxes.items())
        if name in available
    ]
    print(f"正解 box: {sum(len(v) for v in boxes.values())} 件 / 手元にある画像 {len(tasks)} 枚")

    # region_gold_eval.py の出力（規則ベースの部位分割での値）があれば並べる
    previous = {}
    cached = ROOT / "tmp" / "gold" / "gold_regions.json"
    if cached.exists():
        for item in json.loads(cached.read_text(encoding="utf-8"))["items"]:
            previous[(item["name"], item["finding"])] = item["top_share"]

    rows, failures = [], Counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for name, result, error in ex.map(_coverage_worker, tasks, chunksize=4):
            if result is None:
                failures[error] += 1
                continue
            for finding, shares in result:
                rows.append({
                    "name": name, "finding": finding,
                    "lung": shares["LUNG"], "chest": shares["CHEST"],
                    "region5": previous.get((name, finding)),
                })
    if failures:
        print(f"分割の失敗: {dict(failures)}")

    def summarise(items):
        n = len(items)
        frac = lambda key, t: sum(r[key] >= t for r in items) / n  # noqa: E731
        five = [r["region5"] for r in items if r["region5"] is not None]
        return {
            "n": n,
            "region5_ge30": (sum(v >= 0.3 for v in five) / len(five)) if five else None,
            "lung_ge30": frac("lung", 0.3), "chest_ge30": frac("chest", 0.3),
            "lung_ge50": frac("lung", 0.5), "chest_ge50": frac("chest", 0.5),
            "chest_ge80": frac("chest", 0.8),
            "lung_median": float(np.median([r["lung"] for r in items])),
            "chest_median": float(np.median([r["chest"] for r in items])),
        }

    by_finding = defaultdict(list)
    for r in rows:
        by_finding[r["finding"]].append(r)
    table = {f: summarise(v) for f, v in sorted(by_finding.items(), key=lambda kv: -len(kv[1]))}
    table["ALL"] = summarise(rows) if rows else {}

    pct = lambda v: "   -  " if v is None else f"{v:6.1%}"  # noqa: E731
    print("\nbox のうち枠へ入る割合が閾値以上の件数比（5部位 = 規則ベースの最大部位）")
    print(f"{'所見':<20}{'n':>5}  {'5部位≥30':>8}  {'肺野≥30':>8}  {'胸部≥30':>8}"
          f"  {'肺野≥50':>8}  {'胸部≥50':>8}  {'胸部≥80':>8}")
    for finding, s in table.items():
        if not s:
            continue
        print(f"{finding:<20}{s['n']:>5}  {pct(s['region5_ge30']):>8}  {pct(s['lung_ge30']):>8}"
              f"  {pct(s['chest_ge30']):>8}  {pct(s['lung_ge50']):>8}  {pct(s['chest_ge50']):>8}"
              f"  {pct(s['chest_ge80']):>8}")
    print("\n注: box は長方形なので、肺の外の画素を必ず含む。小さい box ほど値が高く出る")

    if args.json:
        path = ROOT / args.json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"profile": args.profile, "failures": dict(failures), "summary": table, "items": rows},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8")
        print(f"保存: {path.relative_to(ROOT)}")
    return 0


# --------------------------------------------------------------------------
# sheet: 無作為抽出の目視一覧
# --------------------------------------------------------------------------


def _sheet_worker(task):
    import cv2

    name, path, profile_name, side = task
    try:
        gray, masks, _, _ = segment(Path(path).read_bytes(), load_profile(profile_name))
    except Exception as exc:  # noqa: BLE001
        tile = np.zeros((side, side, 3), dtype=np.uint8)
        cv2.putText(tile, f"FAIL {type(exc).__name__}", (6, side // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        return name, tile
    tile = cv2.cvtColor(cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_GRAY2BGR)
    for code, color in (("LUNG", (255, 160, 0)), ("CHEST", (0, 200, 255))):
        small = cv2.resize(masks[code], (side, side), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(small, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(tile, contours, -1, color, 1)
    cv2.putText(tile, name.removesuffix(".png"), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1)
    return name, tile


def cmd_sheet(args) -> int:
    import cv2

    check_profile(args.profile)
    samples, image_dir = load_samples(argparse.Namespace(**{**vars(args), "limit": 0}))
    random.Random(args.seed).shuffle(samples)
    picked = samples[: args.count]
    tasks = [(s.name, str(image_dir / s.name), args.profile, args.tile) for s in picked]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        tiles = [tile for _, tile in ex.map(_sheet_worker, tasks)]
    cols = args.columns
    rows = [tiles[i : i + cols] for i in range(0, len(tiles), cols)]
    blank = np.zeros_like(tiles[0])
    sheet = np.vstack([np.hstack(r + [blank] * (cols - len(r))) for r in rows])
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)
    print(f"無作為 {len(tiles)} 枚（乱数種 {args.seed}）: {path.relative_to(ROOT)}")
    print("  青 = 肺野（U-Net）  橙 = 胸部野（肺野＋縦隔＋横隔膜）")
    return 0


# --------------------------------------------------------------------------
# vectors: 全量のベクトル生成（再開可能）
# --------------------------------------------------------------------------


def _vector_worker(task):
    """1 枚ぶんの (画像名, pooling 入力, crop, 面積, エラー)。

    入力の作り方は本番の roi-service と同じ `image_pipeline.chest_model_inputs`。
    """
    name, path, profile_name, pool_side, crop_side = task
    try:
        extraction = extract_chest(Path(path).read_bytes(), profile_name=profile_name)
    except ExtractionError as exc:
        return name, None, None, None, exc.code
    except Exception as exc:  # noqa: BLE001 - 1 枚の失敗で全体を止めない
        return name, None, None, None, type(exc).__name__
    inputs = chest_model_inputs(
        extraction, pool_side=pool_side, crop_side=crop_side, masks=("LUNG", "CHEST", "WHOLE")
    )
    return name, (inputs.pool_image, inputs.pool_masks), inputs.crop, extraction.areas, ""


def encode_with_retry(encoder, crops: list[bytes], label: str) -> np.ndarray:
    """サービスが落ちても待って投げ直す（pooling 側の ServicePoolEncoder と同じ方針）。

    全量の実行中に embedding-service が応答せず切断した（RemoteProtocolError）。
    restart: unless-stopped で自動復帰するが、モデルの再ロードに 1〜2 分かかる。
    deterministic はプロセス内なので、ここを通っても再試行は起きない。
    """
    import httpx

    last: Exception | None = None
    for attempt in range(1, rse.POOL_RETRIES + 1):
        try:
            return encoder.encode(crops)
        except httpx.HTTPError as exc:
            last = exc
            if attempt == rse.POOL_RETRIES:
                break
            wait = rse.POOL_RETRY_WAIT * attempt
            print(f"  !! {label}: embedding-service への送信に失敗 ({type(exc).__name__}). "
                  f"{wait:.0f} 秒待って再試行 ({attempt}/{rse.POOL_RETRIES - 1})", flush=True)
            time.sleep(wait)
    raise RuntimeError(
        f"{label}: embedding-service への送信が {rse.POOL_RETRIES} 回失敗しました。"
        f"同じコマンドで再実行すれば、書き出し済みの shard の続きから処理します。"
        f"原因はサーバ側のログにあります: docker compose logs --tail 200 embedding-service"
    ) from last


def cache_path(variant: str, profile: str, total: int) -> Path:
    return OUT_DIR / f"vec_{variant}_{profile}_{total}.npz"


def load_cache(path: Path) -> tuple[list[str], np.ndarray]:
    blob = np.load(path, allow_pickle=True)
    return list(blob["names"]), blob["vecs"]


def shard_dir(profile: str, total: int) -> Path:
    return OUT_DIR / "shards" / f"{profile}_{total}"


def _done_names(directory: Path, variants: list[str]) -> set[str]:
    """全 variant の shard に載った画像と、失敗が確定した画像。"""
    if not directory.exists():
        return set()
    per_variant = []
    for variant in variants:
        names: set[str] = set()
        for path in sorted(directory.glob(f"{variant}_*.npz")):
            names.update(np.load(path, allow_pickle=True)["names"].tolist())
        per_variant.append(names)
    done = set.intersection(*per_variant) if per_variant else set()
    for path in directory.glob("failed_*.json"):
        done.update(json.loads(path.read_text(encoding="utf-8")))
    return done


def _write_shard(directory: Path, index: int, store: dict, failed: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for variant, (names, vecs) in store.items():
        mat = np.vstack(vecs).astype(np.float32) if vecs else np.zeros((0, 0), np.float32)
        np.savez(directory / f"{variant}_{index:05d}.npz",
                 names=np.array(names, dtype=object), vecs=mat)
    (directory / f"failed_{index:05d}.json").write_text(json.dumps(failed), encoding="utf-8")


def cmd_vectors(args) -> int:
    samples, image_dir = load_samples(args)
    total = len(samples)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"未知の表現: {unknown}（使えるのは {', '.join(VARIANTS)}）")
    pending = [v for v in variants if args.refresh or not cache_path(v, args.profile, total).exists()]
    for v in variants:
        if v not in pending:
            print(f"済: {cache_path(v, args.profile, total).relative_to(ROOT)}")
    if not pending:
        return 0
    check_profile(args.profile)

    encoders: dict[str, object] = {}
    pool_codes = [POOL_VARIANTS[v] for v in pending if v in POOL_VARIANTS]
    if pool_codes:
        encoders["pool"] = rse.ServicePoolEncoder(args.service_url, batch=args.pool_batch)
        print(f"pooling: patch grid {encoders['pool'].patch_grid}, mask {pool_codes}")
    if "crop-chest" in pending:
        encoders["crop-chest"] = rse.ServiceEncoder(
            args.service_url, field="fine_vectors", batch=args.crop_batch
        )
    if "deterministic" in pending:
        encoders["deterministic"] = rse.BackendEncoder("deterministic", 512, "", "")
    for name, encoder in encoders.items():
        print(f"  {name}: {encoder.model_version}")

    directory = shard_dir(args.profile, total)
    stat_variant = "stats"
    done = _done_names(directory, [*pending, stat_variant])
    todo = [s for s in samples if s.name not in done]
    print(f"対象 {total} 枚 / 済 {total - len(todo)} 枚 / 残り {len(todo)} 枚")

    next_index = len(list(directory.glob("failed_*.json"))) if directory.exists() else 0
    tasks = [(s.name, str(image_dir / s.name), args.profile, args.pool_side, args.crop_side)
             for s in todo]
    store = {v: ([], []) for v in [*pending, stat_variant]}
    failed: list[str] = []
    failures: Counter[str] = Counter()
    buffer: list = []
    in_shard = 0
    started, processed = time.perf_counter(), 0

    def encode(batch) -> None:
        names = [b[0] for b in batch]
        if "pool" in encoders:
            payloads = [(b[1][0], {c: b[1][1][c] for c in pool_codes}) for b in batch]
            code_to_variant = {code: v for v, code in POOL_VARIANTS.items()}
            for image_index, code, vector in encoders["pool"].encode_pool(payloads):
                names_, vecs = store[code_to_variant[code]]
                names_.append(names[image_index])
                vecs.append(np.asarray(vector, dtype=np.float32))
        crops = [b[2] for b in batch]
        for variant in CROP_VARIANTS:
            if variant in encoders:
                vectors = encode_with_retry(encoders[variant], crops, variant)
                store[variant][0].extend(names)
                store[variant][1].extend(np.asarray(vectors, dtype=np.float32))
        store[stat_variant][0].extend(names)
        store[stat_variant][1].extend(
            np.array([b[3]["lung"], b[3]["chest"], b[3]["heart"]], dtype=np.float32) for b in batch
        )

    print("分割とベクトル生成を開始", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for name, pool, crop, stats, error in ex.map(_vector_worker, tasks, chunksize=8):
            processed += 1
            in_shard += 1
            if pool is None:
                failures[error] += 1
                failed.append(name)
            else:
                buffer.append((name, pool, crop, stats))
            if len(buffer) >= args.chunk:
                encode(buffer)
                buffer = []
            if in_shard >= args.checkpoint:
                if buffer:
                    encode(buffer)
                    buffer = []
                _write_shard(directory, next_index, store, failed)
                next_index += 1
                store = {v: ([], []) for v in store}
                failed, in_shard = [], 0
                rate = processed / max(1e-9, time.perf_counter() - started)
                print(f"  {total - len(todo) + processed}/{total} 枚 ({rate:.1f} 枚/秒, "
                      f"残り {(len(todo) - processed) / max(rate, 1e-9) / 60:.0f} 分)", flush=True)
        if buffer:
            encode(buffer)
        if in_shard:
            _write_shard(directory, next_index, store, failed)
    if failures:
        print(f"分割の失敗（今回分）: {dict(failures)}")

    # shard を表現ごとの 1 ファイルへまとめる
    for variant in [*pending, stat_variant]:
        # 再開時、前回の途中で一部の表現だけ書かれた画像はもう一度処理されるので、
        # 同じ画像が 2 度現れうる。最初のものを残す。
        names, mats, seen = [], [], set()
        for path in sorted(directory.glob(f"{variant}_*.npz")):
            blob = np.load(path, allow_pickle=True)
            keep = [i for i, n in enumerate(blob["names"].tolist()) if n not in seen]
            if keep:
                picked = [blob["names"][i] for i in keep]
                seen.update(picked)
                names.extend(picked)
                mats.append(blob["vecs"][keep])
        out = cache_path(variant, args.profile, total)
        np.savez_compressed(out, names=np.array(names, dtype=object),
                            vecs=np.vstack(mats) if mats else np.zeros((0, 0), np.float32))
        print(f"保存 [{variant}]: {out.relative_to(ROOT)}（{len(names)} 枚）")
    all_failed = []
    for path in sorted(directory.glob("failed_*.json")):
        all_failed.extend(json.loads(path.read_text(encoding="utf-8")))
    print(f"分割の失敗（累計）: {len(all_failed)} 枚 / {total} 枚")
    return 0


# --------------------------------------------------------------------------
# train: 教師データ・学習・評価
# --------------------------------------------------------------------------


def label_matrix(names: list[str], by_name: dict) -> np.ndarray:
    y = np.zeros((len(names), len(FINDINGS)), dtype=np.float64)
    for i, name in enumerate(names):
        labels = by_name[name].labels
        for f, finding in enumerate(FINDINGS):
            if finding in labels:
                y[i, f] = 1.0
    return y


def standardise(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_idx].mean(axis=0)
    std = x[train_idx].std(axis=0)
    std[std < 1e-6] = 1.0
    return ((x - mean) / std).astype(np.float32), mean, std


def class_weights(y: np.ndarray, power: float, cap: float) -> np.ndarray:
    """陽性の重み (陰性数/陽性数)^power。region_mil_stage_a.py と同じ式（稀な所見を潰さない）。"""
    pos = y.sum(axis=0)
    neg = len(y) - pos
    return np.minimum(cap, (neg / np.maximum(pos, 1.0)) ** power)


def train_head(x, y, idx, args, log=print) -> tuple[np.ndarray, np.ndarray, dict]:
    """画像単位の多ラベルロジスティック回帰（所見ごとの線形分類器 14 本）。

    凍結 backbone の特徴に線形の頭だけを載せる。MIL の bag 集約が要らないこと
    以外は region_mil_stage_a.py と同じ条件（Adam、L2、陽性重み、val の平均 AUC で早期終了）。
    """
    rng = np.random.default_rng(args.seed)
    d, k = x.shape[1], y.shape[1]
    w = np.zeros((d, k), dtype=np.float64)
    b = np.zeros(k, dtype=np.float64)
    pos_w = class_weights(y[idx["train"]], args.class_weight_power, args.class_weight_max)
    opt = mil.Adam(lr=args.lr)
    best = (-1.0, w.copy(), b.copy(), 0)
    stale = 0
    train = idx["train"]
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(train)
        for start in range(0, len(order), args.batch):
            rows = order[start : start + args.batch]
            xb, yb = x[rows].astype(np.float64), y[rows]
            p = mil.sigmoid(xb @ w + b)
            weight = np.where(yb > 0, pos_w, 1.0)
            g = weight * (p - yb) / len(rows)
            opt.step("w", w, xb.T @ g + args.l2 * w)
            opt.step("b", b, g.sum(axis=0))
        val = predict(x[idx["val"]], w, b)
        aucs = [mil.roc_auc(val[:, f], y[idx["val"], f]) for f in range(k)]
        score = float(np.nanmean(aucs))
        if score > best[0] + 1e-4:
            best, stale = (score, w.copy(), b.copy(), epoch), 0
        else:
            stale += 1
        if epoch % args.log_every == 0 or stale == 0:
            log(f"    epoch {epoch:3d}  val 平均 AUC {score:.4f}{'  *' if stale == 0 else ''}")
        if stale >= args.patience:
            break
    return best[1], best[2], {"best_epoch": best[3], "val_auc_mean": best[0]}


def predict(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return mil.sigmoid(x.astype(np.float64) @ w + b)


def bootstrap_mean_auc(probs, y, patients, reps: int, seed: int) -> tuple[float, float]:
    """患者を単位にした cluster bootstrap による平均 AUC の 95% 区間。"""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, p in enumerate(patients):
        groups[p].append(i)
    keys = list(groups)
    stats = []
    for _ in range(reps):
        rows = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        aucs = [mil.roc_auc(probs[rows, f], y[rows, f]) for f in range(y.shape[1])]
        stats.append(np.nanmean(aucs))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def retrieval_test_b(query, index, q_labels, i_labels, q_patients, args) -> dict:
    """画像単位の検索品質。単独所見 F の画像で引き、上位 K に占める F の割合を測る。

    `No Finding` のクエリも含める（正常を引いたら正常が返るか）。index は
    クエリと別の患者だけで作る（同一患者の別撮影が近傍に来ると数字が膨らむ）。
    lift = 上位 K の適合率 / index における F の割合。
    """
    rng = np.random.default_rng(args.seed)
    classes = (*FINDINGS, "No Finding")
    out = {}
    for name in classes:
        if name == "No Finding":
            q_rows = np.nonzero(q_labels.sum(axis=1) == 0)[0]
            hits_of = lambda rows: i_labels[rows].sum(axis=-1) == 0  # noqa: E731
        else:
            f = FINDINGS.index(name)
            q_rows = np.nonzero((q_labels[:, f] > 0) & (q_labels.sum(axis=1) == 1))[0]
            hits_of = lambda rows, f=f: i_labels[rows, f] > 0  # noqa: E731
        if len(q_rows) < args.min_queries:
            continue
        if len(q_rows) > args.max_queries:
            q_rows = rng.choice(q_rows, args.max_queries, replace=False)
        base = float(hits_of(np.arange(len(i_labels))).mean())
        precisions = []
        for start in range(0, len(q_rows), 512):
            rows = q_rows[start : start + 512]
            sims = query[rows] @ index.T
            top = np.argpartition(-sims, args.top_k, axis=1)[:, : args.top_k]
            precisions.extend(hits_of(top).mean(axis=1))
        precisions = np.asarray(precisions)
        # 患者単位の cluster bootstrap
        groups = defaultdict(list)
        for j, r in enumerate(q_rows):
            groups[q_patients[r]].append(j)
        keys = list(groups)
        boot = []
        for _ in range(args.bootstrap):
            picked = rng.integers(0, len(keys), len(keys))
            boot.append(precisions[np.concatenate([groups[keys[p]] for p in picked])].mean())
        out[name] = {
            "queries": int(len(q_rows)),
            "base_rate": base,
            "precision": float(precisions.mean()),
            "lift": float(precisions.mean() / base) if base else float("nan"),
            "lift_ci": [float(np.percentile(boot, 2.5) / base), float(np.percentile(boot, 97.5) / base)]
            if base else [float("nan")] * 2,
        }
    return out


def l2(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    return (x / norm).astype(np.float32)


def cmd_train(args) -> int:
    samples, _ = load_samples(args)
    total = len(samples)
    by_name = {s.name: s for s in samples}
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    caches = {}
    for v in variants:
        path = cache_path(v, args.profile, total)
        if not path.exists():
            print(f"キャッシュが無いので外す [{v}]: {path.relative_to(ROOT)}")
            continue
        caches[v] = load_cache(path)
    if not caches:
        raise SystemExit("使えるキャッシュがありません。先に vectors を実行してください")

    # **全表現で同じ画像に揃える。** 表現ごとに画像集合が違うと、差が表現の差なのか
    # 画像の差なのか分からなくなる。
    common = set.intersection(*(set(n) for n, _ in caches.values()))
    names = sorted(n for n in common if n in by_name)
    print(f"共通の画像: {len(names)} 枚（全 {total} 枚）")
    y = label_matrix(names, by_name)
    patients = np.array([by_name[n].patient for n in names])
    idx = stage_a.split_by_patient(names, by_name, args.seed, args.val_fraction, args.test_fraction)
    print(f"患者単位分割: train {len(idx['train'])} / val {len(idx['val'])} / test {len(idx['test'])}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 教師データ: 画像・患者・14 所見のラベル・分割。ベクトルは表現ごとのキャッシュにある。
    split = np.empty(len(names), dtype=object)
    for key, rows in idx.items():
        split[rows] = key
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)
    teacher = TEACHER_DIR / f"teacher_{args.profile}_{len(names)}.npz"
    np.savez_compressed(teacher, names=np.array(names, dtype=object), patients=patients,
                        labels=y.astype(np.uint8), findings=np.array(FINDINGS, dtype=object),
                        split=split)
    print(f"教師データ: {teacher.relative_to(ROOT)}")
    for key in ("train", "val", "test"):
        pos = y[idx[key]].sum(axis=0).astype(int)
        print(f"  {key:5s} 陽性数 " + " ".join(f"{f[:6]}={p}" for f, p in zip(FINDINGS, pos)))

    index_rows = np.concatenate([idx["train"], idx["val"]])
    test_rows = idx["test"]
    report: dict = {"profile": args.profile, "images": len(names), "split": {k: int(len(v)) for k, v in idx.items()},
                    "findings": list(FINDINGS), "variants": {}}
    for variant, (v_names, vecs) in caches.items():
        print(f"\n[{variant}] 次元 {vecs.shape[1]}")
        pos = {n: i for i, n in enumerate(v_names)}
        x_raw = vecs[[pos[n] for n in names]].astype(np.float32)
        x, mean, std = standardise(x_raw, idx["train"])
        w, b, info = train_head(x, y, idx, args)
        probs = predict(x[test_rows], w, b)
        aucs = {f: mil.roc_auc(probs[:, i], y[test_rows, i]) for i, f in enumerate(FINDINGS)}
        valid = [a for a in aucs.values() if np.isfinite(a)]
        ci = bootstrap_mean_auc(probs, y[test_rows], patients[test_rows], args.bootstrap, args.seed)
        print(f"  test 平均 AUC {np.mean(valid):.4f}  95%区間 [{ci[0]:.4f}, {ci[1]:.4f}]"
              f"  (best epoch {info['best_epoch']})")

        all_probs = predict(x, w, b)
        profile_vecs = mil.profile_vectors(all_probs, args.kappa).astype(np.float32)
        embed = l2(x)
        test_b = {
            "embedding": retrieval_test_b(embed[test_rows], embed[index_rows], y[test_rows],
                                          y[index_rows], patients[test_rows], args),
            "profile": retrieval_test_b(profile_vecs[test_rows], profile_vecs[index_rows],
                                        y[test_rows], y[index_rows], patients[test_rows], args),
        }
        report["variants"][variant] = {
            "dimension": int(vecs.shape[1]), **info,
            "image_auc": aucs, "image_auc_mean": float(np.mean(valid)), "image_auc_ci": list(ci),
            "test_b": test_b,
        }
        head = OUT_DIR / f"head_{variant}_{args.profile}.npz"
        np.savez(head, mean=mean, std=std, w=w, b=b, findings=np.array(FINDINGS, dtype=object),
                 kappa=args.kappa)
        print(f"  分類器: {head.relative_to(ROOT)}")

    print_summary(report)
    if args.json:
        path = ROOT / args.json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n保存: {path.relative_to(ROOT)}")
    return 0


def print_summary(report: dict) -> None:
    variants = list(report["variants"])
    if not variants:
        return
    print("\n== 画像単位 AUC（test、患者単位分割）==")
    print(f"{'所見':<20}" + "".join(f"{v:>15}" for v in variants))
    for f in FINDINGS:
        print(f"{f:<20}" + "".join(f"{report['variants'][v]['image_auc'][f]:>15.3f}" for v in variants))
    print(f"{'平均':<20}" + "".join(f"{report['variants'][v]['image_auc_mean']:>15.3f}" for v in variants))

    print("\n== 検索品質 Test B（画像単位、上位 K の lift。分類器の確率から作ったプロファイルで検索）==")
    print(f"{'クエリ':<20}" + "".join(f"{v:>15}" for v in variants))
    for name in (*FINDINGS, "No Finding"):
        cells = []
        for v in variants:
            item = report["variants"][v]["test_b"]["profile"].get(name)
            cells.append(f"{item['lift']:>15.2f}" if item else f"{'-':>15}")
        print(f"{name:<20}" + "".join(cells))

    if "pool-chest" in variants and "pool-lung" in variants:
        print("\n== 心臓・縦隔・横隔膜を枠へ入れた効果（pool-chest − pool-lung の AUC）==")
        for f in ("Cardiomegaly", "Effusion", "Hernia", "Pneumothorax", "Nodule"):
            delta = (report["variants"]["pool-chest"]["image_auc"][f]
                     - report["variants"]["pool-lung"]["image_auc"][f])
            print(f"  {f:<16}{delta:+.3f}")


# --------------------------------------------------------------------------
# verify-ingest: 本番の取り込み経路が評価と同じ特徴を作るか
# --------------------------------------------------------------------------


def compare_inputs(local, roi: dict) -> list[str]:
    """roi-service が返したモデル入力と、手元で作った入力の食い違いを返す（1 バイト単位）。"""
    import base64

    problems = []
    pairs = (
        ("pool_image", local.pool_image, roi["pool_image_base64"]),
        ("pool_mask", local.pool_masks["CHEST"], roi["pool_mask_base64"]),
        ("crop", local.crop, roi["crop_base64"]),
    )
    for name, mine, theirs in pairs:
        if base64.b64decode(theirs) != mine:
            problems.append(name)
    return problems


def compare_probabilities(service: np.ndarray, local: np.ndarray) -> tuple[float, float]:
    """(確率の最大の差, プロファイルの cosine の最小値)。"""
    diff = float(np.max(np.abs(service - local))) if service.size else 0.0
    sv = mil.profile_vectors(service, 1.0)
    lv = mil.profile_vectors(local, 1.0)
    cos = float(np.min(np.sum(sv * lv, axis=1))) if sv.size else 1.0
    return diff, cos


def cmd_verify_ingest(args) -> int:
    """本番の roi-service と embedding-service が、評価と同じ特徴を作るかを確かめる。

    1. roi-service `/roi/chest` のモデル入力が、評価と同じ関数で手元が作る入力と
       **1 バイトも違わない**こと（U-Net のモデル・profile・前処理の一致）
    2. embedding-service `/embeddings/chest` の確率が、評価のキャッシュの特徴
       （vec_pool-chest_*）に分類器を当てた確率と一致すること
       （RAD-DINO の revision・pooling・分類器の一致）

    どちらかが外れたら、全量の取り込みに進まない（確率が黙ってずれる）。
    """
    import base64

    import httpx
    from nihcc_rag.chest_head import ChestHead

    check_profile(args.profile)
    head = ChestHead.load(ROOT / args.head)
    names, vecs = load_cache(ROOT / args.cache)
    index = {n: i for i, n in enumerate(names)}
    data_dir = rse.resolve_data_dir(args.data_dir)
    image_dir = rse.resolve_input(args.images, data_dir, "images", "images")
    # 手元にある画像だけから選ぶ（全量を持たない環境でも動くように）
    available = {p.name for p in image_dir.iterdir()}
    picked = sorted(n for n in names if n in available)
    if not picked:
        raise SystemExit(f"キャッシュの画像が {image_dir} に 1 枚もありません")
    random.Random(args.seed).shuffle(picked)
    picked = picked[: args.limit]

    roi = httpx.Client(base_url=args.roi_url, timeout=120.0)
    emb = httpx.Client(base_url=args.service_url, timeout=300.0)
    input_mismatch: list[tuple[str, list[str]]] = []
    service_probs, local_probs = [], []
    for start in range(0, len(picked), args.batch):
        batch = picked[start : start + args.batch]
        rois = []
        for name in batch:
            data = (image_dir / name).read_bytes()
            response = roi.post("/api/v1/roi/chest",
                                json={"image_base64": base64.b64encode(data).decode()})
            response.raise_for_status()
            body = response.json()
            local = chest_model_inputs(extract_chest(data, profile_name=args.profile))
            problems = compare_inputs(local, body)
            if problems:
                input_mismatch.append((name, problems))
            rois.append(body)
        response = emb.post("/api/v1/embeddings/chest", json={
            "crops_base64": [r["crop_base64"] for r in rois],
            "images_base64": [r["pool_image_base64"] for r in rois],
            "masks_base64": [r["pool_mask_base64"] for r in rois],
        })
        response.raise_for_status()
        body = response.json()
        if not body.get("probabilities"):
            raise SystemExit("embedding-service が確率を返しません（EMBEDDING_FINE_BACKEND=chest-profile か）")
        if body["findings"] != list(head.findings):
            raise SystemExit("embedding-service の所見の並びが成果物と違います")
        service_probs.extend(body["probabilities"])
        features = np.vstack([vecs[index[n]] for n in batch]).astype(np.float32)
        local_probs.extend(head.probabilities(features).tolist())
        print(f"  {min(start + args.batch, len(picked))}/{len(picked)} 枚", flush=True)

    diff, cos = compare_probabilities(np.asarray(service_probs), np.asarray(local_probs))
    ok = not input_mismatch and diff <= args.max_diff and cos >= args.min_cosine
    print(f"\nモデル入力の不一致: {len(input_mismatch)} / {len(picked)} 枚")
    for name, problems in input_mismatch[:10]:
        print(f"  {name}: {', '.join(problems)}")
    print(f"確率の最大の差: {diff:.5f}（許容 {args.max_diff}）")
    print(f"プロファイルの cosine の最小値: {cos:.6f}（許容 {args.min_cosine}）")
    print("判定: " + ("一致。全量の取り込みに進んでよい" if ok else "**不一致。全量の取り込みに進まない**"))
    if args.json:
        path = ROOT / args.json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "images": len(picked), "input_mismatch": input_mismatch,
            "max_probability_diff": diff, "min_profile_cosine": cos, "ok": ok,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if ok else 1


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--profile", default=DEFAULT_PROFILE)
        p.add_argument("--data-dir", default="")
        p.add_argument("--csv", default="")
        p.add_argument("--images", default="")
        p.add_argument("--view", default=None, help="PA / AP に絞る（既定は絞らない）")
        p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        p.add_argument("--seed", type=int, default=20260920)
        p.add_argument("--limit", type=int, default=0, help="無作為に N 枚だけ使う（較正用）")

    p = sub.add_parser("coverage", help="正解 box が肺野・胸部野へ入る割合")
    common(p)
    p.add_argument("--gold-csv", default="tmp/gold/BBox_List_2017.csv")
    p.add_argument("--json", default="tmp/chest/coverage.json")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("sheet", help="無作為抽出の目視一覧")
    common(p)
    p.add_argument("--count", type=int, default=48)
    p.add_argument("--columns", type=int, default=8)
    p.add_argument("--tile", type=int, default=256)
    p.add_argument("--output", default="tmp/chest/sheet.png")
    p.set_defaults(func=cmd_sheet)

    p = sub.add_parser("vectors", help="全量のベクトル生成（再開可能）")
    common(p)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--service-url", default="http://127.0.0.1:8002")
    p.add_argument("--pool-side", type=int, default=rse.POOL_SIDE)
    p.add_argument("--crop-side", type=int, default=rse.POOL_SIDE)
    p.add_argument("--pool-batch", type=int, default=rse.POOL_MAX_BATCH)
    # crop も 518 四方なので pooling と同じく小さく送る。64 枚を 1 回で送っていた
    # ときに、全量の途中で embedding-service が切断した。
    p.add_argument("--crop-batch", type=int, default=rse.POOL_MAX_BATCH,
                   help="crop-chest が 1 リクエストで送る crop 数")
    p.add_argument("--chunk", type=int, default=64, help="埋め込みへ送る単位")
    p.add_argument("--checkpoint", type=int, default=CHECKPOINT, help="shard を書く間隔（枚）")
    p.add_argument("--refresh", action="store_true")
    p.set_defaults(func=cmd_vectors)

    p = sub.add_parser("train", help="教師データ・学習・評価")
    common(p)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--class-weight-max", type=float, default=10.0)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-queries", type=int, default=20)
    p.add_argument("--max-queries", type=int, default=1000)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--json", default="tmp/chest/train.json")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("verify-ingest", help="本番の取り込み経路が評価と同じ特徴を作るか")
    common(p)
    p.add_argument("--roi-url", default="http://127.0.0.1:8001")
    p.add_argument("--service-url", default="http://127.0.0.1:8002")
    p.add_argument("--head", default="infra/chest/head-pool-chest.npz")
    p.add_argument("--cache", default="tmp/chest/vec_pool-chest_nihcc-unet-local-v1_112120.npz",
                   help="評価の pool-chest キャッシュ（分類器を学習した特徴）")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max-diff", type=float, default=0.02, help="確率の差の許容")
    p.add_argument("--min-cosine", type=float, default=0.999, help="プロファイルの cosine の下限")
    p.add_argument("--json", default="tmp/chest/verify-ingest.json")
    p.set_defaults(func=cmd_verify_ingest, limit=200)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
