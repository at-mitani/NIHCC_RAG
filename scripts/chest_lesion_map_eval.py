"""胸部野の分類器の「区画ごとの地図」が、病変の位置と大きさを示すかを正解 box で測る。

背景
----
検索の単位は画像 1 枚の胸部野なので、病変の位置・大きさを近似検索するには病変の位置を
表す情報が別に要る。
データセットのラベルは画像単位なので、位置の正解は `BBox_List_2017.csv`（880 枚）しか
無い。そこで追加の教師なしで作れる位置の地図を試す。

胸部野の分類器は「胸部野で加重平均した RAD-DINO の patch → 標準化 → 線形」なので、
ロジットは区画（37 × 37）ごとの寄与の加重平均に正確に分解できる。その寄与を地図と
する（embedding-service `/api/v1/embeddings/chest-map`）。

**効くとは限らない。** ViT の patch は自己注意で画像全体の情報を持つので、区画ごとの
寄与が局所の情報だけを表すとは限らない。だから作り込む前に測る。

比べる 4 つ（1 枚・1 所見につき「ここが病変」という点を 1 つ出す）
---------------------------------------------------------------
map     地図の最大の区画（胸部野の中）の中心
bright  肺野の中で「白く浮いて見える点」（white top-hat の最大）。利用者が挙げた
        「肺野の内側に点在する白い画像」を自動で拾う方法
random  胸部野の中に無作為に置いた点。当たる確率 = box の面積 / 胸部野の面積（解析的に）
prior   その所見の box が平均的にある位置（胸部野の外接矩形で正規化。1 枚抜き）

指標
----
hit           点が、その所見のいずれかの box の中に入った割合（pointing game）
enrichment    地図の正の値のうち box に入った割合 ÷ box が胸部野に占める割合（1 が無作為）
size_spearman 地図の最大点の周り（最大値の半分以上でつながる区画）の面積と、box の面積の
              順位相関（大きさを表せているか）

合格線（**実行前に固定**）
--------------------------
所見ごとに、次の 3 つをすべて満たせば「地図はその所見の位置を示す」と判定する。
  1. map の hit が random の 2 倍以上
  2. map の hit が prior を上回る
  3. map の hit の 95% 区間の下限が random を上回る
大きさは参考（size_spearman > 0.3 なら「大きさも表す」とする）。

    python scripts/chest_lesion_map_eval.py --service-url http://127.0.0.1:8002

U-Net の profile を使う（ホストから使うときは LUNG_MODEL_PATH を指定）。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in ("packages/common/src", "packages/image-pipeline/src"):
    sys.path.insert(0, str(ROOT / _path))

from image_pipeline import ExtractionError, chest_model_inputs, extract_chest  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

#: 正解 box がある 4 所見（Fibrosis には正解 box が無い）
FINDINGS = ("Infiltration", "Mass", "Nodule", "Pneumonia")
GOLD_LABEL = {"Infiltrate": "Infiltration"}
DEFAULT_PROFILE = os.environ.get("CHEST_PROFILE", "nihcc-unet-local-v1")
#: 白い点を拾う white top-hat の大きさ（作業解像度 1024 での直径。結節の中央値 70 画素の
#: 約 1.2 倍。これより大きい構造＝肋骨の帯・心陰影は背景として引かれる）
TOPHAT_DIAMETER = 81


def _env_value(key: str) -> str | None:
    env = ROOT / ".env"
    if not env.is_file():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def _init_worker() -> None:
    import cv2

    cv2.setNumThreads(1)


# --------------------------------------------------------------------------
# 幾何
# --------------------------------------------------------------------------


def coverage_grid(mask: np.ndarray, grid: int) -> np.ndarray:
    """2 値 mask を grid × grid の被覆率 [0, 1] へ縮約する（面積平均）。"""
    import cv2

    small = cv2.resize((mask > 0).astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)


def patch_center(row: int, col: int, grid: int, work_size: tuple[int, int], scale: float):
    """区画 (row, col) の中心を原画像の座標で返す。pool 画像は作業画像全体を正方形へ
    縮小したものなので、区画は作業画像を grid 等分した矩形に対応する。"""
    width, height = work_size
    x = (col + 0.5) * width / grid
    y = (row + 0.5) * height / grid
    return x / scale, y / scale


def inside_any(point, boxes) -> bool:
    x, y = point
    return any(bx <= x <= bx + bw and by <= y <= by + bh for bx, by, bw, bh in boxes)


def box_mask(boxes, work_size, scale) -> np.ndarray:
    """原画像座標の box 群を、作業解像度の 2 値 mask にする。"""
    width, height = work_size
    out = np.zeros((height, width), dtype=bool)
    for bx, by, bw, bh in boxes:
        x0, y0 = max(0, int(bx * scale)), max(0, int(by * scale))
        x1, y1 = min(width, int(np.ceil((bx + bw) * scale))), min(height, int(np.ceil((by + bh) * scale)))
        out[y0:y1, x0:x1] = True
    return out


def bright_point(gray: np.ndarray, lung: np.ndarray, scale: float):
    """肺野の中で「白く浮いて見える点」の位置（原画像の座標）。

    white top-hat（画像 − 開処理）は、構造要素より小さい明るい構造だけを残す。肋骨の
    帯や心陰影のように大きい明るい領域は引かれ、結節のような小さい白い点が残る。
    """
    import cv2

    diameter = max(3, int(round(TOPHAT_DIAMETER * gray.shape[1] / 1024)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
    tophat = cv2.GaussianBlur(tophat, (0, 0), sigmaX=diameter / 6)
    tophat[lung == 0] = -1.0
    y, x = np.unravel_index(int(np.argmax(tophat)), tophat.shape)
    return (x + 0.5) / scale, (y + 0.5) / scale


def blob_area(values: np.ndarray, valid: np.ndarray, start: tuple[int, int]) -> int:
    """最大点からつながる「最大値の半分以上」の区画数（4 近傍）。"""
    peak = values[start]
    if peak <= 0:
        return 1
    keep = valid & (values >= 0.5 * peak)
    seen = np.zeros_like(keep)
    stack, count = [start], 0
    while stack:
        r, c = stack.pop()
        if seen[r, c] or not keep[r, c]:
            continue
        seen[r, c] = True
        count += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < keep.shape[0] and 0 <= cc < keep.shape[1]:
                stack.append((rr, cc))
    return count


def spearman(a, b) -> float:
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def bootstrap_ci(values, reps: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(reps)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


# --------------------------------------------------------------------------
# 1 枚の前処理（worker）
# --------------------------------------------------------------------------


def _prepare(task):
    """胸部野を抽出し、地図の評価に要る小さな値だけを返す（mask 全体は持ち帰らない）。"""
    name, path, profile, grid, boxes_by_finding, thumb = task
    try:
        data = Path(path).read_bytes()
        ex = extract_chest(data, profile_name=profile)
    except ExtractionError as exc:
        return name, None, exc.code
    except Exception as exc:  # noqa: BLE001 - 1 枚の失敗で全体を止めない
        return name, None, type(exc).__name__
    inputs = chest_model_inputs(ex)
    height, width = ex.normalized.shape
    work = (width, height)
    chest = ex.chest > 0
    chest_area = float(chest.sum())
    ys, xs = np.nonzero(chest)
    chest_bbox = [xs.min() / ex.scale, ys.min() / ex.scale,
                  (xs.max() + 1) / ex.scale, (ys.max() + 1) / ex.scale]
    per_finding = {}
    for finding, boxes in boxes_by_finding.items():
        inside = box_mask(boxes, work, ex.scale) & chest
        per_finding[finding] = {
            "boxes": boxes,
            # 胸部野に無作為に置いた点が box に入る確率
            "random_hit": float(inside.sum()) / chest_area,
            "box_grid": coverage_grid(box_mask(boxes, work, ex.scale), grid).tolist(),
            "box_area": float(max(bw * bh for _, _, bw, bh in boxes)),
        }
    out = {
        "pool_image": base64.b64encode(inputs.pool_image).decode(),
        "pool_mask": base64.b64encode(inputs.pool_masks["CHEST"]).decode(),
        "work": work,
        "scale": ex.scale,
        "chest_grid": coverage_grid(ex.chest, grid).tolist(),
        "chest_bbox": chest_bbox,
        "bright": bright_point(ex.normalized, ex.lung, ex.scale),
        "findings": per_finding,
    }
    if thumb:
        import cv2

        out["thumb"] = cv2.resize(ex.normalized, (thumb, thumb), interpolation=cv2.INTER_AREA).tolist()
    return name, out, ""


# --------------------------------------------------------------------------
# 評価
# --------------------------------------------------------------------------


def evaluate(records: list[dict], grid: int, args) -> dict:
    """records: 1 枚 1 所見ごとの dict（地図・正解・各方法の点）。"""
    by_finding = defaultdict(list)
    for r in records:
        by_finding[r["finding"]].append(r)

    # prior: その所見の box の中心（胸部野の外接矩形で正規化）の平均。1 枚抜きで求める
    def normalised_center(r):
        bx, by, bw, bh = r["boxes"][0]
        x0, y0, x1, y1 = r["chest_bbox"]
        return ((bx + bw / 2 - x0) / max(x1 - x0, 1), (by + bh / 2 - y0) / max(y1 - y0, 1))

    report = {}
    for finding in FINDINGS:
        rows = by_finding.get(finding, [])
        if not rows:
            continue
        centers = np.array([normalised_center(r) for r in rows])
        hits = {"map": [], "bright": [], "prior": [], "random": []}
        enrich, map_areas, box_areas, probs = [], [], [], []
        for i, r in enumerate(rows):
            hits["map"].append(inside_any(r["map_point"], r["boxes"]))
            hits["bright"].append(inside_any(r["bright"], r["boxes"]))
            hits["random"].append(r["random_hit"])
            others = np.delete(centers, i, axis=0) if len(rows) > 1 else centers
            mx, my = others.mean(axis=0)
            x0, y0, x1, y1 = r["chest_bbox"]
            hits["prior"].append(inside_any((x0 + mx * (x1 - x0), y0 + my * (y1 - y0)), r["boxes"]))
            if r["enrichment"] is not None:
                enrich.append(r["enrichment"])
            map_areas.append(r["map_area"])
            box_areas.append(r["box_area"])
            probs.append(r["probability"])
        rates = {k: float(np.mean(v)) for k, v in hits.items()}
        ci = bootstrap_ci(hits["map"], args.bootstrap, args.seed)
        passed = (
            rates["map"] >= 2 * rates["random"]
            and rates["map"] > rates["prior"]
            and ci[0] > rates["random"]
        )
        size_rho = spearman(map_areas, box_areas)
        report[finding] = {
            "images": len(rows),
            "hit": rates,
            "map_hit_ci": ci,
            "enrichment_median": float(np.median(enrich)) if enrich else float("nan"),
            "size_spearman": size_rho,
            "mean_probability": float(np.mean(probs)),
            "passed": bool(passed),
            "size_passed": bool(size_rho > 0.3) if np.isfinite(size_rho) else False,
        }
    return report


def score_map(r: dict, values: np.ndarray, chest_grid: np.ndarray, box_grid: np.ndarray, grid: int):
    """1 枚 1 所見の地図から、最大点・enrichment・塊の面積を求める。"""
    valid = chest_grid >= 0.5
    masked = np.where(valid, values, -np.inf)
    row, col = np.unravel_index(int(np.argmax(masked)), masked.shape)
    point = patch_center(row, col, grid, tuple(r["work"]), r["scale"])
    positive = np.where(valid, np.clip(values, 0, None), 0.0)
    box_share = float((box_grid * valid).sum() / max(valid.sum(), 1))
    if positive.sum() > 0 and box_share > 0:
        enrichment = float((positive * box_grid).sum() / positive.sum()) / box_share
    else:
        enrichment = None
    area_patches = blob_area(values, valid, (row, col))
    width, height = r["work"]
    patch_area = (width / grid / r["scale"]) * (height / grid / r["scale"])
    return point, enrichment, area_patches * patch_area


def draw_sheet(samples: list[dict], grid: int, path: Path, columns: int = 6) -> None:
    """無作為に選んだ例に、地図（赤）・正解 box（緑）・地図の点（赤丸）・白い点（青丸）を重ねる。"""
    import cv2

    tiles = []
    for s in samples:
        side = len(s["thumb"])
        gray = np.asarray(s["thumb"], dtype=np.uint8)
        tile = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        values = np.where(np.asarray(s["chest_grid"]) >= 0.5, s["values"], np.nan)
        finite = values[np.isfinite(values)]
        if finite.size:
            lo, hi = np.percentile(finite, 50), finite.max()
            heat = np.clip((np.nan_to_num(values, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
            heat = cv2.resize(heat.astype(np.float32), (side, side), interpolation=cv2.INTER_LINEAR)
            tile[:, :, 2] = np.clip(tile[:, :, 2] + heat * 120, 0, 255).astype(np.uint8)
        width, height = s["work"]
        fx, fy = side / (width / s["scale"]), side / (height / s["scale"])
        for bx, by, bw, bh in s["boxes"]:
            cv2.rectangle(tile, (int(bx * fx), int(by * fy)), (int((bx + bw) * fx), int((by + bh) * fy)),
                          (0, 220, 0), 1)
        cv2.circle(tile, (int(s["map_point"][0] * fx), int(s["map_point"][1] * fy)), 5, (0, 0, 255), 2)
        cv2.circle(tile, (int(s["bright"][0] * fx), int(s["bright"][1] * fy)), 5, (255, 128, 0), 2)
        cv2.putText(tile, f"{s['finding'][:6]} p={s['probability']:.2f}", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        tiles.append(tile)
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    rows = [tiles[i : i + columns] for i in range(0, len(tiles), columns)]
    sheet = np.vstack([np.hstack(r + [blank] * (columns - len(r))) for r in rows])
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


def main() -> int:
    import httpx

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--service-url", default="http://127.0.0.1:8002")
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--gold-csv", default="tmp/gold/BBox_List_2017.csv")
    parser.add_argument("--data-dir", default="", help="既定は .env の NIHCC_DATA_DIR")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--sheet-count", type=int, default=36)
    parser.add_argument("--thumb", type=int, default=256)
    parser.add_argument("--json", default="tmp/chest/lesion-map.json")
    parser.add_argument("--sheet", default="tmp/chest/lesion-map-sheet.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir or _env_value("NIHCC_DATA_DIR") or "NIHCC")
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    image_dir = data_dir / "images"
    gold = ROOT / args.gold_csv
    if not gold.exists():
        raise SystemExit(f"正解 box がありません: {gold}")
    boxes: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with gold.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            finding = GOLD_LABEL.get(row[1], row[1])
            if finding in FINDINGS:
                boxes[row[0]][finding].append(tuple(float(v) for v in row[2:6]))
    available = {p.name for p in image_dir.iterdir()}
    names = sorted(n for n in boxes if n in available)
    print(f"対象: {len(names)} 枚（4 所見の正解 box がある画像のうち手元にあるもの）")

    client = httpx.Client(base_url=args.service_url, timeout=300.0)
    info = client.get("/api/v1/model").json()
    if not info.get("chest_head"):
        raise SystemExit("embedding-service の 2 段目が chest-profile ではありません")
    grid = int(info.get("region_pool_patch_grid") or 37)
    print(f"分類器: {info['fine_embedding_model_version']}  grid {grid}")

    rng = random.Random(args.seed)
    sheet_names = set(rng.sample(names, min(args.sheet_count, len(names))))
    tasks = [(n, str(image_dir / n), args.profile, grid,
              {f: list(v) for f, v in boxes[n].items()},
              args.thumb if n in sheet_names else 0) for n in names]

    prepared: dict[str, dict] = {}
    failures: dict[str, int] = defaultdict(int)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        for name, out, error in ex.map(_prepare, tasks, chunksize=2):
            if out is None:
                failures[error] += 1
            else:
                prepared[name] = out
    print(f"胸部野の抽出: {len(prepared)} 枚（失敗 {dict(failures)}）"
          f" {time.perf_counter() - started:.0f} 秒")

    records, samples = [], []
    ordered = sorted(prepared)
    for start in range(0, len(ordered), args.batch):
        batch = ordered[start : start + args.batch]
        response = client.post("/api/v1/embeddings/chest-map", json={
            "images_base64": [prepared[n]["pool_image"] for n in batch],
            "masks_base64": [prepared[n]["pool_mask"] for n in batch],
            "findings": list(FINDINGS),
        })
        response.raise_for_status()
        body = response.json()
        if body["grid"] != grid:
            raise SystemExit(f"地図の grid が想定と違います: {body['grid']}（想定 {grid}）")
        for i, name in enumerate(batch):
            p = prepared[name]
            chest_grid = np.asarray(p["chest_grid"])
            for finding, info_f in p["findings"].items():
                k = body["findings"].index(finding)
                values = np.asarray(body["maps"][i][k], dtype=np.float32).reshape(grid, grid)
                point, enrichment, area = score_map(
                    p, values, chest_grid, np.asarray(info_f["box_grid"]), grid)
                record = {
                    "name": name, "finding": finding, "boxes": info_f["boxes"],
                    "random_hit": info_f["random_hit"], "box_area": info_f["box_area"],
                    "chest_bbox": p["chest_bbox"], "bright": p["bright"],
                    "map_point": point, "enrichment": enrichment, "map_area": area,
                    "probability": float(body["probabilities"][i][k]),
                }
                records.append(record)
                if "thumb" in p:
                    samples.append({**record, "thumb": p["thumb"], "values": values,
                                    "chest_grid": p["chest_grid"], "work": p["work"],
                                    "scale": p["scale"]})
        print(f"  地図 {min(start + args.batch, len(ordered))}/{len(ordered)} 枚", flush=True)

    report = evaluate(records, grid, args)
    print(f"\n{'所見':<14}{'枚':>5}{'map':>8}{'bright':>8}{'prior':>8}{'random':>8}"
          f"{'map 95%区間':>16}{'enrich':>8}{'大きさρ':>8}  判定")
    for finding, r in report.items():
        h = r["hit"]
        print(f"{finding:<14}{r['images']:>5}{h['map']:>8.1%}{h['bright']:>8.1%}{h['prior']:>8.1%}"
              f"{h['random']:>8.1%}   [{r['map_hit_ci'][0]:.1%}, {r['map_hit_ci'][1]:.1%}]"
              f"{r['enrichment_median']:>8.2f}{r['size_spearman']:>8.2f}  "
              f"{'合格' if r['passed'] else '不合格'}{'（大きさも）' if r['size_passed'] else ''}")
    print("\nmap = 地図の最大点 / bright = 肺野の白い点 / prior = 所見の平均位置 / random = 胸部野の無作為点")
    print("合格 = map が random の 2 倍以上・prior 超え・95% 区間の下限が random 超え（実行前に固定）")

    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "classifier": info["fine_embedding_model_version"], "grid": grid,
        "images": len(prepared), "failures": dict(failures), "report": report,
        "records": [{k: v for k, v in r.items()} for r in records],
    }, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    draw_sheet(samples, grid, ROOT / args.sheet)
    print(f"保存: {out.relative_to(ROOT)} / {args.sheet}（無作為 {len(samples)} 例の重ね合わせ）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
