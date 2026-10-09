"""ROI の肺野を、CheXmask の参照 mask と Dice / IoU で比べる。

    python scripts/eval_roi_dice.py [--limit N] [--json 出力.json]
    python scripts/eval_roi_dice.py --inspect 6 --out-dir tmp/roi

ROI の良否を目視ではなく数値で判断するための土台である。
参照 mask は scripts/fetch_reference_masks.py で取り込む。

参照は手描きではなく模型が作ったものなので、**これを真の正解として扱わない。**
使い方は 2 つ。

  1. 破綻の検出   : Dice が極端に低い画像は、どちらかが明確に間違っている
  2. 案どうしの比較 : 同じ参照に対する相対順位は、目視 5 枚よりはるかに信頼できる

画像ごとの品質スコア（Dice RCA Mean）が付いているので、既定では配布元の推奨に
従って 0.7 以上の画像だけを使う。
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "image-pipeline" / "src"))
sys.path.insert(0, str(ROOT / "packages" / "common" / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from image_pipeline import extract, load_profile  # noqa: E402
from nihcc_rag import evaluation  # noqa: E402

REFERENCE = ROOT / "data" / "reference-masks" / "chestxray8.csv"
IMAGES = ROOT / "NIHCC" / "images"

csv.field_size_limit(1 << 30)


def rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    """CheXmask の run-length encoding を mask へ戻す。

    `開始位置 長さ 開始位置 長さ ...` が空白区切りで並ぶ。
    """
    runs = np.fromstring(rle, dtype=np.int64, sep=" ")
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(runs[::2], runs[1::2]):
        mask[start : start + length] = 1
    return mask.reshape((height, width))


def dice(a: np.ndarray, b: np.ndarray) -> float:
    total = int(a.sum()) + int(b.sum())
    if total == 0:
        return 1.0
    return 2.0 * float(np.logical_and(a, b).sum()) / total


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum()) / union


def load_reference(path: Path, min_rca: float) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rca = float(row["Dice RCA (Mean)"])
            if rca < min_rca:
                continue
            rows[row["Image Index"]] = {
                "rca": rca,
                "left": row["Left Lung"],
                "right": row["Right Lung"],
                "height": int(row["Height"]),
                "width": int(row["Width"]),
            }
    return rows


def reference_masks(entry: dict, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """参照 mask を作業解像度へ合わせて返す（画像の左半分, 右半分）。

    CheXmask の Left / Right は患者から見た左右なので、画像上では逆になる。
    念のため重心で確かめ、取り違えていれば入れ替える。
    """
    h, w = entry["height"], entry["width"]
    patient_left = rle_to_mask(entry["left"], h, w)
    patient_right = rle_to_mask(entry["right"], h, w)
    if patient_left.any() and patient_right.any():
        if np.nonzero(patient_left)[1].mean() < np.nonzero(patient_right)[1].mean():
            patient_left, patient_right = patient_right, patient_left
    out = []
    for m in (patient_right, patient_left):   # 画像の左半分, 右半分 の順
        if (h, w) != shape:
            m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        out.append(m.astype(bool))
    return out[0], out[1]


def evaluate(names: list[str], reference: dict[str, dict], profile_name: str,
             profile_dir: str | None = None) -> list[dict]:
    results = []
    for i, name in enumerate(names, 1):
        entry = reference[name]
        try:
            output = extract((IMAGES / name).read_bytes(), profile_name=profile_name,
                             profile_dir=profile_dir)
        except Exception as exc:                      # noqa: BLE001
            results.append({"image": name, "error": str(exc)})
            continue
        gray_shape = output.normalized.shape
        ref_r, ref_l = reference_masks(entry, gray_shape)
        ours_r = np.asarray(output.right_mask) > 0
        ours_l = np.asarray(output.left_mask) > 0
        results.append({
            "image": name,
            "rca": entry["rca"],
            "dice": dice(ours_r | ours_l, ref_r | ref_l),
            "iou": jaccard(ours_r | ours_l, ref_r | ref_l),
            "dice_right": dice(ours_r, ref_r),
            "dice_left": dice(ours_l, ref_l),
            "area_ours": float((ours_r | ours_l).mean()),
            "area_ref": float((ref_r | ref_l).mean()),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(names)} 枚", flush=True)
    return results


def summarise(results: list[dict], run: evaluation.Run | None = None) -> None:
    ok = [r for r in results if "dice" in r]
    failed = [r for r in results if "error" in r]
    if not ok:
        print("評価できた画像がない")
        return
    d = sorted(r["dice"] for r in ok)
    print(f"\n評価 {len(ok)} 枚" + (f"（抽出失敗 {len(failed)} 枚）" if failed else ""))
    print(f"\n  Dice   中央値 {statistics.median(d):.3f}   "
          f"四分位 {np.percentile(d, 25):.3f}-{np.percentile(d, 75):.3f}   "
          f"最小 {d[0]:.3f}")
    print(f"  IoU    中央値 {statistics.median([r['iou'] for r in ok]):.3f}")
    for threshold in (0.5, 0.7, 0.8, 0.9):
        n = sum(1 for x in d if x < threshold)
        print(f"  Dice < {threshold:.1f} : {n:4d} 枚 ({n / len(d) * 100:4.1f}%)")

    if run is not None:
        # **実効的な破綻**（Dice < 0.7 または抽出失敗）が主指標である。中央値だけ
        # 見ていると、少数の派手な失敗を見落とす。
        total = len(ok) + len(failed)
        broken = sum(1 for x in d if x < 0.7) + len(failed)
        run.metric("dice_median", statistics.median(d))
        run.metric("dice_p25", float(np.percentile(d, 25)))
        run.metric("dice_p75", float(np.percentile(d, 75)))
        run.metric("dice_min", d[0])
        run.metric("iou_median", statistics.median([r["iou"] for r in ok]))
        run.metric("extraction_failure_rate", len(failed) / max(total, 1))
        run.metric("breakdown_rate", broken / max(total, 1))
        run.metric("area_median_ours", statistics.median([r["area_ours"] for r in ok]))
        run.metric("area_median_reference", statistics.median([r["area_ref"] for r in ok]))
    print(f"\n  肺野面積（画像に対する割合）  我々 {statistics.median([r['area_ours'] for r in ok]) * 100:.1f}%"
          f"   参照 {statistics.median([r['area_ref'] for r in ok]) * 100:.1f}%")
    worst = sorted(ok, key=lambda r: r["dice"])[:10]
    print("\n  悪い順:")
    for r in worst:
        print(f"    {r['image']}  Dice {r['dice']:.3f}"
              f"  (右 {r['dice_right']:.3f} 左 {r['dice_left']:.3f})"
              f"  面積 我々 {r['area_ours']*100:4.1f}% / 参照 {r['area_ref']*100:4.1f}%")


def draw_outline(rgb: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    """mask の外周を描く（4 近傍の収縮との差を太らせる）。"""
    a = np.asarray(mask) > 0
    eroded = a.copy()
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        eroded &= np.roll(a, shift, axis=axis)
    edge = a & ~eroded
    edge = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return np.where(edge[:, :, None], np.array(colour, np.float32), rgb)


def inspect(results: list[dict], reference: dict[str, dict], profile_name: str,
            count: int, out_dir: Path, profile_dir: str | None = None) -> None:
    """Dice が低い順に、我々の外周（黄）と参照の外周（水色）を重ねて描く。"""
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    worst = sorted((r for r in results if "dice" in r), key=lambda r: r["dice"])[:count]
    tiles = []
    for r in worst:
        output = extract((IMAGES / r["image"]).read_bytes(), profile_name=profile_name,
                         profile_dir=profile_dir)
        gray = output.normalized
        rgb = np.dstack([gray] * 3).astype(np.float32)
        ref_r, ref_l = reference_masks(reference[r["image"]], gray.shape)
        rgb = draw_outline(rgb, ref_r | ref_l, (0, 200, 255))          # 参照: 水色
        rgb = draw_outline(rgb, (np.asarray(output.right_mask) > 0)
                           | (np.asarray(output.left_mask) > 0), (255, 214, 0))   # 我々: 黄
        image = Image.fromarray(rgb.clip(0, 255).astype(np.uint8))
        image.thumbnail((300, 300), Image.LANCZOS)
        tiles.append(np.asarray(image.convert("RGB")))
        print(f"  {r['image']}  Dice {r['dice']:.3f}")
    if not tiles:
        return
    h = min(t.shape[0] for t in tiles)
    w = min(t.shape[1] for t in tiles)
    per_row = 3
    rows = [np.concatenate([t[:h, :w] for t in tiles[i : i + per_row]], axis=1)
            for i in range(0, len(tiles) - len(tiles) % per_row, per_row)]
    if rows:
        path = out_dir / "worst.png"
        Image.fromarray(np.concatenate(rows, axis=0)).save(path)
        print(f"  {path} に保存（黄=我々 水色=参照）")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--profile", default="nihcc-classical-v1")
    parser.add_argument("--profile-dir", default=None,
                        help="package 外の profile を使う（試験用）")
    parser.add_argument("--limit", type=int, default=100, help="評価する枚数（0 で全件）")
    parser.add_argument("--min-rca", type=float, default=0.7, help="参照の品質の下限")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, help="1 枚ごとの結果を書き出す")
    parser.add_argument("--note", default="", help="DB に残す覚書")
    parser.add_argument("--inspect", type=int, default=0, help="Dice が低い N 枚を描く")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "tmp" / "roi-dice")
    args = parser.parse_args()

    if not args.reference.exists():
        print(f"参照 mask が無い: {args.reference}\n"
              f"先に python scripts/fetch_reference_masks.py を実行する", file=sys.stderr)
        return 1

    reference = load_reference(args.reference, args.min_rca)
    names = sorted(n for n in reference if (IMAGES / n).exists())
    print(f"参照 {len(reference)} 枚（品質 {args.min_rca} 以上）、うち手元に画像がある {len(names)} 枚")
    if args.limit and args.limit < len(names):
        rng = np.random.default_rng(args.seed)
        names = [names[i] for i in sorted(rng.choice(len(names), args.limit, replace=False))]
        print(f"無作為に {len(names)} 枚を評価する")

    profile = load_profile(args.profile, args.profile_dir)
    with evaluation.record(
        kind="roi-dice",
        target_version=profile.roi_model_version,
        params={
            "profile": args.profile,
            "limit": args.limit,
            "min_rca": args.min_rca,
            "seed": args.seed,
            "reference": "CheXmask ChestX-Ray8 v1.0.0",
        },
        sample_size=len(names),
        note=args.note or None,
    ) as run:
        results = evaluate(names, reference, args.profile, args.profile_dir)
        summarise(results, run)
    if not run.persisted:
        print("（DB へ記録できなかった。JSON は書き出されている）")
    if args.inspect:
        print(f"\nDice が低い {args.inspect} 枚を描く:")
        inspect(results, reference, args.profile, args.inspect, args.out_dir,
                args.profile_dir)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{args.json} に書き出した")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
