"""CheXmask の参照 mask から、肺野分割モデルの学習データを作る。

    python scripts/build_lung_dataset.py --max 8000 --size 384

`scripts/fetch_reference_masks.py` で取り込んだ CSV と NIHCC の画像を突き合わせ、
学習に使える配列へ落とす。

## 設計上の要点

**前処理は推論時とまったく同じ `image_pipeline.pipeline.normalize` を通す。** ここが
ずれると学習時と推論時で入力の分布が変わり、Dice が落ちる。normalize は百分位の
clip と極性の反転を行うので、単に読み込んで 0-255 に伸ばすのとは違う。

**分割は患者単位で行う。** ChestX-ray14 は 1 人の患者が複数枚を持つ
（`00000001_000` / `00000001_001` ...）。画像単位で分けると同じ患者が学習側と検証側
の両方に入り、成績が実力より良く出る。

**ラベルは 1 枚の uint8 に bit で詰める。** 右肺=1 / 左肺=2 / 心臓=4。心臓は肺と
重なることがあるので、排他的な多値ラベルにはできない。この持ち方なら重なっても
表せるうえ、3 枚別々に持つより小さい。

出力（`--out` 以下）:

    train-images.npy   uint8 [N, S, S]       normalize 済みの画像
    train-labels.npy   uint8 [N, S, S]       bit フラグ（1=右肺 2=左肺 4=心臓）
    train-names.json                          画像名（並びは配列と同じ）
    val-images.npy / val-labels.npy / val-names.json
    meta.json                                 大きさ・件数・作った条件
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "image-pipeline" / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from image_pipeline import load_profile  # noqa: E402
from image_pipeline.pipeline import decode_image, normalize  # noqa: E402

REFERENCE = ROOT / "data" / "reference-masks" / "chestxray8.csv"
IMAGES = ROOT / "NIHCC" / "images"
OUT = ROOT / "data" / "lung-dataset"

RIGHT, LEFT, HEART = 1, 2, 4

csv.field_size_limit(1 << 30)


def rle_to_mask(rle: str, height: int, width: int) -> np.ndarray:
    """CheXmask の run-length encoding を mask へ戻す（行優先）。"""
    runs = np.fromstring(rle, dtype=np.int64, sep=" ")
    mask = np.zeros(height * width, dtype=np.uint8)
    for start, length in zip(runs[::2], runs[1::2]):
        mask[start : start + length] = 1
    return mask.reshape((height, width))


def patient_of(name: str) -> str:
    """`00000001_000.png` -> `00000001`。"""
    return name.split("_", 1)[0]


def build_label(row: dict, size: int) -> np.ndarray:
    """1 枚ぶんのラベルを作る。

    CheXmask の Left / Right は患者から見た左右。画像上では左右が入れ替わるので、
    重心で確かめてから詰める（`nihcc_rag` 側も画像基準で右/左と呼んでいる）。
    """
    height, width = int(row["Height"]), int(row["Width"])
    patient_left = rle_to_mask(row["Left Lung"], height, width)
    patient_right = rle_to_mask(row["Right Lung"], height, width)
    if patient_left.any() and patient_right.any():
        if np.nonzero(patient_left)[1].mean() < np.nonzero(patient_right)[1].mean():
            patient_left, patient_right = patient_right, patient_left
    heart = rle_to_mask(row["Heart"], height, width) if row.get("Heart") else None

    label = np.zeros((height, width), dtype=np.uint8)
    label[patient_right > 0] |= RIGHT      # 画像の左半分
    label[patient_left > 0] |= LEFT        # 画像の右半分
    if heart is not None:
        label[heart > 0] |= HEART
    # bit を保つため最近傍で縮小する（線形だと中間値が出て別の bit になる）
    return cv2.resize(label, (size, size), interpolation=cv2.INTER_NEAREST)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--images", type=Path, default=IMAGES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--profile", default="nihcc-classical-v1")
    parser.add_argument("--size", type=int, default=384, help="学習解像度")
    parser.add_argument("--max", type=int, default=0, help="使う枚数の上限（0 で全件）")
    parser.add_argument("--min-rca", type=float, default=0.8,
                        help="参照の品質の下限。学習では評価時(0.7)より厳しくする")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.reference.exists():
        print(f"参照 mask が無い: {args.reference}\n"
              f"先に python scripts/fetch_reference_masks.py を実行する", file=sys.stderr)
        return 1

    profile = load_profile(args.profile)

    rows: list[dict] = []
    with args.reference.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if float(row["Dice RCA (Mean)"]) < args.min_rca:
                continue
            if not (args.images / row["Image Index"]).exists():
                continue
            rows.append(row)
    print(f"参照 {len(rows)} 枚（品質 {args.min_rca} 以上、手元に画像がある分）")
    if not rows:
        return 1

    # --- 患者単位で分ける ---
    patients = sorted({patient_of(r["Image Index"]) for r in rows})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * args.val_frac))
    val_patients = set(patients[:n_val])
    print(f"患者 {len(patients)} 人を {len(patients) - n_val} / {n_val} に分ける")

    rows.sort(key=lambda r: r["Image Index"])
    if args.max:
        # 患者の偏りを避けるため、患者単位で丸ごと採る
        keep, seen_patients = [], set()
        for row in rows:
            seen_patients.add(patient_of(row["Image Index"]))
            keep.append(row)
            if len(keep) >= args.max:
                break
        rows = keep
        print(f"上限により {len(rows)} 枚（患者 {len(seen_patients)} 人）を使う")

    split = {"train": [], "val": []}
    for row in rows:
        split["val" if patient_of(row["Image Index"]) in val_patients else "train"].append(row)
    print(f"学習 {len(split['train'])} 枚 / 検証 {len(split['val'])} 枚")

    args.out.mkdir(parents=True, exist_ok=True)
    size = args.size
    began = time.time()
    for part, part_rows in split.items():
        if not part_rows:
            continue
        images = np.lib.format.open_memmap(
            args.out / f"{part}-images.npy", mode="w+", dtype=np.uint8,
            shape=(len(part_rows), size, size))
        labels = np.lib.format.open_memmap(
            args.out / f"{part}-labels.npy", mode="w+", dtype=np.uint8,
            shape=(len(part_rows), size, size))
        names = []
        written = 0
        for row in part_rows:
            name = row["Image Index"]
            try:
                raw = decode_image((args.images / name).read_bytes())
                gray, _, _ = normalize(raw, profile)
            except Exception as exc:                            # noqa: BLE001
                print(f"  飛ばす {name}: {exc}")
                continue
            if raw.shape[:2] != (int(row["Height"]), int(row["Width"])):
                # 参照と原画像の大きさが違うと座標が合わない
                print(f"  飛ばす {name}: 大きさが違う "
                      f"{raw.shape[:2]} 対 ({row['Height']}, {row['Width']})")
                continue
            images[written] = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
            labels[written] = build_label(row, size)
            names.append(name)
            written += 1
            if written % 500 == 0:
                print(f"  {part} {written}/{len(part_rows)} 枚"
                      f"（{time.time() - began:.0f} 秒）", flush=True)
        images.flush()
        labels.flush()
        del images, labels
        # 飛ばした分だけ後ろが空なので切り詰める。
        # Windows では memmap を開いたまま同じパスへ書けないので、一度 RAM へ
        # 読み切ってから参照を捨てる。
        for kind in ("images", "labels"):
            path = args.out / f"{part}-{kind}.npy"
            handle = np.load(path, mmap_mode="r")
            trimmed = np.array(handle[:written])
            del handle
            np.save(path, trimmed)
        (args.out / f"{part}-names.json").write_text(
            json.dumps(names, ensure_ascii=False), encoding="utf-8")
        print(f"{part}: {written} 枚を書き出した")

    meta = {
        "size": size,
        "profile": args.profile,
        "min_rca": args.min_rca,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "bits": {"right_lung": RIGHT, "left_lung": LEFT, "heart": HEART},
        "source": "CheXmask Database v1.0.0 (PhysioNet, CC BY 4.0)",
    }
    (args.out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(p.stat().st_size for p in args.out.glob("*.npy"))
    print(f"\n{args.out} に保存（{total / (1 << 20):.0f} MB、{(time.time() - began) / 60:.1f} 分）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
