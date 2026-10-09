"""学習した U-Net で、画像ディレクトリ全体の肺野 mask を作る。

    # 手元の 4,999 枚
    python scripts/segment_lungs.py --images NIHCC/images --out data/lung-masks

    # EC2 の約 11 万枚（GPU があれば provider を指定する）
    python scripts/segment_lungs.py --images /imports/nihcc/images \\
        --out /data/lung-masks --batch 16 --providers CUDAExecutionProvider

取り込み（ingest）は roi-service 経由で mask を作るので、通常運用でこの script は
要らない。使うのは次の場合である。

  - 取り込みの前に、全件の mask を作って品質を確かめたい
  - 形状の再計算だけを行いたい（画像の再取り込みをせずに）
  - 別の用途（統計・図）へ mask を出したい

出力は 1 枚 1 PNG。**bit を詰めた単チャネル**で、1=右肺 2=左肺 4=心臓。
`build_lung_dataset.py` のラベルと同じ持ち方なので、そのまま比較できる。

途中で止めても、既にある出力は飛ばして続きから再開する。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "image-pipeline" / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from image_pipeline import load_profile  # noqa: E402
from image_pipeline.pipeline import decode_image, normalize  # noqa: E402
from image_pipeline.segmenter import (  # noqa: E402
    CHANNELS,
    _largest_component,
    model_input_size,
)

BITS = (1, 2, 4)


def load_session(model_path: Path, providers: list[str]):
    import onnxruntime

    if not model_path.is_file():
        raise SystemExit(f"分割モデルが無い: {model_path}\n"
                         f"先に scripts/train_lung_unet.py で学習する")
    available = set(onnxruntime.get_available_providers())
    chosen = [p for p in providers if p in available]
    if not chosen:
        print(f"指定の provider が使えない（利用可能: {sorted(available)}）。CPU で動かす")
        chosen = ["CPUExecutionProvider"]
    print(f"provider: {chosen}")
    return onnxruntime.InferenceSession(str(model_path), providers=chosen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, default=ROOT / "NIHCC" / "images")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "lung-masks")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "lung-unet" / "lung-unet.onnx")
    parser.add_argument("--profile", default="nihcc-unet-v1",
                        help="前処理と閾値をここから読む。解像度はモデルから読む")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=-1.0,
                        help="既定は profile の値を使う")
    parser.add_argument("--providers", nargs="+",
                        default=["CUDAExecutionProvider", "CPUExecutionProvider"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", help="既存の出力も作り直す")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    section = profile.data.get("lung_segmentation") or {}
    threshold = args.threshold if args.threshold >= 0 else float(section.get("threshold", 0.5))

    paths = sorted(args.images.glob("*.png"))
    if not paths:
        raise SystemExit(f"画像が無い: {args.images}")
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        before = len(paths)
        paths = [p for p in paths if not (args.out / p.name).exists()]
        if before != len(paths):
            print(f"既にある {before - len(paths)} 枚は飛ばす")
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        print("作るものが無い")
        return 0
    session = load_session(args.model, list(args.providers))
    input_name = session.get_inputs()[0].name

    # **解像度はモデルから読む。** profile の input_size を信じると、profile と
    # モデルの組み合わせを間違えたときに onnxruntime の分かりにくい例外になる
    # （`Got: 384 Expected: 160`）。モデルは自分の入力の大きさを知っている。
    size = model_input_size(session)
    configured = section.get("input_size")
    if size is None:
        size = int(configured or 384)
        print(f"モデルの入力が動的なので profile の {size} を使う")
    elif configured is not None and int(configured) != size:
        print(f"注意: profile {args.profile} の input_size は {configured} だが、"
              f"モデルは {size} だった。モデルに合わせる")
    print(f"{len(paths)} 枚、{size}x{size}、閾値 {threshold}")

    began = time.time()
    done = 0
    failures: list[dict] = []
    summary: list[dict] = []
    for start in range(0, len(paths), args.batch):
        chunk = paths[start : start + args.batch]
        batch, kept = [], []
        for path in chunk:
            try:
                gray, _, _ = normalize(decode_image(path.read_bytes()), profile)
            except Exception as exc:                            # noqa: BLE001
                failures.append({"image": path.name, "error": str(exc)})
                continue
            batch.append(cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA))
            kept.append((path, gray.shape))
        if not batch:
            continue

        array = (np.stack(batch).astype(np.float32) / 255.0)[:, None]
        logits = session.run(None, {input_name: array})[0]
        probabilities = 1.0 / (1.0 + np.exp(-logits.astype(np.float32)))

        for index, (path, shape) in enumerate(kept):
            packed = np.zeros(shape, dtype=np.uint8)
            areas = {}
            for channel, bit in enumerate(BITS):
                # 確率のまま元の解像度へ戻し、それから閾値を掛ける
                full = cv2.resize(probabilities[index, channel], (shape[1], shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
                mask = _largest_component(((full > threshold).astype(np.uint8)) * 255)
                packed[mask > 0] |= bit
                areas[CHANNELS[channel]] = int((mask > 0).sum())
            Image.fromarray(packed).save(args.out / path.name, optimize=True)
            total = int(shape[0] * shape[1])
            summary.append({
                "image": path.name,
                "areas": areas,
                "lung_ratio": (areas["right_lung"] + areas["left_lung"]) / total,
            })
            done += 1

        if done and done % (args.batch * 25) < args.batch:
            rate = done / max(time.time() - began, 1e-9)
            remaining = (len(paths) - done) / max(rate, 1e-9) / 60
            print(f"  {done}/{len(paths)} 枚（{rate:.1f} 枚/秒、残り {remaining:.0f} 分）",
                  flush=True)

    ratios = np.array([s["lung_ratio"] for s in summary]) if summary else np.zeros(1)
    report = {
        "model": str(args.model),
        "profile": args.profile,
        "roi_model_version": profile.roi_model_version,
        "input_size": size,
        "profile_input_size": configured,
        "threshold": threshold,
        "count": done,
        "failures": failures,
        "lung_ratio": {
            "median": float(np.median(ratios)),
            "p05": float(np.percentile(ratios, 5)),
            "p95": float(np.percentile(ratios, 95)),
        },
        #: 肺野が極端に小さい・大きいものは要確認。目視の当たりを付けるために残す
        "suspicious": sorted(
            (s["image"] for s in summary if not 0.10 <= s["lung_ratio"] <= 0.40))[:200],
    }
    (args.out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{done} 枚を {args.out} に書き出した（{(time.time() - began) / 60:.1f} 分）")
    print(f"肺野の割合 中央値 {report['lung_ratio']['median'] * 100:.1f}%"
          f"（5-95 パーセンタイル {report['lung_ratio']['p05'] * 100:.1f}-"
          f"{report['lung_ratio']['p95'] * 100:.1f}%）")
    if failures:
        print(f"読めなかった画像 {len(failures)} 枚")
    if report["suspicious"]:
        print(f"肺野の割合が 10-40% を外れる画像 {len(report['suspicious'])} 枚"
              f"（summary.json の suspicious。目視の当たりに使う）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
