"""評価で学習した胸部野の分類器を、本番の成果物として書き出す。

`scripts/chest_image_eval.py train` が書いた `head_pool-chest_<profile>.npz`
（標準化の平均・標準偏差と線形の重み）に、**特徴の作り方**を添えて保存する。
embedding-service は起動時にこれと自分の構成を照合し、違えば起動を止める
（`nihcc_rag.chest_head.ChestHead.check_compatible`）。

    python scripts/export_chest_head.py \\
      --head tmp/chest/ec2-full/head_pool-chest_nihcc-unet-local-v1.npz \\
      --train-json tmp/chest/ec2-full/train.json \\
      --backbone-revision <RAD-DINO の snapshot（40 桁）> \\
      --output infra/chest/head-pool-chest.npz

**backbone の revision は評価に使ったものを指定する**
（評価時の RAD-DINO の snapshot）。分類器はその重みの
特徴で学習しているので、別の revision の RAD-DINO と組み合わせると確率が黙ってずれる。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in ("packages/common/src", "packages/image-pipeline/src"):
    sys.path.insert(0, str(ROOT / _path))

from image_pipeline import chest as chest_spec  # noqa: E402
from nihcc_rag.chest_head import UNPINNED, ChestHead  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--head", required=True, help="chest_image_eval.py train が書いた head_*.npz")
    parser.add_argument("--train-json", required=True, help="同じ実行の train.json（追跡用の記録）")
    parser.add_argument("--variant", default="pool-chest")
    parser.add_argument("--backbone-model", default="microsoft/rad-dino")
    parser.add_argument("--backbone-revision", required=True,
                        help="評価に使った RAD-DINO の snapshot（40 桁）")
    parser.add_argument("--allow-unpinned", action="store_true",
                        help="revision を unpinned のまま書き出す（手元の動作確認用。本番では使わない）")
    parser.add_argument("--roi-profile", default="nihcc-unet-local-v1",
                        help="評価に使った U-Net の profile")
    parser.add_argument("--split-seed", type=int, default=20260920)
    parser.add_argument("--output", default="infra/chest/head-pool-chest.npz")
    args = parser.parse_args()

    revision = args.backbone_revision.strip()
    if revision in ("", UNPINNED) and not args.allow_unpinned:
        raise SystemExit(
            "backbone の revision が固定されていません。評価に使った RAD-DINO の snapshot を "
            "--backbone-revision に指定してください（docs/runbooks/chest-classifier.md §2.1）"
        )
    if revision not in ("", UNPINNED) and len(revision) != 40:
        raise SystemExit(f"revision は 40 桁の commit hash を指定してください: {revision!r}")

    blob = np.load(ROOT / args.head, allow_pickle=True)
    report = json.loads((ROOT / args.train_json).read_text(encoding="utf-8"))
    variant = report["variants"].get(args.variant)
    if variant is None:
        raise SystemExit(f"train.json に {args.variant} がありません")
    findings = tuple(str(v) for v in blob["findings"])
    if list(findings) != list(report["findings"]):
        raise SystemExit("head と train.json の所見の並びが違います（別の実行の組み合わせ）")

    head = ChestHead(
        mean=blob["mean"].astype(np.float64),
        std=blob["std"].astype(np.float64),
        weights=blob["w"].astype(np.float64),
        bias=blob["b"].astype(np.float64),
        findings=findings,
        kappa=float(blob["kappa"]),
        backbone_model=args.backbone_model,
        backbone_revision=revision or UNPINNED,
        pool_side=chest_spec.POOL_SIDE,
        diaphragm_ratio=chest_spec.DIAPHRAGM_RATIO,
        margin_ratio=chest_spec.MARGIN_RATIO,
        roi_profile=args.roi_profile,
        train_images=int(report["split"]["train"]),
        split_seed=args.split_seed,
        test_auc_mean=float(variant["image_auc_mean"]),
    )
    if head.input_dimension != int(variant["dimension"]):
        raise SystemExit("head の入力次元が train.json の記録と違います")

    output = ROOT / args.output
    digest = head.save(output)
    shown = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    print(f"書き出しました: {shown}")
    print(f"  sha256        {digest}")
    print(f"  model_version chest-{digest[:12]}-d{head.dimension}")
    print(f"  backbone      {head.backbone_model}@{head.backbone_revision}")
    print(f"  特徴の作り方  pool {head.pool_side} / diaphragm {head.diaphragm_ratio} / "
          f"margin {head.margin_ratio} / profile {head.roi_profile}")
    print(f"  学習          train {head.train_images} 枚 / test 平均 AUC {head.test_auc_mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
