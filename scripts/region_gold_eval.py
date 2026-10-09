"""正解 bbox を使って「部位レベルの教師データを作れるか」を画像ごとに検証する。

なぜ必要か
----------
region_contrast_eval.py は、正常画像との差分で部位を特定できるかを
**集計した分布**（所見ごとの argmax 部位の分布が対照と違うか）で判定する。
この判定には穴がある。

  病変の実際の分布が部位間でほぼ一様なら、**局在が完璧に働いていても**
  argmax の分布は対照と変わらない。分布比較では両者を区別できない。

正しい検証は「**この画像の**差分最大部位が、**この画像の**病変部位と一致するか」を
画像ごとに突き合わせることである。それには画像ごとの部位レベル正解が要る。

NIH の localization サブセット（`BBox_List_2017.csv`、984 box / 880 画像 / 8 疾患）が
それを与える。取得元:
  https://raw.githubusercontent.com/TRKuan/cxr8/master/dataset/BBox_List_2017.csv
座標は 1024x1024 の PNG 画素系。**検証専用であり学習には使わない。**

測ること
--------
1. **分割の妥当性**: box が 5 部位投影のどれに入るか。複数部位へ大きく跨るなら
   「1 病変 = 1 部位」という前提自体が成り立たず、どの手法を使っても上限になる。
2. **対比手法の画像ごとの正答率**: 同一患者の正常画像との差分が最大だった部位が、
   正解の部位と一致する割合。偶然（20%）および面積事前分布と比較する。

使い方
------
    # まず正解を取得する（リポジトリには含めない）
    mkdir -p tmp/gold && curl -sfL -o tmp/gold/BBox_List_2017.csv \\
      https://raw.githubusercontent.com/TRKuan/cxr8/master/dataset/BBox_List_2017.csv

    # 分割の妥当性だけ（ベクトル不要）
    python scripts/region_gold_eval.py

    # 対比手法の正答率も測る（region_signal_eval.py のキャッシュを使う）
    python scripts/region_gold_eval.py --encoder deterministic,service,service-fine
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in ("packages/common/src", "packages/image-pipeline/src"):
    sys.path.insert(0, str(ROOT / _path))

from image_pipeline import ExtractionError, extract, load_profile  # noqa: E402
from nihcc_rag import evaluation  # noqa: E402
from nihcc_rag.regions import REGION_CODES  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rse = _load("_region_signal_eval", ROOT / "scripts" / "region_signal_eval.py")

#: 正解ラベルとデータセット語彙の対応（bbox 側は表記が一部違う）
GOLD_LABEL = {"Infiltrate": "Infiltration"}


def load_gold(path: Path, available: set[str]) -> list[dict]:
    """bbox を読む。座標は 1024x1024 の PNG 画素系。"""
    out = []
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row in reader:
            if not row or not row[0].strip() or row[0] not in available:
                continue
            out.append(
                {
                    "name": row[0],
                    "finding": GOLD_LABEL.get(row[1].strip(), row[1].strip()),
                    "box": (float(row[2]), float(row[3]), float(row[4]), float(row[5])),
                }
            )
    return out


def region_masks(image_path: Path, profile: str) -> dict[str, np.ndarray] | None:
    """原画像座標系での 5 部位 mask。

    pipeline は作業解像度（短辺 1024）で mask を返し、`metrics['scale']` に倍率を持つ。
    region.mask は region.bbox の内側だけを切り出したものなので、全画面へ戻す。
    """
    try:
        result = extract(image_path.read_bytes(), profile_name=profile, regions=tuple(REGION_CODES))
    except ExtractionError:
        return None
    except Exception:  # noqa: BLE001
        return None

    work_h, work_w = result.normalized.shape[:2]
    scale = result.metrics.get("scale", 1.0) or 1.0
    out: dict[str, np.ndarray] = {}
    for code, region in result.regions.items():
        if region.mask is None:
            continue
        canvas = np.zeros((work_h, work_w), dtype=bool)
        x0, y0, x1, y1 = region.bbox
        canvas[y0:y1, x0:x1] = region.mask > 0
        out[code] = (canvas, scale)
    return out or None


def box_overlaps(masks, box: tuple[float, float, float, float]) -> dict[str, float]:
    """box 面積のうち各部位 mask と重なる割合。合計は 1 を超えない（部位は排他）。"""
    shares: dict[str, float] = {}
    for code, (mask, scale) in masks.items():
        x, y, w, h = box
        # 原画像座標 -> 作業座標
        x0, y0 = int(round(x * scale)), int(round(y * scale))
        x1, y1 = int(round((x + w) * scale)), int(round((y + h) * scale))
        x0, y0 = max(0, x0), max(0, y0)
        x1 = min(mask.shape[1], x1)
        y1 = min(mask.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            shares[code] = 0.0
            continue
        area = (x1 - x0) * (y1 - y0)
        shares[code] = float(mask[y0:y1, x0:x1].sum()) / area if area else 0.0
    return shares


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--gold", default="tmp/gold/BBox_List_2017.csv")
    parser.add_argument("--encoder", default="", help="カンマ区切り。省略すると分割の妥当性のみ")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--images", default="")
    parser.add_argument("--profile", default=rse.DEFAULT_PROFILE)
    parser.add_argument(
        "--min-share", type=float, default=0.30, help="正解部位と認める最小の重なり割合"
    )
    parser.add_argument("--json", default="")
    parser.add_argument("--note", default="", help="DB に残す覚書")
    args = parser.parse_args()

    data_dir = rse.resolve_data_dir(args.data_dir)
    csv_path = rse.resolve_input(args.csv, data_dir, "Data_Entry_2017_v2020.csv", "CSV")
    image_dir = rse.resolve_input(args.images, data_dir, "images", "images")
    gold_path = ROOT / args.gold
    if not gold_path.exists():
        parser.error(f"正解ファイルがありません: {gold_path}\n  docstring の curl で取得してください")

    samples = rse.load_samples(csv_path, image_dir, None)
    by_name = {s.name: s for s in samples}
    by_patient: dict[str, list] = defaultdict(list)
    for s in samples:
        by_patient[s.patient].append(s)

    gold = load_gold(gold_path, set(by_name))
    print(f"正解 box {len(gold)} 件 / {len({g['name'] for g in gold})} 画像", flush=True)

    # --- 1. 分割の妥当性 --------------------------------------------------
    resolved: list[dict] = []
    failed = 0
    for entry in gold:
        masks = region_masks(image_dir / entry["name"], args.profile)
        if masks is None:
            failed += 1
            continue
        shares = box_overlaps(masks, entry["box"])
        top = max(shares, key=shares.get)
        ranked = sorted(shares.values(), reverse=True)
        resolved.append(
            {
                **entry,
                "shares": shares,
                "gold_region": top,
                "top_share": ranked[0],
                "second_share": ranked[1] if len(ranked) > 1 else 0.0,
                "lung_share": float(sum(shares.values())),
            }
        )

    print(f"ROI 抽出に失敗: {failed} 件", flush=True)
    usable = [r for r in resolved if r["top_share"] >= args.min_share]
    print()
    print("=== 1. 分割の妥当性 ===")
    print(f"box が肺野へ入る割合（中央値）    : {np.median([r['lung_share'] for r in resolved]):.1%}")
    print(f"最大部位への重なり（中央値）      : {np.median([r['top_share'] for r in resolved]):.1%}")
    print(f"2 位部位への重なり（中央値）      : {np.median([r['second_share'] for r in resolved]):.1%}")
    print(f"最大重なりが {args.min_share:.0%} 以上   : {len(usable)}/{len(resolved)} "
          f"({len(usable)/max(1,len(resolved)):.0%})  <- 「1 病変 = 1 部位」が成立する割合")
    print()
    print("正解部位の分布（この偏りが局在の目標値）")
    dist = Counter(r["gold_region"] for r in usable)
    for code in REGION_CODES:
        print(f"  {code}  {dist.get(code,0):>4}  {dist.get(code,0)/max(1,len(usable)):>6.1%}")

    report = {
        "boxes": len(gold),
        "resolved": len(resolved),
        "usable": len(usable),
        "min_share": args.min_share,
        "gold_region_share": {c: dist.get(c, 0) / max(1, len(usable)) for c in REGION_CODES},
        "encoders": {},
    }

    # 画像ごとの正解部位をキャッシュする。ROI 抽出をやり直さずに
    # 他のスクリプト（MIL の評価など）から使えるようにする。
    cache = gold_path.parent / "gold_regions.json"
    cache.write_text(
        json.dumps(
            {
                "min_share": args.min_share,
                "profile": args.profile,
                "items": [
                    {
                        "name": r["name"],
                        "finding": r["finding"],
                        "gold_region": r["gold_region"],
                        "top_share": round(r["top_share"], 4),
                        "lung_share": round(r["lung_share"], 4),
                        "usable": r["top_share"] >= args.min_share,
                    }
                    for r in resolved
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"正解部位をキャッシュしました: {cache}（{len(resolved)} 件）")

    # --- 2. 対比手法の画像ごとの正答率 ------------------------------------
    names = [n.strip() for n in args.encoder.split(",") if n.strip()]
    if not names:
        print("\n（--encoder 未指定のため対比手法の評価は行いません）")
    for name in names:
        cache = rse._cache_path(name, args, len(samples))
        if not cache.exists():
            print(f"\n[{name}] キャッシュがありません: {cache}")
            continue
        vectors = rse._load_cache(cache)
        index = {c: {n: i for i, n in enumerate(ns)} for c, (ns, _) in vectors.items()}
        vecs = {c: v for c, (_, v) in vectors.items()}

        hits = 0
        total = 0
        confusion = Counter()
        for r in usable:
            sample = by_name[r["name"]]
            normals = [s for s in by_patient[sample.patient] if s.normal]
            if not normals:
                continue
            base = normals[0]
            d = {}
            for code in REGION_CODES:
                i = index.get(code, {}).get(base.name)
                j = index.get(code, {}).get(sample.name)
                if i is None or j is None:
                    continue
                d[code] = 1.0 - float(vecs[code][i] @ vecs[code][j])
            if len(d) < 3:
                continue
            pred = max(d, key=d.get)
            total += 1
            hits += pred == r["gold_region"]
            confusion[(r["gold_region"], pred)] += 1

        # 偶然の水準: 正解部位の分布に従って当てずっぽうに選んだ場合
        prior = report["gold_region_share"]
        chance = sum(p * p for p in prior.values())
        acc = hits / total if total else float("nan")
        print()
        print(f"=== 2. 対比手法の正答率 [{name}] ===")
        print(f"  評価対象           : {total} 画像（同一患者の正常画像あり）")
        print(f"  正答率             : {acc:.1%}")
        print(f"  一様に選んだ場合   : 20.0%")
        print(f"  正解分布に従う場合 : {chance:.1%}")
        report["encoders"][name] = {"evaluated": total, "accuracy": acc, "chance": chance}

    evaluation.persist(
        run,
        kind="gold-box",
        target_version=load_profile(args.profile).roi_model_version,
        params={"profile": args.profile, "min_share": args.min_share,
                "encoders": names},
        sample_size=len(usable),
        note=args.note or None,
    )
    if not run.persisted:
        print("（DB へ記録できなかった。評価そのものは成功している）")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
