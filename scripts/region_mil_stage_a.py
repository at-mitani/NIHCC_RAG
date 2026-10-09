"""凍結 backbone の部位特徴に MIL ヘッドだけを学習する。

`region_signal_eval.py` が作った部位ベクトルキャッシュ（画像あたり 5 部位 x D 次元）を
そのまま入力にする。**GPU も torch も不要**で、全量 112,120 枚でも数分で終わる。

目的は「MIL の枠組みが機能するか」を最小コストで見ることである。凍結特徴は弱い
（region_contrast_eval.py の画像レベルの対比で AUC 0.55〜0.62）ため、高い性能は
期待しない。得たいのは次の 3 つ:

  1. 分割・損失・評価・比較までの全経路が通ること
  2. backbone まで学習する手法の数字を読むための**下限**
  3. 自明解（全部正常と答える）へ落ちる等の失敗の仕方を安く観察すること

評価
----
画像レベル : 14 所見の AUC（患者単位分割の test）
部位レベル : 正解 bbox（`BBox_List_2017.csv`）との一致率

**部位レベルの比較水準は「偶然 20%」ではない。** 正解部位の分布は偏っており
（RLL 36.3% / RML 4.4%）、常に RLL と答えるだけで 36.3% に達する。対比手法は
19.5〜22.7%（region_gold_eval.py）。これらを並べて表示する。

使い方
------
    # 正解部位のキャッシュを先に作る（region_gold_eval.py が書き出す）
    python scripts/region_gold_eval.py

    python scripts/region_mil_stage_a.py --encoder service-fine
    python scripts/region_mil_stage_a.py --encoder deterministic,service,service-fine
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in ("packages/common/src", "packages/image-pipeline/src"):
    sys.path.insert(0, str(ROOT / _path))

from nihcc_rag import mil  # noqa: E402
from nihcc_rag.csvspec import ALLOWED_FINDINGS  # noqa: E402
from nihcc_rag.regions import REGION_CODES  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rse = _load("_region_signal_eval", ROOT / "scripts" / "region_signal_eval.py")

#: 学習・評価の対象となる 14 所見（`No Finding` を除く）
FINDINGS = tuple(f for f in ALLOWED_FINDINGS if f != "No Finding")


def build_arrays(samples, vectors) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """部位ベクトルを [n, r, d] の密行列へ並べ替える。

    ROI 失敗で欠けている部位があるため、存在フラグを別に持つ。
    """
    index = {c: {n: i for i, n in enumerate(ns)} for c, (ns, _) in vectors.items()}
    dim = next(iter(vectors.values()))[1].shape[1]
    names = [s.name for s in samples if any(s.name in index.get(c, {}) for c in REGION_CODES)]
    pos = {n: i for i, n in enumerate(names)}

    x = np.zeros((len(names), len(REGION_CODES), dim), dtype=np.float32)
    present = np.zeros((len(names), len(REGION_CODES)), dtype=np.float64)
    for r, code in enumerate(REGION_CODES):
        if code not in vectors:
            continue
        code_names, mat = vectors[code]
        for i, name in enumerate(code_names):
            j = pos.get(name)
            if j is not None:
                x[j, r] = mat[i]
                present[j, r] = 1.0

    by_name = {s.name: s for s in samples}
    y = np.zeros((len(names), len(FINDINGS)), dtype=np.float64)
    for j, name in enumerate(names):
        labels = by_name[name].labels
        for f, finding in enumerate(FINDINGS):
            if finding in labels:
                y[j, f] = 1.0
    normal = np.array([1.0 if by_name[n].normal else 0.0 for n in names])
    return x, present, y, normal, names


def whiten_regions(x: np.ndarray, present: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
    """部位ごとに特徴を標準化する（train の統計のみ使う）。

    標準化しないと、RAD-DINO の特徴では正解が下葉の所見でも RUL を予測し続ける
    （RUL を 3.6 倍過剰予測、RML は 0 回）。これは対比手法でも見える RUL 偏りと
    同じで、MIL ヘッドではなく **masked crop 埋め込みが持つ部位固有の偏り**である。

    部位ごとに 5 回別々に符号化しているため、部位の輪郭・面積・黒塗りの形が
    そのまま特徴の分布差になる。共有ヘッド z = x . W は、どの部位の特徴が W と
    揃いやすいかに引きずられる。部位ごとの平均と分散を揃えれば、この偏りは
    入力側で取り除ける。

    部位バイアス b[r,f] では足りない。b はロジットの定数オフセットしか動かせず、
    特徴分布そのものの違い（どの方向にどれだけ散らばるか）は補正できない。
    """
    mean, std = mil.region_stats(x, present, train_idx)
    out = x.astype(np.float32, copy=True)
    for r in range(x.shape[1]):
        rows = np.nonzero(present[:, r] > 0)[0]
        if not len(rows):
            continue
        out[rows, r] = mil.apply_whiten(
            x[rows, r], np.zeros(len(rows), dtype=int), mean[r : r + 1], std[r : r + 1]
        ).astype(np.float32)
    return out


def split_by_patient(names, by_name, seed: int, val: float, test: float):
    patients = sorted({by_name[n].patient for n in names})
    random.Random(seed).shuffle(patients)
    n_test = int(len(patients) * test)
    n_val = int(len(patients) * val)
    groups = {
        "test": set(patients[:n_test]),
        "val": set(patients[n_test : n_test + n_val]),
    }
    idx = {"train": [], "val": [], "test": []}
    for i, name in enumerate(names):
        p = by_name[name].patient
        key = "test" if p in groups["test"] else "val" if p in groups["val"] else "train"
        idx[key].append(i)
    return {k: np.array(v, dtype=int) for k, v in idx.items()}


def train(x, present, y, normal, idx, args) -> mil.MilHead:
    head = mil.MilHead(
        dim=x.shape[2],
        n_regions=x.shape[1],
        n_findings=y.shape[1],
        tau=args.tau,
        seed=args.seed,
    )
    opt = mil.Adam(lr=args.lr)

    tr = idx["train"]
    # 所見ごとの正例頻度の逆数で重み付ける（88 倍の開きがある）
    freq = np.maximum(y[tr].mean(axis=0), 1e-6)
    class_weight = (freq.mean() / freq) ** args.class_weight_power
    class_weight = np.clip(class_weight, 0.1, args.class_weight_max)

    rng = np.random.default_rng(args.seed)
    best = (float("inf"), None)
    patience = 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(tr))
        total = 0.0
        for start in range(0, len(tr), args.batch):
            sel = tr[order[start : start + args.batch]]
            loss, dW, db = head.loss_and_grads(
                x[sel].astype(np.float64),
                present[sel],
                y[sel],
                normal[sel],
                class_weight,
                args.neg_weight,
                args.l2,
            )
            opt.step("W", head.W, dW)
            opt.step("b", head.b, db)
            total += loss * len(sel)
        va = idx["val"]
        val_loss = 0.0
        for start in range(0, len(va), args.eval_batch):
            sel = va[start : start + args.eval_batch]
            part, _, _ = head.loss_and_grads(
                x[sel].astype(np.float64), present[sel], y[sel], normal[sel],
                class_weight, args.neg_weight, args.l2,
            )
            val_loss += part * len(sel)
        val_loss /= max(1, len(va))
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}  train {total/len(tr):.4f}  val {val_loss:.4f}", flush=True)
        if val_loss < best[0] - 1e-5:
            best = (val_loss, (head.W.copy(), head.b.copy()))
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f"  早期終了（epoch {epoch}）", flush=True)
                break
    if best[1] is not None:
        head.W, head.b = best[1]
    return head


def evaluate(head, x, present, y, idx, names, by_name, gold, args) -> dict:
    te = idx["test"]
    # 全量では test が 2 万枚を超え、float64 化した一時配列が 1GB 規模になる。
    # 分割して評価する（学習側は batch で処理済み）。
    chunks = []
    for start in range(0, len(te), args.eval_batch):
        sel = te[start : start + args.eval_batch]
        chunks.append(head.bag_logits(x[sel].astype(np.float64), present[sel]))
    bag = np.vstack(chunks) if chunks else np.zeros((0, y.shape[1]))
    aucs = {}
    for f, finding in enumerate(FINDINGS):
        aucs[finding] = mil.roc_auc(bag[:, f], y[te][:, f])
    valid = [v for v in aucs.values() if np.isfinite(v)]
    report = {
        "image_auc": aucs,
        "image_auc_mean": float(np.mean(valid)) if valid else float("nan"),
        "test_images": int(len(te)),
    }

    # --- 部位レベル: 正解 bbox との一致 ---------------------------------
    pos = {n: i for i, n in enumerate(names)}
    finding_index = {f: i for i, f in enumerate(FINDINGS)}
    hits = total = 0
    per_finding = defaultdict(lambda: [0, 0])
    # 所見ごとの「正解部位の分布」と「モデルが予測した部位の分布」。
    # 前者は所見ごとの正しいベースライン（常に最頻部位と答えた場合）を与え、
    # 後者は自明解（どの画像でも同じ部位を出す）に落ちていないかを見るために要る。
    gold_dist = defaultdict(Counter)
    pred_dist = defaultdict(Counter)
    probs_cache: dict[int, np.ndarray] = {}
    for item in gold:
        if not item["usable"]:
            continue
        j = pos.get(item["name"])
        fi = finding_index.get(item["finding"])
        if j is None or fi is None:
            continue
        if j not in probs_cache:
            probs_cache[j] = head.region_probs(x[j : j + 1].astype(np.float64))[0]
        p = probs_cache[j][:, fi].copy()
        p[present[j] == 0] = -np.inf
        pred = REGION_CODES[int(np.argmax(p))]
        total += 1
        ok = pred == item["gold_region"]
        hits += ok
        per_finding[item["finding"]][0] += ok
        per_finding[item["finding"]][1] += 1
        gold_dist[item["finding"]][item["gold_region"]] += 1
        pred_dist[item["finding"]][pred] += 1

    report["gold_evaluated"] = total
    report["gold_accuracy"] = hits / total if total else float("nan")

    # 所見ごとに「常にその所見の最頻部位と答えた場合」を baseline とする。
    # 全体の 36.3%（常に RLL）は所見をまたいだ値なので、所見別の判定には使えない。
    detail = {}
    for finding, (hit, n) in per_finding.items():
        gd = gold_dist[finding]
        best_region, best_count = (gd.most_common(1) or [(None, 0)])[0]
        detail[finding] = {
            "hits": hit,
            "n": n,
            "accuracy": hit / n if n else float("nan"),
            "baseline_region": best_region,
            "baseline": best_count / n if n else float("nan"),
            "gold_dist": dict(gd),
            "pred_dist": dict(pred_dist[finding]),
        }
    report["gold_per_finding"] = detail

    # 全所見をまとめた予測分布。1 部位へ偏っていれば自明解を疑う。
    overall_pred = Counter()
    for c in pred_dist.values():
        overall_pred.update(c)
    report["pred_region_share"] = {
        c: overall_pred.get(c, 0) / max(1, total) for c in REGION_CODES
    }
    # 正解の部位分布と、それに対する予測の比。部位の偏りを見る主判定である。
    # 偏った特徴では RUL を数倍過剰に予測し RML を一度も予測しないことがあるので、
    # share を目で見比べるのではなく比を出す。
    overall_gold = Counter()
    for c in gold_dist.values():
        overall_gold.update(c)
    report["gold_region_share"] = {
        c: overall_gold.get(c, 0) / max(1, total) for c in REGION_CODES
    }
    report["pred_over_gold"] = {
        c: (
            report["pred_region_share"][c] / report["gold_region_share"][c]
            if report["gold_region_share"][c] > 0
            else float("nan")
        )
        for c in REGION_CODES
    }
    # 所見ごとの最頻部位を並べ、全所見で同じなら定数予測である
    report["pred_mode_per_finding"] = {
        f: (pred_dist[f].most_common(1) or [(None, 0)])[0][0] for f in per_finding
    }
    # 所見ごと baseline の加重平均（モデルが超えるべき水準）
    weighted = sum(d["baseline"] * d["n"] for d in detail.values())
    report["gold_baseline_weighted"] = weighted / total if total else float("nan")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encoder", default="service-fine")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--images", default="")
    parser.add_argument("--profile", default=rse.DEFAULT_PROFILE)
    parser.add_argument("--gold", default="tmp/gold/gold_regions.json")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--eval-batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--tau", type=float, default=4.0)
    parser.add_argument("--neg-weight", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--class-weight-max", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--whiten",
        action="store_true",
        help="部位ごとに特徴を標準化する（部位固有の偏りを入力側で除く）",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    data_dir = rse.resolve_data_dir(args.data_dir)
    csv_path = rse.resolve_input(args.csv, data_dir, "Data_Entry_2017_v2020.csv", "CSV")
    image_dir = rse.resolve_input(args.images, data_dir, "images", "images")
    samples = rse.load_samples(csv_path, image_dir, None)
    by_name = {s.name: s for s in samples}

    gold_path = ROOT / args.gold
    gold = []
    if gold_path.exists():
        gold = json.loads(gold_path.read_text(encoding="utf-8"))["items"]
    else:
        print(f"※ 正解部位キャッシュが無いため部位レベル評価は行いません: {gold_path}")

    print(f"画像 {len(samples)} 枚 / 患者 {len({s.patient for s in samples})} 人", flush=True)

    reports = {}
    for name in [n.strip() for n in args.encoder.split(",") if n.strip()]:
        cache = rse._cache_path(name, args, len(samples))
        if not cache.exists():
            print(f"[{name}] ベクトルキャッシュがありません: {cache}")
            continue
        print(f"\n{'=' * 66}\n encoder: {name}\n{'=' * 66}", flush=True)
        vectors = rse._load_cache(cache)
        x, present, y, normal, names = build_arrays(samples, vectors)
        idx = split_by_patient(names, by_name, args.seed, args.val_fraction, args.test_fraction)
        if args.whiten:
            x = whiten_regions(x, present, idx["train"])
            print("部位ごとに特徴を標準化しました（train 統計のみ使用）", flush=True)
        print(f"次元 {x.shape[2]} / 画像 {x.shape[0]} "
              f"(train {len(idx['train'])} / val {len(idx['val'])} / test {len(idx['test'])})",
              flush=True)
        head = train(x, present, y, normal, idx, args)
        report = evaluate(head, x, present, y, idx, names, by_name, gold, args)
        reports[name] = report

        print()
        print(f"画像レベル 平均 AUC : {report['image_auc_mean']:.4f}  (test {report['test_images']} 枚)")
        top = sorted(report["image_auc"].items(), key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else 0))
        for finding, value in top[:5]:
            print(f"    {finding:<20}{value:.3f}")
        if report["gold_evaluated"]:
            print()
            print(f"部位レベル 正答率   : {report['gold_accuracy']:.1%} "
                  f"({report['gold_evaluated']} 画像)")
            print(f"  所見別 baseline の加重平均（常に最頻部位）: "
                  f"{report['gold_baseline_weighted']:.1%}  <- これを超えたか")
            print("  参考: 一様 20.0% / 対比手法 19.5〜22.7%")
            print()
            print(f"  {'所見':<20}{'n':>5}{'正答率':>8}{'baseline':>10}{'最頻':>6}{'予測最頻':>9}")
            for finding, v in sorted(
                report["gold_per_finding"].items(), key=lambda kv: -kv[1]["n"]
            ):
                mark = " *" if v["accuracy"] > v["baseline"] else "  "
                print(f"  {finding:<20}{v['n']:>5}{v['accuracy']:>8.1%}{v['baseline']:>10.1%}"
                      f"{str(v['baseline_region']):>6}"
                      f"{str(report['pred_mode_per_finding'].get(finding)):>9}{mark}")
            print()
            print("  部位分布（正解 / 予測 / 比）。比が 1.0 から離れるほど偏っている")
            print(f"  {'部位':<6}{'正解':>8}{'予測':>8}{'比':>8}")
            for c in REGION_CODES:
                ratio = report["pred_over_gold"][c]
                # 合格線: RUL の比 2.0 未満、RML の比が 0 でない
                mark = ""
                if c == "RUL" and ratio >= 2.0:
                    mark = "  <- 過剰予測（合格線 2.0 未満）"
                if c == "RML" and ratio == 0.0:
                    mark = "  <- 一度も予測していない"
                print(f"  {c:<6}{report['gold_region_share'][c]:>8.1%}"
                      f"{report['pred_region_share'][c]:>8.1%}{ratio:>8.2f}{mark}")
            modes = set(report["pred_mode_per_finding"].values())
            if len(modes) == 1:
                print(f"    !! 全所見で予測最頻部位が同一（{modes.pop()}）= 定数予測の疑い")

    if args.json and reports:
        Path(args.json).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"\nJSON: {args.json}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
