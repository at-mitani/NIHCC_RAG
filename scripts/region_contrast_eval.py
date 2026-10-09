"""同一患者の正常画像との対比で、所見のある部位を特定できるかを測る。

着想
----
「所見なしの画像と結節などの画像を比較し、差がある部位を取れば教師データになる」
という発想の検証。部位レベルの正解が無い問題を、**正常との差分**で埋められるかを問う。

素朴な形（他人の正常画像と比べる）は効かない見込みが高い。別患者どうしでは体格・
性別・体位・撮影条件の差が病変よりはるかに大きく、region_signal_eval.py の
Test C（部位ベクトルの 70〜96% が部位の輪郭で決まる）と Test A（残差は nuisance）が
それを裏付けている。

そこで**同一患者内**で対比する。ChestX-ray14 は 1 患者平均 3.64 枚の追跡撮影を含み、
「所見なしの回」と「所見ありの回」の両方を持つ患者が多数いる（気胸なら 1,298 人 /
4,943 枚）。同一患者どうしなら解剖学的な個人差がほぼ相殺され、残る差が病変に
近づく、という仮説である。

検定
----
ペアは同一患者内の 2 枚 (N, M)。N は必ず `No Finding`。

  対照   : M も `No Finding`          … 差は撮影条件・経時変化のみ
  陽性   : M に所見 F がある           … 差は上記 + 病変

部位ごとに cosine 距離 d_r = 1 - cos(N_r, M_r) を取り、画像対のスコアを

  max_r d_r          … 「最も変化した部位」の大きさ
  max_r d_r - median … 変化が 1 部位へ集中しているか（局在性）

として、陽性ペアと対照ペアを分離できるか **AUC** で測る。AUC 0.5 は無情報。

さらに**他人の正常画像と比べる場合**（cross）も同じ形で測り、同一患者内（within）が
どれだけ有利かを示す。

局在の妥当性
------------
部位レベルの正解が無いため、`argmax_r d_r` の分布が**解剖学的な事前知識**と
一致するかで見る。たとえば胸水は下葉、心拡大は内側〜下方に出やすい。
分布が一様（各部位 20%）なら、差分は病変を指していない。

使い方
------
    # 帰無（医学的意味を持たない stub）。まずこれを確定させる
    python scripts/region_contrast_eval.py --encoder deterministic

    # 実モデル（region_signal_eval.py のベクトルキャッシュを共有する）
    python scripts/region_contrast_eval.py --encoder deterministic,service,service-fine

ベクトルは scripts/region_signal_eval.py が作るキャッシュをそのまま使う。
先に region_signal_eval.py を実行しておくこと（キャッシュが無ければエラーにする）。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in ("packages/common/src", "packages/image-pipeline/src"):
    sys.path.insert(0, str(ROOT / _path))

from nihcc_rag.regions import REGION_CODES  # noqa: E402


def _load_signal_eval():
    """region_signal_eval.py を読み込む。Sample / キャッシュ規約を共有するため。"""
    spec = importlib.util.spec_from_file_location(
        "_region_signal_eval", ROOT / "scripts" / "region_signal_eval.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_region_signal_eval"] = module
    spec.loader.exec_module(module)
    return module


rse = _load_signal_eval()


# --------------------------------------------------------------------------
# ペアの構成
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """同一患者内の 2 枚。base は必ず `No Finding`。"""

    patient: str
    base: str
    other: str
    findings: frozenset[str]  # other 側の所見（対照なら空）

    @property
    def control(self) -> bool:
        return not self.findings


def build_pairs(samples, rng: random.Random, max_per_patient: int) -> list[Pair]:
    """同一患者内の (正常, 他) ペアを作る。

    1 人が大量の撮影を持つと結果を支配するため、患者ごとに上限を設ける。
    """
    by_patient: dict[str, list] = defaultdict(list)
    for s in samples:
        by_patient[s.patient].append(s)

    pairs: list[Pair] = []
    for patient, items in by_patient.items():
        normals = [s for s in items if s.normal]
        if not normals:
            continue
        others = [s for s in items if s is not normals[0]]
        picked: list[Pair] = []
        for other in others:
            base = normals[0] if other is not normals[0] else None
            if base is None or other.name == base.name:
                continue
            picked.append(
                Pair(
                    patient=patient,
                    base=base.name,
                    other=other.name,
                    findings=frozenset(f for f in other.labels if f != "No Finding"),
                )
            )
        rng.shuffle(picked)
        pairs.extend(picked[:max_per_patient])
    return pairs


# --------------------------------------------------------------------------
# スコア
# --------------------------------------------------------------------------


def region_distances(
    pair: Pair, index: dict[str, dict[str, int]], vectors: dict[str, np.ndarray]
) -> dict[str, float] | None:
    """部位ごとの cosine 距離。両方に存在する部位のみ。"""
    out: dict[str, float] = {}
    for code in REGION_CODES:
        i = index[code].get(pair.base)
        j = index[code].get(pair.other)
        if i is None or j is None:
            continue
        out[code] = 1.0 - float(vectors[code][i] @ vectors[code][j])
    return out if len(out) >= 3 else None


def region_baseline(controls) -> dict[str, tuple[float, float]]:
    """対照ペア（正常→正常）から、部位ごとの距離の平均と標準偏差を求める。

    部位によって「撮影ごとの揺れ幅」が大きく違う。横隔膜の位置で下葉が動きやすい、
    encoder によっては上葉の表現が不安定、といった差である。生の距離で argmax を
    取ると**所見に関係なく同じ部位が選ばれる**（実測: BiomedCLIP はどの所見でも
    RUL が 32〜38%、RAD-DINO は 49〜58%、stub は LLL が 31〜36%）。

    対照ペアは定義上「病変の変化を含まない」ので、その分布が各部位の素の揺れ幅に
    なる。これで標準化すれば、部位固有の偏りを取り除ける。
    """
    values: dict[str, list[float]] = {c: [] for c in REGION_CODES}
    for _pair, d in controls:
        for code, value in d.items():
            values[code].append(value)
    out: dict[str, tuple[float, float]] = {}
    for code, vals in values.items():
        if len(vals) < 2:
            out[code] = (0.0, 1.0)
            continue
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        out[code] = (mu, sd if sd > 1e-9 else 1.0)
    return out


def standardise(d: dict[str, float], baseline: dict[str, tuple[float, float]]) -> dict[str, float]:
    """部位固有の揺れ幅で正規化した z スコア。"""
    return {code: (value - baseline[code][0]) / baseline[code][1] for code, value in d.items()}


def auc(positive: list[float], negative: list[float]) -> float:
    """Mann-Whitney U による AUC。0.5 が無情報。"""
    if not positive or not negative:
        return float("nan")
    values = np.concatenate([np.asarray(positive), np.asarray(negative)])
    order = values.argsort()
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1)
    # 同順位の平均化
    _, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    n_pos = len(positive)
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * len(negative)))


def bootstrap_auc(
    pos: list[tuple[float, str]], neg: list[tuple[float, str]], iterations: int, seed: int
) -> tuple[float, float]:
    """患者単位の cluster bootstrap による AUC の 95% 区間。"""
    if not pos or not neg:
        return (float("nan"), float("nan"))
    pos_by: dict[str, list[float]] = defaultdict(list)
    neg_by: dict[str, list[float]] = defaultdict(list)
    for value, patient in pos:
        pos_by[patient].append(value)
    for value, patient in neg:
        neg_by[patient].append(value)
    pos_keys, neg_keys = list(pos_by), list(neg_by)
    rng = random.Random(seed)
    values = []
    for _ in range(iterations):
        p = [v for _ in pos_keys for v in pos_by[pos_keys[rng.randrange(len(pos_keys))]]]
        n = [v for _ in neg_keys for v in neg_by[neg_keys[rng.randrange(len(neg_keys))]]]
        values.append(auc(p, n))
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return (float("nan"), float("nan"))
    values.sort()
    return (values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))])


# --------------------------------------------------------------------------


def evaluate(encoder: str, samples, vectors_by_code, args) -> dict:
    by_name = {s.name: s for s in samples}
    index = {code: {n: i for i, n in enumerate(names)} for code, (names, _) in vectors_by_code.items()}
    vectors = {code: vecs for code, (_, vecs) in vectors_by_code.items()}
    rng = random.Random(args.seed)
    pairs = build_pairs(samples, rng, args.max_pairs_per_patient)

    # --- within-patient のスコアを貯める --------------------------------
    scored: list[tuple[Pair, np.ndarray]] = []
    for pair in pairs:
        d = region_distances(pair, index, vectors)
        if d is not None:
            scored.append((pair, d))

    controls = [(p, d) for p, d in scored if p.control]
    report: dict = {
        "encoder": encoder,
        "pairs": len(scored),
        "control_pairs": len(controls),
        "patients": len({p.patient for p, _ in scored}),
        "findings": {},
    }

    # 部位固有の揺れ幅は対照ペアから推定する（病変を含まないため素の揺れになる）
    baseline = region_baseline(controls)
    report["region_baseline"] = {c: baseline[c] for c in REGION_CODES}

    def raw_max(d):
        return max(d.values())

    def z_max(d):
        return max(standardise(d, baseline).values())

    neg_max = [(raw_max(d), p.patient) for p, d in controls]
    neg_zmax = [(z_max(d), p.patient) for p, d in controls]

    counts = Counter(f for p, _ in scored for f in p.findings)
    for finding, _n in counts.most_common(args.findings):
        pos = [(p, d) for p, d in scored if finding in p.findings]
        if len(pos) < args.min_pairs:
            continue
        pos_max = [(raw_max(d), p.patient) for p, d in pos]
        pos_zmax = [(z_max(d), p.patient) for p, d in pos]

        # argmax 部位の分布（局在の妥当性を解剖学的事前知識と照らす）。
        # 生の距離と、部位固有の揺れ幅で標準化した z スコアの両方で見る。
        argmax, argmax_z = Counter(), Counter()
        for p, d in pos:
            argmax[max(d, key=d.get)] += 1
            z = standardise(d, baseline)
            argmax_z[max(z, key=z.get)] += 1
        total = sum(argmax.values()) or 1

        # 対照ペアでも同じ分布を取る。これが「病変が無いときの偏り」であり、
        # 所見ごとの分布がこれと変わらなければ局在していない。
        ctrl_z = Counter()
        for _p, d in controls:
            z = standardise(d, baseline)
            ctrl_z[max(z, key=z.get)] += 1
        ctrl_total = sum(ctrl_z.values()) or 1

        report["findings"][finding] = {
            "pairs": len(pos),
            "patients": len({p.patient for p, _ in pos}),
            "auc_max": auc([v for v, _ in pos_max], [v for v, _ in neg_max]),
            "auc_max_ci": bootstrap_auc(pos_max, neg_max, args.bootstrap, args.seed),
            "auc_zmax": auc([v for v, _ in pos_zmax], [v for v, _ in neg_zmax]),
            "auc_zmax_ci": bootstrap_auc(pos_zmax, neg_zmax, args.bootstrap, args.seed),
            "argmax_share": {c: argmax.get(c, 0) / total for c in REGION_CODES},
            "argmax_share_z": {c: argmax_z.get(c, 0) / total for c in REGION_CODES},
            "argmax_share_z_control": {c: ctrl_z.get(c, 0) / ctrl_total for c in REGION_CODES},
        }
    return report


def render(reports: dict[str, dict]) -> str:
    encs = list(reports)
    out = ["", "=" * 78]
    out.append(" 同一患者内の対比で所見のある部位を特定できるか")
    out.append("=" * 78)
    first = reports[encs[0]]
    out.append(
        f"ペア {first['pairs']}（うち対照 {first['control_pairs']}）/ 患者 {first['patients']}"
    )
    out.append("")
    out.append("AUC: 陽性ペア（正常→所見あり）と対照ペア（正常→正常）の分離。0.5 が無情報")
    out.append("")
    header = f"{'所見':<20}{'ペア':>7}" + "".join(f"{e:>18}" for e in encs)
    out.append(header)
    findings = list(first["findings"])
    for finding in findings:
        cells = []
        n = first["findings"][finding]["pairs"]
        for e in encs:
            r = reports[e]["findings"].get(finding)
            if not r:
                cells.append("-".rjust(18))
                continue
            cells.append(f"{r['auc_max']:.3f} / {r['auc_zmax']:.3f}".rjust(18))
        out.append(f"{finding:<20}{n:>7}" + "".join(cells))
    out.append("")
    out.append("  各欄は「生の距離 / 部位標準化後」。標準化は対照ペアの分布で行う。")

    out.append("")
    out.append("最も変化した部位の分布（部位標準化後）")
    out.append("所見によらず同じ形なら、局在していない。対照行と比べること。")
    out.append("")
    for e in encs:
        out.append(f"  [{e}]")
        out.append("    " + f"{'所見':<20}" + "".join(f"{c:>8}" for c in REGION_CODES))
        control = reports[e]["findings"].get(findings[0], {}).get("argmax_share_z_control")
        if control:
            out.append(
                "    " + f"{'(対照: 正常→正常)':<20}"
                + "".join(f"{control[c]:>7.0%} " for c in REGION_CODES)
            )
        for finding in findings:
            r = reports[e]["findings"].get(finding)
            if not r:
                continue
            share = r["argmax_share_z"]
            line = "".join(f"{share[c]:>7.0%} " for c in REGION_CODES)
            if control:
                worst = max(abs(share[c] - control[c]) for c in REGION_CODES)
                line += f"  最大差 {worst * 100:+.1f}pt"
            out.append("    " + f"{finding:<20}" + line)
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encoder", default="deterministic")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--images", default="")
    parser.add_argument("--profile", default=rse.DEFAULT_PROFILE)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--view", default="")
    parser.add_argument("--findings", type=int, default=8)
    parser.add_argument("--min-pairs", type=int, default=100)
    parser.add_argument("--max-pairs-per-patient", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    data_dir = rse.resolve_data_dir(args.data_dir)
    csv_path = rse.resolve_input(args.csv, data_dir, "Data_Entry_2017_v2020.csv", "CSV")
    image_dir = rse.resolve_input(args.images, data_dir, "images", "images")
    samples = rse.load_samples(csv_path, image_dir, args.view or None)
    if args.limit:
        by_patient: dict[str, list] = defaultdict(list)
        for s in samples:
            by_patient[s.patient].append(s)
        keys = sorted(by_patient)
        random.Random(args.seed).shuffle(keys)
        picked = []
        for key in keys:
            if len(picked) >= args.limit:
                break
            picked.extend(by_patient[key])
        samples = picked[: args.limit]

    print(f"画像 {len(samples)} 枚 / 患者 {len({s.patient for s in samples})} 人", flush=True)
    print(f"データ: {image_dir}", flush=True)

    reports: dict[str, dict] = {}
    for name in [n.strip() for n in args.encoder.split(",") if n.strip()]:
        cache = rse._cache_path(name, args, len(samples))
        if not cache.exists():
            parser.error(
                f"ベクトルキャッシュがありません: {cache}\n"
                "  先に scripts/region_signal_eval.py を同じ条件で実行してください"
            )
        print(f"キャッシュを使用 [{name}]: {cache.relative_to(ROOT)}", flush=True)
        reports[name] = evaluate(name, samples, rse._load_cache(cache), args)

    print(render(reports))
    if args.json:
        Path(args.json).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"JSON を書き出しました: {args.json}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
