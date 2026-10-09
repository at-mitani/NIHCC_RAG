"""凍結 backbone 上の MIL ヘッド（region_mil_stage_a.py）が出す部位別所見プロファイルを、
検索用ベクトルとして書き出す。

目的
----
部位レベルの正答率とは別に、**本当の目的である検索品質が改善するか**を
`region_signal_eval.py` の検索ハーネスで測る。

やること
--------
1. region_mil_stage_a.py と同じ手順で MIL ヘッドを学習する
2. 全画像の部位別所見プロファイル（5 部位 x 14 所見の確率）を推論する
3. それを検索用ベクトルとして `region_signal_eval.py` のキャッシュ形式で書き出す

書き出したあとは既存のハーネスがそのまま使える:

    python scripts/region_mil_profile.py --encoder service-fine --whiten
    python scripts/region_signal_eval.py \\
        --encoder deterministic,service,service-fine,mil-profile

リーク防止
----------
検索ハーネスは患者の 20% をクエリ側へ回す。**MIL がその患者で学習していると、
クエリ側の部位が既知になり数字が不当に良くなる。** ハーネスと同じ分割手順で
クエリ患者を先に特定し、MIL の学習対象から完全に除外する。

索引側の患者で学習するのは問題ない。実運用でも「既存コーパスで学習したモデルで
索引を作り、新しい画像を検索する」形になるため、その設定を再現している。

ベクトルの作り方
----------------
所見確率 p（14 次元）をそのまま使うと、正常な部位はほぼ零ベクトルになり cosine が
退化する（正常どうしの類似度が定義できない）。**「正常」を明示的な次元として持たせる。**

    v = normalize([ p_1, ..., p_14, kappa * (1 - max_f p_f) ])

これで正常部位どうしは最後の次元が揃って高い類似度になり、気胸の部位どうしは
p_pneumothorax が揃って高くなる。利用者の要求（病変のない部位では所見なしが、
病変のある部位では同じ所見が上位に来る）を直接表現している。

`--output hybrid` では、外観埋め込みと連結したベクトルも書き出す。実運用の
2 段検索（1 段目で外観、2 段目でプロファイル一致）に近い形の比較ができる。
"""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in ("packages/common/src", "packages/image-pipeline/src"):
    sys.path.insert(0, str(ROOT / _path))

from nihcc_rag import mil  # noqa: E402
from nihcc_rag.regions import REGION_CODES  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rse = _load("_region_signal_eval", ROOT / "scripts" / "region_signal_eval.py")
stage_a = _load("_region_mil_stage_a", ROOT / "scripts" / "region_mil_stage_a.py")


def query_patients(samples, seed: int, query_fraction: float) -> set[str]:
    """`region_signal_eval.run_tests` と同一手順でクエリ患者を再現する。

    手順が食い違うとリーク防止にならないため、実装を写して同じ seed を使う。
    """
    patients = sorted({s.patient for s in samples})
    rng = random.Random(seed)
    rng.shuffle(patients)
    n_query = max(1, int(len(patients) * query_fraction))
    return set(patients[:n_query])


def split_excluding(names, by_name, held_out: set[str], seed: int, val_fraction: float):
    """held_out（検索クエリ患者）を除いた中で train / val を作る。

    test は作らない。ここでの目的は検索用ベクトルの生成であり、
    部位レベルの評価は region_mil_stage_a.py が受け持つ。
    """
    usable = [i for i, n in enumerate(names) if by_name[n].patient not in held_out]
    patients = sorted({by_name[names[i]].patient for i in usable})
    rng = random.Random(seed + 1)
    rng.shuffle(patients)
    n_val = max(1, int(len(patients) * val_fraction))
    val_patients = set(patients[:n_val])
    idx = {"train": [], "val": [], "test": []}
    for i in usable:
        key = "val" if by_name[names[i]].patient in val_patients else "train"
        idx[key].append(i)
    idx["test"] = idx["val"]  # region_mil_stage_a.train() が参照するため形を合わせる
    return {k: np.array(v, dtype=int) for k, v in idx.items()}


def profile_vectors(probs: np.ndarray, kappa: float) -> np.ndarray:
    """所見確率 -> 検索用ベクトル。本番推論と同じ実装を使う。"""
    return mil.profile_vectors(probs, kappa)


def write_cache(path: Path, names: list[str], present: np.ndarray, vectors: np.ndarray) -> None:
    """`region_signal_eval.py` が読めるキャッシュ形式で書き出す。"""
    payload: dict[str, np.ndarray] = {}
    for r, code in enumerate(REGION_CODES):
        live = np.nonzero(present[:, r] > 0)[0]
        if not len(live):
            continue
        payload[f"{code}_names"] = np.array([names[i] for i in live], dtype=object)
        payload[f"{code}_vecs"] = vectors[live, r].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--encoder", default="service-fine", help="入力に使う部位ベクトル")
    parser.add_argument("--output-name", default="mil-profile", help="書き出す encoder 名")
    parser.add_argument(
        "--output", default="profile", choices=["profile", "hybrid", "broadcast"],
        help="profile=所見プロファイルのみ / hybrid=外観埋め込みと連結 / "
             "broadcast=画像レベル確率を全部位へ複製した**対照**",
    )
    parser.add_argument("--hybrid-weight", type=float, default=0.5,
                        help="hybrid でのプロファイル側の重み（0〜1）")
    parser.add_argument("--kappa", type=float, default=1.0, help="「正常らしさ」次元の重み")
    parser.add_argument("--query-fraction", type=float, default=0.2,
                        help="検索ハーネスと同じ値にすること（リーク防止）")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--csv", default="")
    parser.add_argument("--images", default="")
    parser.add_argument("--profile", default=rse.DEFAULT_PROFILE)
    # --- MIL ヘッドの学習設定（region_mil_stage_a.py と揃える）---
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
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--whiten", action="store_true")
    parser.add_argument("--seed", type=int, default=20260919,
                        help="検索ハーネスの既定 seed と一致させること")
    parser.add_argument(
        "--export-artifact", default="",
        help="学習した MIL ヘッドと前処理統計を npz へ書き出す（本番推論で使う）",
    )
    args = parser.parse_args()

    data_dir = rse.resolve_data_dir(args.data_dir)
    csv_path = rse.resolve_input(args.csv, data_dir, "Data_Entry_2017_v2020.csv", "CSV")
    image_dir = rse.resolve_input(args.images, data_dir, "images", "images")
    samples = rse.load_samples(csv_path, image_dir, None)
    by_name = {s.name: s for s in samples}

    cache = rse._cache_path(args.encoder, args, len(samples))
    if not cache.exists():
        parser.error(f"入力ベクトルがありません: {cache}")
    print(f"画像 {len(samples)} 枚 / 患者 {len({s.patient for s in samples})} 人", flush=True)
    print(f"入力: {cache.name}", flush=True)

    vectors = rse._load_cache(cache)
    x, present, y, normal, names = stage_a.build_arrays(samples, vectors)
    appearance = x.copy() if args.output == "hybrid" else None

    held = query_patients(samples, args.seed, args.query_fraction)
    idx = split_excluding(names, by_name, held, args.seed, args.val_fraction)
    print(f"検索クエリ患者 {len(held)} 人を学習から除外", flush=True)
    print(f"学習 {len(idx['train'])} 枚 / 検証 {len(idx['val'])} 枚", flush=True)

    x_raw = x
    if args.whiten:
        x = stage_a.whiten_regions(x, present, idx["train"])
        print("部位ごとに特徴を標準化しました（train 統計のみ使用）", flush=True)

    head = stage_a.train(x, present, y, normal, idx, args)

    if args.export_artifact:
        # 本番の推論は成果物 1 つで完結させる。学習時と同じ前処理統計を同梱しないと、
        # 標準化の有無で部位レベルの正答率が大きく変わる（34.1% と 51.0%）。
        mean, std = (
            mil.region_stats(x_raw, present, idx["train"])
            if args.whiten
            else (np.zeros((len(REGION_CODES), x.shape[2])), np.ones((len(REGION_CODES), x.shape[2])))
        )
        artifact = mil.MilArtifact(
            weights=head.W,
            bias=head.b,
            whiten_mean=mean,
            whiten_std=std,
            regions=tuple(REGION_CODES),
            findings=tuple(stage_a.FINDINGS),
            kappa=args.kappa,
            tau=args.tau,
            backbone_version=args.encoder,
            whitened=bool(args.whiten),
        )
        digest = artifact.save(Path(args.export_artifact))
        print("")
        print(f"成果物を書き出しました: {args.export_artifact}")
        print(f"  sha256   : {digest}")
        print(f"  出力次元 : {artifact.dimension}  -> EMBEDDING_FINE_DIMENSION={artifact.dimension}")
        print(f"  backbone : {args.encoder}（学習時と同じものを本番でも使うこと）")

    # --- 全画像のプロファイルを推論する ---------------------------------
    probs = np.zeros((x.shape[0], x.shape[1], y.shape[1]), dtype=np.float32)
    for start in range(0, x.shape[0], args.eval_batch):
        sel = slice(start, start + args.eval_batch)
        probs[sel] = head.region_probs(x[sel].astype(np.float64)).astype(np.float32)

    if args.output == "broadcast":
        # **部位の情報を意図的に捨てた対照。** 画像レベルの確率（MIL の bag 出力）を
        # 5 部位すべてへ同じ値で配る。
        #
        # Test B のヒット判定は画像単位のラベル（region_signal_eval.py の
        # `by_name[i_names[c]].labels`）なので、「この画像に所見がある」だけを
        # 符号化したベクトルでも高い lift が出うる。この対照が実モデルに匹敵する
        # なら、Test B は部位レベルの良し悪しを測れていないことになる。
        bag = mil.sigmoid(head.bag_logits(x.astype(np.float64), present))
        probs = np.repeat(bag[:, None, :], x.shape[1], axis=1).astype(np.float32)
        print("画像レベル確率を全部位へ複製しました（部位情報なしの対照）", flush=True)

    out = profile_vectors(probs.astype(np.float64), args.kappa)
    covered = int((present > 0).sum())
    print(f"\nプロファイル推論: {out.shape[0]} 枚 x {out.shape[2]} 次元 "
          f"（部位ベクトル {covered} 本）", flush=True)
    print("  平均の所見確率（部位あたり）: "
          + "  ".join(
              f"{f[:12]} {probs[present > 0][:, i].mean():.3f}"
              for i, f in enumerate(stage_a.FINDINGS[:4])
          ), flush=True)

    name = args.output_name
    if args.output == "broadcast":
        name = f"{name}-broadcast"
    if args.output == "hybrid":
        # 外観と所見プロファイルを重み付きで連結する。どちらも L2 正規化済みなので、
        # 重みがそのまま cosine への寄与比になる。
        app = appearance / np.maximum(
            np.linalg.norm(appearance, axis=2, keepdims=True), 1e-9
        )
        w = args.hybrid_weight
        out = np.concatenate([(1 - w) * app, w * out], axis=2)
        norm = np.linalg.norm(out, axis=2, keepdims=True)
        norm[norm < 1e-9] = 1.0
        out = out / norm
        name = f"{name}-hybrid"
        print(f"外観と連結しました（プロファイル重み {w}） -> {out.shape[2]} 次元", flush=True)

    path = rse._cache_path(name, args, len(samples))
    write_cache(path, names, present, out)
    print(f"\n書き出しました: {path}")
    print(f"\n次に検索品質を測定します:")
    print(f"  python scripts/region_signal_eval.py \\")
    print(f"      --encoder deterministic,service,service-fine,{name}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
