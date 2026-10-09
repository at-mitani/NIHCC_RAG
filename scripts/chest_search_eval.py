"""取り込んだ DB に対して、**実際の検索経路**で検索品質（Test B）を測る。

評価（`chest_image_eval.py train`）は手元の numpy で、胸部野の分類器のプロファイル
だけを測る。本番の検索経路（全件の厳密検索、または BiomedCLIP で候補を集めてから
並べ替える 2 段検索）では、候補の取りこぼしや DB 側の処理で数字が変わりうる。
ここではそれを API 経由で測る。

    python scripts/chest_search_eval.py \\
      --teacher tmp/chest/ec2-full/teacher_nihcc-unet-local-v1_112066.npz \\
      --queries-per-finding 50 --json tmp/chest/search.json

測り方
------
- クエリは**評価で test に回した患者の画像だけ**から取る。分類器は train の患者で
  学習しているので、train の画像で測ると数字が膨らむ
- 単独所見（ラベルが 1 つだけ）の画像を所見ごとに N 枚。`No Finding` も含める
- 1 回のクエリにつき 3 通り:
    profile    所見プロファイルで全件を厳密に検索（strategy=profile。既定）
    two_stage  BiomedCLIP で候補を集めてから並べ替え（strategy=coarse_to_fine）
    stage1     BiomedCLIP だけ（rerank=false）
- 上位から**同じ患者の画像を除いて** K 件を見る（同一患者の別撮影は似ていて当然）
- lift = 上位 K に占める同じ所見の割合 / DB 全体での割合。帰無（無作為）は 1.0

合格線（実行前に固定）:
  - 2 段まで: 14 所見の過半で lift ≥ 1.1
  - 1 段目だけ: 過半の所見で lift > 1.0（無作為より良い）

profile を既定とする判定（実行前に固定）:
  - profile: 14 所見の過半で lift ≥ 1.1
  - profile の lift が two_stage 以上の所見が 14 所見の過半
  - profile のベクトル検索時間（aggregator の timing_ms）の p95 ≤ 500 ms
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from load_initial_images import admin_credentials, request  # noqa: E402

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

NO_FINDING = "No Finding"


def env_value(key: str) -> str | None:
    env = ROOT / ".env"
    if not env.is_file():
        return None
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip() or None
    return None


def patient_of(name: str) -> str:
    """NIH の画像名（00000001_000.png）の先頭 8 桁が患者番号。"""
    return name.split("_", 1)[0]


#: 比べる進め方: 名前 -> (rerank, strategy)
VARIANTS = {
    "profile": (True, "profile"),
    "two_stage": (True, "coarse_to_fine"),
    "stage1_only": (False, "auto"),
}
PROFILE_P95_MS = 500


def search_image(api: str, token: str, image: Path, *, rerank: bool, strategy: str,
                 top_k: int) -> dict:
    """multipart で画像検索を投げる（urllib だけで組み立てる）。"""
    import urllib.error
    import urllib.request

    boundary = uuid.uuid4().hex
    body_json = json.dumps({
        "mode": "IMAGE_TO_IMAGE", "top_k": top_k, "candidate_k": min(500, top_k * 5),
        "min_similarity": -1.0, "rerank": rerank, "strategy": strategy,
        "return": {"thumbnail": False, "roi_overlay": False},
    })
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"request\"\r\n\r\n{body_json}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{image.name}\"\r\n"
        "Content-Type: image/png\r\n\r\n".encode() + image.read_bytes() + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        f"{api}/api/v1/searches", data=b"".join(parts), method="POST",
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        return {"error": error.code, "detail": error.read()[:300].decode(errors="replace")}


def hit(labels: set[str], finding: str) -> bool:
    return (not labels or labels == {NO_FINDING}) if finding == NO_FINDING else finding in labels


def lift_table(rows: list[dict], base: dict[str, float], top_k: int) -> dict[str, dict]:
    """クエリごとの適合率を所見ごとに平均し、DB 全体の割合で割る。"""
    by_finding: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        kept = [r for r in row["results"] if patient_of(r["name"]) != row["patient"]][:top_k]
        if not kept:
            continue
        precision = sum(hit(set(r["labels"]), row["finding"]) for r in kept) / len(kept)
        by_finding[row["finding"]].append(precision)
    out = {}
    for finding, values in sorted(by_finding.items()):
        precision = float(np.mean(values))
        out[finding] = {
            "queries": len(values),
            "base_rate": base[finding],
            "precision": precision,
            "lift": precision / base[finding] if base[finding] else float("nan"),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--teacher", required=True, help="chest_image_eval.py train の teacher_*.npz")
    parser.add_argument("--data-dir", default="", help="既定は .env の NIHCC_DATA_DIR")
    parser.add_argument("--queries-per-finding", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--json", default="tmp/chest/search.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir or env_value("NIHCC_DATA_DIR") or "NIHCC")
    data_dir = data_dir if data_dir.is_absolute() else ROOT / data_dir
    labels_by_name: dict[str, set[str]] = {}
    with (data_dir / "Data_Entry_2017_v2020.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels_by_name[row["Image Index"]] = set(row["Finding Labels"].split("|"))

    teacher = np.load(ROOT / args.teacher, allow_pickle=True)
    findings = [str(f) for f in teacher["findings"]] + [NO_FINDING]
    test_names = [str(n) for n, s in zip(teacher["names"], teacher["split"]) if s == "test"]
    # DB 全体での割合（帰無の適合率）
    total = len(labels_by_name)
    base = {f: sum(hit(ls, f) for ls in labels_by_name.values()) / total for f in findings}

    rng = random.Random(args.seed)
    queries: list[tuple[str, str]] = []
    for finding in findings:
        pool = [n for n in test_names
                if (labels_by_name[n] == {finding} if finding != NO_FINDING
                    else labels_by_name[n] == {NO_FINDING})]
        rng.shuffle(pool)
        queries += [(finding, n) for n in pool[: args.queries_per_finding]]
    print(f"クエリ {len(queries)} 件（test の患者から、所見ごとに最大 {args.queries_per_finding} 件）")

    user, password = admin_credentials(args.user, args.password)
    status, body = request("POST", f"{args.api}/api/v1/auth/token",
                           body={"username": user, "password": password})
    if status != 200:
        raise SystemExit(f"ログインできません: {status} {body}")
    token = body["access_token"]

    records: dict[str, list[dict]] = {name: [] for name in VARIANTS}
    timings: dict[str, list[int]] = {name: [] for name in VARIANTS}
    failures = 0
    started = time.time()
    # 同じ患者を除いても K 件残るよう多めに取る
    fetch = args.top_k * 2
    for i, (finding, name) in enumerate(queries, start=1):
        image = data_dir / "images" / name
        for variant, (rerank, strategy) in VARIANTS.items():
            result = search_image(args.api, token, image, rerank=rerank, strategy=strategy,
                                  top_k=fetch)
            if "error" in result:
                failures += 1
                continue
            used = (result.get("query") or {}).get("strategy")
            expected = "coarse_only" if not rerank else strategy
            if used != expected:
                # サーバが古い（strategy を知らない）か、fine が未構成。比べても意味が無い
                raise SystemExit(f"{variant}: サーバが使った進め方が {used!r}（想定 {expected!r}）")
            timings[variant].append(int(result.get("timing_ms", 0)))
            records[variant].append({
                "finding": finding, "name": name, "patient": patient_of(name),
                "rerank_status": result.get("rerank_status"),
                "results": [{"name": r["source_image_name"], "labels": r["finding_labels"]}
                            for r in result.get("results", [])],
            })
        if i % 50 == 0:
            print(f"  {i}/{len(queries)} 件（{(time.time() - started) / i:.1f} 秒/件）", flush=True)

    report = {"top_k": args.top_k, "queries": len(queries), "failures": failures}
    for variant in VARIANTS:
        report[variant] = lift_table(records[variant], base, args.top_k)
    report["timing_ms"] = {
        variant: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95))}
        for variant, v in timings.items() if v
    }
    print(f"\n{'クエリ':<22}{'件数':>6}{'基準率':>9}{'profile':>10}{'2段まで':>10}{'1段目だけ':>11}")
    for finding in findings:
        row = [report[v].get(finding, {}).get("lift", float("nan")) for v in VARIANTS]
        queries_n = report["profile"].get(finding, {}).get("queries", 0)
        if not queries_n:
            continue
        print(f"{finding:<22}{queries_n:>6}{base[finding]:>9.3f}"
              f"{row[0]:>10.2f}{row[1]:>10.2f}{row[2]:>11.2f}")
    for variant, t in report["timing_ms"].items():
        print(f"ベクトル検索時間 {variant:<12} p50 {t['p50']:.0f} ms  p95 {t['p95']:.0f} ms")

    measured = [f for f in findings[:-1] if f in report["two_stage"]]
    lift = lambda v, f: report[v].get(f, {}).get("lift", 0.0)  # noqa: E731
    passed_two = sum(lift("two_stage", f) >= 1.1 for f in measured)
    passed_one = sum(lift("stage1_only", f) > 1.0 for f in measured)
    passed_profile = sum(lift("profile", f) >= 1.1 for f in measured)
    profile_not_worse = sum(lift("profile", f) >= lift("two_stage", f) for f in measured)
    profile_p95 = report["timing_ms"].get("profile", {}).get("p95", float("inf"))
    majority = lambda n: n * 2 > len(measured)  # noqa: E731
    report["verdict"] = {
        "two_stage_ge_1_1": f"{passed_two}/{len(measured)}",
        "stage1_gt_1_0": f"{passed_one}/{len(measured)}",
        "passed": majority(passed_two) and majority(passed_one),
        "profile_ge_1_1": f"{passed_profile}/{len(measured)}",
        "profile_ge_two_stage": f"{profile_not_worse}/{len(measured)}",
        "profile_p95_ms": profile_p95,
        "profile_adopted": majority(passed_profile) and majority(profile_not_worse)
        and profile_p95 <= PROFILE_P95_MS,
    }
    v = report["verdict"]
    print(f"\n2 段まで lift ≥ 1.1: {v['two_stage_ge_1_1']}  ・  1 段目だけ lift > 1.0: "
          f"{v['stage1_gt_1_0']}  ・  失敗 {failures} 件")
    print("判定（検索品質）: " + ("合格" if v["passed"] else "**不合格**"))
    print(f"profile lift ≥ 1.1: {v['profile_ge_1_1']}  ・  profile ≥ 2 段まで: "
          f"{v['profile_ge_two_stage']}  ・  p95 {profile_p95:.0f} ms（上限 {PROFILE_P95_MS}）")
    print("判定（profile を既定にする）: " + ("採用" if v["profile_adopted"] else
          "**不採用**（IMAGE_SEARCH_STRATEGY=coarse_to_fine に戻す）"))

    path = ROOT / args.json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"保存: {path.relative_to(ROOT)}")
    return 0 if report["verdict"]["passed"] and report["verdict"]["profile_adopted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
