"""CheXmask から、ローカル corpus 分の肺野 mask だけを取り込む。

    # 該当する範囲を探す（画像 ID の分布を見る）
    python scripts/fetch_reference_masks.py --probe

    # 範囲を指定して取り込む（手元の 4,999 枚はこの範囲に全部入る）
    python scripts/fetch_reference_masks.py --from-mb 1480 --to-mb 1600

    # 全件を取り込む（EC2 の約 11 万枚。帯域が出る環境で使う）
    python scripts/fetch_reference_masks.py --all

ROI の良否を目視ではなく数値で測るための土台である。

出典: CheXmask Database v1.0.0 (PhysioNet, CC BY 4.0)
  Gaggion et al., "CheXmask: a large-scale dataset of anatomical segmentation
  masks for multi-center chest x-ray images", Scientific Data (2024)

**これは専門家の手描き mask ではない。** HybridGNet という模型が作り、Reverse
Classification Accuracy（RCA）で品質を推定したものである。ただし

  - **我々と同じ ChestX-ray8 の画像**に付いているので領域差が無い
  - 画像ごとに品質スコア（Dice RCA）があるので、信頼できる分だけを選べる

ので、目視 5 枚よりはるかに良い基準になる。手描きの正解が要るなら Montgomery
（138 枚）を別途当たる（NLM の配布先は落ちていることがある）。

## なぜ範囲を指定するのか

配布元の CSV は全 112k 枚ぶんで 2.2 GB ある。PhysioNet からの実効速度は 1 本あたり
70 KB/秒ほどで、全部流すと 9 時間かかる。

幸い、**この CSV は画像 ID が近いものが大きな塊で並んでいる。** ローカル corpus は
`00000001`〜`00001335` という狭い ID 帯なので、該当する塊（実測で 1490〜1595 MB
付近の約 105 MB）だけを取れば足りる。`--probe` でその位置を調べられる。

範囲の先頭と末尾の行は途中で切れているので捨てる。

EC2 のように帯域が出る環境では `--all` で全件を流す。こちらは範囲を気にしなくて
よい代わりに 2.2 GB を読み切る。手元には corpus 分しか残らないのは同じ。
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    "https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/"
    "OriginalResolution/ChestX-Ray8.csv"
)
DEST = ROOT / "data" / "reference-masks" / "chestxray8.csv"
HEADER = ["Image Index", "Dice RCA (Mean)", "Dice RCA (Max)", "Landmarks",
          "Left Lung", "Right Lung", "Heart", "Height", "Width"]
MB = 1 << 20
_NAME = re.compile(r"\b(\d{8})_\d{3}\.png")

csv.field_size_limit(1 << 30)


def fetch_range(source: str, start: int, end: int, retries: int = 3) -> bytes:
    """[start, end] バイトを取ってくる。"""
    request = urllib.request.Request(source, headers={"Range": f"bytes={start}-{end}"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.read()
        except Exception:                                  # noqa: BLE001
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    return b""


def probe(source: str, step_mb: int, span_mb: int) -> None:
    """file 全体を走査して、どこにどの画像 ID があるかを示す。"""
    print("位置        その範囲にある画像 ID")
    for mb in range(0, span_mb, step_mb):
        chunk = fetch_range(source, mb * MB, mb * MB + 80_000).decode("utf-8", "replace")
        ids = sorted(int(m) for m in _NAME.findall(chunk))
        span = f"{ids[0]:08d}-{ids[-1]:08d}" if ids else "(不明)"
        print(f"{mb:6d} MB  {span}")


def corpus_names() -> set[str]:
    return {p.name for p in (ROOT / "NIHCC" / "images").glob("*.png")}


def fetch_all(args, wanted: set[str]) -> int:
    """全件を流しながら corpus 分だけ書き出す。"""
    print(f"全件（約 2.2 GB）を流す。{args.dest} へ書き出す")
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    began = time.time()
    seen = kept = 0
    with urllib.request.urlopen(args.source, timeout=300) as response:
        stream = io.TextIOWrapper(response, encoding="utf-8", newline="")
        reader = csv.reader(stream)
        next(reader)                                   # ヘッダ
        with args.dest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for row in reader:
                seen += 1
                if len(row) == len(HEADER) and row[0] in wanted:
                    writer.writerow(row)
                    kept += 1
                if seen % 10000 == 0:
                    elapsed = time.time() - began
                    print(f"  {seen} 行 / {kept} 件採取（{seen / max(elapsed, 1):.0f} 行/秒、"
                          f"{elapsed / 60:.1f} 分経過）", flush=True)
    print(f"\n{kept} / {len(wanted)} 枚ぶんを保存（{args.dest.stat().st_size / MB:.0f} MB、"
          f"{(time.time() - began) / 60:.1f} 分）")
    if kept < len(wanted):
        print(f"注意: {len(wanted) - kept} 枚は CheXmask に無い")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--dest", type=Path, default=DEST)
    parser.add_argument("--probe", action="store_true", help="ID の分布だけ調べる")
    parser.add_argument("--all", action="store_true",
                        help="範囲を使わず全件（2.2 GB）を流す。帯域が出る環境向け")
    parser.add_argument("--probe-step-mb", type=int, default=100)
    parser.add_argument("--probe-span-mb", type=int, default=2100)
    parser.add_argument("--from-mb", type=int, default=1480)
    parser.add_argument("--to-mb", type=int, default=1600)
    parser.add_argument("--jobs", type=int, default=6, help="並列数（1 本 70 KB/秒ほど）")
    args = parser.parse_args()

    if args.probe:
        probe(args.source, args.probe_step_mb, args.probe_span_mb)
        return 0

    wanted = corpus_names()
    if not wanted:
        print("NIHCC/images に画像がない", file=sys.stderr)
        return 1
    print(f"ローカル corpus: {len(wanted)} 枚")

    if args.all:
        return fetch_all(args, wanted)

    print(f"{args.from_mb}-{args.to_mb} MB を {args.jobs} 本並列で取得する")

    start, end = args.from_mb * MB, args.to_mb * MB
    size = end - start
    bounds = [(start + i * size // args.jobs,
               start + (i + 1) * size // args.jobs - 1) for i in range(args.jobs)]
    began = time.time()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        chunks = list(pool.map(lambda b: fetch_range(args.source, *b), bounds))
    blob = b"".join(chunks)
    elapsed = time.time() - began
    print(f"  {len(blob) / MB:.0f} MB 受信（{elapsed:.0f} 秒、{len(blob) / MB / max(elapsed, 1):.2f} MB/秒）")

    text = blob.decode("utf-8", "replace")
    # 先頭と末尾の行は途中で切れている
    body = text.split("\n", 1)[1].rsplit("\n", 1)[0] if "\n" in text else ""
    rows = [r for r in csv.reader(io.StringIO(body)) if len(r) == len(HEADER)]
    print(f"  完全な行 {len(rows)} 件")

    keep = [r for r in rows if r[0] in wanted]
    args.dest.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, list[str]] = {}
    if args.dest.exists():
        with args.dest.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            existing = {r[0]: r for r in reader if len(r) == len(HEADER)}
        print(f"  既存 {len(existing)} 件に追記する")
    for row in keep:
        existing[row[0]] = row

    with args.dest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for name in sorted(existing):
            writer.writerow(existing[name])

    print(f"\n今回 {len(keep)} 件、累計 {len(existing)} / {len(wanted)} 枚ぶんを "
          f"{args.dest} に保存（{args.dest.stat().st_size / MB:.0f} MB）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
