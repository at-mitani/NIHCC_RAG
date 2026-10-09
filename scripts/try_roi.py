"""ROI 抽出パイプラインを実データで手早く確認する開発用スクリプト。

    python scripts/try_roi.py <画像ディレクトリ> [枚数] [出力ディレクトリ]

コンテナを起動せずに抽出品質と処理時間を確認する。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "image-pipeline" / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from image_pipeline import ExtractionError, extract  # noqa: E402

COLOURS = {
    "RUL": (255, 128, 0),
    "RML": (0, 200, 255),
    "RLL": (0, 128, 255),
    "LUL": (128, 255, 0),
    "LLL": (255, 0, 200),
}


def render(result, path: Path) -> None:
    canvas = cv2.cvtColor(result.normalized, cv2.COLOR_GRAY2BGR)
    for code, region in result.regions.items():
        x0, y0, x1, y1 = region.bbox
        overlay = np.zeros_like(canvas)
        overlay[y0:y1, x0:x1][region.mask > 0] = COLOURS[code]
        canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.45, 0)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), COLOURS[code], 2)
        cv2.putText(canvas, f"{code} {region.confidence:.2f}", (x0 + 4, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOURS[code], 2, cv2.LINE_AA)
    for y in range(0, canvas.shape[0], 4):
        cv2.circle(canvas, (int(result.spine_x[y]), y), 1, (255, 255, 255), -1)
    cv2.imwrite(str(path), canvas)


def main() -> int:
    image_dir = Path(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "tmp" / "roi-preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(image_dir.rglob("*.png"))[:limit]
    ok = failed = 0
    statuses: dict[str, int] = {}
    confidences: list[float] = []
    elapsed: list[float] = []

    for path in files:
        data = path.read_bytes()
        started = time.perf_counter()
        try:
            result = extract(data)
        except ExtractionError as exc:
            failed += 1
            print(f"{path.name}: FAILED {exc.code} {exc}")
            continue
        elapsed.append(time.perf_counter() - started)
        ok += 1
        statuses[result.overall_status] = statuses.get(result.overall_status, 0) + 1
        confidences.extend(r.confidence for r in result.regions.values())
        render(result, out_dir / f"{path.stem}_preview.png")
        print(
            f"{path.name}: {result.overall_status} regions={len(result.regions)} "
            f"conf={ {c: round(r.confidence, 2) for c, r in result.regions.items()} } "
            f"flags={result.quality.flags} {elapsed[-1]*1000:.0f}ms"
        )

    print("\n--- summary ---")
    print(json.dumps({
        "images": len(files),
        "ok": ok,
        "failed": failed,
        "statuses": statuses,
        "mean_confidence": round(float(np.mean(confidences)), 3) if confidences else None,
        "p50_ms": round(float(np.percentile(elapsed, 50)) * 1000, 1) if elapsed else None,
        "p95_ms": round(float(np.percentile(elapsed, 95)) * 1000, 1) if elapsed else None,
        "preview_dir": str(out_dir),
    }, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
