#!/usr/bin/env bash
# 胸部野（肺野＋縦隔＋横隔膜）を 1 単位にした評価を、被覆率 -> 目視一覧 ->
# 全量のベクトル -> 学習・評価まで通しで実行する。
# スクリプトの一覧と使い方は docs/OPERATIONS.md。
#
#   nohup bash scripts/run_chest_eval.sh > tmp/chest/run.log 2>&1 &
#   tail -f tmp/chest/run.log
#
# **途中から再開できる。** 被覆率は出力の有無で飛ばし、ベクトル生成は 2,000 枚ごとの
# shard から続きを処理する。落ちたら原因を直して同じコマンドを投げ直せばよい。
set -euo pipefail

cd "$(dirname "$0")/.."

SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:8002}"
PROFILE="${CHEST_PROFILE:-nihcc-unet-local-v1}"
export LUNG_MODEL_PATH="${LUNG_MODEL_PATH:-infra/lung-unet/lung-unet.onnx}"
LIMIT="${LIMIT:-0}"       # 較正は LIMIT=1000
OUT=tmp/chest

step() { printf '\n========== %s ==========\n' "$1"; date '+%Y-%m-%d %H:%M:%S'; }
die()  { printf '\n!! %s\n' "$1" >&2; exit 1; }

mkdir -p "$OUT"

# ---------------------------------------------------------------- 事前確認
# 数時間の工程に入る前に落ちる条件を潰す。
step "0. 事前確認"

# venv が無効なのか、venv にモジュールが足りないのかを区別して伝える。
# 一律に「venv が無効」と出すと、有効にした後も同じ表示が出て原因に辿り着けない。
if [ -z "${VIRTUAL_ENV:-}" ]; then
  die "venv が有効になっていません。同じシェルで先に: source ~/.venv-eval/bin/activate
   （nohup で起動するプロセスは、起動したシェルの環境を引き継ぐ）"
fi
if ! IMPORT_ERR="$(python -c 'import numpy, httpx, cv2, yaml, PIL' 2>&1)"; then
  die "venv（${VIRTUAL_ENV}）に必要なモジュールがありません:
   $(printf '%s\n' "$IMPORT_ERR" | tail -1)
   docs/runbooks/chest-classifier.md §1.1 の pip install を実行してください"
fi
python -c 'import onnxruntime' 2>/dev/null \
  || die "onnxruntime がありません: pip install onnxruntime"
[ -f "$LUNG_MODEL_PATH" ] \
  || die "U-Net のモデルがありません: ${LUNG_MODEL_PATH}（docs/runbooks/chest-classifier.md §1.2）"
[ -f tmp/gold/BBox_List_2017.csv ] \
  || die "正解 box がありません: tmp/gold/BBox_List_2017.csv（docs/runbooks/chest-classifier.md §1.2）"

# 胸部野の mask 名（CHEST）を受け付ける embedding-service か。image が古いと
# 契約が CHEST を許さず、全量の最初の要求で 422 になる。
PROBE="$(python - "$SERVICE_URL" <<'PY'
import base64, io, json, sys, urllib.request
from PIL import Image
buf = io.BytesIO(); Image.new("L", (64, 64), 255).save(buf, format="PNG")
png = base64.b64encode(buf.getvalue()).decode()
body = json.dumps({"images_base64": [png], "masks_base64": [{"CHEST": png}]}).encode()
req = urllib.request.Request(sys.argv[1] + "/api/v1/embeddings/region-pool", data=body,
                             headers={"Content-Type": "application/json"})
try:
    urllib.request.urlopen(req, timeout=120)
    print("OK")
except urllib.error.HTTPError as exc:
    print(f"HTTP {exc.code}")
except Exception as exc:
    print("ERROR " + type(exc).__name__)
PY
)"
case "$PROBE" in
  OK) echo "  embedding-service: 胸部野の pooling に対応" ;;
  "HTTP 422") die "embedding-service が古い image です（CHEST を受け付けない）。
   このブランチで docker compose build embedding-service し直してください（docs/runbooks/chest-classifier.md §2）" ;;
  *) die "embedding-service へ届きません ($PROBE)。docs/runbooks/chest-classifier.md §2 の構成を確認してください" ;;
esac

AVAIL_GB="$(df -P . | awk 'NR==2{printf "%d", $4/1048576}')"
echo "  空き容量: ${AVAIL_GB:-?} GB（キャッシュと shard で約 3 GB 使う）"
if [ -n "${AVAIL_GB:-}" ] && [ "$AVAIL_GB" -lt 8 ]; then
  die "空き容量が足りません（${AVAIL_GB} GB）"
fi

LIMIT_ARGS=()
SUFFIX=""
if [ "$LIMIT" != "0" ]; then
  LIMIT_ARGS=(--limit "$LIMIT")
  SUFFIX="-${LIMIT}"
  echo "  較正: 無作為 ${LIMIT} 枚"
fi

# ---------------------------------------------------------------- 1
step "1. 正解 box の被覆率（肺野 / 胸部野）"
if [ -f "${OUT}/coverage.json" ]; then
  echo "  済: ${OUT}/coverage.json"
else
  python scripts/chest_image_eval.py coverage --profile "$PROFILE" --json "${OUT}/coverage.json"
fi

# ---------------------------------------------------------------- 2
step "2. 無作為抽出の目視一覧"
python scripts/chest_image_eval.py sheet --profile "$PROFILE" --count 64 \
  --output "${OUT}/sheet.png"

# ---------------------------------------------------------------- 3
step "3. ベクトル生成（全量は数時間。再開可能）"
python scripts/chest_image_eval.py vectors --profile "$PROFILE" \
  --service-url "$SERVICE_URL" "${LIMIT_ARGS[@]}"

# ---------------------------------------------------------------- 4
step "4. 教師データ・学習・評価"
python scripts/chest_image_eval.py train --profile "$PROFILE" "${LIMIT_ARGS[@]}" \
  --json "${OUT}/train${SUFFIX}.json"

printf '\n結果ファイル（手元の tmp/chest/ へ戻してください）:\n'
for f in coverage.json "train${SUFFIX}.json" sheet.png; do
  if [ -f "${OUT}/${f}" ]; then printf '  %s\n' "${OUT}/${f}"; fi
done
