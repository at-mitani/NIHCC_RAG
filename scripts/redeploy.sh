#!/usr/bin/env bash
# 現在の作業ツリーの内容を、動作中のコンテナへ反映する。
# ローカルでも EC2 でも同じ手順で使える。
#
#   bash scripts/redeploy.sh            # 手元の HEAD を反映
#   bash scripts/redeploy.sh --pull     # git pull してから反映
#   bash scripts/redeploy.sh --yes      # 確認を省く（自動実行用）
#
# **全イメージを再ビルドする。** packages/common（契約・部位定義・MIL）が全ての
# python イメージへ COPY される作りなので、1 つだけ直しても他が古いままになる。
# db-migrate は migrate プロファイルに属し、引数なしの build では対象外なので
# 個別にビルドする。
set -euo pipefail

cd "$(dirname "$0")/.."

PULL=0; ASSUME_YES=0; FORCE=0
for arg in "$@"; do
  case "$arg" in
    --pull) PULL=1 ;;
    --yes|-y) ASSUME_YES=1 ;;
    --force) FORCE=1 ;;
    *) echo "不明な引数: $arg" >&2; exit 2 ;;
  esac
done

step() { printf '\n========== %s ==========\n' "$1"; }
die()  { printf '\n!! %s\n' "$1" >&2; exit 1; }

# ------------------------------------------------------------------ 事前確認
step "0. 事前確認"

if [ -n "$(git status --porcelain)" ]; then
  echo "  注意: 作業ツリーに未コミットの変更があります"
  git status --short | sed 's/^/    /'
fi

# **評価の実行中は再ビルドしない。** embedding-service を作り直すと、進行中の
# 推論要求が切れる。全量の評価は数時間かかり、やり直しになる。
# 実際に起動しているものだけを拾う。ファイル名だけで照合すると、エディタで
# 開いているだけの状態にも反応して再反映を止めてしまう。
EVAL_PATTERN='python[0-9.]* .*(region_signal_eval|region_mil_[a-z_]*)\.py'
if command -v pgrep >/dev/null 2>&1; then
  RUNNING="$(pgrep -fa "$EVAL_PATTERN" 2>/dev/null || true)"
elif ps -ef >/dev/null 2>&1; then
  RUNNING="$(ps -ef 2>/dev/null | grep -E "$EVAL_PATTERN" | grep -v grep || true)"
else
  # 検出できない環境（Git Bash など）では黙って素通りさせず、その旨を伝える。
  RUNNING=""
  echo "  注意: 実行中プロセスを確認できません（pgrep / ps が使えない）。"
  echo "        評価を回していないことを手元で確かめてください"
fi
if [ -n "$RUNNING" ] && [ "$FORCE" -eq 0 ]; then
  echo "$RUNNING" | sed 's/^/    /'
  die "評価スクリプトが実行中です。終了を待つか、承知のうえで --force を付けてください"
fi

if [ "$PULL" -eq 1 ]; then
  step "1. git pull"
  git pull --ff-only origin main
fi
echo "  HEAD: $(git log --oneline -1)"

if [ "$ASSUME_YES" -eq 0 ]; then
  printf '\n全イメージを再ビルドしてコンテナを入れ替えます。15〜30 分かかります。続けますか? [y/N] '
  # 対話端末が無い環境（nohup / CI / 一部のシェル）では /dev/tty を開けない。
  # 黙って続行すると意図しない再構築になるので、--yes を案内して止める。
  answer=""
  # [ -r /dev/tty ] は通っても open に失敗する環境がある。実際に開いて確かめる。
  # **stdin へは読みにいかない。** 端末でないときの read は EOF が来るまで
  # 待ち続けることがあり、nohup や CI では無言で固まる。
  if ! { : < /dev/tty; } 2>/dev/null; then
    die "対話端末がありません。--yes を付けて実行してください"
  fi
  answer=""
  read -r answer < /dev/tty || answer=""
  case "$answer" in
    y|Y|yes) ;;
    *) die "中止しました" ;;
  esac
fi

# ------------------------------------------------------------------ 取り込み停止
step "2. 取り込みの停止"
# 取り込み中にイメージを入れ替えると、処理中の item が失敗する。
# 手順 5 の `docker compose up -d` が既定プロファイル全体を起動し直すため、
# ここで止めておけば「ビルドと migration の間だけ止まっている」状態になる。
docker compose stop ingest-worker
echo "  停止しました（手順 5 の up -d で自動的に戻ります）"

# ------------------------------------------------------------------ ビルド
step "3. ビルド"
docker compose build db-migrate
docker compose build

# ------------------------------------------------------------------ migration
step "4. スキーマ適用"
if ! docker compose run --rm db-migrate; then
  cat >&2 <<'MSG'

!! migration が失敗しました。

   "already applied with a different checksum" で止まった場合は
   docs/OPERATIONS.md §3.6 を参照してください。SQL が変わっていないことを
   確かめたうえで、記録だけを 1 回更新できます:

     REPAIR='<版名>.sql'
     docker compose run --rm -e MIGRATION_REPAIR_CHECKSUMS="$REPAIR" db-migrate

MSG
  exit 1
fi

# ------------------------------------------------------------------ 起動
step "5. 起動と health 待ち"
# ポートの解放待ちで 1 つのサービスが bind に失敗すると、そこで up が止まり
# **残りが古いイメージのまま動き続ける**。Docker Desktop の wslrelay が前の
# コンテナのポートを握ったままになることがあり、少し待つと通る。
if ! docker compose up -d; then
  echo "  up に失敗しました。30 秒待って再試行します"
  sleep 30
  docker compose up -d
fi

wait_healthy() {
  # embedding-service はモデルのロードに 1〜2 分かかる。
  local deadline=$(( $(date +%s) + 600 )) bad
  while :; do
    bad="$(docker compose ps -a --format '{{.Service}} {{.State}} {{.Health}}' \
           | awk '$2 != "running" || ($3 != "" && $3 != "healthy")' || true)"
    # set -e の下で `[ -z x ] && break` と書くと、偽のときに異常終了扱いになる
    if [ -z "$bad" ]; then break; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "$bad" | sed 's/^/    /'
      die "10 分待っても健全になりませんでした。docker compose logs で確認してください"
    fi
    sleep 10
  done
  echo "  全サービスが healthy です"
}

wait_healthy

# **健全性だけでは足りない。** 古いイメージのままでも health check は通るため、
# 取り残しに気づけない（古いイメージのまま何日も動き続けうる）。
# 稼働中のイメージ ID が、いまビルドしたものと一致するかを確かめる。
STALE=""
for s in frontend backend-api ingest-worker search-worker search-aggregator roi-service embedding-service; do
  cid="$(docker compose ps -q "$s" 2>/dev/null | head -1)"
  if [ -z "$cid" ]; then continue; fi
  running="$(docker inspect "$cid" --format "{{.Image}}")"
  latest="$(docker image inspect "local/nihcc-${s}:1.0.0" --format "{{.Id}}" 2>/dev/null || true)"
  if [ -n "$latest" ] && [ "$running" != "$latest" ]; then
    STALE="$STALE $s"
  fi
done
if [ -n "$STALE" ]; then
  echo "  古いイメージで動いているサービス:$STALE"
  echo "  作り直します"
  # 同じタグだと compose が差分なしと判断することがあるため強制する
  docker compose up -d --force-recreate --no-deps $STALE
  wait_healthy
fi
echo "  全サービスが最新のイメージです"

# ------------------------------------------------------------------ 確認
step "6. 確認"
docker compose exec -T backend-api python -c "
import json, urllib.request
d = json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/ready', timeout=10))
print('  status:', d['status'])
for k, v in d.get('checks', {}).items():
    print(f'    {k}: {v}')
raise SystemExit(0 if d['status'] == 'ok' else 1)
"

ING="$(docker compose ps --format '{{.Service}} {{.State}}' | grep '^ingest-worker' | head -1)"
echo "  ingest-worker: ${ING:-（起動していません。docker compose up -d で確認してください）}"

cat <<'MSG'

残りの確認（任意）:
  SMOKE_PASSWORD='<admin のパスワード>' python3 tests/integration/smoke_test.py
MSG
