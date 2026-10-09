#!/usr/bin/env bash
# PostgreSQL のメジャーバージョンアップ。
#
#   bash scripts/upgrade_postgres.sh dump      # 1. 今のデータを退避する
#   bash scripts/upgrade_postgres.sh recreate  # 2. 新しい版で作り直す（**データが消える**）
#   bash scripts/upgrade_postgres.sh restore   # 3. 退避したデータを戻す
#   bash scripts/upgrade_postgres.sh verify    # 行数だけを数え直す
#
# **メジャーバージョンアップはデータファイルの互換性が無い。** PG16 のボリュームを
# PG17 のコンテナに読ませることはできず、dump と restore が要る。
#
# metadata とベクトル用クラスタで扱いが違う。
#
#   metadata            スキーマごと dump して、そのまま戻す
#   ベクトル用クラスタ  **データだけ** dump する。スキーマは migration が作り直す
#                       （`create_distributed_table` を通さないと分散されないため）
#
# ベクトル用クラスタは 1 つ（vector-coordinator + vector-worker-1..5）。
#
# 退避先は `tmp/pg-upgrade/`。**git の対象外**なので、終わったら消してよい。
set -euo pipefail

cd "$(dirname "$0")/.."

DUMP_DIR="${DUMP_DIR:-tmp/pg-upgrade}"
VECTOR_SERVICE=vector-coordinator
VECTOR_DB=vector
VECTOR_TABLE=image_embedding

step() { printf '\n========== %s ==========\n' "$1"; }
die()  { printf '\n!! %s\n' "$1" >&2; exit 1; }

password() {
  [ -f secrets/db_password.txt ] || die "secrets/db_password.txt がありません"
  cat secrets/db_password.txt
}

cmd_dump() {
  local pw; pw="$(password)"
  mkdir -p "$DUMP_DIR"

  step "metadata を退避する（スキーマ + データ）"
  docker compose exec -T metadata-db sh -c \
    "PGPASSWORD='$pw' pg_dump -U postgres -d metadata --no-owner --no-privileges" \
    > "$DUMP_DIR/metadata.sql"
  printf '  %s (%s)\n' "$DUMP_DIR/metadata.sql" "$(du -h "$DUMP_DIR/metadata.sql" | cut -f1)"

  step "ベクトル用クラスタを退避する（データのみ）"
  # **スキーマは dump しない。** 分散テーブルは create_distributed_table を通す
  # 必要があり、それは migration が行う。ここでは中身だけを運ぶ。
  docker compose exec -T "$VECTOR_SERVICE" sh -c \
    "PGPASSWORD='$pw' pg_dump -U postgres -d $VECTOR_DB \
       --data-only --no-owner --table=$VECTOR_TABLE" \
    > "$DUMP_DIR/$VECTOR_DB.sql"
  printf '  %s (%s)\n' "$DUMP_DIR/$VECTOR_DB.sql" "$(du -h "$DUMP_DIR/$VECTOR_DB.sql" | cut -f1)"

  step "行数を控える（戻したあとの突き合わせに使う）"
  {
    printf 'metadata.source_image\t'
    docker compose exec -T metadata-db sh -c \
      "PGPASSWORD='$pw' psql -U postgres -d metadata -tAc 'SELECT count(*) FROM source_image'"
    printf '%s.%s\t' "$VECTOR_DB" "$VECTOR_TABLE"
    docker compose exec -T "$VECTOR_SERVICE" sh -c \
      "PGPASSWORD='$pw' psql -U postgres -d $VECTOR_DB -tAc \
        'SELECT count(*) FROM $VECTOR_TABLE'"
  } | tr -d '\r' > "$DUMP_DIR/counts.txt"
  cat "$DUMP_DIR/counts.txt" | sed 's/^/  /'

  printf '\n退避しました。次は recreate です。\n'
}

cmd_recreate() {
  [ -f "$DUMP_DIR/metadata.sql" ] || die "先に dump を実行してください"

  step "確認"
  printf '**ボリュームを削除します。退避していないデータは失われます。**\n'
  printf '退避先: %s\n' "$DUMP_DIR"
  if [ "${ASSUME_YES:-0}" != "1" ]; then
    printf '続けますか? [y/N] '
    read -r answer < /dev/tty || answer=""
    case "$answer" in y|Y|yes) ;; *) die "中止しました" ;; esac
  fi

  step "停止してボリュームを消す"
  docker compose down
  # DB のボリュームだけを消す。モデル cache・監視のデータは残す。
  #
  # **名前は compose の定義から引く。** 手で書いた正規表現だと取りこぼし、旧版の
  # ボリュームが残ったまま新版が起動して `database files are incompatible` で落ちる。
  # compose 自身の規則で project 名を決める（環境変数 > .env > ディレクトリ名）。
  local project pattern env_project
  env_project="$(sed -n 's/^COMPOSE_PROJECT_NAME=//p' .env 2>/dev/null | tr -d '\r' | head -1)"
  project="${COMPOSE_PROJECT_NAME:-$env_project}"
  [ -n "$project" ] || project="$(basename "$PWD" | tr 'A-Z' 'a-z' | tr -cd 'a-z0-9_-')"
  [ -n "$project" ] || die "compose の project 名を決められません"
  pattern="^${project}_(metadata-data|vector-(coordinator|worker-[0-9])-data)$"

  docker volume ls -q | grep -E "$pattern" | while read -r v; do
    printf '  削除: %s\n' "$v"
    docker volume rm "$v" >/dev/null
  done

  # 取りこぼしが無いことを確かめる。残っていると次の起動で必ず落ちる。
  local left
  left="$(docker volume ls -q | grep -E "$pattern" || true)"
  [ -z "$left" ] || { echo "$left" | sed 's/^/    /'; die "消えていないボリュームがあります"; }
  printf '  DB のボリュームをすべて削除しました\n'

  step "新しい版でビルドする"
  docker compose build db-migrate
  docker compose build

  step "起動する"
  docker compose up -d
  printf '  health を待ちます（数分）\n'
  local deadline=$(( $(date +%s) + 600 ))
  while :; do
    local bad
    bad="$(docker compose ps -a --format '{{.Service}} {{.State}} {{.Health}}' \
           | awk '$2 != "running" || ($3 != "" && $3 != "healthy")' || true)"
    [ -z "$bad" ] && break
    [ "$(date +%s)" -ge "$deadline" ] && { echo "$bad" | sed 's/^/    /'; die "健全になりませんでした"; }
    sleep 10
  done
  printf '  全サービスが healthy です\n'

  step "スキーマを作る"
  docker compose run --rm db-migrate

  printf '\n作り直しました。次は restore です。\n'
}

#: 復元のあいだ止めるサービス。**DB に繋ぎっぱなしのものを止めないと
#: `database is being accessed by other users` で DROP できない。** 止めた
#: ついでに、復元の途中で取り込みが走って中途半端な状態に書き込むのも防げる。
APP_SERVICES=(backend-api ingest-worker search-worker search-aggregator frontend)

#: 退避時の行数と、いま DB に入っている行数を突き合わせる。
#: 復元のあとに呼ぶほか、`verify` で単独でも実行できる。**復元は数時間
#: かかるので、数え直しだけを再実行できる必要がある。**
verify_counts() {
  local pw; pw="$(password)"
  [ -f "$DUMP_DIR/counts.txt" ] || die "counts.txt がありません: $DUMP_DIR"

  step "行数を突き合わせる"
  # **`counts.txt` を先に読み切る。** ループの中の `docker compose exec -T` は
  # 標準入力を読むので、`done < counts.txt` にすると 2 行目以降を食われ、
  # 1 行だけ検査して終わったように見える。
  local ok=1 line
  local -a rows=()
  while IFS= read -r line; do
    [ -n "$line" ] && rows+=("$line")
  done < "$DUMP_DIR/counts.txt"
  [ "${#rows[@]}" -gt 0 ] || die "counts.txt が空です"

  for line in "${rows[@]}"; do
    local target expected db table actual
    target="${line%%$'\t'*}"; expected="${line#*$'\t'}"
    db="${target%%.*}"; table="${target#*.}"
    local service="$VECTOR_SERVICE"
    [ "$db" = "metadata" ] && service=metadata-db
    actual="$(docker compose exec -T "$service" sh -c \
      "PGPASSWORD='$pw' psql -U postgres -d $db -tAc 'SELECT count(*) FROM $table'" \
      < /dev/null | tr -d '\r')"
    if [ "$actual" = "$expected" ]; then
      printf '  %-34s %s 件  一致\n' "$target" "$actual"
    else
      printf '  %-34s 退避 %s / 復元 %s  **食い違い**\n' "$target" "$expected" "$actual"
      ok=0
    fi
  done
  [ "$ok" = "1" ] || die "行数が合いません。復元を確認してください"
  printf '  %d 件すべて一致\n' "${#rows[@]}"

  step "版を確認する"
  docker compose exec -T metadata-db sh -c \
    "PGPASSWORD='$pw' psql -U postgres -d metadata -tAc 'SHOW server_version'" | sed 's/^/  metadata: /'
  docker compose exec -T "$VECTOR_SERVICE" sh -c \
    "PGPASSWORD='$pw' psql -U postgres -d $VECTOR_DB -tAc \
      \"SELECT 'PG ' || current_setting('server_version') ||
               ' / citus ' || (SELECT extversion FROM pg_extension WHERE extname='citus') ||
               ' / pgvector ' || (SELECT extversion FROM pg_extension WHERE extname='vector')\"" \
    | sed "s/^/  $VECTOR_DB: /"

}

cmd_restore() {
  local pw; pw="$(password)"
  [ -f "$DUMP_DIR/metadata.sql" ] || die "退避したファイルがありません: $DUMP_DIR"

  step "アプリを止める（DB の接続を切る）"
  docker compose stop "${APP_SERVICES[@]}"
  # 止めたら必ず戻す。途中で失敗しても落としっぱなしにしない。
  trap 'printf "\n-- アプリを戻します --\n"; docker compose start "${APP_SERVICES[@]}" >/dev/null 2>&1 || true' EXIT

  step "metadata を戻す"
  # migration が作ったスキーマを捨てて、退避したものをそのまま入れる。
  # **dump にはスキーマも schema_migration の記録も入っている。**
  # `WITH (FORCE)` は残った接続を切る。サービスを止めても評価用の psql などが
  # 繋がっていることがあり、1 本残るだけで DROP が通らない。
  docker compose exec -T metadata-db sh -c \
    "PGPASSWORD='$pw' psql -U postgres -d postgres -q -v ON_ERROR_STOP=1 \
       -c 'DROP DATABASE IF EXISTS metadata WITH (FORCE)' \
       -c 'CREATE DATABASE metadata'"
  docker compose exec -T metadata-db sh -c \
    "PGPASSWORD='$pw' psql -U postgres -d metadata -q -v ON_ERROR_STOP=1" \
    < "$DUMP_DIR/metadata.sql"
  printf '  入れました\n'

  step "ベクトル用クラスタを戻す（データのみ）"
  docker compose exec -T "$VECTOR_SERVICE" sh -c \
    "PGPASSWORD='$pw' psql -U postgres -d $VECTOR_DB -q -v ON_ERROR_STOP=1" \
    < "$DUMP_DIR/$VECTOR_DB.sql"
  printf '  入れました\n'

  step "権限を付け直す"
  # **dump は `--no-privileges` で取っている。** 権限は migration（`grant_schema`）
  # が唯一の定義元で、dump に入れると定義が 2 か所になる。metadata を DROP した
  # 時点で `app_meta` への GRANT も消えるので、ここで流し直す。
  #
  # 付け直さないと、起動はするのにログインだけが 500 になる。
  # `psycopg.errors.InsufficientPrivilege: permission denied for table app_user`
  #
  # migration 自体は `schema_migration`（dump に含まれる）を見て飛ばすので、
  # スキーマは触らない。
  docker compose run --rm db-migrate

  verify_counts

  printf '\n戻しました。smoke test で確認してください。\n'
}

case "${1:-}" in
  dump)     cmd_dump ;;
  recreate) cmd_recreate ;;
  restore)  cmd_restore ;;
  verify)   verify_counts ;;
  *) die "使い方: $0 {dump|recreate|restore|verify}" ;;
esac
