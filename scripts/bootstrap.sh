#!/usr/bin/env bash
# 初期セットアップ（設計書 11.3）: secret 生成と .env 作成。
# 秘密情報は Git に置かず、Docker secret としてのみ配布する。
set -euo pipefail

cd "$(dirname "$0")/.."

# secret はディレクトリ側で保護する（0700）。ファイルを 0600 にすると、
# Docker Compose がホストの uid / mode のまま bind mount するため、非 root で動く
# コンテナ（postgres uid 999 / app uid 10001）から読めず起動に失敗する。
mkdir -p secrets
chmod 700 secrets 2>/dev/null || true
touch secrets/.gitkeep

random() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -base64 33 | tr -d '\n=+/' | cut -c1-40
  else
    python -c "import secrets;print(secrets.token_urlsafe(32)[:40])"
  fi
}

for name in db_password app_db_password rabbitmq_password jwt_secret grafana_password; do
  file="secrets/${name}.txt"
  if [ -s "$file" ]; then
    echo "keep    ${file}"
  else
    printf '%s' "$(random)" > "$file"
    echo "created ${file}"
  fi
  # 既存ファイルにも適用する（0600 で作られた古い checkout の修復を兼ねる）
  chmod 644 "$file" 2>/dev/null || true
done

if [ ! -f .env ]; then
  cp .env.example .env
  # ローカル IdP の初期パスワードもランダム化する
  admin="$(random)"; operator="$(random)"; viewer="$(random)"
  python - "$admin" "$operator" "$viewer" <<'PY'
import pathlib, sys
admin, operator, viewer = sys.argv[1:4]
path = pathlib.Path(".env")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "LOCAL_USERS=admin:admin-change-me:admin,operator:operator-change-me:operator,viewer:viewer-change-me:viewer",
    f"LOCAL_USERS=admin:{admin}:admin,operator:{operator}:operator,viewer:{viewer}:viewer",
)
if "COMPOSE_FILE=" not in text:
    text += (
        "\n# 部位別 Citus クラスタの compose を併せて読み込む\n"
        "COMPOSE_FILE=docker-compose.yml:infra/compose/docker-compose.vectors.yml\n"
        "COMPOSE_PATH_SEPARATOR=:\n"
    )
path.write_text(text, encoding="utf-8")
PY
  echo "created .env（初期ユーザーのパスワードは .env の LOCAL_USERS を参照）"
  grep '^LOCAL_USERS=' .env
else
  echo "keep    .env"
fi

echo
echo "次の手順:"
echo "  1) .env の NIHCC_DATA_DIR がデータセットの場所を指しているか確認する"
echo "  2) docker compose build"
echo "  3) docker compose up -d"
echo "  4) docker compose run --rm db-migrate"
