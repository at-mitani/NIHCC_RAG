# NIHCC 胸部X線画像検索ローカルRAGシステム 運用コマンド
# Windows では PowerShell から `bash -lc "make <target>"`、または各コマンドを直接実行する。

SHELL := /bin/bash
COMPOSE := docker compose

.DEFAULT_GOAL := help
.PHONY: help bootstrap build up migrate down clean logs ps status test smoke lint \
        model-fetch reconcile observability

help: ## このヘルプを表示する
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

bootstrap: ## secret と .env を生成する（初回のみ）
	bash scripts/bootstrap.sh

build: ## 全イメージをビルドする
	$(COMPOSE) build

up: ## 全コンテナを起動する（ベクトル DB 6 + アプリ）
	$(COMPOSE) up -d

migrate: ## スキーマ適用と Citus Worker 登録（全ノード起動後に実行）
	$(COMPOSE) run --rm db-migrate

down: ## コンテナを停止する（データは保持）
	$(COMPOSE) down

clean: ## コンテナとボリュームを削除する（全データ消去）
	$(COMPOSE) down -v

ps: ## コンテナ一覧
	$(COMPOSE) ps

logs: ## ログを追う（例: make logs S=ingest-worker）
	$(COMPOSE) logs -f $(S)

status: ## クラスタ状態を表示する
	@curl -s localhost:$${BACKEND_API_PORT:-8000}/api/v1/health/ready | python -m json.tool || true

test: ## 単体テストを実行する
	python -m pytest tests/unit -q

smoke: ## 結合スモークテスト（起動中のシステムに対して実行）
	python tests/integration/smoke_test.py --limit $${LIMIT:-30}

lint: ## Python の構文チェック
	python -m compileall -q apps packages services db scripts tests

model-fetch: ## BiomedCLIP を model volume へ取得する（要インターネット）
	bash scripts/fetch_model.sh

reconcile: ## ベクトルとメタデータの整合性を検査する
	$(COMPOSE) exec -T backend-api python /srv/scripts/reconcile.py

observability: ## Prometheus と Grafana を起動する
	$(COMPOSE) --profile observability up -d prometheus grafana
