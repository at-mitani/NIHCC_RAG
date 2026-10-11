"""docs/diagrams/ の詳しい構成図・フロー図（01〜06 の PNG と SVG）を作り直す。

    python -B docs/diagrams/build_diagrams.py [図の名前 ...]

Playwright（headless Chromium）が要る:
    pip install playwright && python -m playwright install chromium

内容は compose ファイル（docker-compose.yml・infra/compose/*.yml）・.env.example・
各サービスのコードと DESIGN.md / MULTIMODAL.md / OPERATIONS.md に合わせてある。
構成やフローを変えたら、ここも直して作り直すこと（SVG も同じ場所に出す）。
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from svgkit import NET_COLORS, PALETTE, Sequence, Svg  # noqa: E402


# ============================================================================ 1. システム構成図


def system_architecture() -> str:
    s = Svg(1800, 1150, "システム構成図",
            "胸部X線画像検索ローカル RAG（胸部野 1 単位・所見プロファイル検索。EMBEDDING_FINE_BACKEND=chest-profile の構成）")

    s.box(405, 100, 230, 74, "利用者", ["ブラウザ（研究用途）"], kind="user")
    s.arrow([(520, 174), (520, 318)], "HTTP :8080", label_pos=0.12, label_dx=52, label_dy=4)

    s.zone(40, 210, 1720, 920, "ホスト（EC2 g5.2xlarge・NVIDIA A10G 24 GB ／ 手元 PC は CPU）",
           color="#b59b2b")
    s.zone(60, 250, 1250, 860, "Docker Compose", color="#475569",
           sub="アプリのコンテナは read_only・cap_drop ALL・no-new-privileges。医療情報を外部 API へ送らない")

    bands = [
        (300, 140, "画面・API", "edge"),
        (460, 150, "非同期処理・検索", "worker"),
        (630, 160, "AI 推論", "ai"),
        (810, 190, "データ", "data"),
        (1020, 75, "運用", "ops"),
    ]
    for y, h, label, kind in bands:
        fill, stroke, head = PALETTE[kind]
        s.raw(f'<rect x="80" y="{y}" width="1210" height="{h}" rx="10" fill="{fill}" '
              f'fill-opacity="0.35" stroke="none"/>')
        s.text(96, y + 26, label, size=15, weight="bold", color=head)

    # 画面・API
    s.box(250, 320, 300, 100, "frontend", ["nginx・画面（静的ファイル）", "/api を backend-api へ中継"],
          kind="edge")
    s.box(600, 320, 420, 100, "backend-api", ["FastAPI・REST API・認証（JWT）・監査",
                                              "取り込みの受付・検索の実行（同期）"], kind="edge")
    s.arrow([(550, 370), (598, 370)])
    # 非同期処理・検索
    s.box(250, 480, 230, 110, "queue", ["RabbitMQ 3.13", "取り込み・検索のジョブ"], kind="worker")
    s.box(505, 480, 230, 110, "ingest-worker", ["取り込み（台数は", "INGEST_WORKER_REPLICAS）"], kind="worker",
          badge="×N")
    s.box(760, 480, 230, 110, "search-worker", ["非同期の検索", "（respond-async）"], kind="worker")
    s.box(1015, 480, 255, 110, "search-aggregator", ["画像: プロファイルの全件検索", "文章: BiomedCLIP の HNSW"],
          kind="worker")
    # AI 推論
    s.box(330, 650, 380, 125, "roi-service", ["U-Net（ONNX）で肺・心臓を分割", "→ 胸部野の mask・crop",
                                              "胸郭の形（16 次元）"], kind="ai")
    s.box(760, 650, 510, 125, "embedding-service", ["BiomedCLIP（画像・文 512 次元）",
                                                   "RAD-DINO → 胸部野の分類器 → 所見プロファイル（15 次元）",
                                                   "所見の推定（14 所見。研究用・診断ではない）"],
          kind="ai", badge="GPU")
    # データ
    s.cylinder(330, 830, 380, 150, "metadata-db", ["PostgreSQL 17", "画像・ラベル・胸部野・所見の推定",
                                                   "取り込みジョブ・検索の監査・利用者"])
    s.cylinder(760, 830, 510, 150, "ベクトル用クラスタ", ["Citus + pgvector（coordinator 1 + worker 5、30 shard）",
                                                     "image_embedding（1 画像 1 行）",
                                                     "BiomedCLIP 512 ・プロファイル 15 ・胸郭の形 16"])
    # 運用
    s.box(250, 1033, 300, 52, "db-migrate", ["スキーマ適用（migrate プロファイル）"], kind="ops",
          title_size=14, line_size=11.5, dashed=True)
    s.box(575, 1033, 300, 52, "prometheus / grafana", ["監視（observability プロファイル）"], kind="ops",
          title_size=14, line_size=11.5, dashed=True)
    s.box(900, 1033, 200, 52, "mailpit", ["開発用メール受け"], kind="ops", title_size=14,
          line_size=11.5)

    # 層の間の矢印
    s.arrow([(810, 420), (810, 458), (365, 458), (365, 478)], "ジョブ投入（AMQP）", label_pos=0.62)
    s.arrow([(1000, 420), (1000, 450), (1142, 450), (1142, 478)], "検索", label_pos=0.6)
    s.arrow([(620, 590), (620, 648)], "推論（HTTP）", label_dx=46, label_dy=4)
    s.arrow([(875, 590), (875, 648)], "推論（HTTP）", label_dx=46, label_dy=4)
    s.arrow([(1270, 535), (1292, 535), (1292, 800), (1150, 800), (1150, 828)], "SQL（検索）",
            label_pos=0.82)
    s.arrow([(725, 590), (725, 615), (735, 615), (735, 800), (900, 800), (900, 828)],
            "SQL（UPSERT）", label_pos=0.86)
    s.arrow([(640, 420), (640, 432), (228, 432), (228, 905), (328, 905)], "")
    s.text(220, 802, "SQL（メタデータ）", size=12.5, color="#4b5563", anchor="end")

    # ホスト上のファイル
    s.zone(1340, 260, 400, 520, "ホスト上のファイル（読み取り専用でマウント）", color="#8a8580")
    s.box(1360, 310, 360, 95, "NIH ChestX-ray14", ["112,120 枚 ・ ラベル CSV", "/imports/nihcc（取り込み元）"],
          kind="file")
    s.box(1360, 420, 360, 80, "infra/lung-unet", ["U-Net（ONNX）→ roi-service"], kind="file")
    s.box(1360, 515, 360, 95, "infra/chest", ["胸部野の分類器 head-pool-chest.npz",
                                              "（特徴の作り方を記録）→ embedding-service"], kind="file")
    s.box(1360, 625, 360, 135, "model-cache（volume）", ["BiomedCLIP・RAD-DINO の重み",
                                                       "版（revision）を .env で固定", "初回に取得（オフライン運用も可）"],
          kind="file")

    s.zone(1340, 800, 400, 310, "ホストのスクリプト（Python の venv）", color="#b59b2b")
    s.box(1360, 850, 360, 240, "評価・取り込みのスクリプト",
          ["load_initial_images.py（一括取り込み）", "run_chest_eval.sh（教師データ・学習）",
           "export_chest_head.py（分類器の書き出し）", "chest_search_eval.py（検索品質）",
           "chest_lesion_map_eval.py（病変の位置）", "load_initial_users.py（初期利用者）"],
          kind="host", align="left")

    s.legend(1480, 110, [("画面・API", "edge"), ("処理・ジョブ", "worker"), ("AI 推論", "ai"),
                         ("データ", "data")])
    s.legend(1640, 110, [("ファイル", "file"), ("ホストの処理", "host"), ("運用", "ops")])
    return s.render()


# ============================================================================ 2. プロセス構成図


def process_architecture() -> str:
    s = Svg(1900, 1180, "プロセス構成図",
            "各コンテナで動くプロセスと、プロセス間の通信（HTTP / AMQP / SQL）")

    s.box(40, 110, 220, 70, "ブラウザ", ["画面（app.js）"], kind="user")
    s.box(40, 290, 220, 150, "frontend", ["nginx（1 プロセス）", "静的ファイルの配信", "/api・/docs を中継",
                                          ":8080"], kind="edge")
    s.arrow([(150, 180), (150, 288)], "HTTP", label_dx=26, label_dy=4)

    # backend-api
    s.zone(300, 250, 400, 560, "backend-api（uvicorn）", color="#3b6fd8", dashed=False,
           fill="#f4f8ff")
    s.box(320, 300, 360, 150, "REST API", ["認証（JWT・bcrypt）・利用者管理", "取り込み: 検証 → ジョブ・item 作成",
                                           "検索: respond-async 指定なら非同期へ", "画像・胸部野・サムネイルの配信"],
          kind="edge", title_size=15, line_size=12)
    s.box(320, 470, 360, 140, "検索の実行（execute_search）", ["日本語 → 英語（承認済み辞書）",
                                                               "胸部野抽出 → ベクトル生成",
                                                               "進め方の決定（profile / 2 段 / 1 段）",
                                                               "メタデータ・所見の推定を結合"],
          kind="edge", title_size=15, line_size=12)
    s.box(320, 630, 360, 160, "常駐スレッド（30 秒ごと）", ["outbox の再送（取り込みジョブ）",
                                                           "期限切れの検索元画像を削除",
                                                           "期限切れの再設定トークンを削除"],
          kind="edge", title_size=15, line_size=12, dashed=True)
    s.arrow([(260, 360), (318, 360)], "HTTP\n/api", label_dy=-10)

    # queue / workers
    s.box(760, 250, 300, 170, "queue（RabbitMQ）", ["ingest.items", "search.requests",
                                                   "（失敗は *.dlq へ）"], kind="worker")
    s.box(760, 480, 300, 140, "ingest-worker", ["Python のキュー消費プロセス", "1 枚 = 1 メッセージ",
                                                "失敗は item に残し続行"], kind="worker", badge="×N")
    s.box(760, 670, 300, 140, "search-worker", ["Python のキュー消費プロセス",
                                                "backend-api と同じ execute_search",
                                                "（roi / embedding / aggregator を呼ぶ）"],
          kind="worker")
    s.arrow([(680, 330), (758, 330)], "AMQP\npublish", label_dy=-30)
    s.arrow([(910, 420), (910, 478)], "consume", label_dx=40, label_dy=4)
    s.arrow([(1060, 395), (1085, 395), (1085, 740), (1062, 740)], "consume", label_pos=0.82,
            label_dx=6, label_anchor="start")

    # AI 推論
    s.box(1150, 250, 330, 160, "roi-service（uvicorn）", ["POST /api/v1/roi/chest",
                                                         "U-Net（onnxruntime）で肺・心臓",
                                                         "胸部野 mask・crop・pooling 入力",
                                                         "胸郭の形（16 次元）"], kind="ai",
          title_size=15, line_size=12)
    s.box(1150, 450, 330, 250, "embedding-service（uvicorn）", ["POST /api/v1/embeddings/chest",
                                                               "POST /api/v1/embeddings/texts",
                                                               "POST …/chest-map・…/region-pool（評価用）",
                                                               "BiomedCLIP（crop → 512 次元）",
                                                               "RAD-DINO（518 px・fp16）",
                                                               "→ 分類器 → 15 次元・確率",
                                                               "起動時に分類器の前提を照合"],
          kind="ai", badge="GPU", title_size=15, line_size=12)
    s.arrow([(1060, 520), (1110, 520), (1110, 330), (1148, 330)], "HTTP", label_pos=0.55,
            label_dx=-24)
    s.arrow([(1060, 575), (1148, 575)], "HTTP", label_dy=-6)

    # aggregator / citus
    s.box(1540, 250, 330, 160, "search-aggregator（uvicorn）", ["POST /api/v1/vector/search",
                                                               "  stage1=fine: 全件の厳密検索",
                                                               "  stage1=embedding: HNSW（→ 再スコア）",
                                                               "POST /api/v1/vector/lung-shape",
                                                               "circuit breaker・504/503 の区別"],
          kind="worker", title_size=15, line_size=12, align="left")
    s.zone(1540, 450, 330, 330, "ベクトル用クラスタ（Citus + pgvector）", color="#6d55c9",
           dashed=False, fill="#f6f4fe")
    s.box(1560, 500, 290, 80, "vector-coordinator", ["postgres・分散計画・結果の集約"], kind="data",
          title_size=14, line_size=11.5)
    for i in range(5):
        s.box(1560 + i * 58, 640, 52, 80, f"w{i + 1}", ["6 shard"], kind="data", title_size=13,
              line_size=10.5)
    s.arrow([(1705, 580), (1705, 636)], "SQL（shard ごと）", label_dx=70, label_dy=4)
    s.text(1560, 760, "image_embedding（30 shard・1 画像 1 行）", size=12, color="#3b2a86")
    s.arrow([(1705, 410), (1705, 496)], "SQL", label_pos=0.3, label_dx=26, label_dy=4)

    # 検索（同期）の呼び出し: 上を回す
    s.arrow([(620, 250), (620, 205), (1315, 205), (1315, 248)], "", color="#3b6fd8")
    s.arrow([(1315, 205), (1705, 205), (1705, 248)], "", color="#3b6fd8")
    s.arrow([(1315, 205), (1125, 205), (1125, 448), (1148, 448)], "", color="#3b6fd8")
    s.text(640, 196, "検索（同期）: roi-service → embedding-service → search-aggregator の順に HTTP",
           size=12.5, color="#3b6fd8")

    # ingest-worker → vector cluster（取り込み時の UPSERT）
    s.arrow([(1000, 620), (1000, 645), (1130, 645), (1130, 815), (1515, 815), (1515, 540),
             (1558, 540)], "", color="#6d55c9")
    s.text(1322, 808, "取り込み: image_embedding へ UPSERT（SQL）", size=12.5, color="#6d55c9",
           anchor="middle")

    # metadata-db
    s.cylinder(300, 880, 760, 130, "metadata-db（PostgreSQL 17）",
               ["source_image・image_binary・image_finding・chest_asset・finding_estimate",
                "ingest_job / ingest_item・outbox・search_result・search_audit・app_user"])
    s.arrow([(500, 810), (500, 878)], "SQL", label_dx=24, label_dy=4)
    s.arrow([(850, 620), (850, 650), (740, 650), (740, 878)], "SQL", label_pos=0.8, label_dx=-24)
    s.arrow([(960, 810), (960, 878)], "SQL（search_result）", label_dx=74, label_dy=4)

    # ホスト
    s.zone(1150, 840, 720, 300, "ホストの Python プロセス（venv）", color="#b59b2b")
    s.box(1170, 885, 330, 230, "一括取り込み・運用", ["load_initial_images.py",
                                                     "  → backend-api の REST（:8000）",
                                                     "reconcile.py（backend-api のイメージで実行）",
                                                     "smoke_test.py（結合確認）"],
          kind="host", align="left", title_size=15, line_size=12)
    s.box(1520, 885, 330, 230, "評価（EC2）", ["chest_image_eval.py",
                                              "  U-Net をホストで実行し",
                                              "  embedding-service（:8002）へ",
                                              "chest_search_eval.py → REST",
                                              "chest_lesion_map_eval.py",
                                              "※ :8001/:8002 は評価用の上書きで",
                                              "  127.0.0.1 にだけ公開"],
          kind="host", align="left", title_size=15, line_size=12)
    s.text(40, 1150, "uvicorn = Python の ASGI サーバ（FastAPI を動かす）。アプリのコンテナは read_only で、"
           "書けるのは /tmp（tmpfs）だけ。", size=12.5, color="#6b7280")
    return s.render()


# ============================================================================ 3. コンテナ関連図


def container_relations() -> str:
    s = Svg(2000, 1130, "コンテナ関連図",
            "docker-compose.yml + infra/compose/docker-compose.vectors.yml（＋ gpu.yml）。"
            "帯 = Docker ネットワーク、箱が重なる帯 = 所属するネットワーク")

    bands = [("edge", 110, 120, "外部に開く唯一の面"),
             ("app", 230, 270, "サービス間（下請けは外から叩けない）"),
             ("data", 500, 190, "DB に到達できるものを絞る"),
             ("cluster-vector", 690, 200, "Citus の coordinator と worker の間")]
    band_y = {}
    for name, y, h, desc in bands:
        color = NET_COLORS[name]
        s.raw(f'<rect x="30" y="{y}" width="1940" height="{h}" fill="{color}" fill-opacity="0.07" '
              f'stroke="{color}" stroke-opacity="0.5" stroke-dasharray="6 5"/>')
        s.text(42, y + 24, name, size=16, weight="bold", color=color)
        s.text(42, y + 44, desc, size=11.5, color="#6b7280")
        band_y[name] = (y, h)

    def span(first: str, last: str) -> tuple[int, int]:
        y0 = band_y[first][0] + 12
        y1 = band_y[last][0] + band_y[last][1] - 12
        return y0, y1 - y0

    def container(x, w, first, last, name, lines, kind, *, badge=None, ports=None):
        y, h = span(first, last)
        s.box(x, y, w, h, name, lines, kind=kind, title_size=15, line_size=11.5, badge=badge)
        if ports:
            s.chip(x + 8, y + h - 26, ports, "#b45309")

    container(230, 170, "edge", "app", "frontend", ["nginx", "← backend-api"], "edge",
              ports=":8080")
    container(420, 190, "edge", "data", "backend-api", ["REST API", "← metadata-db", "← queue",
                                                       "← mailpit", "bind: NIHCC（ro）"], "edge",
              ports=":8000")
    container(630, 175, "app", "data", "ingest-worker", ["← metadata-db・queue", "← roi-service",
                                                       "← embedding-service", "bind: NIHCC（ro）"],
              "worker", badge="×N")
    container(825, 175, "app", "data", "search-worker", ["← metadata-db", "← queue"], "worker")
    container(1020, 190, "app", "data", "search-aggregator", ["← vector-coordinator"], "worker")

    # app だけのコンテナ（2 段）
    ay = band_y["app"][0]
    s.box(1240, ay + 14, 220, 116, "queue", ["rabbitmq:3.13", "vol: queue-data"], kind="worker",
          title_size=15, line_size=11.5)
    s.chip(1248, ay + 104, ":15672 管理画面", "#b45309")
    s.box(1480, ay + 14, 220, 116, "roi-service", ["U-Net（ONNX）", "bind: infra/lung-unet（ro）"],
          kind="ai", title_size=15, line_size=11.5)
    s.box(1720, ay + 14, 230, 116, "embedding-service", ["vol: model-cache", "bind: infra/chest（ro）"],
          kind="ai", badge="GPU", title_size=15, line_size=11.5)
    s.box(1240, ay + 142, 220, 116, "mailpit", ["開発用メール"], kind="ops", title_size=15,
          line_size=11.5)
    s.chip(1248, ay + 232, ":8025", "#b45309")
    s.box(1480, ay + 142, 220, 116, "prometheus", ["vol: prometheus-data", "observability"],
          kind="ops", title_size=15, line_size=11.5)
    s.chip(1488, ay + 232, ":9090", "#b45309")
    s.box(1720, ay + 142, 230, 116, "grafana", ["vol: grafana-data", "← prometheus", "observability"], kind="ops",
          title_size=15, line_size=11.5)
    s.chip(1728, ay + 232, ":3000", "#b45309")

    # data
    dy, dh = band_y["data"]
    s.cylinder(1240, dy + 18, 260, dh - 36, "metadata-db", ["postgres:17-alpine",
                                                            "vol: metadata-data"])
    container(1530, 190, "data", "cluster-vector", "vector-coordinator",
              ["Citus + pgvector", "vol: vector-coordinator-data"], "data")
    container(1740, 210, "data", "cluster-vector", "db-migrate",
              ["スキーマ適用", "migrate プロファイル", "← metadata-db", "← vector-*（6 台）"], "ops")
    # cluster-vector の worker
    cy, ch = band_y["cluster-vector"]
    for i in range(5):
        x = 230 + i * 205
        s.cylinder(x, cy + 30, 185, ch - 60, f"vector-worker-{i + 1}",
                   ["Citus + pgvector", f"vol: vector-worker-{i + 1}-data", "6 shard"])
    s.arrow([(1530, cy + 100), (1262 + 0, cy + 100)], "分散実行（SQL）", label_dy=-8,
            color=NET_COLORS["cluster-vector"])

    # 凡例・注記
    ly = 925
    s.text(42, ly, "読み方", size=16, weight="bold", color="#111827")
    notes = [
        "・「← X」は depends_on（X が起動・healthy になってから起動する）。:数字 はホストへ公開するポート。×N は INGEST_WORKER_REPLICAS",
        "・roi-service と embedding-service は app だけに居るので DB へ到達できない（胸部野と推論は DB を持たない）",
        "・vector-worker は cluster-vector だけに居る。アプリから届くのは coordinator だけ",
        "・評価のときだけ docker-compose.eval.yml で roi-service（:8001）・embedding-service（:8002）・"
        "metadata-db（:55432）を 127.0.0.1 に公開する",
        "・アプリのコンテナ（frontend・backend-api・worker・aggregator・roi・embedding）は read_only: true（書けるのは /tmp の tmpfs）。"
        "永続するのは vol: の名前付き volume だけ",
        "・bind（ro）はホストのファイルを読み取り専用で見せる: NIHCC = 画像とラベル CSV、infra/lung-unet = U-Net、"
        "infra/chest = 分類器",
    ]
    for i, line in enumerate(notes):
        s.text(42, ly + 28 + i * 24, line, size=13.5, color="#374151")
    s.legend(1700, 925, [("画面・API", "edge"), ("処理", "worker"), ("AI 推論", "ai"),
                         ("データ", "data"), ("運用", "ops")])
    return s.render()


# ============================================================================ 4. 教師データ作成


def training_flow() -> str:
    s = Svg(1900, 1260, "処理フロー図（教師データ作成時）",
            "EC2 で 1 回だけ行う。run_chest_eval.sh（chest_image_eval.py）→ export_chest_head.py"
            "（docs/runbooks/chest-classifier.md）")

    lanes = [("入力データ", 40, 300, "file"), ("ホストの評価スクリプト（venv・CPU 並列）", 360, 560, "host"),
             ("embedding-service（GPU・rad-dino・:8002 を公開）", 940, 440, "ai"),
             ("成果物（tmp/chest・infra/chest）", 1400, 460, "data")]
    for label, x, w, kind in lanes:
        fill, stroke, head = PALETTE[kind]
        s.raw(f'<rect x="{x}" y="100" width="{w}" height="1140" rx="12" fill="{fill}" '
              f'fill-opacity="0.3" stroke="{stroke}" stroke-opacity="0.6"/>')
        s.text(x + w / 2, 128, label, size=15, weight="bold", color=head, anchor="middle")

    # 入力
    s.box(60, 160, 260, 95, "胸部X線画像", ["NIH ChestX-ray14", "112,120 枚（PNG）"], kind="file")
    s.box(60, 300, 260, 95, "Data_Entry_2017_v2020.csv", ["画像単位のラベル（NLP 由来）", "患者番号"],
          kind="file", title_size=14)
    s.box(60, 440, 260, 95, "BBox_List_2017.csv", ["正解 box（880 枚・984 個）", "被覆率の確認だけに使う"],
          kind="file", title_size=14)
    s.box(60, 580, 260, 95, "U-Net（ONNX）", ["infra/lung-unet/lung-unet.onnx", "profile nihcc-unet-local-v1"],
          kind="file")

    # ホスト: ROI
    s.box(390, 160, 500, 150, "① 胸部野の抽出（image_pipeline.chest）",
          ["U-Net で右肺・左肺・心臓の mask", "胸部野 = 肺＋心臓の凸包 ＋ 下方へ肺の高さ 10% の帯",
           "＋ 周囲 2% の余白（縦隔・横隔膜を含める）", "肺野が小さすぎる画像は除外（54 枚）"],
          kind="host", title_size=15, line_size=12)
    s.arrow([(320, 207), (388, 207)])
    s.arrow([(320, 627), (350, 627), (350, 280), (388, 280)], "", color="#8a8580")
    s.box(390, 350, 500, 105, "② モデル入力を作る", ["pooling 用: 518 px 四方の画像と胸部野 mask",
                                                  "（評価と本番で同じ関数。1 画素違うと確率がずれる）"],
          kind="host", title_size=15, line_size=12)
    s.arrow([(640, 310), (640, 348)])

    # embedding
    s.box(970, 330, 380, 190, "③ RAD-DINO の patch 特徴", ["/api/v1/embeddings/region-pool",
                                                          "ViT-B/14・518 px → 37×37 の patch",
                                                          "胸部野 mask の被覆率で重み付け平均",
                                                          "→ 768 次元（pool-chest）",
                                                          "対照: pool-lung・pool-whole・crop-chest"],
          kind="ai", title_size=15, line_size=12)
    s.arrow([(890, 400), (968, 400)], "HTTP", label_dy=-6)
    s.cylinder(1440, 350, 380, 120, "特徴のキャッシュ", ["vec_pool-chest_*.npz・shards/",
                                                       "2,000 枚ごとに保存・途中から再開"])
    s.arrow([(1350, 425), (1438, 425)])

    # 正解 box の被覆率
    s.box(390, 500, 500, 95, "④ 被覆率（coverage）", ["正解 box が胸部野へ 50% 以上入る割合",
                                                   "胸部野 99.7% ／ 肺野だけ 58.7%"],
          kind="host", title_size=15, line_size=12)
    s.arrow([(320, 487), (388, 540)])
    s.box(1440, 510, 380, 70, "coverage.json", ["枠が所見を含むかの判定材料"], kind="data",
          title_size=14)
    s.arrow([(890, 547), (1438, 547)], "", color="#8a8580")

    # 教師データと学習
    s.box(390, 640, 500, 140, "⑤ 教師データ（teacher）", ["画像単位の 14 所見ラベル（多ラベル）",
                                                       "患者単位で分割（同じ患者を跨がせない）",
                                                       "train 77,648 / val 11,435 / test 22,983 枚",
                                                       "112,066 枚（99.95%）で作成"],
          kind="host", title_size=15, line_size=12)
    s.arrow([(320, 347), (340, 347), (340, 700), (388, 700)], "", color="#8a8580")
    s.cylinder(1440, 640, 380, 110, "ec2-full/teacher_*.npz", ["画像名・14 所見・split（git で追跡）"])
    s.arrow([(890, 700), (1438, 700)])

    s.box(390, 820, 500, 150, "⑥ 分類器の学習（ChestHead）", ["標準化 → 線形（768 × 14）→ sigmoid",
                                                           "train で学習・val で epoch を選ぶ",
                                                           "所見プロファイル = 14 確率 ＋ 正常らしさ",
                                                           "→ L2 正規化（15 次元）"],
          kind="host", title_size=15, line_size=12)
    s.arrow([(640, 780), (640, 818)])
    s.arrow([(1440, 455), (1393, 455), (1393, 838), (892, 838)], "特徴を読む",
            label_pos=0.8, color="#6d55c9")

    s.box(390, 1010, 500, 150, "⑦ 評価（test の患者だけ）", ["画像単位 AUC 0.833 [0.824, 0.839]",
                                                         "帰無（画像統計量）0.696 を上回る",
                                                         "Test B: 上位 20 件の lift が 14 所見すべて ≥ 1.33",
                                                         "所見別 AUC 0.70（浸潤影）〜0.905（心拡大）"],
          kind="judge", title_size=15, line_size=12)
    s.arrow([(640, 970), (640, 1008)])
    s.box(1440, 820, 380, 100, "head_pool-chest_*.npz", ["学習した分類器", "train.json（AUC・Test B）"],
          kind="data", title_size=14)
    s.arrow([(890, 870), (1438, 870)])

    s.box(970, 1010, 380, 200, "⑧ 本番用に書き出す", ["export_chest_head.py",
                                                     "特徴の作り方を一緒に記録:",
                                                     "  RAD-DINO の版（revision 固定）",
                                                     "  pooling 518 px・mask 10% / 2%",
                                                     "  U-Net の profile",
                                                     "未固定（unpinned）は書き出せない"],
          kind="host", title_size=15, line_size=12, align="left")
    s.arrow([(890, 1085), (968, 1085)], "合格したら", label_dy=-8)
    s.box(1440, 1000, 380, 110, "infra/chest/head-pool-chest.npz", ["本番の分類器（git には入れない）"],
          kind="data", title_size=14)
    s.arrow([(1350, 1055), (1438, 1055)])
    s.note(1440, 1130, 380, ["→ embedding-service（chest-profile）が起動時に読み、",
                             "  backbone の版・解像度を照合。ingest-worker は",
                             "  mask の作り方・profile を照合（違えば止まる）"])
    s.note(60, 720, 270, ["教師は「画像単位のラベル」", "そのもの。部位単位の正解は", "無いので、検索の単位も画像",
                          "1 枚の胸部野にしている"])
    return s.render()


# ============================================================================ 5. 取り込み


def ingest_flow() -> str:
    seq = Sequence(
        "処理フロー図（データ取り込み時）",
        "1 枚 = 1 メッセージ。1 枚の失敗で全体を止めない・同じ画像の再送は冪等（docs/DESIGN.md §3.1）",
        [
            ("user", "利用者 / スクリプト", "画面・load_initial_images.py", "user"),
            ("api", "backend-api", "REST・検証", "edge"),
            ("mq", "queue", "RabbitMQ", "worker"),
            ("iw", "ingest-worker", "×N", "worker"),
            ("roi", "roi-service", "U-Net", "ai"),
            ("emb", "embedding-service", "GPU", "ai"),
            ("meta", "metadata-db", "PostgreSQL 17", "data"),
            ("vec", "ベクトル用クラスタ", "Citus + pgvector", "data"),
        ],
        width=1900,
    )
    seq.section("受付（同期）")
    seq.msg("user", "api", "POST /api/v1/ingestions/validate\nCSV と画像フォルダの指定")
    seq.self_msg("api", "CSV と画像の突合\n（件数・欠損・重複）→ ingest_validation")
    seq.msg("api", "user", "検証結果（validation_id・件数・警告）", dashed=True)
    seq.msg("user", "api", "POST /api/v1/ingestions（validation_id・重複時 SKIP・Idempotency-Key）")
    seq.msg("api", "meta", "ingest_job・idempotency_key・ingest_item（1 枚 1 行）・outbox を 1 トランザクションで")
    seq.msg("api", "mq", "publish ingest.item（outbox 経由。失敗しても 30 秒ごとに再送）")
    seq.msg("api", "user", "202 Accepted（job_id）→ 画面は進捗を polling", dashed=True)
    seq.section("1 枚ごとの処理（非同期）")
    seq.msg("mq", "iw", "consume ingest.items")
    seq.self_msg("iw", "/imports/nihcc から画像を読み、形式を検査\n（ホストの NIHCC を読み取り専用でマウント）")
    seq.msg("iw", "meta", "patient（患者 ID はハッシュ）・source_image・image_binary・image_finding\n（登録済みで SKIP 指定ならここで終了）")
    seq.msg("iw", "roi", "POST /api/v1/roi/chest（画像）")
    seq.self_msg("roi", "U-Net で肺・心臓 → 胸部野 mask\ncrop・pooling 入力（どちらも 518 px）\n胸郭の形（16 次元）・品質判定")
    seq.msg("roi", "iw", "胸部野 mask・crop・pooling 画像と mask・胸郭の形・信頼度", dashed=True)
    seq.msg("iw", "emb", "POST /api/v1/embeddings/chest\n（roi-service が作った入力をそのまま渡す）")
    seq.self_msg("emb", "BiomedCLIP: crop → 512 次元\nRAD-DINO → 胸部野で pooling → 分類器\n→ 所見プロファイル 15 次元・14 確率")
    seq.msg("emb", "iw", "vectors・fine_vectors・probabilities", dashed=True)
    seq.msg("iw", "meta", "model_version・chest_asset（mask・bbox・品質）・finding_estimate（14 確率）・\nimage_vector_state = PENDING を 1 トランザクションで")
    seq.msg("iw", "vec", "image_embedding へ UPSERT（1 画像 1 行）\nembedding 512・embedding_fine 15・embedding_lung 16")
    seq.msg("iw", "meta", "image_vector_state = STORED・ingest_item = SUCCEEDED")
    seq.note("iw", "vec", ["失敗した場合: ingest_item に理由を残してジョブは続行（例: LUNG_TOO_SMALL = 肺野が小さすぎる → 手動確認へ）",
                           "ベクトル登録だけ失敗した場合は image_vector_state = FAILED を残し、reconcile.py で突合して補う"],
             kind="warn")
    seq.note("api", "iw", ["起動時の照合: ingest-worker は roi-service の mask の作り方と分類器の前提、",
                           "ベクトル列の幅と .env の次元が一致するかを確かめ、違えば取り込みを始めない"])
    return seq.render()


# ============================================================================ 6. 検索


def search_flow() -> str:
    seq = Sequence(
        "処理フロー図（検索時）",
        "画像だけの検索は所見プロファイルで全件を厳密に比べる（既定）。文章を含む検索は BiomedCLIP の HNSW"
        "（MULTIMODAL.md §4.1）",
        [
            ("user", "利用者", "ブラウザ", "user"),
            ("api", "backend-api", "execute_search", "edge"),
            ("roi", "roi-service", "U-Net", "ai"),
            ("emb", "embedding-service", "GPU", "ai"),
            ("agg", "search-aggregator", "", "worker"),
            ("vec", "ベクトル用クラスタ", "Citus + pgvector", "data"),
            ("meta", "metadata-db", "PostgreSQL 17", "data"),
        ],
        width=1900,
    )
    seq.section("要求と前処理")
    seq.msg("user", "api", "POST /api/v1/searches（画像・文・モード・Top-K・絞り込み）")
    seq.self_msg("api", "日本語の所見文 → 英語\n（承認済みのローカル辞書だけ。辞書に無ければ 422）")
    seq.msg("api", "roi", "POST /api/v1/roi/chest（検索元画像）")
    seq.msg("roi", "api", "胸部野 mask・crop・pooling 入力・胸郭の形・信頼度", dashed=True)
    seq.msg("api", "emb", "POST /api/v1/embeddings/chest（画像）・/embeddings/texts（文）")
    seq.msg("emb", "api", "BiomedCLIP 512・所見プロファイル 15（画像だけ）・検索元の所見の推定", dashed=True)
    seq.self_msg("api", "進め方を決める（IMAGE_SEARCH_STRATEGY / 要求の strategy）\n"
                        "画像だけ → profile（既定）か coarse_to_fine ／ 文章を含む・rerank=false → coarse_only")
    seq.section("ベクトル検索")
    seq.msg("api", "agg", "POST /api/v1/vector/search（stage1 = fine | embedding）")
    seq.note("agg", "meta", ["【画像だけ・既定】stage1 = fine: embedding_fine（15 次元）で全行を厳密に比較",
                            "  ORDER BY embedding_fine <=> q LIMIT k を 30 shard へ並列に押し下げ（索引なし・p95 24 ms）"])
    seq.msg("agg", "vec", "SELECT … ORDER BY embedding_fine <=> q LIMIT candidate_k")
    seq.note("agg", "meta", ["【文章を含む】embedding（BiomedCLIP 512）の HNSW で上位 k 件（所見プロファイルは文章を扱えない）",
                            "【2 段検索（coarse_to_fine）】HNSW で candidate_k × 4 件 → embedding_fine で厳密に再スコア"])
    seq.msg("vec", "agg", "image_id・距離（・roi_confidence）", dashed=True)
    seq.msg("agg", "api", "候補・rerank_status（PROFILE_EXACT / APPLIED / DISABLED …）・timing_ms", dashed=True)
    seq.msg("api", "agg", "POST /api/v1/vector/lung-shape（胸郭の形 16 次元。別欄）")
    seq.msg("agg", "api", "胸郭の形が似た症例", dashed=True)
    seq.section("結果の組み立て")
    seq.msg("api", "meta", "メタデータ・データセットのラベル・所見の推定・胸部野の品質を取得")
    seq.self_msg("api", "絞り込み（付随ラベル・撮影方向）・同一画像の除外\n検索元の信頼度が低ければ結果を抑制し警告")
    seq.msg("api", "meta", "search_query_chest・search_query_outline（検索元の輪郭。期限つき）・search_audit（監査）")
    seq.msg("api", "user", "結果一覧（類似度・ラベル・所見の推定〈研究用・診断ではない〉・研究用途の警告）",
            dashed=True)
    seq.note("user", "meta", ["画面で非同期実行を選ぶと（Prefer: respond-async）: backend-api は 202 を返して search.requests へ投入",
                              "→ search-worker が同じ execute_search を実行して search_result へ書き込み、画面は search_id で polling する"],
             kind="warn")
    return seq.render()


# ============================================================================ 出力


DIAGRAMS = {
    "01-system-architecture": system_architecture,
    "02-process-architecture": process_architecture,
    "03-container-relations": container_relations,
    "04-flow-training-data": training_flow,
    "05-flow-ingest": ingest_flow,
    "06-flow-search": search_flow,
}


def main() -> int:
    from playwright.sync_api import sync_playwright

    names = sys.argv[1:] or list(DIAGRAMS)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(device_scale_factor=1.5)
        for name in names:
            svg = DIAGRAMS[name]()
            (HERE / f"{name}.svg").write_text(svg, encoding="utf-8")
            page.set_content(f"<html><body style='margin:0'>{svg}</body></html>")
            page.locator("svg").first.screenshot(path=str(HERE / f"{name}.png"))
            print(f"wrote {name}.png / .svg")
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
