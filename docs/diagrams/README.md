# 構成図・フロー図

[English](README_EN.md) | 日本語

このフォルダにある図の一覧。どの図も PNG と SVG の 2 つがあり、同じ名前で並んでいる。
図は所見プロファイル（`EMBEDDING_FINE_BACKEND=chest-profile`）を有効にした構成を描いている。

## 詳しい図

サービス間のやり取り・ネットワーク・ポート・volume まで書いた図。

| 図 | 内容 |
|---|---|
| [01-system-architecture.png](01-system-architecture.png) | システム構成図: ホスト（EC2）上の Docker Compose のコンテナを役割ごと（画面・API／非同期処理・検索／AI 推論／データ／運用）に並べ、ホスト上のファイルとスクリプトとの関係を示す |
| [02-process-architecture.png](02-process-architecture.png) | プロセス構成図: 各コンテナで動くプロセス（nginx・uvicorn・キュー消費・常駐スレッド）と、HTTP・AMQP・SQL の通信、エンドポイント、キュー名、metadata のテーブル |
| [03-container-relations.png](03-container-relations.png) | コンテナ関連図: Docker ネットワーク（edge・app・data・cluster-vector）への所属、depends_on、公開ポート、volume と bind mount |
| [04-flow-training-data.png](04-flow-training-data.png) | 処理フロー図（教師データ作成時）: 胸部野の抽出 → RAD-DINO の特徴 → 教師データ → 分類器の学習・評価 → 本番用の書き出し |
| [05-flow-ingest.png](05-flow-ingest.png) | 処理フロー図（データ取り込み時）: 検証・ジョブ登録から、1 枚ごとの胸部野の抽出・ベクトル化・DB への登録までのシーケンス |
| [06-flow-search.png](06-flow-search.png) | 処理フロー図（検索時）: 英語化・胸部野の抽出・ベクトル化・ベクトル検索（全件検索／BiomedCLIP／2 段検索）・結果の組み立てのシーケンス |

## 説明用の簡略図

発表や説明に使う、文字を大きく・少なくした図。

| 図 | 内容 |
|---|---|
| [stage3-current.png](stage3-current.png) | 教師データによる学習を含むシステム構成: EC2 1 台の Docker Compose（Citus はコーディネータ 1 台 + ワーカ 5 台）。検索・取り込み・教師データ作成の流れを色分けで示す |
| [containers-processes.png](containers-processes.png) | コンテナ構成とプロセスの関連: 各コンテナで動くプロセスと、HTTP・AMQP・SQL のつながり |
| [preprocess-normalize.png](preprocess-normalize.png) | 画像の正規化: 読み込み → 濃度の切り詰め（0.5%〜99.5%）→ 0〜255 → 白黒反転の判定 → 長辺 1,024 px 以下へ縮小 |
| [preprocess-roi.png](preprocess-roi.png) | 肺野の抽出（ROI）: U-Net で右肺・左肺・心臓 → 二値化 → 品質判定 → 胸部野 mask（凸包・下へ 10%・全周 2%）→ モデルへの入力 |
| [training-teacher-data.png](training-teacher-data.png) | 教師データの作成: 画像 → 胸部野 → RAD-DINO の特徴（768 次元）＋ 14 所見のラベル → 患者単位で train / val / test |
| [training-usage.png](training-usage.png) | 教師データの利用: train で分類器を学習・val で止め時・test で評価 → 本番で所見プロファイルと所見の推定に使う |
| [flow-ingest-simple.png](flow-ingest-simple.png) | 画像データの取り込み: 指定 → 検証 → ジョブ登録 → 胸部野の抽出 → ベクトル化 → 登録 |
| [flow-search-simple.png](flow-search-simple.png) | 画像検索と結果表示: 入力 → 英語化 → 胸部野の抽出 → ベクトル化 → 検索 → 組み立て → 表示（結果カードの見本つき） |

## 作り直し方

図はこのフォルダのスクリプトで作る。描画に Playwright（headless Chromium）を使う。

```bash
pip install playwright && python -m playwright install chromium
python -B docs/diagrams/build_diagrams.py      # リポジトリのルートで実行
```

| スクリプト | 作る図 |
|---|---|
| `build_diagrams.py` | 01〜06（名前を引数に渡すとその図だけ。例: `05-flow-ingest`） |
| `build_samples.py` | stage3-current |
| `build_containers.py` | containers-processes |
| `build_preprocess.py` | preprocess-normalize・preprocess-roi |
| `build_training.py` | training-teacher-data・training-usage |
| `build_flows.py` | flow-ingest-simple・flow-search-simple |

共通の描画部品は `svgkit.py` にある。構成やフローを変えたら、スクリプトを直して作り直す
（PNG と SVG がこのフォルダに上書きされる）。
