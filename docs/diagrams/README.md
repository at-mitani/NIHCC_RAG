# 説明用の簡略図

発表や説明に使う簡略図。詳しい図は [../README.md](../README.md)。

## システム構成

| 図 | 内容 |
|---|---|
| [stage3-current.png](stage3-current.png) | 教師データによる学習を加えたシステム構成: EC2 1 台の Docker Compose（フロントエンド ｜ バックエンド ｜ データベース。Citus はコーディネータ 1 台 + ワーカ 5 台）。胸部野を画像 1 単位にし、画像単位のラベルで分類器を学習する。教師データ作成と取り込みの流れを含む |

作り直し: `.venv-dev/Scripts/python.exe -B docs/diagrams/build_samples.py`（SVG も同じ場所に出る）。

## 前処理の処理フロー

| 図 | 内容 |
|---|---|
| [preprocess-normalize.png](preprocess-normalize.png) | 画像の正規化: 読み込み → 濃度の切り詰め（0.5%〜99.5%）→ 0〜255 → 白黒反転の判定 → 長辺 1,024 px へ縮小 |
| [preprocess-roi.png](preprocess-roi.png) | 肺野の抽出（ROI）: U-Net で右肺・左肺・心臓 → 二値化 → 品質判定 → 胸部野 mask（凸包・下へ 10%・全周 2%）→ モデルへの入力 |

作り直し: `.venv-dev/Scripts/python.exe -B docs/diagrams/build_preprocess.py`

## 教師データの処理フロー

| 図 | 内容 |
|---|---|
| [training-teacher-data.png](training-teacher-data.png) | 教師データの作成: 画像 → 胸部野 → RAD-DINO 特徴（768 次元）＋ 14 所見のラベル → 患者単位で train / val / test |
| [training-usage.png](training-usage.png) | 教師データの利用: train で分類器を学習・val で止め時・test で評価 → 本番で所見プロファイル・所見の推定・病変の位置の検証に使う |

作り直し: `.venv-dev/Scripts/python.exe -B docs/diagrams/build_training.py`

## 取り込みと検索の処理フロー（説明用）

| 図 | 内容 |
|---|---|
| [flow-ingest-simple.png](flow-ingest-simple.png) | 画像データの取り込み: 指定 → 検証 → ジョブ登録 → 胸部野の抽出 → ベクトル化 → 登録 |
| [flow-search-simple.png](flow-search-simple.png) | 画像検索と結果表示: 入力 → 英語化 → 胸部野の抽出 → ベクトル化 → 検索 → 組み立て → 表示（結果カードの見本つき） |

作り直し: `.venv-dev/Scripts/python.exe -B docs/diagrams/build_flows.py`。
サービス間のやり取りまで書いた詳しい図は [../05-flow-ingest.png](../05-flow-ingest.png)・[../06-flow-search.png](../06-flow-search.png)。

## コンテナ構成とプロセスの関連（説明用）

| 図 | 内容 |
|---|---|
| [containers-processes.png](containers-processes.png) | 各コンテナで動くプロセス（nginx・uvicorn・キュー消費・PostgreSQL など）と、HTTP・AMQP・SQL のつながりを色分けで示す |

作り直し: `.venv-dev/Scripts/python.exe -B docs/diagrams/build_containers.py`。
ネットワーク・ポート・volume まで書いた詳しい図は [../03-container-relations.png](../03-container-relations.png)・[../02-process-architecture.png](../02-process-architecture.png)。

