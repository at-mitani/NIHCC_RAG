# 文書の索引

文書は**設計仕様書**と**運用マニュアル**の 2 種類に分ける。

---

## 設計仕様書（開発者向け）

| 文書 | 内容 |
|---|---|
| [GLOSSARY.md](GLOSSARY.md) | **略語と用語**（ROI / MIL / U-Net など）。最初に読むとよい |
| [DESIGN.md](DESIGN.md) | システム構成、コンテナ構成、プロセスフロー、DB とテーブル仕様 |
| [MULTIMODAL.md](MULTIMODAL.md) | 使用モデルとライブラリ、マルチモーダル設計、胸部X線のハーネス、多段検索、ROI / MIL / U-Net |

DDL は [db/metadata-migrations/](../db/metadata-migrations/) と
[db/vector-migrations/](../db/vector-migrations/)。

## 運用マニュアル（運用者向け）

| 文書 | 内容 |
|---|---|
| [../README.md](../README.md) | 入口。起動停止と**画面の使い方** |
| [../SETUP_LOCAL.md](../SETUP_LOCAL.md) | 手元の PC への構築 |
| [../SETUP_EC2.md](../SETUP_EC2.md) | GPU 付き EC2 への構築 |
| [diagrams/](diagrams/README.md) | **構成図・フロー図**（システム・プロセス・コンテナ、教師データ作成・取り込み・検索） |
| [OPERATIONS.md](OPERATIONS.md) | 日々の運用、更新の反映、**障害対応** |
| [user-management.md](user-management.md) | 利用者とパスワードの管理 |

### 個別の手順書（runbooks/）

滅多に行わない作業の手順。

| 文書 | 内容 | 所要 |
|---|---|---|
| [runbooks/lung-unet.md](runbooks/lung-unet.md) | 肺野分割モデル（U-Net）を学習して配置する | 2〜3 時間 |
| [runbooks/chest-classifier.md](runbooks/chest-classifier.md) | 胸部野の教師データを作り、分類器を学習・評価して書き出す | 4〜5 時間 |
| [runbooks/chest-profile-setup.md](runbooks/chest-profile-setup.md) | 所見プロファイル（chest-profile）を有効にして EC2 で構築し、全量を取り込む | 1 日 |
| [runbooks/chest-evaluation.md](runbooks/chest-evaluation.md) | 検索品質と、分類器の地図が病変の位置を示すかを測る | 3〜4 時間 |

---

## どこに書くか

| 書きたいこと | 置き場所 |
|---|---|
| 今の構成・仕様 | `DESIGN.md` / `MULTIMODAL.md` |
| 手順（繰り返す） | `OPERATIONS.md` |
| 手順（滅多に行わない） | `runbooks/` |
| 画面の使い方 | `../README.md` |
