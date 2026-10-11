# 検索品質と病変の位置の地図の測定（EC2）

[English](chest-evaluation_EN.md) | 日本語

所見プロファイルを有効にして全量を取り込んだ環境（[chest-profile-setup.md](chest-profile-setup.md)）で
行う測定。どちらも取り込み直しや migration は要らない。

| | 測定 | スクリプト | 所要 |
|---|---|---|---|
| §1 | 実際の検索経路での検索品質（Test B） | `chest_search_eval.py` | 2〜3 時間 |
| §2 | 分類器の区画ごとの地図が病変の位置・大きさを示すか | `chest_lesion_map_eval.py` | 30 分 |

```bash
cd ~/NIHCC_RAG && source ~/.venv-eval/bin/activate   # chest-classifier.md §1.1 の仮想環境
```

---

## 1. 検索品質を測る

分類器の評価（`chest_image_eval.py train`）は numpy で 2 段目だけを測る。ここでは
**backend-api の API を通して**、本番と同じ検索経路で測る。

### 1.1 測り方

- クエリは**分類器の評価で test に回した患者の画像だけ**から取る（`--teacher` の分割を使う）。
  分類器は train の患者で学習しているので、train の画像で測ると数字が膨らむ
- 単独所見（ラベルが 1 つだけ）の画像を所見ごとに最大 N 枚。`No Finding` も含める
- 1 枚につき 3 通りを投げる

  | 名前 | 検索 | 要求 |
  |---|---|---|
  | `profile` | 所見プロファイルで全件を厳密に検索（既定） | `strategy=profile` |
  | `two_stage` | BiomedCLIP で候補を集めてから並べ替え | `strategy=coarse_to_fine` |
  | `stage1` | BiomedCLIP だけ | `rerank=false` |

- 上位から**同じ患者の画像を除いて** K 件を見る（同一患者の別撮影は似ていて当然）
- lift = 上位 K に占める同じ所見の割合 ÷ DB 全体での割合。無作為なら 1.0

### 1.2 実行

先に smoke test が通ることを確かめる。

```bash
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```

```bash
python scripts/chest_search_eval.py \
  --teacher tmp/chest/ec2-full/teacher_nihcc-unet-local-v1_112066.npz \
  --queries-per-finding 50 --top-k 20 --json tmp/chest/search.json
```

- 接続先は `--api`（既定 `http://localhost:8000`）。利用者は `.env` の `LOCAL_USERS` の
  admin を使う（`--user` / `--password` で上書きできる）
- `--teacher` は git で追跡している教師データ。分類器を作り直したときは、その実行の
  `tmp/chest/ec2-full/teacher_*.npz` を指定する
- 750 クエリ × 3 通りで 2〜3 時間（GPU）。tmux の中で動かす

最後に所見ごとの lift の表、ベクトル検索時間（aggregator の `timing_ms`）の p50 / p95、
判定が出る。判定の基準はスクリプトの docstring にある。

このリポジトリの構成での実測（目安）:

| 項目 | 値 |
|---|---|
| `profile` の lift ≥ 1.1 | 14 / 14 所見 |
| `profile` の lift が `two_stage` 以上 | 13 / 14 所見 |
| ベクトル検索時間の p95 | `profile` 24 ms / `two_stage` 290 ms |

`two_stage` が `profile` に届かないのは、1 段目（BiomedCLIP の HNSW）が所見の合う画像を
候補から取りこぼすためである。

### 1.3 全件検索の速さを DB で直接確かめる（任意）

```bash
docker compose exec vector-coordinator psql -U postgres -d vector -c "
EXPLAIN (ANALYZE, SUMMARY) SELECT image_id FROM image_embedding
WHERE embedding_fine IS NOT NULL
ORDER BY embedding_fine <=> (SELECT embedding_fine FROM image_embedding WHERE embedding_fine IS NOT NULL LIMIT 1)
LIMIT 200;" | tail -3
# Execution Time: 数十〜数百 ms
```

画像検索の既定の進め方は `.env` の `IMAGE_SEARCH_STRATEGY` で変えられる
（[OPERATIONS.md](../OPERATIONS.md) §3.4）。

---

## 2. 病変の位置の地図を測る

胸部野の分類器は「胸部野で加重平均した RAD-DINO の patch → 標準化 → 線形」なので、
ロジットは区画（37 × 37）ごとの寄与に正確に分解できる。embedding-service の
`/api/v1/embeddings/chest-map` がその寄与を地図として返す。この地図が病変の位置と
大きさを示すかを、正解 box（`BBox_List_2017.csv`）で測る。

### 2.1 測り方

対象は正解 box がある 4 所見（Fibrosis には正解 box が無い）。4 所見のどれかに正解 box が
ある 388 枚を使う。

| 所見 | 正解 box | 大きさ（面積の中央値） |
|---|---:|---:|
| Infiltration | 123 | 5.7% |
| Mass | 85 | 1.5% |
| Nodule | 79 | 0.5% |
| Pneumonia | 120 | 5.7% |

1 枚・1 所見につき「ここが病変」という点を 1 つ出し、正解 box に入るか（hit）を比べる。

| 方法 | 内容 |
|---|---|
| **map** | 分類器の地図の最大の区画（胸部野の中） |
| bright | 肺野の中で白く浮いて見える点（white top-hat の最大） |
| prior | その所見の box が平均的にある位置（1 枚抜き） |
| random | 胸部野の中の無作為な点（当たる確率を解析的に計算） |

ほかに、地図の正の値のうち box に入った割合（enrichment、1 が無作為）と、地図の塊の面積と
box の面積の順位相関（大きさρ）を出す。

所見ごとに、map の hit が random の 2 倍以上・prior を上回る・95% 区間の下限が random を
上回る、の 3 つを満たせば「地図はその所見の位置を示す」と判定する。大きさρ が 0.3 を
超えれば「大きさも表す」とする。

### 2.2 embedding-service を公開する

測定のあいだだけ、評価用の上書きで embedding-service をホストへ公開する。embedding-service が
数分止まるので、**取り込みや §1 の測定が動いている間はやらない**。

```bash
ls tmp/gold/BBox_List_2017.csv                 # git で追跡している
cp .env tmp/env.before-lesion-map              # §2.4 で戻す

setenv() { if grep -q "^$1=" .env; then sed -i "s#^$1=.*#$1=$2#" .env; else echo "$1=$2" >> .env; fi; }
setenv COMPOSE_FILE docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.gpu.yml:infra/compose/docker-compose.eval.yml
setenv ROI_SERVICE_REPLICAS 1                  # 上書きは roi-service も公開するので 1 台

docker compose up -d
docker compose ps embedding-service            # (healthy) になるまで待つ（1〜2 分）
curl -s http://127.0.0.1:8002/api/v1/model | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['fine_embedding_model_version'], bool(d.get('chest_head')))"
# chest-…-d15 True と出ること
```

### 2.3 測る

胸部野の抽出はホストで行う（U-Net）。

```bash
LUNG_MODEL_PATH=infra/lung-unet/lung-unet.onnx \
  python scripts/chest_lesion_map_eval.py --service-url http://127.0.0.1:8002
```

最後に所見ごとの表と判定が出る。

```
所見            枚     map  bright   prior  random     map 95%区間  enrich  大きさρ  判定
Infiltration  ...
```

- `tmp/chest/lesion-map.json` … 所見ごとの結果と 1 枚ごとの記録
- `tmp/chest/lesion-map-sheet.png` … **無作為 36 例**に、地図（赤）・正解 box（緑）・地図の点（赤丸）・
  白い点（青丸）を重ねたもの。数字と合わせて目で確かめる

### 2.4 戻す

```bash
cp tmp/env.before-lesion-map .env              # 上書きが外れてポートが閉じ、roi-service の台数も戻る
docker compose up -d --remove-orphans
docker compose ps                              # すべて (healthy) / Up に戻ること
```

---

## 結果を手元へ戻す

```bash
# 手元の PC で（リポジトリ直下で）。Windows の scp は {a,b} を展開できないので 1 ファイルずつ
scp <ec2-host>:/home/ubuntu/NIHCC_RAG/tmp/chest/search.json tmp/chest/
scp <ec2-host>:/home/ubuntu/NIHCC_RAG/tmp/chest/lesion-map.json tmp/chest/
scp <ec2-host>:/home/ubuntu/NIHCC_RAG/tmp/chest/lesion-map-sheet.png tmp/chest/
```
