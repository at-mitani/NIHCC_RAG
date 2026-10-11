# Building with the finding profile enabled and ingesting the full dataset (EC2)

English | [日本語](chest-profile-setup.md)

The runbook for installing the chest field classifier, enabling `EMBEDDING_FINE_BACKEND=chest-profile`, and ingesting all 112,120
images. Once enabled, it works as follows ([MULTIMODAL_EN.md](../MULTIMODAL_EN.md) §4.1).

| Search | How it proceeds |
|---|---|
| Image only | Exact exhaustive search with the finding profile (15 dimensions) (`IMAGE_SEARCH_STRATEGY=profile`, default) |
| Text / image + text | Collect candidates with BiomedCLIP (chest field crop, 512 dimensions), then re-rank with the finding profile |

The screen shows finding estimates (「所見の推定（研究用・診断ではありません）」, "Finding estimates (for research use only, not for diagnosis)"). Probabilities are computed and stored at ingestion time.

- Time required: 1 day (ingesting the full dataset takes about 4 hours)
- Prerequisite: [SETUP_EC2_EN.md](../../SETUP_EC2_EN.md) §1–8 are done (GPU, Docker, repository, images, `.env`)

---

## 0. What you need

| Item | Location | How to make it |
|---|---|---|
| U-Net model | `infra/lung-unet/lung-unet.onnx` | [lung-unet_EN.md](lung-unet_EN.md) |
| Chest field classifier | `infra/chest/head-pool-chest.npz` | [chest-classifier_EN.md](chest-classifier_EN.md) |
| RAD-DINO revision used when training the classifier | Your note (40 chars) | [chest-classifier_EN.md](chest-classifier_EN.md) §2.1 |
| Feature cache | `tmp/chest/vec_pool-chest_nihcc-unet-local-v1_112120.npz` | Same, §4 (used for the check in §4) |

```bash
cd ~/NIHCC_RAG && source ~/.venv-eval/bin/activate
df -h /                                         # aim for 120 GB or more free
ls NIHCC/images | wc -l                         # 112120
nvidia-smi                                      # the A10G should be visible
ls infra/lung-unet/lung-unet.onnx infra/chest/head-pool-chest.npz
```

Disk breakdown (estimate):

| Item | Size |
|---|---|
| Images | 42 GB |
| Docker images | about 25 GB |
| metadata-db (copies of original images and masks) | about 50 GB |
| Vector DB | about 3 GB |

> **If you already have data ingested with `EMBEDDING_FINE_BACKEND=none`**, those rows have
> neither profiles nor finding estimates. Delete them with `python scripts/reset_data.py` and re-ingest in §6
> (users are kept).

---

## 1. `.env`

```bash
python3 scripts/sync_env.py --write             # add keys present in .env.example but missing from .env
setenv() { if grep -q "^$1=" .env; then sed -i "s#^$1=.*#$1=$2#" .env; else echo "$1=$2" >> .env; fi; }
RADDINO_REV=<RAD-DINO revision used when training the classifier (40 chars)>

# add the evaluation override (exposes roi-service and embedding-service to the host) only during the §4 check
setenv COMPOSE_FILE docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.gpu.yml:infra/compose/docker-compose.eval.yml
setenv COMPOSE_PATH_SEPARATOR :
setenv ROI_SERVICE_REPLICAS 1                   # 1 replica during the check, since it is exposed to the host

setenv ROI_PROFILE nihcc-unet-local-v1          # the profile the classifier was trained with
setenv EMBEDDING_FINE_BACKEND chest-profile
setenv EMBEDDING_FINE_DIMENSION 15
setenv EMBEDDING_FINE_MODEL_REVISION "$RADDINO_REV"
setenv CHEST_HEAD_ALLOW_UNPINNED false
setenv EMBEDDING_MODEL_REVISION 9f341de24bfb00180f1b847274256e9b65a3a32e   # BiomedCLIP
setenv CITUS_SHARD_COUNT 30
grep -E '^(COMPOSE_FILE|ROI_PROFILE|EMBEDDING_FINE_|EMBEDDING_MODEL_REVISION|CHEST_HEAD|CITUS_SHARD_COUNT|PG_MAJOR)=' .env
```

> **Decide `EMBEDDING_FINE_DIMENSION` and `CITUS_SHARD_COUNT` before the migration.**
> The column width and shard count are fixed when the migration runs.

---

## 2. Building and starting

```bash
tmux attach -t work || tmux new -s work
docker compose build
docker compose build db-migrate                 # it has a profile, so a build without arguments does not include it
bash scripts/fetch_model.sh                     # only when HF_HUB_OFFLINE=1. Fetches the pinned revisions
docker compose up -d
docker compose ps -a | grep -E 'unhealthy|starting|Restarting|Exited|Created'   # wait until this prints nothing
docker compose run --rm db-migrate
```

At startup, embedding-service compares the classifier with its own configuration (backbone revision, pooling resolution,
mask settings) and refuses to start if they differ. ingest-worker compares how the chest field features are built
(roi-service profile and mask settings) against the classifier's record.

Check the Citus configuration.

```bash
docker compose exec vector-coordinator psql -U postgres -d vector -c \
  "SELECT nodename, isactive FROM pg_dist_node ORDER BY nodename;"
# 6 rows: vector-coordinator and vector-worker-1..5. All workers have isactive = t

docker compose exec vector-coordinator psql -U postgres -d vector -c \
  "SELECT nodename, count(*) FROM citus_shards GROUP BY nodename ORDER BY nodename;"
# 6 shards per worker (30 total)
```

---

## 3. Connectivity check

```bash
python scripts/load_initial_users.py
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
# all items PASS (WARN is acceptable only for response time)
```

---

## 4. Checking that the features match

**Before ingesting the full dataset**, check that the production roi-service and embedding-service build the same features as
classifier training. If the way features are built differs by even one pixel, the classifier raises no exception and only the probabilities drift.
It uses the image files directly, so no ingestion is needed.

```bash
LUNG_MODEL_PATH=infra/lung-unet/lung-unet.onnx \
  python scripts/chest_image_eval.py verify-ingest --limit 200
# モデル入力の不一致: 0 / 200 枚 (model input mismatches: 0 / 200 images)
# 確率の最大の差: 0.0xxx（許容 0.02） (max probability difference: 0.0xxx (tolerance 0.02))
# プロファイルの cosine の最小値: 0.999x（許容 0.999） (min profile cosine: 0.999x (tolerance 0.999))
# 判定: 一致。全量の取り込みに進んでよい (verdict: match. OK to proceed to full ingestion)
```

| Check | What it verifies |
|---|---|
| Model input mismatches | The input to roi-service `/roi/chest` does not differ by a single byte from the input built by the same function as in evaluation (U-Net model, profile, preprocessing) |
| Probability difference / cosine | The probabilities from embedding-service `/embeddings/chest` match the probabilities from applying the classifier to the cached features (RAD-DINO revision, pooling, classifier) |

**If they do not match, do not ingest the full dataset** (§8).

---

## 5. Ingesting a small batch and removing the override

```bash
python scripts/load_initial_images.py --limit 1000 --yes
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```

If it passes, remove the evaluation override and close the ports (the policy is that internal services stay closed within the Docker
network). If you want 2 or more roi-service replicas, raise `ROI_SERVICE_REPLICAS` here.

```bash
setenv COMPOSE_FILE docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.gpu.yml
docker compose up -d --remove-orphans
```

---

## 6. Ingesting the full dataset

```bash
nohup python scripts/load_initial_images.py --yes > tmp/load.log 2>&1 &   # without --yes it stops waiting for confirmation
tail -f tmp/load.log
```

- About 4 hours (measured 231 minutes on g5.2xlarge)
- If it crashes midway, rerunning the same command skips images already ingested (`on_duplicate=SKIP`)

Checking completion:

```bash
docker compose exec vector-coordinator psql -U postgres -d vector -c "SELECT count(*) FROM image_embedding;"
# around 112,066 (112,120 minus the roughly 54 images that failed segmentation)

# reconcile.py runs inside the backend-api image (the host does not have nihcc_rag)
docker compose run --rm --no-deps --entrypoint python backend-api /srv/scripts/reconcile.py
```

Of reconcile's `issue_count`, `MISSING_VECTOR` entries are the images that failed segmentation
(`LUNG_TOO_SMALL`, around 50) and are expected. `ORPHAN_VECTOR` / `MISSING_FINE_VECTOR` /
`MISSING_FINDING_ESTIMATE` are not expected.

To measure search quality, see [chest-evaluation_EN.md](chest-evaluation_EN.md) §1.

---

## 7. Checking on screen

Open `http://<ec2-host>:8080` (or via SSH port forwarding) and check the following.

- Searching by image shows a single result list and "Cases with similar thoracic shape" (「胸郭の形が似た症例」)
- The label "Exhaustive search with finding profile" (「所見プロファイルで全件検索」) appears above the results
- The chest field outline is overlaid on the cards, and "Finding estimates (for research use only, not for diagnosis)" (「所見の推定（研究用・診断ではありません）」) appears
- Searching by text (e.g. 「胸水」 (pleural effusion)) returns results
- The cluster table on the admin screen shows 1 coordinator and 5 workers

---

## 8. When things go wrong

| Symptom | Where to look |
|---|---|
| embedding-service stops with `胸部野の分類器がありません` (chest field classifier not found) | `infra/chest/head-pool-chest.npz` and `CHEST_HEAD_DIR` / `CHEST_HEAD_PATH` |
| embedding-service stops with `特徴の作り方と、いまの構成が違います` (the feature build settings differ from the current configuration) | The items shown (backbone version, pooling resolution, mask settings). Align `EMBEDDING_FINE_MODEL_REVISION` with the classifier, or re-export the classifier |
| embedding-service stops with `EMBEDDING_FINE_DIMENSION=... が分類器の出力 15 次元と一致しない` (does not match the classifier's 15-dimensional output) | `EMBEDDING_FINE_DIMENSION=15` |
| ingest-worker stops with `vector 列の幅が設定と違います` (vector column width differs from the setting) | [OPERATIONS_EN.md](../OPERATIONS_EN.md) §3.1 |
| ingest-worker stops due to a difference in how features are built | The roi-service profile / mask settings and the classifier's record differ. Set `ROI_PROFILE` to the profile the classifier was trained with |
| db-migrate stops while registering workers | Check that `COMPOSE_FILE` includes `docker-compose.vectors.yml` and that `vector-worker-{1..5}` are all running |
| The feature match check fails on "model input mismatches" (「モデル入力の不一致」) | The roi-service image is old, or the U-Net model differs. Rebuild all images with `bash scripts/redeploy.sh` and check `infra/lung-unet/`. **Do not ingest the full dataset** |
| The feature match check fails on "probability difference" (「確率の差」) | The RAD-DINO version differs from the classifier's (`EMBEDDING_FINE_MODEL_REVISION`), or the classifier is from a different run |
| embedding-service disconnects during ingestion | Lower `EMBEDDING_POOL_BATCH_SIZE`. Check the cause with `docker inspect` (OOMKilled) and `dmesg` |
| Search is slow | [OPERATIONS_EN.md](../OPERATIONS_EN.md) §3.4 |
