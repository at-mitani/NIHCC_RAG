# Operations manual
English | [日本語](OPERATIONS.md)

Day-to-day operation, applying updates, and troubleshooting. **For operators.**

For setup see [SETUP_LOCAL_EN.md](../SETUP_LOCAL_EN.md) / [SETUP_EC2_EN.md](../SETUP_EC2_EN.md),
for how to use the UI see [README_EN.md](../README_EN.md), and for the design see [DESIGN_EN.md](DESIGN_EN.md).

---

## 1. Daily operations

### 1.1 Start and stop

```bash
docker compose up -d        # start
docker compose stop         # stop (data is kept)
docker compose ps           # status
docker compose logs -f backend-api
```

### 1.2 Health checks

```bash
docker compose ps           # are all services running / healthy?
```

How to check differs by service.

```bash
# backend-api (reachable from outside)
curl -s localhost:8000/api/v1/health/ready

# roi-service / embedding-service (ports not published)
docker compose exec roi-service python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/ready').read().decode())"
```

> **roi-service and embedding-service cannot be reached from outside.** They live only on
> the `app` network, so `localhost:8000` points to backend-api.

### 1.3 End-to-end smoke test

```bash
# The admin password is in LOCAL_USERS in .env (comma-separated user:password:role)
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```

> **A password is required.** Without it, the script prints
> `--password または SMOKE_PASSWORD が必要です` (--password or SMOKE_PASSWORD is required) and
> stops with exit code 2. You can also pass `--user` / `--password` directly.
> The default user is `admin` and the target is `http://localhost:8000` (change with `BASE_URL`).

Actually runs ingestion and search end to end. A little over 40 checks (the number varies with configuration).
**Run it after every update.**

It really ingests 30 images, so it **adds rows to production data**. If you only want to try it, make
`--limit` small.

---

## 2. Applying updates

```bash
git pull --ff-only origin main
python scripts/sync_env.py          # check the diff
python scripts/sync_env.py --write  # merge into .env
bash scripts/redeploy.sh --yes
python tests/integration/smoke_test.py
```

`redeploy.sh --pull` starts with `git pull` (pulls `origin main`).

What `redeploy.sh` does:

1. Checks that no evaluation is running (stops if one is)
2. Stops ingestion
3. Rebuilds **all images, including `db-migrate`**
4. Applies migrations
5. Starts services and waits for health
6. Detects containers running on old images and replaces them

> ## Do not build selected services
>
> **Migration files and the UI static files are baked into images.**
>
> - `docker compose build` (no arguments) **does not include the `migrate` profile**
> - `up --force-recreate` **does not rebuild images**
>
> If you build only selected services, new migrations may not be applied while the output only says
> "already applied", or an old `app.js` keeps being served and the UI breaks.

### Verifying that the update took effect

```bash
# Were the migrations actually applied?
docker compose exec -T metadata-db sh -c \
  'PGPASSWORD=$(cat /run/secrets/db_password) psql -U postgres -d metadata -tAc \
   "SELECT version FROM schema_migration ORDER BY version DESC LIMIT 3"'

# Is the served UI the new one? (example)
curl -s localhost:8080/app.js | grep -c findingLabel
```

---

## 3. Troubleshooting

### 3.1 ingest-worker does not start with `vector 列の幅が設定と違います` (vector column width differs from the configuration)

**The dimensions in `.env` and the DB column width do not match.** The columns (`embedding` /
`embedding_fine` / `embedding_lung` in `image_embedding`) are fixed to the values at migration time, so this
happens when only `.env` is changed later. At startup, ingest-worker compares the actual column widths with `.env`,
and if they differ it stops without starting ingestion.

```bash
grep -E "^EMBEDDING_(DIMENSION|FINE_BACKEND|FINE_DIMENSION|LUNG_DIMENSION)=" .env

docker compose exec -T vector-coordinator sh -c \
  'PGPASSWORD=$(cat /run/secrets/db_password) psql -U postgres -d vector -tAc \
   "SELECT attname, format_type(atttypid, atttypmod) FROM pg_attribute
     WHERE attrelid='"'"'image_embedding'"'"'::regclass AND attname LIKE '"'"'embedding%'"'"'"'
```

There are two remedies.

| | Setting | Result |
|---|---|---|
| **Recommended** | Match `.env` to the columns (`EMBEDDING_FINE_DIMENSION=15` for `chest-profile` / `none`) | Recovers as is |
| Not recommended | Change the column width | Requires a new `V00x` in `db/vector-migrations` and discards existing vectors |

```bash
docker compose up -d --force-recreate embedding-service ingest-worker
```

### 3.2 Investigating per-image failures

Failure reasons are recorded in two places.

```bash
# Vector write failures
docker compose exec -T metadata-db sh -c \
  'PGPASSWORD=$(cat /run/secrets/db_password) psql -U postgres -d metadata -tAc \
   "SELECT image_id, left(last_error, 200) FROM image_vector_state
     WHERE state = '"'"'FAILED'"'"' ORDER BY updated_at DESC LIMIT 5"'

# Ingestion as a whole
docker compose logs --tail 400 ingest-worker | grep -i "item failed"
```

You can also view them by reason in the UI's Ingestion history (「取り込み履歴」) and download them as CSV.

### 3.3 embedding-service does not start

```bash
docker compose logs --tail 40 embedding-service
```

| Message | Cause |
|---|---|
| `胸部野の分類器がありません` (no chest-field classifier) | `chest-profile` was selected but `infra/chest/head-pool-chest.npz` is missing |
| `胸部野の分類器が前提とする特徴の作り方と、いまの構成が違います` (the current configuration differs from the feature recipe the chest-field classifier assumes) | The RAD-DINO revision, pooling resolution, or mask settings differ from what the classifier recorded ([runbooks/chest-profile-setup_EN.md](runbooks/chest-profile-setup_EN.md) §8) |
| `vector shape (1, 768) != (1, 15)` | Backend and dimension do not match |
| `produces the same model_version as the coarse backend` | Points to the same model as the first stage |

In every case you can start it as single-stage search with `EMBEDDING_FINE_BACKEND=none` (finding estimates are then no longer produced).

> Startup takes about 3 minutes (model loading). `unhealthy` right after startup may simply mean
> you did not wait long enough. If the log shows no exception, wait.

### 3.4 Search returns 503 / 504

| Status | Meaning | Action |
|---|---|---|
| 503 | A dependent service is not responding | Check health with `docker compose ps` |
| 504 | Timeout | CPU contention. Check whether it overlaps with ingestion or training |

By default, image search is an exhaustive search over finding profiles (`IMAGE_SEARCH_STRATEGY=profile`). Measured on EC2,
the vector search p95 is 24 ms, faster than two-stage search (290 ms). If only the exhaustive search is slow,
isolate the problem by setting `IMAGE_SEARCH_STRATEGY=coarse_to_fine` (collect candidates with BiomedCLIP, then re-rank)
and running `docker compose up -d` (no build needed). The value must be either `profile` or `coarse_to_fine`;
a misspelling causes an error on every search (it does not silently fall back to the default).

If two-stage search is slow, review `HNSW_MAX_SCAN_TUPLES` and `HNSW_EF_SEARCH`.

### 3.5 The UI behaves like an old version

```
Cannot read properties of undefined (reading 'length')
```

**The frontend image has not been rebuilt.** Static files are baked into the image.

```bash
bash scripts/redeploy.sh --yes      # rebuild all images; for frontend only, use the next line
docker compose build frontend && docker compose up -d --force-recreate frontend
```

The browser cache may also be stale. Check with a hard reload.

### 3.6 Migration stops with `already applied with a different checksum`

**An applied migration file has been modified.** Applied migrations are pinned by
checksum and changes are rejected. Restore the file and add the change as a new `V00x`.

Only when the SQL content is identical and just the recorded checksum differs can you update the record alone
(it is not re-run).

```bash
docker compose run --rm -e MIGRATION_REPAIR_CHECKSUMS='<version name>.sql' db-migrate
```

---

## 4. Data management

### 4.1 Ingestion

Use the UI (Data ingestion tab, 「データ取り込み」) or a script.

```bash
python scripts/load_initial_images.py
```

First check the reconciliation of the CSV and the images with a dry-run.

### 4.2 Reconciliation (daily)

```bash
docker compose run --rm --no-deps --entrypoint python backend-api /srv/scripts/reconcile.py
```

Detects discrepancies between metadata and vectors and records them in `reconciliation_report`.

### 4.3 Deleting

```bash
python scripts/reset_data.py        # delete ingested data (users are kept)
```

**This cannot be undone.** You are asked to confirm.

### 4.4 Retention of query images

Query images and images overlaid with the chest-field outline are deleted after `QUERY_IMAGE_RETENTION_MINUTES` (default 60 minutes).
**These are patient images, so everything related to query images is deleted with the same deadline.**
A resident job in backend-api deletes them.

---

## 5. Replacing models

### 5.1 Lung-field segmentation model (U-Net)

roi-service segments the lung fields and heart with a U-Net (**required**; the rule-based method does not output the heart).
The procedure is in [runbooks/lung-unet_EN.md](runbooks/lung-unet_EN.md). Only an outline is given here.

1. Ingest reference masks (`scripts/fetch_reference_masks.py`)
2. Build the training data (`scripts/build_lung_dataset.py`)
3. Train (`scripts/train_lung_unet.py`; on GPU, the dedicated image `infra/train/Dockerfile.lung-unet`)
4. **Measure Dice before deploying** (`scripts/eval_roi_dice.py`)
5. Place it in `infra/lung-unet/` and recreate roi-service

> **The compose containers cannot be used for training.** With `read_only: true` they cannot write anywhere except `/tmp`,
> and the run-as user is fixed to uid 10001. Add `--user` and `--shm-size=2g` to
> `docker run`.

Changing the model or `ROI_PROFILE` changes `roi_model_version`, so re-ingestion is required to keep the searchable
data consistent. **The chest-field classifier is paired with the profile used for training**, so if you change the profile,
rebuild the classifier too (§5.2).

### 5.2 Finding profiles (chest-field classifier)

| Task | Procedure |
|---|---|
| Build the classifier (training data, training, evaluation, export) | [runbooks/chest-classifier_EN.md](runbooks/chest-classifier_EN.md) |
| Install the classifier, build, and ingest the full dataset | [runbooks/chest-profile-setup_EN.md](runbooks/chest-profile-setup_EN.md) |
| Measure search quality and the lesion-location map | [runbooks/chest-evaluation_EN.md](runbooks/chest-evaluation_EN.md) |

Three key points:

1. Export the classifier together with the RAD-DINO version (`export_chest_head.py`)
2. Set `EMBEDDING_FINE_BACKEND=chest-profile` and `EMBEDDING_FINE_MODEL_REVISION` in `.env`
3. Before ingesting, confirm that the ingestion path produces the same features as training (`chest_image_eval.py verify-ingest`)

Once installed, image search becomes an exhaustive search over finding profiles. Searches that include text stay a single-stage
BiomedCLIP search (the profile cannot handle text; [MULTIMODAL_EN.md](MULTIMODAL_EN.md) §4.1). If you replace the classifier, the
`embedding_fine` and finding estimates of existing rows come from the old classifier, so re-ingestion is required.

### 5.3 Fetching the embedding models

```bash
bash scripts/fetch_model.sh
```

In a closed network, fetch them on a machine with connectivity and bring in the whole model volume.

---

## 6. Recording evaluation results

### 6.1 Why keep them in the DB

`tmp/` is in `.gitignore`, so results left in files never reach your local machine from EC2,
you would not notice if they were deleted, and you cannot query "how does this compare with the previous version".
Evaluation results are **not reproducible** (the same run can never be repeated) yet **small**, so they go into the DB.

### 6.2 What goes into the DB and what does not

| Kind | Example | Size | Location | Reason |
|---|---|---|---|---|
| **Measurements** | Dice, AUC, breakdown rate | 1.7 MB | **DB** | Not reproducible. Used for comparison |
| Model artifacts | `.onnx` `.npz` | 7–15 MB | Files + **location and SHA in DB** | Containers read them via read-only mounts |
| Intermediate caches | Vector `.npz` | 58 MB | Files | NumPy arrays. Worthless once the version changes |
| Training data | `data/lung-dataset/` | 539 MB | Files | Deterministically reproducible from the source images |

**Artifacts themselves are not stored in the DB.** Containers read them via read-only mounts, so storing them in the DB
would require extracting and writing them to files on every startup, which does not fit with `read_only: true`.

### 6.3 Recording an evaluation

**Run the evaluation scripts on the host.** They cannot complete inside any single container.

| | `cv2` (ROI extraction) | DB access |
|---|---|---|
| roi-service | Yes | **No** |
| backend-api / ingest-worker | **No** | Yes |

So the host needs to reach the DB. The evaluation overlay
(`infra/compose/docker-compose.eval.yml`) publishes `metadata-db` **on loopback only**,
so add it and start.

```bash
# .env
COMPOSE_FILE=docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.eval.yml
COMPOSE_PATH_SEPARATOR=:
```

```bash
docker compose up -d metadata-db

METADATA_DB_HOST=127.0.0.1 METADATA_DB_PORT=55432 \
POSTGRES_PASSWORD="$(cat secrets/app_db_password.txt)" \
EVAL_GIT_REVISION="$(git rev-parse --short HEAD)" \
LUNG_MODEL_PATH="$PWD/infra/lung-unet/lung-unet.onnx" \
  python scripts/eval_roi_dice.py --limit 1000 --profile nihcc-unet-local-v1 --note "<note>"
```

`EVAL_RAN_ON` overrides the location (`ec2` / `local`).

Three evaluations support recording.

| script | `kind` | Main metrics |
|---|---|---|
| `eval_roi_dice.py` | `roi-dice` | `dice_median` / `breakdown_rate` |
| `region_gold_eval.py` | `gold-box` | `gold_box_accuracy` |
| `region_signal_eval.py` | `region-signal` | `test_a_lift` / `test_b_lift` / `test_c_lift` |

`region_signal_eval.py` is recorded as **one run per encoder**. Combining them would mean
embedding the encoder into the metric names, and you could no longer filter by condition.
`deterministic` is the null baseline, so its value goes into `baseline` for the other encoders' metrics.

> **Publish the port only while evaluating.** By design, internal services stay inside the Docker network.
> When done, remove it from `COMPOSE_FILE` and restart.
> Do not open this port externally in the security group.

> **Recording is auxiliary.** If the DB cannot be reached, it prints a warning and gives up without stopping the evaluation itself.
> It explicitly says "evaluation succeeded, only recording failed", so there is no confusion.

### 6.4 Viewing results

```sql
-- trend per version
SELECT r.target_version, r.ran_on, r.sample_size,
       max(CASE WHEN m.name = 'dice_median' THEN m.value END)    AS dice_median,
       max(CASE WHEN m.name = 'breakdown_rate' THEN m.value END) AS breakdown
  FROM evaluation_run r JOIN evaluation_metric m USING (run_id)
 WHERE r.kind = 'roi-dice'
 GROUP BY r.run_id, r.target_version, r.ran_on, r.sample_size, r.started_at
 ORDER BY r.started_at DESC;
```

`evaluation_metric.baseline` holds **the null-baseline value**. The column exists so that nothing is claimed to
"work" without measuring the null baseline. The ratio to the null baseline is computed from it.

```sql
-- per finding, how many times the null baseline was exceeded
-- (帰無比 = ratio to null baseline)
SELECT r.params->>'encoder' AS encoder, m.finding, m.region_code,
       round((m.value / NULLIF(m.baseline, 0))::numeric, 2) AS 帰無比
  FROM evaluation_run r JOIN evaluation_metric m USING (run_id)
 WHERE r.kind = 'region-signal' AND m.name = 'test_b_lift' AND m.baseline IS NOT NULL
 ORDER BY 4 DESC NULLS LAST LIMIT 20;
```

### 6.5 Registering model artifacts

```bash
METADATA_DB_HOST=127.0.0.1 METADATA_DB_PORT=55432 \
POSTGRES_PASSWORD="$(cat secrets/app_db_password.txt)" \
  python scripts/register_model_artifact.py \
  --type LUNG_SEGMENTATION --path infra/lung-unet/lung-unet.onnx \
  --version unet-lung-spine-local-v2.0.0 --activate
```

Match `--version` to the profile's `algorithm_version` (`unet-lung-spine-local-v2.0.0` for
`nihcc-unet-local-v1`).

As in §6.3, publish `metadata-db` with the evaluation overlay beforehand.

The location and SHA-256, and for ONNX the input shape as well, go into `model_version`. Without this, you would not notice
if different artifacts were running on EC2 and locally (`roi_model_version` is the profile version and
does not reflect the contents of the `.onnx`).

---

## 7. Script list

### Operations

| Script | Purpose |
|---|---|
| `bootstrap.sh` | Generate secrets and `.env` (**once, first**) |
| `redeploy.sh` | Apply updates (**use this**) |
| `sync_env.py` | Merge the diff of `.env.example` into `.env` |
| `fetch_model.sh` | Fetch the embedding models |
| `load_initial_users.py` | Register the initial users |
| `load_initial_images.py` | Ingest images |
| `reset_data.py` | Delete ingested data |
| `upgrade_postgres.sh` | PostgreSQL major version upgrade (§8) |

### Run from a container

Included in the `backend-api` image (`/srv/scripts/`).

| Script | Purpose |
|---|---|
| `reconcile.py` | Reconciliation of metadata and vectors (§4.2) |

### Lung-field segmentation (U-Net)

The procedure is in [runbooks/lung-unet_EN.md](runbooks/lung-unet_EN.md).

| Script | Purpose |
|---|---|
| `fetch_reference_masks.py` | Ingest reference masks from CheXmask |
| `build_lung_dataset.py` | Build the training data |
| `train_lung_unet.py` | Train and export ONNX |
| `eval_roi_dice.py` | Measure segmentation accuracy by Dice / IoU against the reference masks |
| `segment_lungs.py` | Generate masks in bulk (optional) |
| `register_model_artifact.py` | Record the location and SHA-256 of model artifacts in the DB (§6.5) |

### Chest-field classifier and finding profiles

| Script | Purpose |
|---|---|
| `chest_image_eval.py` | Generate chest-field features, train and evaluate the classifier (`coverage` / `sheet` / `vectors` / `train`), verify features from the ingestion path (`verify-ingest`) |
| `run_chest_eval.sh` | Run `coverage` → `sheet` → `vectors` → `train` of `chest_image_eval.py` in one go (procedure in [runbooks/chest-classifier_EN.md](runbooks/chest-classifier_EN.md)) |
| `export_chest_head.py` | Export the trained classifier, together with its feature recipe, as a production artifact (same as above) |
| `chest_search_eval.py` | Measure search quality (Test B) on the ingested DB via the actual search path. Compares exhaustive search over finding profiles, two-stage search, and first stage only (procedure in [runbooks/chest-evaluation_EN.md](runbooks/chest-evaluation_EN.md)) |
| `chest_lesion_map_eval.py` | Measure with ground-truth boxes whether the classifier's per-cell map indicates lesion location and size (same as above) |

The procedure for building and ingesting the full dataset is in [runbooks/chest-profile-setup_EN.md](runbooks/chest-profile-setup_EN.md).

### Per-region signal evaluation

Evaluation tools that measure finding signals with a representation splitting the lung fields into regions. Not used in the ingestion or search paths.
`region_signal_eval.py` and `region_mil_stage_a.py` are imported by `chest_image_eval.py` as shared components for the ROI path and
classifier training.

| Script | Purpose |
|---|---|
| `region_signal_eval.py` | Measure whether region vectors encode findings by comparison with base rates (Test A / B / C). Caches vectors in `tmp/region-signal/` |
| `region_gold_eval.py` | Measure with ground-truth boxes whether the region of a finding can be identified per image |
| `region_contrast_eval.py` | Measure whether the region with a finding can be identified by contrast with normal images of the same patient |
| `region_mil_stage_a.py` | Train a MIL head on frozen features and measure image-level AUC and region accuracy |
| `region_mil_profile.py` | Export the MIL head's per-region finding profiles as search vectors so they can be measured with `region_signal_eval.py` |

### Development

| Script | Purpose |
|---|---|
| `try_roi.py` | Quick visual check of ROI extraction |

> **Do not keep unused scripts.** Anything not listed here was forgotten when it should have been deleted.
> Evaluation tools that are reusable as methods are kept, with their purpose documented.

---

## 8. PostgreSQL version upgrade

**Data files are not compatible across major versions.** A PG16 volume cannot be read by a
PG17 container, so dump and restore are required.

### 8.1 How far can we upgrade

**Citus sets the upper limit.** The official image has no PG18 version yet
(as of 2026-09, `citusdata/citus:*-pg18` does not exist). pgvector itself already supports PG18,
and its source even contains branches for PG19, so **upgrading to PG18 has to wait until
Citus catches up**.

### 8.2 Procedure

The version is set in one place, `PG_MAJOR` in `.env`. The metadata DB and the vector cluster use the same version.

```bash
# 0. Pull the update (on EC2)
git pull --ff-only origin main
python3 scripts/sync_env.py --write      # PG_MAJOR etc. into .env

# 1. Back up the current data (to tmp/pg-upgrade/)
bash scripts/upgrade_postgres.sh dump

# 2. Change the version in .env (not needed if sync_env.py already added it)
#    PG_MAJOR=17

# 3. Recreate with the new version (**deletes volumes**)
bash scripts/upgrade_postgres.sh recreate

# 4. Restore the backed-up data
bash scripts/upgrade_postgres.sh restore
```

`recreate` deletes **only the 7 DB volumes** (metadata + the vector cluster's
1 coordinator + 5 workers). Model cache, Grafana, Prometheus, and the queue are kept.
If any volume is missed, it stops on the spot. **If you start with a volume left over, it crashes with
`database files are incompatible with server`.**

`restore` stops the apps, reloads the data, **re-grants privileges**, compares row counts,
and brings the apps back (→ §10.4). Finally it shows the actual version.

```bash
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```

### 8.3 What to check after upgrading

Matching row counts is not enough. **Check distribution and internal connections.**

```bash
PGPW="$(cat secrets/db_password.txt)"

# 1. Are there 30 shards spread over 5 workers? (shard数 = shard count)
docker compose exec -T -e PGPASSWORD="$PGPW" vector-coordinator \
  psql -U postgres -d vector -c \
  "SELECT nodename, count(*) AS shard数 FROM citus_shards
     WHERE table_name = 'image_embedding'::regclass GROUP BY 1 ORDER BY 1"

# 2. **Do distributed queries work with the application role?** (a superuser would not reveal the problem)
docker compose exec -T vector-coordinator \
  psql "host=vector-coordinator user=app_ingest dbname=vector" \
  -tAc "SELECT count(*) FROM image_embedding"

unset PGPW
```

> ## Do not substitute a superuser for check 2
>
> Internal Coordinator → Worker connections are made **with the role running the query**.
> Counting as `postgres` may succeed while `app_ingest` fails.
> You would only find out when every ingestion fails.
>
> If it fails, first suspect leftovers from the initial startup.
>
> ```bash
> docker compose exec -T vector-coordinator sh -c \
>   "tr '\0' '\n' < /proc/1/environ | grep -c '^PGPASSWORD='"
> ```
>
> If it prints `1`, that postmaster has `PGPASSWORD` and **sends the postgres password
> for every role**. Check whether the image includes the fix
> (`infra/postgres/zzz-unset-pgpassword.sh`), and if it does, replace the container with
> `docker compose up -d`.

### 8.4 Why there are two ways of handling

| | What is dumped | How it is restored |
|---|---|---|
| metadata | **Schema + data** | Loaded as is |
| Vector cluster | **Data only** | Loaded after migrations create the schema |

The schema of the cluster is not dumped because **distributed tables are only distributed when created through
`create_distributed_table`**. `pg_dump` does not output that call,
so the schema is created by migrations.

### 8.5 Duration and cautions

Estimates on a local machine (4,999 images / 4.3 GB dump).

| Step | Time |
|---|---|
| `dump` | About 10 minutes (mostly metadata) |
| `recreate` | About 4 minutes (build + startup + migration) |
| `restore` | About 5 minutes |

For EC2 (110,000 images / dump about 52 GB), expect **ten-plus times longer**. Check free disk space at the backup destination
**beforehand**. The dump is about as large as the original DB.

> **Run it locally first.** EC2 holds 110,000 images, and a failure means redoing everything
> from ingestion.

---

## 9. Monitoring

Use Grafana (`localhost:3000`). Main metrics:

| Metric | What to look for |
|---|---|
| `ingest_stage_duration_seconds{stage="roi"}` | Whether ROI has become the bottleneck |
| Vector write success/failure | Whether failures are increasing |
| Search response time | When slow, use asynchronous execution from the UI (`Prefer: respond-async`) |

If `roi` grows, increase `ROI_SERVICE_REPLICAS`.

---

## 10. Backup and restore

### 10.1 What to save

| Target | Method | Size (110,000 images) |
|---|---|---|
| metadata | `pg_dump` (schema + data) | About 47 GB |
| Vector cluster | **Data-only** `pg_dump` (see below) | About 3 GB (estimate) |
| Model artifacts | Save `infra/chest/` `infra/lung-unet/` as is | A few MB |
| Configuration | `.env` and `secrets/` | Small |

> **`secrets/` contains secret information.** Do not put it in git, and restrict where backups are stored.

The source images are in `image_binary`, so they are included in the metadata backup.
**This accounts for most of the size (44 GB).** If you keep the dataset itself separately, it is lighter to exclude it with
`--exclude-table=image_binary` and re-ingest.

### 10.2 Citus distributed tables are handled differently

**This is the key point.** `image_embedding` in the vector cluster is a Citus distributed table,
and its actual data is spread across shards on 5 workers.

```
  vector-coordinator (table definitions and distribution metadata)
      └→ vector-worker-1 … vector-worker-5 (actual data spread over 30 shards)
```

| | Possible / not possible |
|---|---|
| `pg_dump` on the coordinator to **get the data** | **Possible.** It collects from all workers and outputs it |
| Restore that schema dump to **reproduce the distribution** | **Not possible** |

`pg_dump` outputs `CREATE TABLE`, but **does not output the `create_distributed_table()` call**.
Restoring it as is only creates one ordinary, non-distributed table on the coordinator,
and the workers stay empty. Search may work, but without parallelism and without shards.

**So the schema is created by migrations.**

| | What is dumped | Restore order |
|---|---|---|
| metadata | Schema + data | Loaded as is |
| Vector cluster | **Data only** (`--data-only --table=image_embedding`) | Migrations → data |

The end of the migration (`db/vector-migrations/V001`) calls `create_distributed_table`.
**Distributing the empty table first** and then loading data spreads it correctly across the shards.

> Do not dump the workers directly. Shards are spread under names like `image_embedding_102008`,
> and collecting them individually cannot be restored. **Always go through the coordinator.**

### 10.3 Taking a backup

`scripts/upgrade_postgres.sh dump` performs exactly this procedure. It is not specific to version upgrades,
so **it can also be used for regular backups**.

```bash
DUMP_DIR=/backup/$(date +%F) bash scripts/upgrade_postgres.sh dump
```

The output is as follows. `counts.txt` contains row counts, **used for comparison when restoring**.

```
metadata.sql   schema + data
vector.sql     data only
counts.txt     row count of each table
```

### 10.4 Restoring

**This assumes restoring into an empty environment.** The schema must be created first.

```bash
docker compose up -d
docker compose run --rm db-migrate      # create the schema and distribute it

DUMP_DIR=/backup/<date> bash scripts/upgrade_postgres.sh restore
```

What `restore` does:

1. Stops the apps (**metadata cannot be DROPped while connections are open**)
2. Recreates metadata and loads it
3. Loads data into the vector cluster
4. **Re-runs `db-migrate` to re-grant privileges**
5. Compares row counts with `counts.txt` (stops if they do not match)
6. Brings the apps back

> ## Privileges are not in the dump
>
> The dump is taken with `--no-privileges`. **Privileges are defined in one place, the migration's
> `grant_schema`**; including them in the dump would create a second definition.
>
> When metadata is DROPped, the GRANTs to `app_meta` disappear too, so
> **restoring alone is not enough**. Without re-granting, all services stay healthy
> while **only login returns 500**.
>
> ```
> psycopg.errors.InsufficientPrivilege: permission denied for table app_user
> ```
>
> `restore` re-runs it automatically in step 4. If you restore by hand, do not forget it.
> The migrations themselves check `schema_migration` (included in the dump) and skip,
> so they do not touch the schema.

To recount row counts only, use `verify`. **Restoring takes hours on EC2, so do not redo it
just to check.**

```bash
DUMP_DIR=/backup/<date> bash scripts/upgrade_postgres.sh verify
```

### 10.5 Verifying

Do not stop at restoring. **Check that distribution is in effect.**

```bash
PGPW="$(cat secrets/db_password.txt)"
# テーブル = table, shard数 = shard count
docker compose exec -T -e PGPASSWORD="$PGPW" vector-coordinator \
  psql -U postgres -d vector -c \
  "SELECT logicalrelid AS テーブル, count(*) AS shard数
     FROM pg_dist_shard GROUP BY 1"
unset PGPW
```

If `image_embedding` shows `30`, it is distributed. **If there are 0 rows, distribution failed**
(for example, when the schema was restored without going through migrations).

Finally, run an end-to-end check.

```bash
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```
