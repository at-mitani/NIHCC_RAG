# Building the chest field classifier (training data, training, evaluation, export)

English | [日本語](chest-classifier.md)

The runbook for building the classifier `infra/chest/head-pool-chest.npz` used by the finding profile
(`EMBEDDING_FINE_BACKEND=chest-profile`). It builds chest field features
(lung fields + mediastinum + diaphragm) from all 112,120 images, trains a classifier on image-level labels (14 findings),
and exports it as the production artifact.

The procedure for putting the built classifier into production is [chest-profile-setup_EN.md](chest-profile-setup_EN.md).

- Where to run: EC2 (g5.2xlarge; features are sent to the GPU embedding-service)
- Time required: half a day (generating features for the full dataset takes 3–5 hours)
- The DB is not used

---

## 0. What gets built

ChestX-ray14 labels are per image. The search unit is aligned to the chest field of one image,
so those labels serve directly as training data.

`scripts/run_chest_eval.sh` runs the following end to end (internally, subcommands of `scripts/chest_image_eval.py`).

| Step | Subcommand | Output (`tmp/chest/`) |
|---|---|---|
| 1 | `coverage` | `coverage.json` (fraction of ground-truth boxes falling inside the lung fields / chest field) |
| 2 | `sheet` | `sheet.png` (overview of 64 random images with outlines overlaid) |
| 3 | `vectors` | `vec_<representation>_<profile>_<count>.npz`, `shards/` (features; resumable) |
| 4 | `train` | `train.json` (AUC, Test B), `head_<representation>_<profile>.npz` (classifier), `ec2-full/teacher_<profile>_<count>.npz` (training data and split) |

The following 5 representations are compared. Production uses `pool-chest`.

| Name | Contents | Role |
|---|---|---|
| `pool-chest` | RAD-DINO patch tokens pooled with the chest field mask | **Production** |
| `pool-lung` | Pooled with the lung field mask only | Control that excludes the heart |
| `pool-whole` | Pooled over the whole image | Control without narrowing the frame |
| `crop-chest` | RAD-DINO on a crop with everything outside the chest field blacked out | Control that physically erases outside the frame |
| `deterministic` | Image statistics of the chest field crop | **Null baseline** (no medical meaning) |

Training and evaluation split by patient (train / val / test). `ec2-full/teacher_*.npz` is
used by the search quality measurement ([chest-evaluation_EN.md](chest-evaluation_EN.md) §1) to pick queries from test patients,
so it is tracked in git.

---

## 1. Prerequisites

### 1.1 Virtual environment for evaluation

The scripts run on the host (ROI extraction uses cv2 and the U-Net; only the embeddings are sent over HTTP to
embedding-service).

```bash
cd ~/NIHCC_RAG
python3 -m venv ~/.venv-eval && source ~/.venv-eval/bin/activate
pip install numpy httpx opencv-python-headless pyyaml pillow onnxruntime
```

`run_chest_eval.sh` first checks that these can be imported and stops if any are missing.
**A process started with nohup inherits the environment of the shell that started it**, so activate the venv
in the same shell before running.

### 1.2 U-Net and ground-truth boxes

```bash
ls infra/lung-unet/lung-unet.onnx              # if missing, see lung-unet.md
ls tmp/gold/BBox_List_2017.csv                 # tracked in git
```

The default profile is `nihcc-unet-local-v1` (160px). When using the 384px model, add
`CHEST_PROFILE=nihcc-unet-v1` to the subsequent commands. **The classifier is paired with the profile used for training**,
so set production `ROI_PROFILE` to the same.

The ground-truth boxes are for validation only and are not used for training. Where to get them if deleted:

```bash
mkdir -p tmp/gold && curl -sfL -o tmp/gold/BBox_List_2017.csv \
  https://raw.githubusercontent.com/TRKuan/cxr8/master/dataset/BBox_List_2017.csv
```

### 1.3 Disk

The feature cache and shards use about 3 GB (it stops if less than 8 GB is free).

---

## 2. Starting embedding-service for feature generation

Run RAD-DINO as the second stage and expose it to the host. **With this `.env`, do not start anything
other than embedding-service.** `EMBEDDING_FINE_DIMENSION=768` does not match the DB column width (15), so
ingest-worker stops at startup, and running `db-migrate` would create the column with width 768.

```bash
cp .env tmp/env.before-chest-classifier          # restored in §7

setenv() { if grep -q "^$1=" .env; then sed -i "s#^$1=.*#$1=$2#" .env; else echo "$1=$2" >> .env; fi; }
setenv COMPOSE_FILE docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.gpu.yml:infra/compose/docker-compose.eval.yml
setenv COMPOSE_PATH_SEPARATOR :
setenv EMBEDDING_FINE_BACKEND rad-dino
setenv EMBEDDING_FINE_DIMENSION 768
setenv EMBEDDING_FINE_MODEL_REVISION <RAD-DINO revision (40 chars)>   # §2.1

docker compose build embedding-service
docker compose up -d --no-deps --force-recreate embedding-service
```

Check that pooling is available.

```bash
curl -s http://127.0.0.1:8002/api/v1/model | python3 -m json.tool \
  | grep -E 'region_pool|fine_embedding_model_revision'
# "fine_embedding_model_revision": "<40 chars>",
# "region_pool_backend": "rad-dino",
# "region_pool_patch_grid": 37,
```

### 2.1 Pinning the RAD-DINO version

**The classifier is trained on RAD-DINO features, so if the RAD-DINO weights change, the classifier silently breaks.**
Pin the revision before building features, and use the same value for the export (§6) and for production
`EMBEDDING_FINE_MODEL_REVISION`. `export_chest_head.py` does not export without a revision,
and the production embedding-service refuses to start if its configuration differs from the artifact's revision.

If you have not decided which revision to use, start once with an empty revision (the model is
downloaded into the cache), then read it from the cache. The cache is in the Docker volume
`nihcc-rag_model-cache` (`/models/hf-cache` inside the container).

```bash
docker run --rm --entrypoint sh -v nihcc-rag_model-cache:/m:ro \
  local/nihcc-embedding-service:1.0.0 -c '
for repo in models--microsoft--rad-dino models--microsoft--BiomedCLIP-PubMedBERT_256-vit_base_patch16_224; do
  echo "== $repo"
  echo "refs/main: $(cat /m/hub/$repo/refs/main 2>/dev/null || echo なし)"   # なし = none
  ls -l /m/hub/$repo/snapshots/
  for f in /m/hub/$repo/snapshots/*/*.safetensors /m/hub/$repo/snapshots/*/*.bin; do
    [ -e "$f" ] && sha256sum "$(readlink -f "$f")" | sed "s#  .*#  ${f#/m/hub/}#"
  done
done' | tee tmp/chest/model-revisions.txt
```

- Write the 40-character value of `refs/main` into `EMBEDDING_FINE_MODEL_REVISION` and recreate embedding-service
- The SHA-256 of the weight files is also recorded, so you can verify by content, not only by version value
- Pin BiomedCLIP the same way with `EMBEDDING_MODEL_REVISION`. The classifier and search measurements in this repository
  were taken with `9f341de24bfb00180f1b847274256e9b65a3a32e`

---

## 3. Calibration (1,000 images first)

```bash
mkdir -p tmp/chest
LIMIT=1000 bash scripts/run_chest_eval.sh 2>&1 | tee tmp/chest/calib.log
```

What to check:

- All preflight checks pass (it prints `胸部野の pooling に対応` (chest field pooling supported))
- `分割の失敗` (segmentation failures) is within a few %
- Speed (`枚/秒`, images/s). Time for the full dataset is 112,120 ÷ this value
- Look at `tmp/chest/sheet.png` by eye. **Blue = lung fields, orange = chest field**. The orange should include the mediastinum, cardiac shadow,
  and diaphragm domes, and should not extend deep into the neck or abdomen

Do not read the AUC from 1,000 images; there are too few positives per finding.

---

## 4. Full dataset

```bash
nohup bash scripts/run_chest_eval.sh > tmp/chest/run.log 2>&1 &
tail -f tmp/chest/run.log
```

- **It can resume midway.** Features are written to `tmp/chest/shards/` every 2,000 images. If it crashes,
  fix the cause and rerun the same command; it continues from where it left off
- If embedding-service disconnects, it waits for the reload (1–2 minutes) and resends. If this repeats,
  lower `EMBEDDING_POOL_BATCH_SIZE` (check the cause with OOMKilled in `docker inspect` and `dmesg`)
- Speed on EC2 (g5.2xlarge) is 9.2–9.6 images/s. The bottleneck is the U-Net and preprocessing (CPU)

---

## 5. Checking the results

`tmp/chest/train.json` contains the AUC per representation (per finding, mean, 95% interval) and Test B.
The values from the run that built this repository's classifier are shown as a reference.

| Item | Value |
|---|---|
| Images successfully segmented | 112,066 / 112,120 (all 54 failures are `LUNG_TOO_SMALL`) |
| Patient-level split | train 77,648 / val 11,435 / test 22,983 images |
| Ground-truth boxes with 50% or more inside the chest field | 99.7% (58.7% for lung fields only) |

Image-level AUC on test (mean and 95% interval):

| `pool-lung` | **`pool-chest`** | `pool-whole` | `crop-chest` | `deterministic` (null baseline) |
|---:|---:|---:|---:|---:|
| 0.829 [0.821, 0.835] | **0.833 [0.824, 0.839]** | 0.826 [0.816, 0.832] | 0.815 [0.805, 0.821] | 0.696 [0.686, 0.705] |

Per-finding AUC for `pool-chest` ranges from 0.70 (Infiltration) to 0.905 (Cardiomegaly). Test B
(lift of the top 20) is 1.33 or higher for all 14 findings.

Notes when reading:

- **The null baseline is high.** Image statistics alone reach AUC 0.696. This is because acquisition conditions (portable radiography,
  AP views, etc.) correlate with findings; the difference from the null baseline is what was gained from image content
- If the null baseline approaches the main representation, suspect feature generation (U-Net, RAD-DINO version, pooling)

---

## 6. Exporting

Export the `pool-chest` classifier together with **how the features were built** (backbone revision, pooling resolution, how the chest field
mask is built, ROI profile, split random seed). embedding-service compares this with its own configuration at startup
and refuses to start if they differ. The artifact is not committed to git.

```bash
python scripts/export_chest_head.py \
  --head tmp/chest/head_pool-chest_nihcc-unet-local-v1.npz \
  --train-json tmp/chest/train.json \
  --backbone-revision <RAD-DINO revision from §2.1> \
  --output infra/chest/head-pool-chest.npz
# should print 「test 平均 AUC 0.8328」 (test mean AUC 0.8328) (for this repository's classifier; equals the value in train.json)
```

If you trained with the 384px profile, add `--roi-profile nihcc-unet-v1`.

The check that the production ingestion path builds the same features ([chest-profile-setup_EN.md](chest-profile-setup_EN.md) §4)
uses the `tmp/chest/vec_pool-chest_*.npz` built here. Keep it; do not delete it.

---

## 7. Restoring

```bash
cp tmp/env.before-chest-classifier .env
docker compose up -d --force-recreate embedding-service
```

If you go on to enable the finding profile, proceed to [chest-profile-setup_EN.md](chest-profile-setup_EN.md).
Check that the evaluation override (`docker-compose.eval.yml`) has been removed (the policy is that internal services stay
closed within the Docker network).
