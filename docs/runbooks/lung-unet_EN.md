# Training and deploying the lung field segmentation model (U-Net)

English | [日本語](lung-unet.md)

roi-service segments the lung fields and heart with a U-Net trained on CheXmask, and builds the chest field
(lung fields + mediastinum + diaphragm). This document is the runbook for training and deploying that model.

**The U-Net is required.** The chest field needs a heart mask, and the rule-based profile
(`nihcc-classical-v1`) does not produce the heart. If `ROI_PROFILE` is rule-based, roi-service
stops at startup. `nihcc-classical-v1` is used only as the baseline for Dice comparison (§5).

The same procedure works locally (about 5,000 images) and on EC2 (about 110,000 images). The only differences are bandwidth and
whether a GPU is available, and these are absorbed by command arguments.

## Profiles and models

**Models with different input resolutions are not compatible**, so they have separate profiles.

| profile | How lung fields are obtained | Input | Use |
|---|---|---|---|
| `nihcc-unet-local-v1` | U-Net (reduced version trainable on CPU) | 160px | **Default** (`.env.example` / `docker-compose.yml`) |
| `nihcc-unet-v1` | U-Net (trained on GPU) | 384px | High-resolution version |
| `nihcc-classical-v1` | Rule-based | - | Dice comparison baseline only |

**The chest field classifier is tied to the profile.** The classifier ([chest-classifier_EN.md](chest-classifier_EN.md)) was
trained on the chest field of `nihcc-unet-local-v1`, and ingest-worker checks this match at startup.
If you change the profile, retrain the classifier as well and re-ingest the full dataset.

Measured with the default 160px model on 1,000 random images: median Dice 0.968, failures 0.5%.

## Python to use

Every `python` in this document is the **project's development environment** (`.venv-dev`).
The bare `python` has neither cv2 nor numpy, so it fails with `ModuleNotFoundError`.

```powershell
# PowerShell
.\.venv-dev\Scripts\Activate.ps1
```

```bash
# bash / EC2
source .venv-dev/bin/activate      # on EC2, adjust to your environment
```

Only GPU training (§3) runs inside a container, so it is unrelated to this development environment.

## Overview

| | Step | What it does | Local | EC2 |
|---|---|---|---|---|
| 1 | [Fetching reference masks](#1-fetching-reference-masks) | Get ground-truth masks from CheXmask | 8 min | 10–40 min |
| 2 | [Building the training dataset](#2-building-the-training-dataset) | Images and masks into arrays | 5 min | 20 min |
| 3 | [Training](#3-training) | Train the U-Net and export ONNX | 50 min (160px) | 30–60 min (384px) |
| 4 | [Deploying the model](#4-deploying-the-model) | Place it in `infra/lung-unet/` | 1 min | 1 min |
| 5 | [Checking accuracy](#5-checking-accuracy) | Measure with Dice | 10 min | 10 min |
| 6 | [Applying to roi-service](#6-applying-to-roi-service) | Start it and check the version | 5 min | 5 min |
| 7 | [Batch mask generation](#7-batch-mask-generation-optional) | Output masks separately from ingestion (optional) | 20 min | 3 hours |

For inference (§4 onward), onnxruntime on CPU is enough in both cases.

---

## 1. Fetching reference masks

The source is **CheXmask Database v1.0.0** (PhysioNet, CC BY 4.0, no credentialing required). ChestX-ray8,
i.e. the very images this system uses, comes with left lung, right lung, and heart masks.

### Local (about 5,000 images)

The distributed CSV is 2.2 GB, but images with nearby IDs are laid out in large blocks. The local
corpus (`images_001`, `00000001`–`00001335`) is a narrow band, so fetching only the relevant block is enough.

```bash
python scripts/fetch_reference_masks.py --from-mb 1480 --to-mb 1600
```

If your corpus has a different ID band, first find its position.

```bash
python scripts/fetch_reference_masks.py --probe
```

### EC2 (about 110,000 images)

```bash
python scripts/fetch_reference_masks.py --all
```

Both write to `data/reference-masks/chestxray8.csv`. Only rows for images you have locally
are kept, so for 110,000 images it is about 2 GB.

---

## 2. Building the training dataset

```bash
# default 160px (nihcc-unet-local-v1)
python scripts/build_lung_dataset.py --max 1200 --size 160 --out data/lung-dataset-160

# 384px (nihcc-unet-v1)
python scripts/build_lung_dataset.py --max 8000 --size 384
```

The 384px output goes to the default `data/lung-dataset/`. There are three key points.

- **Preprocessing goes through the same `normalize` as inference.** If they differ, the input distribution differs between training and inference
- **The split is by patient.** If images of the same patient span the training and validation sides, the results look better than they really are
- **Filter by reference quality.** The default is Dice RCA 0.8 or higher (stricter than the 0.7 used for evaluation)

`--max` is the upper limit on the number of images. Lung field segmentation is an easy task, so 8,000 images are enough.
Using all 110,000 images barely changes Dice, and building the dataset and training take more than 10 times as long.

---

## 3. Training

### 160px (about 50 minutes on CPU)

The development environment includes torch (CPU), so it runs as is.

```bash
python scripts/train_lung_unet.py --dataset data/lung-dataset-160 --epochs 15 --batch 8 --base 16 --workers 0
```

Measured on CPU (8 threads, base=16):

| Resolution | Per image | 1 epoch with 1,000 images |
|---|---|---|
| 128px | 0.14 s | 2.3 min |
| 160px | 0.21 s | 3.5 min |
| 256px | 0.55 s | 9.2 min |

Raising `base` from 16 to 24 makes it 1.6 times slower. Training at 256px or above on CPU is not practical.

### 384px (30–60 minutes on GPU)

**Build a training-only image.** The base is embedding-service, with `onnx` / `onnxscript` /
`opencv-python-headless` added. On EC2, `TORCH_INDEX_URL` is cu121, so the base
torch is the CUDA build. torch is not reinstalled, so the versions do not drift between training and inference.

```bash
docker compose build embedding-service        # base (if not built yet)
docker build -f infra/train/Dockerfile.lung-unet -t local/nihcc-lung-unet-train:1.0.0 .
```

```bash
mkdir -p models/lung-unet
docker run --rm --gpus all --shm-size=2g \
  -v "$PWD:/work" -w /work \
  --user "$(id -u):$(id -g)" \
  local/nihcc-lung-unet-train:1.0.0 \
  python -u scripts/train_lung_unet.py --epochs 40 --batch 16
```

- **`--shm-size` is required.** DataLoader workers pass tensors through shared memory, but Docker's
  `/dev/shm` is only 64 MB by default, which is not enough for 384px with batch 16
- **`docker compose run` cannot be used.** The operational containers are `read_only: true`, cannot write anywhere
  but `/tmp`, and the run user is fixed at uid 10001. Passing the host uid to `docker run` with `--user`
  makes the output owned by the host user
- Check that the GPU is usable with `docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi`

To check only the plumbing, add `--smoke` (2 epochs, 16 images, tens of seconds).

### Output

Written to `models/lung-unet/` (can be changed with `--out`).

| File | Contents |
|---|---|
| `lung-unet.onnx` | Used for inference. **A single file including the weights** |
| `best.pt` | The weights with the best validation Dice. For further training |
| `metrics.json` | Validation Dice per epoch (right lung, left lung, heart) |

**The number to watch is `val_lung_dice`** (average of right and left lungs). It normally rises to around 0.95;
if it is below 0.90, suspect the training data or preprocessing.

---

## 4. Deploying the model

compose mounts `${LUNG_MODEL_DIR:-./infra/lung-unet}` read-only at `/models/lung-unet`.
The model is not committed to git.

```bash
# bash / EC2
cp models/lung-unet/lung-unet.onnx infra/lung-unet/
sha256sum infra/lung-unet/lung-unet.onnx
```

```powershell
# PowerShell
Copy-Item models\lung-unet\lung-unet.onnx infra\lung-unet\
Get-FileHash infra\lung-unet\lung-unet.onnx -Algorithm SHA256
```

When bringing in a model trained in another environment, note the hash and compare it at the destination.
To record the location and SHA-256 in the DB, use `register_model_artifact.py` ([OPERATIONS_EN.md](../OPERATIONS_EN.md) §6.5).

---

## 5. Checking accuracy

**Measure before deploying.** Use the reference masks fetched in §1.

Run it on the host. The `model_path` held by the profile is a path inside the container (`/models/lung-unet/...`),
so **tell it the location with `LUNG_MODEL_PATH`**. If you forget, it stops with `分割モデルが無い` (segmentation model not found).

```bash
# bash / EC2
python scripts/eval_roi_dice.py --limit 500 --profile nihcc-classical-v1 \
    --json tmp/roi-dice/classical.json

LUNG_MODEL_PATH="$PWD/models/lung-unet/lung-unet.onnx" \
python scripts/eval_roi_dice.py --limit 500 --profile nihcc-unet-local-v1 \
    --json tmp/roi-dice/unet.json --inspect 9
```

```powershell
# PowerShell
$env:LUNG_MODEL_PATH = "$PWD\models\lung-unet\lung-unet.onnx"
python scripts/eval_roi_dice.py --limit 500 --profile nihcc-unet-local-v1 --json tmp/roi-dice/unet.json --inspect 9
```

Match the profile to the model's resolution (`nihcc-unet-local-v1` for 160px, `nihcc-unet-v1` for 384px).

`--inspect` draws outlines in order of worst results (yellow = this system, light blue = reference). Measure on a random sample,
then check the bad ones by eye.

Rule-based values (1,000 random images):

```
Dice 中央値 0.828   四分位 0.725-0.879
(median Dice 0.828   quartiles 0.725-0.879)
実効的な破綻（Dice < 0.7 または抽出失敗）: 23.8%
(effective failures (Dice < 0.7 or extraction failure): 23.8%)
```

> **The reference was made by a model, not hand-drawn by experts.** The U-Net is also trained on the same CheXmask,
> so Dice measures "how similar it is to the reference" and **is biased upward.**
> Look at the difference from the rule-based method and at how visible failures decrease.

The procedure for saving results to the DB is [OPERATIONS_EN.md](../OPERATIONS_EN.md) §6.3.

---

## 6. Applying to roi-service

Check that `ROI_PROFILE` in `.env` matches the model's resolution, then recreate.

```bash
grep '^ROI_PROFILE=' .env                       # default nihcc-unet-local-v1
docker compose up -d --force-recreate roi-service
docker compose ps roi-service                   # about 10 seconds until healthy
```

roi-service does not expose a port, so check from inside the container.

```bash
docker compose exec roi-service python -c \
  "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/ready').read().decode())"
```

```
{"status":"ok","profile":"nihcc-unet-local-v1",
 "roi_model_version":"unet-lung-spine-local-v2.0.0+256051fe3a84", ...}
```

`roi_model_version` differs per profile. After replacing the model, re-ingest so the search targets are consistent
(`python scripts/load_initial_images.py`).

---

## 7. Batch mask generation (optional)

Use this when you want to output only the masks as images, separately from ingestion. It is for quality checks and making figures.
Ingestion creates masks via roi-service, so normal operation does not need this.

```bash
# local
python scripts/segment_lungs.py --images NIHCC/images --out data/lung-masks --profile nihcc-unet-local-v1

# EC2 (about 110,000 images). If a GPU is available, specify the provider
python scripts/segment_lungs.py \
    --images /imports/nihcc/images --out /data/lung-masks \
    --batch 16 --providers CUDAExecutionProvider CPUExecutionProvider
```

The output is one PNG per image, a **bit-packed single channel** (1=right lung 2=left lung 4=heart).
If stopped midway, it resumes and skips what already exists.

`summary.json` contains the distribution of lung field fractions and `suspicious` (images whose fraction is outside 10–40%).
Use it to pick candidates for visual inspection.

**The resolution is read from the model, not the profile.** It runs even if `--profile` is wrong,
and reports the mismatch. Only the thresholds and preprocessing come from the profile.

GPU inference requires `onnxruntime-gpu`. Without it, it falls back to CPU automatically.
Running 110,000 images on CPU takes about 3 hours.

---

## Common pitfalls

**`onnxruntime が無い`** (onnxruntime not found)
: Rebuild the roi-service image. `docker compose build roi-service`

**`Got invalid dimensions for input: image ... Got: 384 Expected: 160`**
: The profile and model resolution do not match. roi-service stops at startup with `SegmenterError`.
  Set `ROI_PROFILE` in `.env` to match.

**`unknown extraction profile: ...`**
: Profiles are baked into the image. `docker compose build roi-service`

**`分割モデルが無い: /models/lung-unet/lung-unet.onnx`** (segmentation model not found)
: Check that you placed it in `infra/lung-unet/` and that `LUNG_MODEL_DIR` is correct.
  `docker compose exec roi-service ls -l /models/lung-unet`

**`No module named 'onnxscript'` / `No module named 'cv2'` (training)**
: The training image has not been built.
  `docker build -f infra/train/Dockerfile.lung-unet -t local/nihcc-lung-unet-train:1.0.0 .`

**`Read-only file system: '/srv/.local'` / `Permission denied: '/work/models'`**
: You are trying to train with `docker compose run`. Use the `docker run` in §3.

**`Unexpected bus error encountered in worker ... insufficient shared memory (shm)`**
: `--shm-size=2g` is missing. As a stopgap `--workers 0` also works, but the GPU waits for input.

**ONNX becomes two files (`lung-unet.onnx` and `lung-unet.onnx.data`)**
: `train_lung_unet.py` repacks it into one file, so this normally does not happen. If you exported it by hand,
  pass it through `onnx.save_model(..., save_as_external_data=False)`.

**Training Dice stays at 0 and does not rise**
: Suspect that the left/right labels are swapped. `build_lung_dataset.py` checks this using centroids, but
  when you change `--size`, draw a few images and check.

**`UnicodeEncodeError: 'cp932'`**
: Happens when torch prints emoji on the Windows console. Add `PYTHONIOENCODING=utf-8`.

**`ModuleNotFoundError: No module named 'cv2'` (host)**
: You are using the bare `python`. See "[Python to use](#python-to-use)".

**`sha256sum` / `cp` not available in PowerShell**

| Unix | PowerShell |
|---|---|
| `sha256sum F` | `Get-FileHash F -Algorithm SHA256` |
| `cp A B` | `Copy-Item A B` |
| `ls -l` | `Get-ChildItem` |
| `export VAR=x; cmd` | `$env:VAR = "x"; cmd` |
