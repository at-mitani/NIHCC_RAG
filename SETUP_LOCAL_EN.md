# Setting Up the Environment on Your Own PC

English | [日本語](SETUP_LOCAL.md)

Build the full stack on a Windows (Docker Desktop) or Linux workstation.

For building on a GPU EC2 instance see [SETUP_EC2_EN.md](SETUP_EC2_EN.md); for day-to-day operations see
[docs/OPERATIONS_EN.md](docs/OPERATIONS_EN.md).

---

## 1. Requirements

| | Minimum | Recommended |
|---|---|---|
| Memory | 16 GB (8 GB or more for Docker) | 32 GB |
| Disk | 60 GB | 200 GB |
| CPU | 4 cores | 8 cores |

**No GPU is needed.** Everything runs on CPU. However, ingestion is 5–10x slower.

> **A local Intel integrated GPU cannot be used.** CUDA is NVIDIA-only, and `/dev/dri` is not visible
> from Docker Desktop's WSL2. If you need to train models, do it on EC2 and
> bring only the artifacts over.

### Number of containers started

There are **about 16**. Six of them are the vector DB (1 coordinator + 5 workers).
A Docker Desktop memory allocation of 8 GB or more is recommended.

---

## 2. Prerequisite software

### Windows

1. Enable **WSL2** (`wsl --install`)
2. Install **Docker Desktop** and use the WSL2 backend
3. In Settings → Resources, set memory to **8 GB or more** and disk to 60 GB or more
4. **Git for Windows** (includes Git Bash)

> Many procedures in this repository assume bash. **Running them from Git Bash** is the safest.
> To run them in PowerShell you need to translate commands, e.g. `sha256sum` → `Get-FileHash`,
> `cp` → `Copy-Item`.

### Linux

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git python3-venv
sudo usermod -aG docker $USER      # requires logging out and back in
```

---

## 3. Place the dataset

Obtain NIH ChestX-ray14 and place it in the following layout.

```
NIHCC/
  images/                       *.png
  Data_Entry_2017_v2020.csv
```

**You do not need all 112,120 images.** It works with a few thousand. Narrow it down to fit your disk and time.

The location can be changed with `NIHCC_DATA_DIR` in `.env` (default `./NIHCC`).

---

## 4. Build

```bash
git clone git@github.com:at-mitani/NIHCC_RAG.git
cd NIHCC_RAG
```

### 4.1 Create secrets and `.env`

```bash
bash scripts/bootstrap.sh
```

**The initial user password is displayed. Write it down.** It cannot be shown again.

Keys for the DB, JWT, etc. are created in `secrets/`. **They are not committed to git** (already in `.gitignore`).

### 4.2 Build the images

```bash
docker compose build db-migrate     # first; not included in a no-argument build
docker compose build
```

The DB image builds pgvector from source, so the first build takes 15–30 minutes.

> **Do not forget to specify `db-migrate` separately.** It belongs to the `migrate` profile,
> so a no-argument `build` does not build it. The migration files are baked into the image,
> so if you forget to build it, the new schema is not applied.

### 4.3 Start

```bash
docker compose up -d
docker compose ps        # a few minutes until everything is healthy
```

On the first start, BiomedCLIP is downloaded from the internet. To run in a closed network, fetch it first
with `bash scripts/fetch_model.sh`.

### 4.4 Apply the schema

Run this **after all containers are healthy**.

```bash
docker compose run --rm db-migrate
```

Creates the schema in metadata and in the vector cluster, and registers the 5 Citus workers.

> The width of the `embedding_fine` column is **fixed** at this point by `EMBEDDING_FINE_DIMENSION` in `.env`.
> Changing only `.env` afterward does not change the column, and ingestion fails for every image.
> Check the value before running (default 15).

### 4.5 Create users

```bash
python scripts/load_initial_users.py
```

Registers the initial users created by `bootstrap.sh`.

### 4.6 Verify

```bash
python tests/integration/smoke_test.py
```

Open `http://localhost:8080` in a browser.

---

## 5. Ingest images

The easiest way is from the Data ingestion tab (「データ取り込み」) on the screen. First check the reconciliation of the CSV and
images with a **dry-run**, and run it if there are no problems.

A script also works.

```bash
python scripts/load_initial_images.py
```

On CPU each image takes 1–2 seconds. Try with about 100 images first.

---

## 6. Development environment (optional)

To run scripts directly you need a Python virtual environment.

```bash
python -m venv .venv-dev
.venv-dev/Scripts/activate      # Windows
source .venv-dev/bin/activate   # Linux

pip install -e packages/common -e packages/image-pipeline
pip install pytest httpx
```

> **Plain `python` does not work.** Neither `cv2` nor `numpy` is installed, so you get
> `ModuleNotFoundError`. Activate the virtual environment when running scripts.

To evaluate or train lung field segmentation locally, you also need the following.

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install onnxruntime onnx onnxscript
```

---

## 7. Stop and cleanup

```bash
docker compose stop            # stop (data is kept)
docker compose down            # remove containers (data is kept)
docker compose down -v         # remove including volumes (**data is deleted**)
```

---

## 8. Common pitfalls

**Does not finish starting / containers keep crashing**
: Check Docker's memory allocation. With less than 8 GB, the roughly 16 containers cannot all come up.

**`ModuleNotFoundError: No module named 'cv2'`**
: You are using plain `python`. Activate the virtual environment (§6).

**`port is already allocated`**
: One of 8080 / 8000 / 8025 / 15672 / 9090 / 3000 is in use. Change `FRONTEND_PORT` etc.
  in `.env`.

**Migrations are not applied**
: Did you run `docker compose build db-migrate`? The migration files are baked into
  the image.

**Ingestion fails for every image**
: The dimensions in `.env` and the DB column widths do not match. [OPERATIONS_EN.md](docs/OPERATIONS_EN.md) §3.1.

**The screen behaves like an old version**
: Did you run `docker compose build frontend`? The static files are also baked into the image.

**`invalid option name: pipefail`**
: You are running a bash script with `sh`. Invoke it explicitly as `bash scripts/....sh`.
