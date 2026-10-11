# EC2 (g5.2xlarge / Ubuntu 26.04) Setup Guide

English | [日本語](SETUP_EC2.md)

This is the procedure for building the NIHCC Chest X-ray Image Search Local RAG System from scratch on an AWS EC2 GPU instance,
ingesting all 112,120 images, and shutting it down. For the meaning of each setting and the design, see
[README_EN.md](README_EN.md).

> **Building on your own PC** is covered in [SETUP_LOCAL_EN.md](SETUP_LOCAL_EN.md), and **operations after the build** in
> [docs/OPERATIONS_EN.md](docs/OPERATIONS_EN.md). This document is limited to the EC2-specific
> steps (GPU driver, Container Toolkit, exposure scope).

> Ubuntu 26.04 is a new release, so the package versions of the NVIDIA driver and Docker change
> over time. This guide does not hard-code version numbers; it detects them with commands before installing.

## Overall flow

| # | Task | Estimated time |
|---|---|---|
| 1 | [AWS preparation](#1-aws-preparation) | 15 min (quota request separately) |
| 2 | [OS initial setup](#2-os-initial-setup) | 10 min |
| 3 | [NVIDIA driver](#3-nvidia-driver) | 15 min |
| 4 | [Docker Engine / Compose](#4-docker-engine--compose) | 10 min |
| 5 | [NVIDIA Container Toolkit](#5-nvidia-container-toolkit) | 10 min |
| 6 | [Clone the repository](#6-clone-the-repository) | 5 min |
| 7 | [Download and extract the image data](#7-download-and-extract-the-image-data) (about 42 GB) | 1–2 hours |
| 8 | [Configure `.env`](#8-configure-env) | 10 min |
| 9 | [Build the containers](#9-build-the-containers) | 30–40 min |
| 10 | [Startup and schema setup](#10-startup-and-schema-setup) | 10 min |
| 11 | [Check the global IP](#11-check-the-global-ip) | 5 min |
| 12 | [Access from a browser](#12-access-from-a-browser) | 5 min |
| 13 | [Data ingestion](#13-data-ingestion) (112,120 images) | 4–6 hours |
| 14 | [Stop the containers and the instance](#14-stop-the-containers-and-the-instance) | 5 min |

**The order of the steps matters.** In particular, if you do not follow the order "kernel update → reboot → driver install",
the driver module and the running kernel get out of sync and the GPU becomes unusable ([3.4](#34-if-it-fails-kernel-and-module-mismatch)).

---

## 1. AWS preparation

### 1.1 Check the vCPU quota

g5 is subject to the "**Running On-Demand G and VT instances**" quota. On new accounts it may be
0, in which case you cannot launch the instance.

In the EC2 console → Service Quotas → Amazon Elastic Compute Cloud (Amazon EC2) →
"Running On-Demand G and VT instances", confirm that the applied value is **8 or more**.
If it is not enough, request an increase (approval can take from several hours to several days).

```bash
# when using the AWS CLI
aws service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query 'Quota.Value'
```

### 1.2 Security group

| Type | Port | Source | Purpose |
|---|---:|---|---|
| SSH | 22 | My IP | SSH for administration |
| Custom TCP | 8080 | My IP | Screen (not needed if you use SSH port forwarding in [12](#12-access-from-a-browser)) |

**Do not open 8080 to `0.0.0.0/0`.** Traffic is plaintext HTTP.

### 1.3 Create the instance

| Item | Setting |
|---|---|
| AMI | Ubuntu Server 26.04 LTS (HVM), SSD Volume Type — 64-bit (x86) |
| Instance type | **g5.2xlarge** (8 vCPU / 32 GiB / NVIDIA A10G 24 GB) |
| Key pair | Create new or use existing (save the `.pem` locally) |
| Security group | The one created in 1.2 |
| Storage | **200 GiB / gp3** |
| Elastic IP (optional) | Assign if you do not want the IP to change on stop/start |

Approximate storage breakdown:

| Use | Size |
|---|---:|
| OS and Docker images (including CUDA PyTorch) | about 25 GB |
| Images (extracted) | about 42 GB |
| Image archives (temporary space until extraction finishes) | up to about 42 GB |
| metadata-db (copies of original images, masks) | about 50 GB |
| Vector DB, WAL, logs | about 5 GB (1 row per image) |

### 1.4 SSH connection

```bash
chmod 400 key.pem
ssh -i key.pem ubuntu@<public IP>
```

From Windows PowerShell you can connect with `ssh -i .\key.pem ubuntu@<public IP>`.

---

## 2. OS initial setup

**Before the driver, update the kernel to the latest and reboot.** If you install the driver first and apt then
updates the kernel, the driver will not load on the next reboot.

```bash
sudo apt-get update
sudo apt-get -y full-upgrade
sudo reboot
```

After reconnecting, install the required packages.

```bash
sudo apt-get install -y git curl ca-certificates gnupg tmux sysstat pciutils \
  python-is-python3 ubuntu-drivers-common
sudo timedatectl set-timezone Asia/Tokyo
```

- `python-is-python3`: `scripts/bootstrap.sh` and the README commands call `python`.
  Ubuntu only has `python3` by default, so this makes `python` available.
- `tmux`: Used so that long-running tasks (7, 9, 13) are not stopped by an SSH disconnect.

```bash
tmux new -s work        # create a session. Detach with Ctrl+b → d
tmux attach -t work     # reattach
```

---

## 3. NVIDIA driver

### 3.1 Is the GPU visible?

```bash
lspci | grep -i nvidia
```

If a line like `NVIDIA Corporation ... A10G` appears, all is well. If nothing appears, the instance
is not a g5 instance (check `instance-type` using the method in [11](#11-check-the-global-ip)).

### 3.2 Install

```bash
sudo ubuntu-drivers list --gpgpu
V=$(sudo ubuntu-drivers list --gpgpu 2>/dev/null | grep -oP 'nvidia-driver-\K[0-9]+(?=-server)' | sort -n | tail -1)
echo "driver version: ${V}"
```

If `driver version:` is empty, look at the number of the `nvidia-driver-XXX-server` shown in the list and set `V=XXX` manually.

```bash
sudo ubuntu-drivers install --gpgpu                 # kernel modules and libraries
sudo apt-get install -y nvidia-utils-${V}-server    # nvidia-smi (not included by the command above)
sudo reboot
```

> Installing only the `sudo apt install nvidia-utils-XXX` that Ubuntu suggests when you run `nvidia-smi`
> does not work (there is no kernel driver, so you get `couldn't communicate with the NVIDIA driver`).
> Always install it together with `ubuntu-drivers install --gpgpu`.

### 3.3 Verify

```bash
nvidia-smi              # A10G and the Driver Version (525.60 or later) should be shown
lsmod | grep nvidia     # the nvidia kernel modules should be loaded
```

### 3.4 If it fails (kernel and module mismatch)

This applies when `nvidia-smi` reports `couldn't communicate with the NVIDIA driver`, or `lsmod` shows nothing.

```bash
V=$(dpkg -l | grep -oP 'nvidia-utils-\K[0-9]+(?=-server)' | head -1)
uname -r                                            # running kernel
dpkg -l | grep -E 'linux-modules-nvidia|nvidia-dkms' | awk '{print $2, $3}'
sudo modprobe nvidia                                # "Module nvidia not found" means a mismatch
```

Try these in order from the top.

```bash
# A. Add the modules for the running kernel (nothing is removed)
sudo apt-get install -y linux-modules-nvidia-${V}-server-$(uname -r)
sudo modprobe nvidia && nvidia-smi
```

```bash
# B. If A gives "Unable to locate package": align the latest kernel and modules, then reboot
sudo apt-get install -y linux-aws linux-modules-nvidia-${V}-server-aws
sudo reboot
```

```bash
# C. If they still do not match: build for the running kernel with DKMS
sudo apt-get install -y linux-headers-$(uname -r) nvidia-dkms-${V}-server
sudo reboot
```

### 3.5 Stop automatic kernel updates

Ubuntu installs new kernels through automatic updates (unattended-upgrades), so the 3.4 mismatch
can recur on the next reboot. Pin the kernel until ingestion is finished.

```bash
sudo apt-mark hold linux-aws linux-image-aws linux-headers-aws
```

When updating the kernel for maintenance, do it in this order: `sudo apt-mark unhold linux-aws linux-image-aws linux-headers-aws`
→ `full-upgrade` → reboot → verify with 3.3 → regenerate the CDI definition in [5.3](#53-cdi-definition-file).

---

## 4. Docker Engine / Compose

**Do not use the snap version of Docker.** The snap version does not read the NVIDIA Container Toolkit configuration
(`/etc/docker/daemon.json`), so it cannot use the GPU.

```bash
snap list docker 2>/dev/null && sudo snap remove docker
```

### 4.1 Official Docker repository (recommended)

First, check whether Docker publishes a repository for Ubuntu 26.04.

```bash
CODENAME=$(. /etc/os-release && echo "$VERSION_CODENAME")
curl -fsSL https://download.docker.com/linux/ubuntu/dists/ | grep -q "href=\"${CODENAME}/\"" \
  && echo "docker-ce: ${CODENAME} に対応" || echo "docker-ce: ${CODENAME} は未対応 → 4.2 へ"
# output: "に対応" = supported; "は未対応 → 4.2 へ" = not supported → go to 4.2
```

If it is supported, install the official packages.

```bash
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

### 4.2 If not supported: Ubuntu packages

```bash
sudo apt-get install -y docker.io docker-compose-v2 docker-buildx
```

### 4.3 Common settings and verification

```bash
sudo usermod -aG docker ubuntu
exit                     # log out once and reconnect over SSH so the group takes effect
```

```bash
docker version --format 'Engine {{.Server.Version}}'
docker compose version   # should be v2.20 or later
docker run --rm hello-world
```

---

## 5. NVIDIA Container Toolkit

### 5.1 Install

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
apt-cache policy nvidia-container-toolkit    # a version should appear under Candidate:
sudo apt-get install -y nvidia-container-toolkit
nvidia-ctk --version
```

### 5.2 Register with Docker

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
cat /etc/docker/daemon.json                  # should contain "runtimes": { "nvidia": ... }
```

### 5.3 CDI definition file

Newer Docker discovers GPUs through CDI (Container Device Interface). Without the definition file you get
`failed to discover GPU vendor from CDI: no known GPU vendor found`.

```bash
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
nvidia-ctk cdi list                          # nvidia.com/gpu=0 and nvidia.com/gpu=all should appear
```

The definition file records the paths of the driver libraries, so **regenerate it whenever you update the driver**.

### 5.4 Verify

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If the A10G table appears from inside the container, GPU preparation is complete.

| Error | Cause and remedy |
|---|---|
| `nvidia-ctk: command not found` | Toolkit not installed. Redo 5.1 |
| `failed to discover GPU vendor from CDI` | No CDI definition from 5.3, or the toolkit is not installed |
| `failed to initialize NVML` (when running 5.3) | The driver is not loaded. Go back to [3.3](#33-verify) / [3.4](#34-if-it-fails-kernel-and-module-mismatch) |
| `could not select device driver "" with capabilities: [[gpu]]` | Redo the registration in 5.2 and the Docker restart |

---

## 6. Clone the repository

The repository URL is in SSH form (`git@github.com:at-mitani/NIHCC_RAG.git`).
Create a key for EC2 and register it on GitHub as a **read-only deploy key**.

```bash
ssh-keygen -t ed25519 -C "ec2-nihcc" -f ~/.ssh/id_ed25519 -N ""
cat ~/.ssh/id_ed25519.pub
```

Register the displayed public key on GitHub under the repository → Settings → Deploy keys → Add deploy key
(leave Allow write access off).

```bash
ssh -o StrictHostKeyChecking=accept-new -T git@github.com    # should say "successfully authenticated"
cd ~
git clone git@github.com:at-mitani/NIHCC_RAG.git
cd ~/NIHCC_RAG
```

Unless otherwise noted, run the following steps in `~/NIHCC_RAG`.

---

## 7. Download and extract the image data

The diagnosis mapping CSV (`NIHCC/Data_Entry_2017_v2020.csv`) is included in the repository.
The images (`NIHCC/images/`) are not under Git, so download them from the NIH distribution site.

```bash
tmux attach -t work || tmux new -s work
cd ~/NIHCC_RAG/NIHCC
python batch_download_zips.py                # images_01.tar.gz to images_12.tar.gz (about 42 GB total)
```

Extract one file at a time and delete each archive once it is done, to keep disk usage down.
Each archive extracts into the `images/` directory.

```bash
for f in images_*.tar.gz; do echo "extract $f"; tar -xzf "$f" && rm -f "$f"; done
ls images | wc -l                            # 112120
ls -l Data_Entry_2017_v2020.csv
df -h /
```

**If you only want to check that it works**, download just the first file (4,999 images).

```bash
URL=$(grep -oP "https://nihcc\.box\.com/[^']+" batch_download_zips.py | head -1)
curl -L -o images_01.tar.gz "$URL" && tar -xzf images_01.tar.gz && rm -f images_01.tar.gz
ls images | wc -l                            # 4999
```

---

## 8. Configure `.env`

### 8.1 Generate secrets and `.env`

```bash
cd ~/NIHCC_RAG
bash scripts/bootstrap.sh
ls -l secrets/                               # directory 700 / files 644
```

`secrets/*.txt` are created with mode 0644 so that non-root containers (postgres uid 999 / app uid 10001) can read them.
With 0600, the DB does not start, failing with `cat: /run/secrets/app_db_password: Permission denied`.

### 8.2 GPU and exposure settings

Define a function that replaces values in `.env`, then apply the settings (you need to redefine it in a new shell).

```bash
setenv() { if grep -q "^$1=" .env; then sed -i "s#^$1=.*#$1=$2#" .env; else echo "$1=$2" >> .env; fi; }

# layer the GPU compose file on top
setenv COMPOSE_FILE docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.gpu.yml
setenv COMPOSE_PATH_SEPARATOR :
setenv NIHCC_DATA_DIR ./NIHCC

# keep administrative ports inside the host (expose only the screen on 8080)
setenv BACKEND_API_PORT 127.0.0.1:8000
setenv RABBITMQ_UI_PORT 127.0.0.1:15672
setenv MAILPIT_UI_PORT 127.0.0.1:8025
setenv PROMETHEUS_PORT 127.0.0.1:9090
setenv GRAFANA_PORT 127.0.0.1:3000
```

#### Two-stage search and finding estimates (disabled by default)

Search compares chest fields (lung fields + mediastinum + diaphragm). The second stage (re-ranking
by the finding profile from the chest field classifier) and the screen's "Finding estimates" (「所見の推定」) are enabled only when
a trained classifier is in place. **They are disabled by default**, and you can leave them that way for the initial build.

```bash
grep -E '^(ROI_PROFILE|EMBEDDING_FINE_BACKEND|EMBEDDING_FINE_DIMENSION|CHEST_HEAD_DIR)=' .env
# ROI_PROFILE=nihcc-unet-local-v1  <- U-Net is required (the chest field needs a heart mask)
# EMBEDDING_FINE_BACKEND=none      <- single-stage search
# EMBEDDING_FINE_DIMENSION=15      <- column width. Fixed when the migration runs
# CHEST_HEAD_DIR=./infra/chest     <- referenced only when the second stage is used
```

> **Decide `EMBEDDING_FINE_DIMENSION` before the migration in [10](#10-startup-and-schema-setup).**
> The width of the `embedding_fine` column is fixed to the value at the time the migration runs,
> and changing it later requires adding a new `V00x`. The default of 15 is for the chest field
> classifier's profile (14 findings + normality).
>
> For the steps to enable the second stage, see [docs/runbooks/chest-profile-setup_EN.md](docs/runbooks/chest-profile-setup_EN.md)
> (exporting the classifier and pinning the RAD-DINO version are required).

### 8.3 Verify

```bash
grep -E '^(COMPOSE_FILE|NIHCC_DATA_DIR|EMBEDDING_(DEVICE|DTYPE|MODEL_REPLICAS|TORCH_THREADS)|TORCH_INDEX_URL|INGEST_WORKER_REPLICAS|INGEST_STORE_REGION_CROPS|PREPROCESS_VERSION)=' .env
docker compose config | grep -A5 'reservations:'        # driver: nvidia should appear
```

On g5.2xlarge the settings are correct if they have the following values (the defaults in `.env.example`).

| Variable | Value |
|---|---|
| `EMBEDDING_DEVICE` | `cuda` |
| `EMBEDDING_DTYPE` | `float16` |
| `EMBEDDING_MODEL_REPLICAS` | `1` |
| `EMBEDDING_TORCH_THREADS` | `4` |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu121` |
| `INGEST_WORKER_REPLICAS` | `6` |
| `INGEST_STORE_REGION_CROPS` | `false` |
| `PREPROCESS_VERSION` | `prep-v1.1.0` |

Write down the password of the initial user for logging in to the screen (in the form `admin:<password>:admin`).

```bash
grep '^LOCAL_USERS=' .env
```

---

## 9. Build the containers

This takes time because the DB image builds pgvector from source and the embedding-service downloads CUDA PyTorch.

```bash
tmux attach -t work || tmux new -s work
cd ~/NIHCC_RAG
docker compose build
docker compose build db-migrate      # it has a profile, so a no-argument build does not build it
docker images | grep -E 'nihcc|citus'
df -h /
```

---

## 10. Startup and schema setup

If `.env` has **`HF_HUB_OFFLINE=1`** (offline operation that does not download models), fetch the model into
the cache **before starting**. On the first build and after `docker compose down -v` the cache is empty, so if you skip this,
the embedding-service fails to start and keeps restarting, and the ingest-worker and frontend stay in `Created` and never start.
With the default `HF_HUB_OFFLINE=0` this is unnecessary (the embedding-service fetches the model automatically on first startup).

```bash
cd ~/NIHCC_RAG
grep '^HF_HUB_OFFLINE=' .env
bash scripts/fetch_model.sh      # only when HF_HUB_OFFLINE=1 (goes online only while running to fetch the model)
```

```bash
cd ~/NIHCC_RAG
docker compose up -d
```

Wait until all containers are healthy. Startup is complete when the following command **prints nothing**
(the embedding-service takes a few minutes the first time because it downloads the model). Without `-a`,
containers stuck in `Created` do not appear in the list, so you would miss them.

```bash
docker compose ps -a | grep -E 'unhealthy|starting|Restarting|Exited|Created'
```

Once started, apply the schema and register the Citus workers.

```bash
# the embedding_fine column width is fixed to the value at run time. Check it first
grep '^EMBEDDING_FINE_DIMENSION=' .env      # default 15

docker compose run --rm db-migrate
```

`V001`–`V012` are applied to metadata, and `V001` to the vector cluster (`vector-coordinator` + `vector-worker-1..5`);
6 shards are placed on each of the 5 workers.

The procedure for updating an EC2 instance that already has data ingested is in [docs/OPERATIONS_EN.md](docs/OPERATIONS_EN.md) §2.

> When `git pull` adds migrations (`db/metadata-migrations/` etc.), **first run
> `docker compose build db-migrate`** and then `run`. The SQL is copied into the image,
> so with the old image the new migrations are not applied and ingestion fails for every image.

**Always confirm that the GPU is being used.** If the GPU is not visible, the embedding-service logs a warning and
keeps running on CPU, making ingestion extremely slow.

```bash
docker compose exec embedding-service python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`2.5.1+cu121 True NVIDIA A10G` means it is working correctly.

| Result | Remedy |
|---|---|
| `2.5.1+cpu` | Check `TORCH_INDEX_URL` and redo `docker compose build embedding-service` |
| `2.5.1+cu121 False` | gpu.yml is not in `COMPOSE_FILE`, or [5](#5-nvidia-container-toolkit) is not complete. After fixing, run `docker compose up -d --force-recreate embedding-service` |
| Stays unhealthy | Check `docker compose logs --tail=80 <service name>` ([docs/OPERATIONS_EN.md](docs/OPERATIONS_EN.md) §3) |

---

## 11. Check the global IP

```bash
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')
PUBLIC_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type ; echo
echo "${PUBLIC_IP}"
```

Set `APP_BASE_URL`, which is used in the URL of password reset emails, to the actual address.

```bash
setenv() { if grep -q "^$1=" .env; then sed -i "s#^$1=.*#$1=$2#" .env; else echo "$1=$2" >> .env; fi; }
setenv APP_BASE_URL "http://${PUBLIC_IP}:8080"
docker compose up -d backend-api
```

Stopping and starting the instance changes the public IP (unless you assigned an Elastic IP).
Redo this step each time.

---

## 12. Access from a browser

### Method A: SSH port forwarding (recommended)

You do not need to open 8080 in the security group, and traffic is encrypted by SSH. Run this on your own PC.

```bash
ssh -i key.pem -L 8080:localhost:8080 -L 8025:localhost:8025 ubuntu@<public IP>
```

Open <http://localhost:8080> in your local browser (8025 is the Mailpit inbox).

### Method B: Open directly via the global IP

Open 8080 in the security group restricted to My IP, then open it in a browser.

```
http://<public IP>:8080          screen
http://<public IP>:8080/docs     OpenAPI
```

> **HTTP is plaintext.** Login IDs, passwords and access tokens travel unencrypted.
> Even with Method B, always restrict the source to your own IP, and if you expose it on an ongoing basis, put an ALB + ACM certificate in front.

### Login

Log in with `admin` and the password you wrote down in 8.3. **After logging in, change the password from the menu.**

---

## 13. Data ingestion

### 13.1 Smoke check (50 images)

```bash
tmux attach -t work || tmux new -s work
cd ~/NIHCC_RAG
python scripts/load_initial_images.py --limit 50 --yes
```

It shows dry-run validation → job registration → waiting for completion → result summary, in that order. Confirm that it finishes with 0 failures.

### 13.2 Ingest everything

```bash
python scripts/load_initial_images.py --yes
```

Ingestion continues even if you detach from tmux (Ctrl+b → d). Even if you stop it with `Ctrl+C`, the server-side job continues.

To ingest from the screen, press **dry-run validation** (「dry-run 検証」) → **Start ingestion** (「取り込み開始」) in the Data ingestion tab (「データ取り込み」).
The progress bar shows the count, elapsed time and remaining time.

### 13.3 Check progress and performance

```bash
nvidia-smi -l 5                               # GPU utilization and python processes
docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}' | sort -k2 -hr | head -12
iostat -xm 5                                  # disk saturation (%util)
```

Check the number of images processed per minute from the DB. To finish in 3 hours you need **about 10.4 images/second**.

```bash
PSQL() { docker compose exec -T -e PGPASSWORD="$(cat secrets/db_password.txt)" metadata-db \
  psql -U postgres -d metadata -h 127.0.0.1 -X -c "$1"; }

PSQL "SELECT status, succeeded_count + failed_count AS done, started_at FROM ingest_job ORDER BY started_at DESC NULLS LAST LIMIT 1;"
sleep 60
PSQL "SELECT status, succeeded_count + failed_count AS done, started_at FROM ingest_job ORDER BY started_at DESC NULLS LAST LIMIT 1;"
```

If throughput is low, isolate the cause in this order: GPU check ([10](#10-startup-and-schema-setup)) → ROI (CPU) saturation → DB waits.

---

## 14. Stop the containers and the instance

### 14.1 Stop the containers

```bash
cd ~/NIHCC_RAG
docker compose down          # stop (ingested data remains in Docker volumes)
```

**Only if you want to delete all data**, use the following (in addition to ingested images, vectors and users,
**the model cache is also deleted**).

```bash
docker compose down -v
```

When starting again after `down -v`, if `HF_HUB_OFFLINE=1`, run `bash scripts/fetch_model.sh` **first**
and then `docker compose up -d` ([10](#10-startup-and-schema-setup)).

If you only want to fix the source and re-ingest the data, delete just the data with the following command instead of `down -v`.
The model cache is kept, so the model does not need to be fetched again.

```bash
python scripts/reset_data.py --all
```

### 14.2 Stop the instance

A GPU instance is billed just for running. Stop it when not in use.

```bash
sudo shutdown -h now         # an EBS-backed instance goes to "stopped" (not terminated)
```

"Instance state → Stop instance" in the EC2 console does the same. EBS (200 GiB) is still billed while stopped.

### 14.3 Resuming

1. Start the instance in the EC2 console and connect via SSH
2. Check the GPU with `nvidia-smi` (normally fine if the kernel is pinned)
3. Start the containers

```bash
cd ~/NIHCC_RAG
docker compose up -d
docker compose ps -a | grep -E 'unhealthy|starting|Restarting|Exited|Created'
```

4. If the public IP has changed, redo [11](#11-check-the-global-ip)

If you stopped the instance without running `docker compose down`, the containers start automatically together with Docker
because of the `restart: unless-stopped` setting.

---

## Troubleshooting quick reference

| Symptom | Cause | See |
|---|---|---|
| `Command 'nvidia-smi' not found` | Driver not installed, or `nvidia-utils-XXX-server` not installed | [3.2](#32-install) |
| `couldn't communicate with the NVIDIA driver` | Kernel and driver modules do not match | [3.4](#34-if-it-fails-kernel-and-module-mismatch) |
| `nvidia-ctk: command not found` | Container Toolkit not installed | [5.1](#51-install) |
| `failed to discover GPU vendor from CDI` | No CDI definition | [5.3](#53-cdi-definition-file) |
| `torch.cuda.is_available()` is `False` | gpu.yml not loaded, or toolkit setup incomplete | [10](#10-startup-and-schema-setup) |
| `cat: /run/secrets/...: Permission denied` | Secrets are 0600 | [8.1](#81-generate-secrets-and-env) |
| `dependency failed to start: ... is unhealthy` | Secret permissions, memory, disk, etc. | [OPERATIONS_EN.md](docs/OPERATIONS_EN.md) §3 |
| Ingestion fails for every image / `invalid input value for enum` on cancel | The `db-migrate` image is old and new migrations are not applied | [10](#10-startup-and-schema-setup) (`docker compose build db-migrate` → `run`) |
| embedding-service keeps restarting / ingest-worker and frontend stay in `Created` | `HF_HUB_OFFLINE=1` but the model cache is empty (e.g. after `down -v`) | [10](#10-startup-and-schema-setup) (`bash scripts/fetch_model.sh` → `docker compose up -d`) |
| `crypto.randomUUID is not a function` when starting ingestion | Cached old screen | Hard reload in the browser (Ctrl+Shift+R) |
| Ingestion is slow | Check in order: GPU not used → ROI (CPU) → DB | [13.3](#133-check-progress-and-performance) |
| `python: command not found` | `python-is-python3` not installed | [2](#2-os-initial-setup) |
| `Permission denied while trying to connect to the Docker daemon` | docker group membership not applied | [4.3](#43-common-settings-and-verification) (log in again) |
