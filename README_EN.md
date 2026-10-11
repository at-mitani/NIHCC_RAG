# NIHCC Chest X-ray Image Search Local RAG System

English | [日本語](README.md)

A system that searches chest X-ray images by comparing their **chest fields** (lung fields + mediastinum + diaphragm).
Submit an image and similar cases are listed; submit a finding sentence and likely matching cases are listed. Each candidate
comes with **finding estimates** from a classifier trained on chest fields (for research use only, not for diagnosis).

> Because there is no region-level ground-truth data, the lungs are not split into regions; the chest field of one image is treated as one unit.

No medical information is sent to external services. Everything runs entirely in your own environment.

> ## For research use only
>
> **Do not use for diagnosis.** The dataset labels were derived by NLP from radiology reports,
> not assigned by experts looking at the images. Search results always show a warning to that effect.

---

## Document map

| What you want to read | Document |
|---|---|
| **Just get it running** | This page (§1–§3) |
| **How to use the screens** | This page (§4) |
| Build on your own PC | [SETUP_LOCAL_EN.md](SETUP_LOCAL_EN.md) |
| Build on a GPU EC2 instance | [SETUP_EC2_EN.md](SETUP_EC2_EN.md) |
| Day-to-day operations and incident handling | [docs/OPERATIONS_EN.md](docs/OPERATIONS_EN.md) |
| **Meaning of abbreviations** (ROI / MIL / U-Net …) | [docs/GLOSSARY_EN.md](docs/GLOSSARY_EN.md) |
| System design | [docs/DESIGN_EN.md](docs/DESIGN_EN.md) |
| Model and image-processing design | [docs/MULTIMODAL_EN.md](docs/MULTIMODAL_EN.md) |
| User and password management | [docs/user-management_EN.md](docs/user-management_EN.md) |

---

## 1. Requirements

| | Minimum | Recommended |
|---|---|---|
| OS | Linux / Windows (Docker Desktop) | Ubuntu 24.04 or later |
| CPU | 4 cores | 8 cores or more |
| Memory | 16 GB | 32 GB |
| Disk | 60 GB | 200 GB (to load all 110,000 images) |
| GPU | Not required (runs on CPU) | NVIDIA (ingestion is 5–10x faster) |

Docker and Docker Compose v2 are required. If you use a GPU, the NVIDIA Container Toolkit is also required.

Obtain the dataset (NIH ChestX-ray14) separately and place it in `NIHCC/`.

---

## 2. Installation

The build steps are split by environment.

- Your own PC → **[SETUP_LOCAL_EN.md](SETUP_LOCAL_EN.md)**
- GPU EC2 instance → **[SETUP_EC2_EN.md](SETUP_EC2_EN.md)**

Only the shortest path is shown here.

```bash
git clone git@github.com:at-mitani/NIHCC_RAG.git
cd NIHCC_RAG

bash scripts/bootstrap.sh          # create secrets and .env (prints the initial password)
docker compose build db-migrate    # the migrate profile is not included in a no-argument build
docker compose build
docker compose up -d
docker compose run --rm db-migrate # apply the schema and register the Citus workers

python scripts/load_initial_users.py
```

Open `http://localhost:8080` in a browser.

> **Write down the initial password that `bootstrap.sh` prints.** It cannot be shown again.

---

## 3. Start and stop

| What you want to do | Command |
|---|---|
| Start | `docker compose up -d` |
| Stop (data is kept) | `docker compose stop` |
| Stop and clean up (data is kept) | `docker compose down` |
| Check status | `docker compose ps` |
| View logs | `docker compose logs -f backend-api` |
| **Apply updates** | `bash scripts/redeploy.sh` |

> **Do not `build` only selected services.** Both the migration files and the screen's static files
> are baked into the images, so any service you leave out keeps running the old version. `redeploy.sh`
> rebuilds everything including `db-migrate`, waits for health, and detects and replaces containers
> running on old images.

### Exposed ports

| Port | What | Environment variable |
|---|---|---|
| 8080 | Screen (web UI) | `FRONTEND_PORT` |
| 8000 | API (OpenAPI at `/docs`) | `BACKEND_API_PORT` |
| 8025 | Mailpit (development mail inbox) | `MAILPIT_UI_PORT` |
| 15672 | RabbitMQ management UI | `RABBITMQ_UI_PORT` |
| 9090 / 3000 | Prometheus / Grafana | `PROMETHEUS_PORT` / `GRAFANA_PORT` |

---

## 4. Using the screens

The screen is divided into five tabs: **Search** (「検索」) / **Data ingestion** (「データ取り込み」) / **Ingestion history** (「取り込み履歴」) /
**Administration** (「管理」) / **Change password** (「パスワード変更」). Which tabs are visible depends on the role.

| Role | What it can do |
|---|---|
| `viewer` | Search |
| `operator` | Search + ingestion |
| `admin` | Everything + user management |

The **display language** is switched between 日本語 / English / 中文（简体） with the pull-down in the header.
The selected language is saved in the browser and used next time as well (on first use it follows the browser language).
The screen text is in `apps/frontend/static/i18n/{ja,en,zh}.json`. When changing text,
update all three (key consistency is checked by `tests/unit/test_frontend_i18n.py`).
Text returned by the server (warnings, error details, notifications, standard findings) is **returned in English and translated
by the screen into the display language**. Errors are translated via `message_id` and `params`, and search warnings via `warning_details`, using
`server.<id>` in the dictionary; anything not in the dictionary is shown in English as is. When you add text on the server,
add it to the dictionaries as well (omissions are detected by the same test).

### 4.1 Search

#### Input fields

| Field | Meaning | Default | When to use |
|---|---|---|---|
| **Search mode** (「検索モード」) | What to search by | Image → image | See the table below |
| **Query image** (「検索対象画像」) | The image to search from | - | Drag & drop or select |
| **Finding text** (「所見テキスト」) | A finding sentence | - | Japanese allowed. Can be chosen from a list of standard phrases |
| **Input language** (「入力言語」) | Japanese / English | Japanese | Japanese is converted to English with an approved dictionary |
| **Image weight** (「画像の重み」) | Mix of image and text | 0.70 | Only effective in hybrid mode |
| **Number of results** (「表示件数」) | Number of results returned (API `top_k`) | 4 | Larger values are slower |
| **Minimum similarity** (「最小類似度」) | Results below this are discarded | 0.0 | Raise it when there is a lot of noise |
| **Dataset labels** (「付随ラベル」) | Filter by finding | - | **Entered in Japanese** (結節影 (nodule), 腫瘤 (mass) …) |
| **View position** (「撮影方向」) | Filter by PA / AP | - | List them like `PA, AP` |
| **Show thumbnails** (「サムネイルを表示」) | Whether to show images in the list | on | Turning it off is lighter |
| **Overlay chest field outline** (「胸部野の輪郭を重ねて表示」) | Whether to overlay the compared area (orange line) | on | |
| **Re-ranking** (「再ランキング」) | Whether to use the finding profile | on | Turn it off to compare with a BiomedCLIP-only search |

#### Search modes

| Mode | Input | What happens |
|---|---|---|
| **Image → image** (「画像 → 画像」) | Image | Crops the chest field of the query image and searches all images for those whose finding estimates (profile) are close |
| **Text → image** (「テキスト → 画像」) | Sentence | Converts the sentence to English and searches chest fields in the same space (no re-ranking) |
| **Hybrid** (「ハイブリッド」) | Both | Mixes image and text scores by weight |

#### Usage examples

**Example 1: Find similar cases**

1. Set the mode to Image → image (「画像 → 画像」)
2. Drag & drop an image
3. Click Search (「検索」)

Candidates are listed in a single row. The leftmost is the query image, and every image has an **orange line (the compared chest field)**
overlaid, so you can see what was compared with what.

**Example 2: Search from a finding sentence**

1. Set the mode to Text → image (「テキスト → 画像」)
2. Enter `右上肺野の結節影` (nodular shadow in the right upper lung field) in Finding text (you can also pick it from the list of standard phrases)
3. Click Search (「検索」)

> Free text that is not in the dictionary is **rejected without translation**. Rather than running a search with a translation
> whose quality cannot be guaranteed, we choose to tell the user it cannot be done. "View registered standard Japanese findings"
> (「登録済みの定型日本語所見を見る」) at the bottom of the screen shows the phrases you can use.

**Example 3: Suspected pneumothorax — list cases with a similar thorax**

1. Run an image search
2. Look at "**Cases with a similar thoracic shape**" (「胸郭の形が似た症例」) at the bottom of the results

Cases are ordered by the outline of the whole lung field. The top-left is the query image, and **the yellow line is the outer boundary of the lung field used for comparison**.
In pneumothorax the lung collapses and shrinks, and in cardiomegaly the cardiac silhouette widens and pushes the lungs outward, so these show up
in the shape rather than in how the pixels look.

> **This is not finding discrimination.** It is an axis for lining up cases with similar build and similar appearance to compare them side by side.

#### Reading the results

| Display | Meaning |
|---|---|
| **Dataset labels** (「データセットのラベル」) | The image's labels, derived by NLP from radiology reports |
| **Finding estimates (for research use only, not for diagnosis)** (「所見の推定（研究用・診断ではありません）」) | Probabilities from the chest field classifier. Top 3. Values below 0.5 are shown faded |
| Chest field extraction uncertain (「胸部野の抽出が不確か」) | One lung not found, large left-right difference, etc. Results are for reference only |
| Exhaustive search by finding profile (「所見プロファイルで全件検索」) | Default for image search. Ranking from an exact comparison with the finding estimates of all images |
| Re-ranked (「再ランキング済」) | In two-stage search (image search with `IMAGE_SEARCH_STRATEGY=coarse_to_fine`), candidates re-ranked by closeness of finding estimates |

> **Labels and estimates are different things.** Labels came with the dataset; estimates are the output of a classifier
> trained on images. Neither is a diagnosis. The classifier's image-level AUC is 0.833
> (evaluated on the full dataset; 0.70–0.91 per finding), and some findings, such as infiltration, are weak.

### 4.2 Data ingestion

| Field | Meaning | Example |
|---|---|---|
| **Image directory** (「画像ディレクトリ」) | Location of the images to ingest (path inside the container) | `/imports/nihcc/images` |
| **Diagnosis mapping CSV** (「診断紐づけ CSV」) | Label CSV | `/imports/nihcc/Data_Entry_2017_v2020.csv` |
| **Dataset version** (「データセット版」) | Version name recorded | `nihcc-2017-v2020` |
| **CSV schema version** (「CSV schema 版」) | CSV column layout | `nihcc-data-entry-2017-v1` |
| **Limit** (「件数上限」) | Narrow down when trying things out | `100` |
| **On duplicate** (「重複時」) | How to handle images that already exist | `SKIP` (keep) / `REPLACE` (redo) |

First check the reconciliation of the CSV and images with a **dry-run**, and run it if there are no problems.
A single failed image does not stop the job. Failures can be viewed by reason in Ingestion history (「取り込み履歴」).

### 4.3 Ingestion history

Shows job progress and a breakdown of failures. Errors can be downloaded as CSV.
Jobs can also be cancelled and re-run from here.

### 4.4 Administration

Add users, change roles, disable users. You can also view the system status (model versions, second-stage fill rate,
cluster health).

### 4.5 Change password

Change your own password. If you forget it, request a reset from the login screen and
a link is sent to your registered email address (in the development environment, Mailpit receives it).

---

## 5. Configuration

Settings are consolidated in `.env`. `.env.example` is the template, and each item has an explanation.

### Commonly changed settings

| Variable | Meaning | Default |
|---|---|---|
| `NIHCC_DATA_DIR` | Location of the dataset | `./NIHCC` |
| `ROI_PROFILE` | How lung fields are extracted (U-Net is required) | `nihcc-unet-local-v1` |
| `EMBEDDING_BACKEND` | First-stage model | `biomedclip` |
| `EMBEDDING_FINE_BACKEND` | Finding profile model (the adopted configuration is `chest-profile`) | `none` |
| `IMAGE_SEARCH_STRATEGY` | How image search proceeds (`profile` exhaustive search / `coarse_to_fine` two-stage search) | `profile` |
| `ROI_MIN_CONFIDENCE` | Results are suppressed if the confidence of the query image's chest field is below this | `0.35` |
| `QUERY_IMAGE_RETENTION_MINUTES` | Retention period for query images | `60` |
| `INGEST_WORKER_REPLICAS` | Ingestion parallelism | `2` |

> **When `.env.example` is updated, bring the changes in with `python scripts/sync_env.py`.**
> It keeps existing values and adds only the new keys. New keys use the compose default values,
> so **adding them alone never changes behavior**.

### Keep dimension-related settings consistent with the schema

`EMBEDDING_DIMENSION` / `EMBEDDING_FINE_DIMENSION` / `EMBEDDING_LUNG_DIMENSION`
**must match the width of the DB columns**. The columns are fixed to the values at the time the migration runs,
so if you later change only `.env`, ingestion fails for every image.

The ingest-worker checks them against each other at startup and stops, so this never goes unnoticed.

---

## 6. Troubleshooting

[docs/OPERATIONS_EN.md](docs/OPERATIONS_EN.md) collects remedies by symptom. Only the common ones are listed here.

| Symptom | Where to look |
|---|---|
| Blank screen / old behavior | Did you rebuild the frontend image (`redeploy.sh`)? |
| Ingestion fails for every image | `.env` dimensions vs. column widths (OPERATIONS §3) |
| Search returns 503 | Health of dependent services (`docker compose ps`) |
| Search returns 504 | CPU contention. Searches are slower during ingestion |
| Migrations are not applied | Did you rebuild `db-migrate`? |

---

## 7. Development

```bash
python -m pytest tests/unit          # unit tests

# smoke test against the running system (requires the admin password)
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```

Passwords are in `LOCAL_USERS` in `.env` (comma-separated `user:password:role`).

For the list of scripts and their purposes, see [docs/OPERATIONS_EN.md](docs/OPERATIONS_EN.md) §7.
