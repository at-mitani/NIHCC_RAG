# Design specification
English | [日本語](DESIGN.md)

Structure and design of the NIHCC chest X-ray image search local RAG system. **For developers.**

**Abbreviations are collected in [GLOSSARY_EN.md](GLOSSARY_EN.md).**

The multimodal design (models, ROI, classifier, U-Net, multi-stage search) is split out into [MULTIMODAL_EN.md](MULTIMODAL_EN.md).
Operating procedures are in [OPERATIONS_EN.md](OPERATIONS_EN.md).

---

## 1. What the system does

It searches by comparing the **chest field** (lung field + mediastinum + diaphragm) of chest X-ray images with each other. Submit an image and
similar cases are listed; submit a finding sentence and likely matching cases are listed. Each candidate carries **finding estimates**
(for research use only, not for diagnosis) from a classifier trained on the chest field.

The target is NIH ChestX-ray14 (112,120 images / 30,805 patients).

### 1.1 Why the unit is the chest field of a single image

The dataset labels are **per image**; there is no ground truth per lung region (upper lobe, lower lobe, etc.).
Region-level search can be neither trained nor evaluated. Unsupervised appearance embeddings only capture
region shape and imaging conditions, and cannot predict region-level findings.

With the unit aligned to a single image, the per-image labels become the training data directly. On the full dataset evaluation,
per-image AUC is 0.833, and similar-case search (Test B) is at least 1.33 times random for all 14 findings.

The frame includes the mediastinum and diaphragm, not just the lung field, because **cardiomegaly appears in the heart itself and
pleural effusion appears at the costophrenic angle**. The share of ground-truth boxes with more than half inside the frame is
58.7% for the lung field alone and 99.7% for the chest field.

The trade-off is that you cannot search by "which region has the finding".

### 1.2 Research use only

**Do not use for diagnosis.** The dataset labels were derived by NLP from radiology reports, not
assigned by experts looking at the images. Search results always carry a warning to that effect.

### 1.3 Nothing leaves the system

Medical information is not sent to external APIs. Japanese finding text is also converted to English only with an
**approved local dictionary**, not an external translation API (free text not in the dictionary is rejected, not translated).

---

## 2. System architecture

Diagrams (PNG) are in [diagrams/](diagrams/README_EN.md) (system architecture, process architecture, container relationships,
and the flows for training data creation, ingestion, and search).

### 2.1 Overall

```
                    ┌──────────┐
  user ───────────→ │ frontend │  static files (nginx)
                    └────┬─────┘
                         │
                    ┌────┴────────┐
                    │ backend-api │  REST API, auth, audit
                    └──┬───┬───┬──┘
           ┌───────────┘   │   └───────────┐
           │               │               │
    ┌──────┴─────┐   ┌─────┴──────┐  ┌─────┴──────┐
    │ metadata-db│   │   queue    │  │   search / ingestion
    │ PostgreSQL │   │ RabbitMQ   │  │   backend services
    └────────────┘   └─────┬──────┘  └─────┬──────┘
                           │               │
              ┌────────────┴───┐    ┌──────┴────────────┐
              │ ingest-worker  │    │ search-worker     │
              │ search-worker  │    │ search-aggregator │
              └────────┬───────┘    └──────┬────────────┘
                       │                   │
         ┌─────────────┴───┬───────────────┴──┐
         │                 │                  │
   ┌─────┴──────┐  ┌───────┴────────┐  ┌──────┴──────────────┐
   │ roi-service│  │embedding-service│ │ vector cluster       │
   │ chest field│  │ vectors and    │  │ Citus + pgvector × 1 │
   │ extraction │  │ estimates      │  │                      │
   └────────────┘  └────────────────┘  └─────────────────────┘
```

### 2.2 Three separate networks

| Name | Who is on it | Intent |
|---|---|---|
| `edge` | frontend / backend-api | The only surface exposed externally |
| `app` | Between services | Backend services cannot be called directly from outside |
| `data` | DB and cluster | **Restrict what can reach the DB** |

`roi-service` and `embedding-service` are only on `app`. **They cannot reach the DB.**
DB reads and writes are done by ingest-worker / backend-api / search-aggregator;
roi-service and embedding-service only process the images they receive and return the result.

### 2.3 Container list

| Service | Image | Role | Exposed |
|---|---|---|---|
| `frontend` | nginx | UI (static files) | 8080 |
| `backend-api` | custom | REST API, auth, audit | 8000 |
| `ingest-worker` | custom | Ingestion (2 instances by default) | - |
| `search-worker` | custom | Asynchronous search | - |
| `search-aggregator` | custom | Search against the vector cluster (profile exhaustive search / two-stage search) | - |
| `roi-service` | custom | Chest field extraction (from U-Net lungs and heart) | - |
| `embedding-service` | custom | Vector generation and finding estimates (torch) | - |
| `metadata-db` | postgres:17 | Metadata | - |
| `queue` | rabbitmq:3.13 | Jobs | 15672 (management UI) |
| `db-migrate` | custom | Schema application (`migrate` profile) | - |
| `mailpit` | mailpit | Mail receiver for development | 8025 |
| `prometheus` / `grafana` | official | Monitoring | 9090 / 3000 |
| `vector-coordinator` / `vector-worker-1..5` | Citus + pgvector | Vectors (1 coordinator + 5 workers) | - |

**The application containers (frontend, backend-api, the workers, search-aggregator, roi-service, embedding-service) are `read_only: true`**; only `/tmp` (tmpfs) is writable.
They also have `cap_drop: ALL` and `no-new-privileges`. This hardening does not fit work that writes, such as training or
backfill, so that work is done with separate images and `docker run`
([OPERATIONS_EN.md](OPERATIONS_EN.md)).

### 2.4 One vector cluster

The search unit is a single image and 1 row = 1 image, so there is one cluster (1 coordinator +
5 workers, 30 shards, 6 per node). Scans are distributed across the 5 workers.
Defined in [infra/compose/docker-compose.vectors.yml](../infra/compose/docker-compose.vectors.yml).

---

## 3. Process flow

### 3.1 Ingestion

```
  user → backend-api          validation (reconciliation of CSV and images) → ingest_job / ingest_item
             │
             └→ queue         1 image = 1 message
                  │
                  ↓
             ingest-worker
                  │
                  ├→ roi-service        U-Net lungs and heart → chest field mask, crop, shape vector
                  ├→ embedding-service  first stage (BiomedCLIP), second stage (profile), finding estimates
                  ├→ vector cluster     UPSERT into image_embedding (1 row per image)
                  └→ metadata-db        chest_asset / finding_estimate / image_vector_state
```

Key points.

- **One failed image does not stop the whole job.** Failures are recorded in `ingest_item` and the job continues
- **Idempotent.** Resending the same image does not insert it twice (`idempotency_key`,
  `ON CONFLICT DO UPDATE`)
- **The chest field is centralized in roi-service.** roi-service also builds the input to the embedding model,
  and callers do not rebuild it. The classifier was trained on features built by the evaluation script, so
  a difference of even one pixel silently shifts the probabilities (evaluation and production use the same function in `image_pipeline.chest`)
- At startup, check that the way roi-service builds masks matches the classifier's assumptions
- Before writing vectors, check at startup that **the column widths match `.env`**

### 3.2 Search

```
  user → backend-api
             │
             ├→ roi-service         chest field of the query image (for image search)
             ├→ embedding-service   query vector and finding estimates of the query image
             │
             └→ search-aggregator → vector cluster
                  │
                  ├→ image only: compare all images exactly by finding profile (15 dimensions) (default)
                  ├→ includes text: single-stage HNSW search (BiomedCLIP; the profile cannot handle text)
                  └→ thoracic shape (embedding_lung)
```

Image-only search uses exhaustive search because in two-stage search, images with matching findings that the first stage
(BiomedCLIP) did not pick up are never considered for re-ranking (nodule lift is 1.91 for two-stage search versus
2.43 for exhaustive search; vector search p95 is 290 ms versus 24 ms). `IMAGE_SEARCH_STRATEGY`
switches it to two-stage search.

If the request carries `Prefer: respond-async`, it is accepted asynchronously; `search-worker` takes over and
writes to `search_result`. The UI polls by `search_id`.

Details of multi-stage search are in [MULTIMODAL_EN.md](MULTIMODAL_EN.md) §4.

### 3.3 Authentication

JWT. `app_user` holds a bcrypt hash. Roles are `viewer` / `operator` / `admin`.
Password reset goes via email (in the development environment, mailpit receives it).
Details in [user-management_EN.md](user-management_EN.md).

---

## 4. Database structure

### 4.1 Two kinds of database

| | metadata | vector cluster |
|---|---|---|
| Implementation | PostgreSQL 17 × 1 | Citus + pgvector (1 coordinator + 5 workers) |
| Holds | Images, metadata, chest field, estimates, jobs, audit | Vectors only |
| DDL | [db/metadata-migrations/](../db/metadata-migrations/) | [db/vector-migrations/](../db/vector-migrations/) |

Vectors alone are placed in a separate DB so that vector scans (HNSW) can be distributed across workers.

### 4.2 PostgreSQL version

**PG17 is used.** The version is set in one place, `PG_MAJOR` in `.env`, and metadata and the vector
cluster use the same version.

| | Version | Constraint |
|---|---|---|
| PostgreSQL | 17 | **Latest supported by Citus** (as of 2026-09 there is no PG18 release) |
| Citus | 14.2.0 | |
| pgvector | 0.8.6 | Supports PG13 to 18. No feature differences between versions |

> **Citus sets the upper limit.** pgvector already supports PG18, and its source even has
> branches for PG19. Moving to PG18 is possible only once Citus supports it.

A major version upgrade has **no data file compatibility**, so dump and restore are
required. The procedure is `scripts/upgrade_postgres.sh` ([OPERATIONS_EN.md](OPERATIONS_EN.md) §8).

### 4.3 Schema management

DDL lives only in migrations.

| | Location | First version |
|---|---|---|
| metadata | `db/metadata-migrations/` | `V001__initial_schema.sql` (all tables and the dataset label vocabulary) |
| vector cluster | `db/vector-migrations/` | `V001__image_embedding.sql` (dimensions substituted from `.env` values) |

- `db-migrate` applies only unapplied versions in order and records the version name and a checksum of the body in `schema_migration`.
  Safe to re-run
- **Do not rewrite an applied version.** To change the schema, add a new `V00x`

> **Migration files are baked into the image.** If you run `db-migrate` without rebuilding it,
> it finishes by only reporting "already applied", without the new migrations.
> Use `scripts/redeploy.sh`.

### 4.4 metadata tables

#### Images and metadata

| Table | Role |
|---|---|
| `patient` | Patient. **Does not hold the original ID**; only a hash and an internal sequence number |
| `source_image` | Metadata of one image (view position, dimensions, source) |
| `image_binary` | Bytes of the original image and SHA-256 |
| `finding` | Finding master (14 findings + `No Finding`) |
| `image_finding` | Image-to-finding mapping (dataset labels; **per image**) |
| `chest_asset` | Chest field and lung field masks, bbox, area, quality. Holds the ROI version |
| `finding_estimate` | Finding estimates from the chest field classifier (probabilities for 14 findings). Holds the classifier version |
| `region_asset` | **Not used** (per-lung-region table; the current processing neither reads nor writes it) |

#### Ingestion

| Table | Role |
|---|---|
| `ingest_job` | Progress and totals per job |
| `ingest_item` | Result per image. Records the failure code and details |
| `ingest_validation` | dry-run results |
| `idempotency_key` | Prevents double registration of the same request |
| `outbox` | Commits queue publishing in the same transaction as the DB change |

#### Vector state

| Table | Role |
|---|---|
| `image_vector_state` | Write state per image and the last failure reason |
| `model_version` | Versions of the models in use (the classifier is `EMBEDDING_FINE`) |
| `region_vector_state` | **Not used** (per-region table) |

The vectors themselves are not here (§4.5).

#### Search

| Table | Role | Retention |
|---|---|---|
| `search_audit` | Who searched what and when. **Images as SHA only** | Indefinite |
| `search_result` | Results of asynchronous searches | Indefinite |
| `search_query_image` | The query image itself | Time-limited |
| `search_query_chest` | Query image with the chest field outline overlaid (for the result list display) | Time-limited |
| `search_query_outline` | Lung field outline of the query image (for the thorax ranking display) | Time-limited |
| `search_query_region` | **Not used** (per-region table; only expired-row deletion runs) | Time-limited |

**Everything around the query image is patient imagery, so it is all deleted on the same expiry**
(`QUERY_IMAGE_RETENTION_MINUTES`, default 60 minutes). A resident job deletes expired rows.

#### Evaluation

| Table | Role |
|---|---|
| `evaluation_run` | One evaluation run. What was measured under which conditions |
| `evaluation_metric` | Its results. One row per metric. **Also holds the null baseline values** |

**Kept in the DB because they cannot be regenerated.** `tmp/` is in `.gitignore`, so results measured on EC2
cannot be shared with the local machine as files. Training data (539 MB) and the
intermediate cache (58 MB) are not stored (they can be regenerated / become worthless when versions change).
Details in [OPERATIONS_EN.md](OPERATIONS_EN.md) §6.

#### Users and operations

| Table | Role |
|---|---|
| `app_user` / `password_reset_token` | Authentication |
| `audit_log` | Logins, searches, original image views, ingestion, settings changes |
| `reconciliation_report` | Daily reconciliation results |
| `schema_migration` | Applied migrations |

### 4.5 Vector cluster tables

Only one table, `image_embedding`. 1 row = 1 image.

| Column | Content | Index |
|---|---|---|
| `embedding` | First stage of searches that include text (512 dimensions / BiomedCLIP, chest field crop) | **HNSW** |
| `embedding_fine` | Exhaustive search and second stage of image search (15 dimensions / chest field classifier profile) | None (exact scan) |
| `embedding_lung` | Thoracic shape (16 dimensions) | None |

The primary key includes `embedding_model_version` and `preprocess_version`, so changing the model or
preprocessing version does not mix new vectors with old ones.

> **Not indexing `embedding_fine` is intentional.** 15 dimensions × 110,000 rows fits within tens of ms with a sequential scan
> per shard, and there is no loss from HNSW approximation. In two-stage search, only the candidates narrowed by the
> first stage are re-scored, so there is nothing to approximate in the first place.
> Details in [MULTIMODAL_EN.md](MULTIMODAL_EN.md) §4.

### 4.6 Version handling

| Version | Changed when | Effect |
|---|---|---|
| `roi_model_version` | ROI profile (`algorithm_version` + SHA of the parameters) | Old and new ROIs do not mix |
| `preprocess_version` | How the crop is built | Does not mix with old vectors |
| `embedding_model_version` | Model and revision | Same as above |

**The SHA-256 of the whole profile becomes `parameter_hash`.** There is no inheritance mechanism, in order to keep
the property that the reproducibility key covers the whole profile.

---

## 5. Main design decisions

| Decision | Reason |
|---|---|
| Use the chest field of a single image as the unit | There is no per-region ground truth, so it cannot be trained. Per image, there is supervision |
| Include the mediastinum and diaphragm in the frame | The evidence for cardiomegaly and pleural effusion lies outside the lung field (99.7% vs 58.7% of ground-truth boxes) |
| Use one vector cluster | 1 row per image; there is no unit to split by |
| Image-only search is a profile exhaustive search | Two-stage search misses images with matching findings that the first stage drops. Even exhaustive, p95 is 24 ms |
| Do not index `embedding_fine` | A sequential scan is enough for 15 dimensions, and there is no approximation loss |
| Compare images by the classifier's 15 dimensions | Compare by learned finding probabilities. Finds similar cases better than raw appearance |
| The classifier carries how its features are built | Different versions or preprocessing silently shift probabilities. Checked at startup, which stops on mismatch |
| Show finding estimates in the UI | For the user's judgment. Always accompanied by a "not a diagnosis" heading and a distinction from the labels |
| Lung field extraction uses U-Net | Rule-based extraction fails on 23.8% of the corpus |
| Compare shape on the whole lung field (thorax) | Regions are cut out of the thorax by geometric rules, so they depend on the thoracic shape |
| Japanese translation by dictionary only | Do not send medical information to external APIs |
| Keep query images only for a limited time | They are patient images |

---

## 6. Quality and monitoring

### 6.1 Automatic quality checks

Chest field extraction assigns a `quality_status` to each image. If the lung field is less than 2% of the image, extraction fails
(422, `LUNG_TOO_SMALL`). If one lung is not found, left-right asymmetry is large, or the lung field is too wide, a
flag is raised and it becomes `ROI_UNCERTAIN` (confidence 0.5), but ingestion is not stopped (because this also happens with
collapse of one lung from pneumothorax or massive pleural effusion). If the query image's confidence is below `ROI_MIN_CONFIDENCE`, results are suppressed;
if it is `ROI_UNCERTAIN`, a warning is attached.

### 6.2 Monitoring

Prometheus collects `/metrics` from each service and Grafana displays them. The main metrics are
ingestion time per stage, ROI success/failure, vector write success/failure, and search response time.

### 6.3 Tests

| Type | Location | Content |
|---|---|---|
| Unit | `tests/unit/` | Consistency of contracts, preprocessing, ROI, classifier, and schema |
| Integration | `tests/integration/` | Smoke tests against the running system |

Integration tests actually run ingestion and search. `python tests/integration/smoke_test.py`.
