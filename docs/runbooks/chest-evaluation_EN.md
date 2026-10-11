# Measuring search quality and the lesion location map (EC2)

English | [日本語](chest-evaluation.md)

Measurements performed in an environment where the finding profile is enabled and the full dataset has been ingested ([chest-profile-setup_EN.md](chest-profile-setup_EN.md)).
Neither requires re-ingestion or a migration.

| | Measurement | Script | Time |
|---|---|---|---|
| §1 | Search quality through the actual search path (Test B) | `chest_search_eval.py` | 2–3 hours |
| §2 | Whether the classifier's per-cell map indicates lesion location and size | `chest_lesion_map_eval.py` | 30 min |

```bash
cd ~/NIHCC_RAG && source ~/.venv-eval/bin/activate   # the virtual environment from chest-classifier.md §1.1
```

---

## 1. Measuring search quality

The classifier evaluation (`chest_image_eval.py train`) measures only the second stage with numpy. Here we measure
**through the backend-api API**, using the same search path as production.

### 1.1 Method

- Queries are taken **only from images of patients assigned to test in the classifier evaluation** (using the `--teacher` split).
  The classifier is trained on train patients, so measuring with train images inflates the numbers
- Up to N images per finding with a single finding (only one label). `No Finding` is included too
- Three variants are issued per image

  | Name | Search | Request |
  |---|---|---|
  | `profile` | Exact exhaustive search with the finding profile (default) | `strategy=profile` |
  | `two_stage` | Collect candidates with BiomedCLIP, then re-rank | `strategy=coarse_to_fine` |
  | `stage1` | BiomedCLIP only | `rerank=false` |

- Look at K results from the top **excluding images of the same patient** (other studies of the same patient are naturally similar)
- lift = fraction of the same finding in the top K ÷ its fraction in the whole DB. Random gives 1.0

### 1.2 Running

First check that the smoke test passes.

```bash
SMOKE_PASSWORD="$(grep '^LOCAL_USERS=' .env | cut -d= -f2- | cut -d, -f1 | cut -d: -f2)" python tests/integration/smoke_test.py
```

```bash
python scripts/chest_search_eval.py \
  --teacher tmp/chest/ec2-full/teacher_nihcc-unet-local-v1_112066.npz \
  --queries-per-finding 50 --top-k 20 --json tmp/chest/search.json
```

- The endpoint is `--api` (default `http://localhost:8000`). The user is the admin
  from `LOCAL_USERS` in `.env` (can be overridden with `--user` / `--password`)
- `--teacher` is the training data tracked in git. If you rebuilt the classifier, specify that run's
  `tmp/chest/ec2-full/teacher_*.npz`
- 750 queries × 3 variants takes 2–3 hours (GPU). Run it inside tmux

At the end it prints a table of lift per finding, p50 / p95 of vector search time (aggregator `timing_ms`),
and a verdict. The verdict criteria are in the script's docstring.

Measured with this repository's configuration (reference):

| Item | Value |
|---|---|
| `profile` lift ≥ 1.1 | 14 / 14 findings |
| `profile` lift ≥ `two_stage` | 13 / 14 findings |
| p95 of vector search time | `profile` 24 ms / `two_stage` 290 ms |

`two_stage` falls short of `profile` because the first stage (BiomedCLIP HNSW) misses images with matching findings
from its candidates.

### 1.3 Checking exhaustive search speed directly in the DB (optional)

```bash
docker compose exec vector-coordinator psql -U postgres -d vector -c "
EXPLAIN (ANALYZE, SUMMARY) SELECT image_id FROM image_embedding
WHERE embedding_fine IS NOT NULL
ORDER BY embedding_fine <=> (SELECT embedding_fine FROM image_embedding WHERE embedding_fine IS NOT NULL LIMIT 1)
LIMIT 200;" | tail -3
# Execution Time: tens to hundreds of ms
```

The default strategy for image search can be changed with `IMAGE_SEARCH_STRATEGY` in `.env`
([OPERATIONS_EN.md](../OPERATIONS_EN.md) §3.4).

---

## 2. Measuring the lesion location map

The chest field classifier is "RAD-DINO patches weighted-averaged over the chest field → standardization → linear", so
the logit decomposes exactly into per-cell (37 × 37) contributions. embedding-service's
`/api/v1/embeddings/chest-map` returns those contributions as a map. We measure whether this map indicates lesion location and
size using the ground-truth boxes (`BBox_List_2017.csv`).

### 2.1 Method

The targets are the 4 findings with ground-truth boxes (Fibrosis has no ground-truth boxes). We use the 388 images that have a ground-truth box
for any of the 4 findings.

| Finding | Ground-truth boxes | Size (median area) |
|---|---:|---:|
| Infiltration | 123 | 5.7% |
| Mass | 85 | 1.5% |
| Nodule | 79 | 0.5% |
| Pneumonia | 120 | 5.7% |

For each image and finding, output one point saying "the lesion is here", and compare whether it falls inside a ground-truth box (hit).

| Method | Description |
|---|---|
| **map** | The maximum cell of the classifier's map (inside the chest field) |
| bright | The point that stands out white within the lung fields (maximum of white top-hat) |
| prior | The average location of that finding's boxes (leave-one-out) |
| random | A random point inside the chest field (hit probability computed analytically) |

In addition, it outputs the fraction of the map's positive values that fall inside boxes (enrichment; 1 is random) and the rank correlation between
the area of the map's blob and the box area (size ρ).

For each finding, if map's hit is at least 2× random, exceeds prior, and the lower bound of its 95% interval exceeds
random — all three — the verdict is "the map indicates that finding's location". If size ρ exceeds 0.3,
it is "also represents size".

### 2.2 Exposing embedding-service

Only during the measurement, expose embedding-service to the host with the evaluation override. embedding-service
stops for a few minutes, so **do not do this while ingestion or the §1 measurement is running**.

```bash
ls tmp/gold/BBox_List_2017.csv                 # tracked in git
cp .env tmp/env.before-lesion-map              # restored in §2.4

setenv() { if grep -q "^$1=" .env; then sed -i "s#^$1=.*#$1=$2#" .env; else echo "$1=$2" >> .env; fi; }
setenv COMPOSE_FILE docker-compose.yml:infra/compose/docker-compose.vectors.yml:infra/compose/docker-compose.gpu.yml:infra/compose/docker-compose.eval.yml
setenv ROI_SERVICE_REPLICAS 1                  # the override also exposes roi-service, so 1 replica

docker compose up -d
docker compose ps embedding-service            # wait until (healthy) (1–2 minutes)
curl -s http://127.0.0.1:8002/api/v1/model | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['fine_embedding_model_version'], bool(d.get('chest_head')))"
# should print chest-…-d15 True
```

### 2.3 Measuring

Chest field extraction runs on the host (U-Net).

```bash
LUNG_MODEL_PATH=infra/lung-unet/lung-unet.onnx \
  python scripts/chest_lesion_map_eval.py --service-url http://127.0.0.1:8002
```

At the end it prints a table per finding and a verdict.

```
所見            枚     map  bright   prior  random     map 95%区間  enrich  大きさρ  判定
Infiltration  ...
```

(Columns: finding, images, map, bright, prior, random, map 95% interval, enrich, size ρ, verdict.)

- `tmp/chest/lesion-map.json` … results per finding and a record per image
- `tmp/chest/lesion-map-sheet.png` … **36 random cases** with the map (red), ground-truth boxes (green), the map's point (red circle), and
  the bright point (blue circle) overlaid. Check by eye together with the numbers

### 2.4 Restoring

```bash
cp tmp/env.before-lesion-map .env              # removes the override, closes the ports, and restores the roi-service replica count
docker compose up -d --remove-orphans
docker compose ps                              # everything should be back to (healthy) / Up
```

---

## Copying results back to your machine

```bash
# on your local PC (at the repository root). Windows scp cannot expand {a,b}, so one file at a time
scp <ec2-host>:/home/ubuntu/NIHCC_RAG/tmp/chest/search.json tmp/chest/
scp <ec2-host>:/home/ubuntu/NIHCC_RAG/tmp/chest/lesion-map.json tmp/chest/
scp <ec2-host>:/home/ubuntu/NIHCC_RAG/tmp/chest/lesion-map-sheet.png tmp/chest/
```
