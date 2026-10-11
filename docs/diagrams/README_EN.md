# Architecture and flow diagrams

English | [日本語](README.md)

An index of the diagrams in this folder. Every diagram exists as both PNG and SVG, side by side under the same name.
The diagrams show the configuration with finding profiles enabled (`EMBEDDING_FINE_BACKEND=chest-profile`).

## Detailed diagrams

Diagrams that go down to the interactions between services, networks, ports, and volumes.

| Diagram | Contents |
|---|---|
| [01-system-architecture.png](01-system-architecture.png) | System architecture: the Docker Compose containers on the host (EC2), grouped by role (UI and API / asynchronous processing and search / AI inference / data / operations), and their relationship to files and scripts on the host |
| [02-process-architecture.png](02-process-architecture.png) | Process architecture: the processes running in each container (nginx, uvicorn, queue consumers, background thread) and the HTTP, AMQP, and SQL communication between them, with endpoints, queue names, and metadata tables |
| [03-container-relations.png](03-container-relations.png) | Container relationships: membership in the Docker networks (edge, app, data, cluster-vector), depends_on, published ports, volumes, and bind mounts |
| [04-flow-training-data.png](04-flow-training-data.png) | Process flow (training data creation): chest field extraction → RAD-DINO features → training data → classifier training and evaluation → export for production |
| [05-flow-ingest.png](05-flow-ingest.png) | Process flow (data ingestion): a sequence from validation and job registration through per-image chest field extraction, vectorization, and registration in the databases |
| [06-flow-search.png](06-flow-search.png) | Process flow (search): a sequence of translation into English, chest field extraction, vectorization, vector search (exhaustive search / BiomedCLIP / two-stage search), and result assembly |

## Simplified explanation diagrams

Diagrams with larger and fewer words, for presentations and explanations.

| Diagram | Contents |
|---|---|
| [stage3-current.png](stage3-current.png) | System architecture including training with training data: Docker Compose on a single EC2 instance (Citus with 1 coordinator + 5 workers). The search, ingestion, and training data creation flows are color-coded |
| [containers-processes.png](containers-processes.png) | Containers and their processes: the processes running in each container and the HTTP, AMQP, and SQL connections between them |
| [preprocess-normalize.png](preprocess-normalize.png) | Image normalization: load → intensity clipping (0.5%–99.5%) → 0–255 → polarity (black/white inversion) check → downscale to a long side of at most 1,024 px |
| [preprocess-roi.png](preprocess-roi.png) | Lung field extraction (ROI): U-Net for right lung, left lung, and heart → binarization → quality check → chest field mask (convex hull, 10% downward, 2% all around) → model input |
| [training-teacher-data.png](training-teacher-data.png) | Creating the training data: image → chest field → RAD-DINO features (768 dimensions) + labels for 14 findings → train / val / test split by patient |
| [training-usage.png](training-usage.png) | Using the training data: train the classifier on train, choose when to stop on val, evaluate on test → used in production for finding profiles and finding estimates |
| [flow-ingest-simple.png](flow-ingest-simple.png) | Image data ingestion: specify → validate → register job → extract chest field → vectorize → register |
| [flow-search-simple.png](flow-search-simple.png) | Image search and result display: input → translate into English → extract chest field → vectorize → search → assemble → display (with a sample result card) |

## How to regenerate

The diagrams are produced by the scripts in this folder. Rendering uses Playwright (headless Chromium).

```bash
pip install playwright && python -m playwright install chromium
python -B docs/diagrams/build_diagrams.py      # run from the repository root
```

| Script | Diagrams |
|---|---|
| `build_diagrams.py` | 01–06 (pass names as arguments to build only those, e.g. `05-flow-ingest`) |
| `build_samples.py` | stage3-current |
| `build_containers.py` | containers-processes |
| `build_preprocess.py` | preprocess-normalize, preprocess-roi |
| `build_training.py` | training-teacher-data, training-usage |
| `build_flows.py` | flow-ingest-simple, flow-search-simple |

The shared drawing helpers are in `svgkit.py`. When the architecture or a flow changes, fix the script and regenerate
(the PNG and SVG files in this folder are overwritten).
