# Glossary
English | [日本語](GLOSSARY.md)

Abbreviations and terms that appear in this system's documents. **For first-time readers.** The Japanese term used in the UI,
code comments, and the Japanese documents is shown in parentheses.

At the end of each entry is where to find a detailed explanation.

---

## The three core concepts

### ROI — Region of Interest (関心領域)

A general term for the **part of an image you want to focus on**. In medical imaging it is used in the sense of "measure this area"
or "cut out only this area".

In this system it refers to the **chest field (胸部野)** (lung field (肺野) + mediastinum (縦隔) + diaphragm (横隔膜)). From one chest X-ray,
the process that builds

- the lung field mask (where the lungs are)
- the chest field mask, bbox, and crop (lungs, heart, the space between, down to the diaphragm)

is called "ROI extraction (ROI 抽出)". Search compares chest fields with each other. The lung field is not split into regions (部位) because
there is no per-region ground-truth data.

→ [MULTIMODAL_EN.md §5](MULTIMODAL_EN.md) / processing flow in [DESIGN_EN.md §3.1](DESIGN_EN.md)

### Chest field classifier (胸部野の分類器) (finding estimates (所見の推定))

A trained linear classifier that outputs a probability for each of the 14 findings (所見) from the image features of the chest field. The dataset
labels (データセットのラベル) are per image, so if the unit is a single image they become the training data (教師データ) directly.
The probabilities are used for the second stage of search (re-ranking) and for the UI's "finding estimates (for research use only, not a diagnosis)"
(「所見の推定（研究用・診断ではありません）」). **They are separate from the dataset labels, and they are not a diagnosis.**

→ [MULTIMODAL_EN.md §6](MULTIMODAL_EN.md)

### MIL — Multiple Instance Learning (多重インスタンス学習)

A framework for problems where **only the "bag" has a label, and it is unknown which "instances" inside it apply**.
This system uses a single image as the unit and trains directly on per-image labels, so MIL is not used on the search path.

### U-Net

**Not an abbreviation.** The name comes from the network's U shape (contract, then expand, connecting layers at the same level
sideways). It was proposed by Ronneberger et al. (2015) for biomedical image
**segmentation (セグメンテーション)** (per-pixel classification).

In this system it is used for **lung field extraction (肺野の切り出し)**. Rule-based extraction by intensity thresholds fails on about
one quarter of the corpus and cannot output a heart mask either, so U-Net is required.

→ [MULTIMODAL_EN.md §7](MULTIMODAL_EN.md)

---

## Image processing

| Term | Meaning |
|---|---|
| **mask** | A black-and-white image indicating per pixel "this is the target" |
| **bbox** | bounding box. The smallest rectangle enclosing the target |
| **crop** | A part cut out of an image. The unit compared in search |
| **Segmentation (セグメンテーション)** | Classifying "what it is" for each pixel |
| **Normalization (正規化, normalize)** | Preprocessing that aligns the brightness range, orientation, and size |
| **Dice / IoU** | Metrics of how much two masks overlap. 1.0 is a perfect match |

→ preprocessing in [MULTIMODAL_EN.md §3.1](MULTIMODAL_EN.md), evaluation in [§8](MULTIMODAL_EN.md)

## Search and vectors

| Term | Meaning |
|---|---|
| **Embedding (埋め込み)** | An image or sentence converted into a sequence of numbers (a vector) |
| **Dimension (次元)** | The length of a vector. 512 dimensions means 512 numbers |
| **Cosine distance (cosine 距離)** | How close vectors are in direction. Smaller means more similar |
| **Approximate nearest neighbor (近似最近傍, ANN)** | A method that quickly finds "roughly close" items without examining everything |
| **HNSW** | A type of ANN index. Used in the first stage (BiomedCLIP) of searches that include text |
| **Multi-stage search (多段検索) (two-stage search (2 段検索))** | Narrow candidates in the first stage (1 段目), then re-rank them exactly in the second stage (2 段目). Used for searches that include text |
| **Exhaustive search (全件検索) (profile)** | Compare exactly against all rows without an index. The default for image-only search (15-dimensional finding profile (所見プロファイル)) |
| **Re-ranking (再ランキング, rerank)** | Re-assigning the order in the second stage |
| **backfill** | Filling in values afterwards for rows that already exist |

→ [MULTIMODAL_EN.md §4](MULTIMODAL_EN.md)

## Models

| Term | Meaning |
|---|---|
| **BiomedCLIP** | A model that maps images and sentences into the same space. Used in the first stage |
| **RAD-DINO** | An image model specialized for chest X-rays. Produces the input to the chest field classifier |
| **backbone** | The base model that extracts features. Used frozen in this system |
| **Freezing (凍結, freeze)** | Not updating the weights during training |
| **ONNX** | A model storage format independent of the training framework. Light if used only for inference |
| **mask pooling** | Taking a weighted average of patch tokens by mask coverage, collapsing the features inside the frame into one vector |

→ [MULTIMODAL_EN.md §1](MULTIMODAL_EN.md) / [§6.2](MULTIMODAL_EN.md)

## Data and evaluation

| Term | Meaning |
|---|---|
| **ChestX-ray14** | Chest X-ray dataset published by NIH (112,120 images) |
| **CheXmask** | Public data adding lung field masks to ChestX-ray8 and others |
| **gold box (正解 box, ground-truth box)** | Lesion boxes drawn by humans as ground truth (880 images). Used to measure the validity of the frame (whether boxes fall inside the chest field) |
| **Null baseline (帰無, null)** | Performance "when nothing works". The baseline for comparison |
| **AUC** | A metric of how well classes are separated. 0.5 is a random guess |
| **lift** | How many times better than the null baseline |

→ [MULTIMODAL_EN.md §3.2](MULTIMODAL_EN.md)

> **Always measure the null baseline.** Even image statistics alone (no medical meaning) reach AUC 0.696. Claiming
> "it worked" without measuring it means mistaking correlations with imaging conditions for finding detection.

## Infrastructure

| Term | Meaning |
|---|---|
| **Citus** | An extension that distributes PostgreSQL across multiple nodes |
| **pgvector** | An extension for handling vectors in PostgreSQL |
| **shard** | A fragment of a distributed table |
| **coordinator / worker** | Citus's control node and working nodes |
| **migration** | A mechanism for changing the schema step by step |
| **profile** | The full set of ROI extraction parameters. The SHA-256 of the whole becomes the reproducibility key |

→ [DESIGN_EN.md §2](DESIGN_EN.md) / [§4](DESIGN_EN.md)

---

## Version terms

For reproducibility, which version to bump when something changes is defined.

| Version | Bumped when this changes |
|---|---|
| `roi_model_version` | ROI profile (`algorithm_version` + SHA of the parameters) |
| `preprocess_version` | How the crop is built |
| `embedding_model_version` | Model and its revision |

**Vectors of different versions do not mix.** This is because the versions are included in the primary key.

→ [DESIGN_EN.md §4.6](DESIGN_EN.md)
