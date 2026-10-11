# Multimodal design specification
English | [日本語](MULTIMODAL.md)

Design of the parts that handle images and language in the same space, and of the processing specific to chest X-rays (chest field, classifier, U-Net,
multi-stage search). **For developers.**

**Abbreviations are collected in [GLOSSARY_EN.md](GLOSSARY_EN.md).**

The overall system architecture is in [DESIGN_EN.md](DESIGN_EN.md); operating procedures are in [OPERATIONS_EN.md](OPERATIONS_EN.md).

---

## 1. Models and libraries

### 1.1 Models

| Purpose | Default | Dimensions | License |
|---|---|---|---|
| First stage (image and text) | **BiomedCLIP**<br>`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | 512 | MIT |
| Second stage backbone | **RAD-DINO**<br>`microsoft/rad-dino` | 768 | MIT |
| Second stage (adopted) | **Chest field classifier profile** (custom; trained on RAD-DINO features) | 15 | - |
| Lung field segmentation | **U-Net** (custom; trained with CheXmask as training data) | - | - |

All of these are published for **research purposes**. The publisher of RAD-DINO states explicitly that it is "not intended
for use in clinical practice". This system is likewise limited to research use.

They can be switched with `EMBEDDING_BACKEND` / `EMBEDDING_FINE_BACKEND`.

| Value | Content |
|---|---|
| `deterministic` | Deterministic stub. For checking wiring and schema. **No medical meaning** |
| `biomedclip` | BiomedCLIP |
| `rad-dino` | RAD-DINO alone (768 dimensions; cannot handle text) |
| `chest-profile` | **Adopted configuration.** RAD-DINO features pooled over the chest field, mapped to 15 dimensions by the classifier |
| `none` | Do not use the second stage |

### 1.2 Libraries

| Layer | Main components |
|---|---|
| Inference (training) | torch 2.5.1 / torchvision / open_clip_torch / timm / transformers |
| Inference (serving) | **onnxruntime** (lung field segmentation; does not bring in torch) |
| Image processing | OpenCV (headless) / scikit-image / Pillow / NumPy |
| API | FastAPI / uvicorn / pydantic |
| DB | psycopg 3 / pgvector / Citus |
| Queue | pika (RabbitMQ) |

> **Training and inference have separate dependencies.** roi-service is a light image with only OpenCV and NumPy;
> adding torch would push it over 2 GB. Lung field segmentation is exported to ONNX and loaded with onnxruntime
> (about 50 MB).

---

## 2. Multimodal design

### 2.1 What goes into the same space

BiomedCLIP maps images and text into the same 512-dimensional space. This means

- **Image → image**: compare chest field crops with each other (the first stage when `IMAGE_SEARCH_STRATEGY=coarse_to_fine`.
  By default, image-only search does not use BiomedCLIP and compares by finding profile; §4)
- **Text → image**: find images from a finding sentence
- **Both**: blend with weights (`HYBRID`)

are all served by a single index.

### 2.2 Handling Japanese

**Medical information is not sent to external translation APIs.** BiomedCLIP's text encoder is an English model,
so Japanese input is converted to English with an approved local dictionary (`ja_en_medical.json`, 72 terms and phrases).

```
「右上肺野の結節影」 → "pulmonary nodule in the right upper lung field"
```

(`右上肺野の結節影` = nodular shadow in the right upper lung field)

**Free text the dictionary cannot handle is not translated; it is returned with 422.** Rather than search with a translation whose quality
cannot be guaranteed, it is better to say it cannot be done. The UI lists the registered standard findings.

Finding labels (such as `Fibrosis`) are also displayed in Japanese, and filters accept Japanese too. Translations and
paraphrases are kept in one place only, `nihcc_rag.csvspec`, and are distributed from the API to the UI.

### 2.3 Some configurations have no text support in the second stage

`rad-dino` and `chest-profile` have no text encoder. Therefore for
`TEXT_TO_IMAGE` / `HYBRID` **the second stage is automatically disabled** and a single-stage search answers.
The response's `models.rerank` indicates this (`disabled-single-stage`).

Conversely, image-only search (`IMAGE_TO_IMAGE`) by default does not use BiomedCLIP and compares all images exactly by
finding profile (`profile-exact-full-scan`; §4.1).

---

## 3. Chest X-ray harness design

### 3.1 Preprocessing

```
  original image → percentile clip (0.5% / 99.5%) → to 0-255 → unify polarity → downscale long side to 1024
```

**Polarity unification** is needed. In a normal chest X-ray, the sides (lung fields) are darker than the center (spine, mediastinum).
Some images are inverted, so this relationship is checked and the image is inverted if needed.

This preprocessing exists in one place only, `image_pipeline.pipeline.normalize`. **Both training and
inference go through the same function.** If they diverge, the input distribution changes.

### 3.2 Evaluation metrics

The fact that labels exist only **per image** is the essential constraint of this problem. Because the search unit was made
a single image, metrics built from per-image labels serve directly as the primary metrics.

| Metric | What it measures | Treatment |
|---|---|---|
| **Per-image AUC** | Whether the classifier can predict findings | **Primary metric** (test set of a patient-level split) |
| **Test B (finding enrichment)** | Whether the same finding appears in the top K similar cases | **Primary metric.** lift = precision / base rate |
| Ground-truth box coverage | Whether ground-truth boxes fall inside the chest field | Validity of the frame (chest field 99.7%) |
| Test A (same-patient reproduction) | Saturates at the ceiling | **Not usable** |

> **Always measure the null baseline.** Even a classifier on image statistics alone (no medical meaning) reaches
> AUC 0.696. Without measuring the null baseline, correlations with imaging conditions are mistaken for finding detection.

---

## 4. Vector multi-stage search

### 4.1 How search proceeds

The procedure differs by search type. The response's `query.strategy` and `models.rerank` show the procedure
actually used.

| Search | Procedure (`query.strategy`) | `models.rerank` |
|---|---|---|
| Image only (default) | `profile`: compare **all images exactly** by finding profile (15 dimensions) | `profile-exact-full-scan` |
| Image only (`IMAGE_SEARCH_STRATEGY=coarse_to_fine` or the request's `strategy`) | `coarse_to_fine`: candidates via BiomedCLIP HNSW → re-score by profile | `two-stage-coarse-to-fine-exact` |
| Text, image + text | `coarse_only`: BiomedCLIP only (the second stage cannot handle text; §2.3) | `disabled-single-stage` |
| `rerank=false` | `coarse_only`: BiomedCLIP only | `disabled-single-stage` |

```
  image only (default)
    └→ compare all rows exactly by the 15-dimensional profile → top k (no index)

  two-stage search (image search with IMAGE_SEARCH_STRATEGY=coarse_to_fine)
    ├→ first stage: k×4 candidates via HNSW (approximate; BiomedCLIP, chest field crop)
    └→ second stage: re-score only the candidates exactly by the 15-dimensional profile → top k
```

**Why image-only search uses exhaustive search**: in two-stage search, images that the first stage did not pick up are
never considered for re-ranking, however well their findings match. The first stage returns at most 1,000 candidates
(about 0.9% of the total). Results of comparing the two on the actual search path (via the API)
(lift of the top 20, 50 queries per finding):

| Finding | Two-stage search | **Exhaustive search** |
|---|---:|---:|
| Nodule | 1.91 | **2.43** |
| Pneumonia | 1.25 | **2.51** |
| Emphysema | 5.93 | **8.96** |
| Hernia | 38.2 | **139.2** |

Exhaustive search wins on 13 of 14 findings, and vector search p95 is 24 ms versus 290 ms for two-stage search
(because the HNSW exploration with ef_search 1,000 is not needed). The sequential scan of 15 dimensions × 110,000 rows is
split by Citus across 30 shards and run in parallel.

Searches that include text do not use exhaustive search because the profile is built from the image, so the meaning
of the text would no longer affect the ranking.

### 4.2 Why the profile has 15 dimensions

**Higher dimensions are not necessarily better.** On the full dataset evaluation of the chest field, Test B (lift of the top 20) when comparing
the same features as raw 768 dimensions versus comparing by the classifier's 15-dimensional profile is
as follows.

| Finding | 768-dim features (standardized) | **15-dim profile** |
|---|---:|---:|
| Cardiomegaly | 4.60 | **10.37** |
| Pneumothorax | 4.61 | **5.56** |
| Mass | 2.83 | **4.76** |
| Nodule | 2.01 | **3.07** |
| Atelectasis | 1.74 | **2.41** |

There are three reasons.

1. **Profile comparison is not approximate search.** There is no index, so the advantage "higher dimensions improve index accuracy"
   has nowhere to apply
2. **The 15 dimensions are the 768-dimensional space viewed from the direction of the findings.** The classifier is a single-layer
   linear map (`W` is 768×14) that projects onto the 14 directions that linearly predict the findings.
   What it discards are the 754 directions orthogonal to those
3. **The number 15 is determined by the label space.** It is 14 findings + normality, not
   a search hyperparameter

> **This comparison is confounded.** What was measured is "768-dimensional raw self-supervised features" versus
> "15-dimensional supervised projection"; no comparison was done that changes only the output dimension under the same supervision.
> All that can be said is "the supervised projection beat the raw high-dimensional features".

### 4.3 Score handling

There is a single ranking, so no calibration to align distributions is needed. Candidates re-ranked by two-stage search have `reranked` set, and the first-stage similarity
(`coarse_similarity`) is also returned. Candidates from exhaustive search do not have a BiomedCLIP similarity computed,
so `coarse_similarity` is null and `reranked` is false.

### 4.4 Search by thoracic shape

As a separate axis, there is a section that ranks by the shape of the whole lung field (16 dimensions). In pneumothorax the lung collapses and shrinks;
in cardiomegaly the cardiac silhouette widens and pushes the lungs outward. It is an axis for findings that **appear in shape rather than in
how the pixels look**.

**It is not a finding discriminator.** Its discriminative power has been measured to be worse than other shape descriptors;
it is positioned as a tool for "lining up cases with similar build and similar imaging for side-by-side comparison".

In the UI, both candidates and the query image are shown **with the lung field outline drawn**. Without seeing what was compared as
the "lung field", you cannot check whether the spine, cardiac silhouette, and diaphragm were excluded correctly.

---

## 5. Chest field data design

**The search unit is the chest field of a single image** (lung field + mediastinum + diaphragm). This is because the labels are per image
and there is no per-region ground truth ([DESIGN_EN.md](DESIGN_EN.md) §1.1).

**ROI** (Region of Interest) here refers to the chest field.

### 5.1 What is built

```
  normalize → U-Net (right lung, left lung, heart) → convex hull of the union of the 3 → band downward → thin margin all around
                                                  ↓
                     chest field mask, lung field mask, shape vector, input to the embedding model
```

1. Fill with the **convex hull** of the union of right lung, left lung, and heart. This includes the mediastinum, cardiac silhouette, trachea, and the area above the line
   joining the costophrenic angles on both sides (the diaphragm dome)
2. Add 10% of the lung field height **downward only**. This includes blunted costophrenic angles and the diaphragm
3. Expand by 2% of the lung field width all around. This includes the pleura (the outer edge of a pneumothorax)

On a frontal view the spine overlaps the mediastinum, so **the spine alone cannot be excluded**.

Spine estimation and lobe segmentation are not done. On the full dataset, extraction failed on 54 / 112,120 images, all of them
images whose lung field was too small.

### 5.2 Evaluation and production use the same function

Normalization → segmentation → mask → downscale → PNG encoding is enclosed in `extract_chest` / `chest_model_inputs` in
`image_pipeline.chest`, and both the evaluation script and roi-service
call it. **The classifier was trained on features built by the evaluation script**, so if the production
input differs by even one pixel, only the probabilities shift, without raising an exception.

- roi-service builds the input to the embedding model, and ingest-worker and backend-api
  pass it on as is without rebuilding it
- In production, `chest_image_eval.py verify-ingest` checks that roi-service's input matches the evaluation input
  (byte for byte) and that the probabilities match

### 5.3 U-Net is required for lung field extraction

| profile | Method | Input |
|---|---|---|
| `nihcc-unet-local-v1` | U-Net (reduced version; the classifier was trained with this) | 160px |
| `nihcc-unet-v1` | U-Net (for production) | 384px |
| `nihcc-classical-v1` | Rule-based. **Does not output the heart, so it cannot be used for the chest field** | - |

**Models with different input resolutions are not compatible**, so they have separate profiles. The classifier records and carries the profile
used for training, and the ingestion worker checks it against roi-service at startup.

### 5.4 The whole profile is the reproducibility key

`algorithm_version` + the SHA-256 of the whole profile is `roi_model_version`.
**There is no inheritance mechanism**, in order to keep the property that the reproducibility key covers the whole profile;
as a consequence, sections are duplicated across profiles.

### 5.5 Quality checks

If the lung field is less than 2% of the image, extraction fails (422, `LUNG_TOO_SMALL`). If one lung is not found,
left-right asymmetry is large, or the lung field is too wide, a flag is raised and it becomes `ROI_UNCERTAIN` (confidence 0.5),
but ingestion is not stopped. This is because it also happens with collapse of one lung from pneumothorax or massive pleural effusion.

---

## 6. Chest field classifier

### 6.1 Why training is needed

Unsupervised appearance embeddings (raw features of BiomedCLIP and RAD-DINO) only capture imaging conditions and
build, and barely retrieve matching findings. Because the unit is a single image, per-image labels
become the training data directly, and a linear classifier per finding can be trained. MIL (multiple instance learning) to estimate findings per region
is not needed.

### 6.2 Structure

```
  whole image (518 square) + chest field mask → RAD-DINO (frozen) → mask-weighted average of patch tokens
                                                        ↓
                                      768 dimensions (L2 normalized) → standardize (train statistics)
                                                        ↓
                                      linear + sigmoid per finding → probabilities for 14 findings
                                                        ↓
                                      + "normality" → 15 dimensions (L2 normalized) = profile
```

- The backbone is frozen and only the linear head was trained (patient-level split 70 / 10 / 20)
- The probabilities are used both for the second-stage profile and for the "finding estimates" in the UI
- Estimates are computed at ingestion time and stored in `finding_estimate`, not computed on every search

### 6.3 Results (full dataset; test 22,983 images)

| Representation | Mean per-image AUC | 95% interval |
|---|---:|---|
| **Pooling over chest field mask (adopted)** | **0.833** | [0.824, 0.839] |
| Pooling over lung field mask | 0.829 | [0.821, 0.835] |
| Pooling over whole image | 0.826 | [0.816, 0.832] |
| Crop with the chest field blacked out | 0.815 | [0.805, 0.821] |
| Null baseline (image statistics) | 0.696 | [0.686, 0.705] |

The null baseline is high because imaging conditions (portable imaging, etc.) correlate with findings; it is reasonable to read
what comes from the image content as the gain over the null baseline (0.137).

### 6.4 Artifact

`nihcc_rag.chest_head.ChestHead`. In addition to the standardization mean and standard deviation, the linear weights, and the finding order,
it carries **how the features are built**.

| Checked item | Reason |
|---|---|
| Backbone model name and revision | If the weights change, the same name gives different features. **Unpinned (`unpinned`) is rejected** |
| Pooling resolution (518) | The patch grid and mask reduction change |
| How the chest field mask is built (10% / 2%) | If the frame changes, the pooling region changes |
| U-Net profile | The lung and heart masks change |

embedding-service checks the backbone and resolution, and ingest-worker checks the mask and profile, at startup,
and stops on mismatch. Export is done with `scripts/export_chest_head.py`.

### 6.5 The classifier barely indicates lesion location

Because the classifier is linear, a finding's logit decomposes **exactly** into the sum of contributions from each cell of a 37×37 grid
(embedding-service's `/api/v1/embeddings/chest-map`). We measured how often the maximum cell of this map falls inside a ground-truth box
(`BBox_List_2017.csv`) (388 images).

| Finding | Map maximum | Random point in chest field | Bright point in lung field | Verdict |
|---|---:|---:|---:|---|
| Mass | **51.8%** | 8.0% | 5.9% | Indicates location |
| Infiltration | 39.0% | 20.8% | 11.4% | Does not reach the 2x line |
| Pneumonia | 35.8% | 19.1% | 12.5% | Same as above |
| Nodule | 0.0% | 1.0% | 0.0% | Does not indicate |

- **Only Mass indicates location.** Size is not indicated for any of the 4 findings (rank correlation ≤ 0.28). Shape cannot be measured because
  the ground truth is only rectangles. Fibrosis has no ground-truth boxes
- **"Bright point in lung field" is below random and is no clue to location.** Overlapping ribs, vessels, and clavicles
  are also bright
- This is the limit of a classifier trained on per-image labels and of frozen ViT patch features.
  To use location, size, and shape in search, a lesion detection model would have to be trained on external data with ground-truth boxes or contours.
  Adding more search stages does not solve it (because there is no representation of location)

---

## 7. U-Net design

**U-Net is not an abbreviation.** The name comes from the network's U shape (contract, then expand, connecting layers at the same level
sideways). It was proposed by Ronneberger et al. (2015) for biomedical image
segmentation (per-pixel classification), and here it is used for **lung field
extraction**.

### 7.1 Why U-Net

Rule-based lung field extraction by intensity thresholds (`nihcc-classical-v1`) **fails on 23.8% of the
corpus** (Dice < 0.7 or extraction failure).

The failure mode is about "location", not "size". It grabs the neck, shoulders, mediastinum, or abdomen instead of the lungs.
Tuning the rules does not fix it.

### 7.2 Training data

**CheXmask Database v1.0.0** (PhysioNet, CC BY 4.0). ChestX-ray8, that is, the very images this system
uses, come with left lung, right lung, and **heart** masks.

> **Not hand-drawn by experts.** They were produced by a model called HybridGNet, with quality estimated by Reverse Classification
> Accuracy. Therefore do not read the absolute Dice value as true performance.
> What matters is the difference from the rule-based method and the visible reduction in failures.

### 7.3 Structure

| | |
|---|---|
| Input | 1 channel, 384×384 (preprocessing is the same `normalize` as at inference) |
| Output | 3 channels (right lung, left lung, heart) |
| Depth | 4 levels, base width 32 (about 7.8 M parameters) |
| Loss | BCE + soft Dice |
| Augmentation | rotation ±10°, scaling 0.9 to 1.1, translation, brightness, **horizontal flip** |

### 7.4 Implementation notes

- **The split is by patient.** If images of the same patient straddle the training and validation sides, results look
  better than true performance
- **Horizontal flips also swap the right lung and left lung channels.** Without the swap, the model
  confuses left and right
- **Labels are packed as bits into a single uint8** (1=right lung 2=left lung 4=heart). The heart can overlap
  the lungs, so mutually exclusive multi-class values are not possible
- **Upscale back to the original resolution while still probabilities, then apply the threshold.** Thresholding first and then enlarging
  makes the boundary staircase-shaped and degrades the shape features

### 7.5 Results

| | Rule-based | U-Net (160px reduced version) |
|---|---|---|
| Median Dice | 0.828 | **0.968** |
| Dice < 0.7 | 22.2% | **0.4%** |
| Extraction failure | 2.0% | **0.1%** |
| Effective failure | **23.8%** | **0.5%** |

1,000 random images. On the 748 unseen patients alone, the median is still 0.968, unchanged.

> Because it was trained on CheXmask and evaluated against CheXmask as reference, **these numbers are biased
> upward.** What matters is the difference: 0.5% failures versus 23.8%.

### 7.6 Serving

Training uses torch (a dedicated image); inference uses **onnxruntime only**. The ONNX is packed into a single file including the weights
and mounted read-only into the model volume.

If the profile's `input_size` and the model's input disagree, onnxruntime throws a confusing exception,
so **they are checked against each other at startup and it stops with a named error.**

---

## 8. How evaluation is done

### 8.1 Do not judge on a few hand-picked images

Adoption decisions are made by measuring on a **random sample**. Visual inspection of a few hand-picked images does not show what share of the whole
fails. **Hand-picked images are used only to reproduce symptoms, never for adoption decisions.**

### 8.2 Measurement tools

| Target | Tool |
|---|---|
| ROI | `scripts/eval_roi_dice.py` (Dice / IoU with CheXmask as reference) |
| Chest field, classifier | `scripts/chest_image_eval.py` (coverage, training data, AUC, Test B, null baseline) |
| Production search path | `scripts/chest_search_eval.py` (Test B via the API; compares exhaustive search, two-stage search, and first stage only) |
| Lesion location | `scripts/chest_lesion_map_eval.py` (classifier maps and ground-truth boxes) |
| Production feature match | `scripts/chest_image_eval.py verify-ingest` |
| Per-region signal (not used on the production path) | `scripts/region_*.py` (per-lung-region signal, MIL evaluation) |

All of them include **comparison with the null baseline**. Do not claim "it worked" without measuring the null baseline.
