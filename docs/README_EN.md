# Documentation index
English | [日本語](README.md)

The documentation is split into two kinds: the **design specification** and the **operations manual**.

---

## Design specification (for developers)

| Document | Contents |
|---|---|
| [GLOSSARY_EN.md](GLOSSARY_EN.md) | **Abbreviations and terms** (ROI / MIL / U-Net, etc.). Good to read first |
| [DESIGN_EN.md](DESIGN_EN.md) | System architecture, container layout, process flows, DB and table specifications |
| [MULTIMODAL_EN.md](MULTIMODAL_EN.md) | Models and libraries used, multimodal design, chest X-ray harness, multi-stage search, ROI / MIL / U-Net |

The DDL is in [db/metadata-migrations/](../db/metadata-migrations/) and
[db/vector-migrations/](../db/vector-migrations/).

## Operations manual (for operators)

| Document | Contents |
|---|---|
| [../README_EN.md](../README_EN.md) | Entry point. Start/stop and **how to use the UI** |
| [../SETUP_LOCAL_EN.md](../SETUP_LOCAL_EN.md) | Setup on a local PC |
| [../SETUP_EC2_EN.md](../SETUP_EC2_EN.md) | Setup on an EC2 instance with GPU |
| [diagrams/](diagrams/README_EN.md) | **Architecture and flow diagrams** (system, process, containers; training data creation, ingestion, search) |
| [OPERATIONS_EN.md](OPERATIONS_EN.md) | Day-to-day operation, applying updates, **troubleshooting** |
| [user-management_EN.md](user-management_EN.md) | Managing users and passwords |

### Individual runbooks (runbooks/)

Procedures for tasks that are rarely performed.

| Document | Contents | Time required |
|---|---|---|
| [runbooks/lung-unet_EN.md](runbooks/lung-unet_EN.md) | Train and deploy the lung-field segmentation model (U-Net) | 2–3 hours |
| [runbooks/chest-classifier_EN.md](runbooks/chest-classifier_EN.md) | Create chest-field training data, then train, evaluate, and export the classifier | 4–5 hours |
| [runbooks/chest-profile-setup_EN.md](runbooks/chest-profile-setup_EN.md) | Enable finding profiles (chest-profile), build on EC2, and ingest the full dataset | 1 day |
| [runbooks/chest-evaluation_EN.md](runbooks/chest-evaluation_EN.md) | Measure search quality and whether the classifier's map indicates lesion location | 3–4 hours |

---

## Where to write what

| What you want to write | Where |
|---|---|
| Current architecture and specifications | `DESIGN.md` / `MULTIMODAL.md` |
| Procedures (recurring) | `OPERATIONS.md` |
| Procedures (rarely performed) | `runbooks/` |
| How to use the UI | `../README.md` |
