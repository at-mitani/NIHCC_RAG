# 胸部野の分類器の置き場
[English](README_EN.md) | 日本語

`scripts/export_chest_head.py` が書き出す学習済みの分類器（`head-pool-chest.npz`）を
置く。`docker-compose.yml` が `/models/chest` へ **read-only** で mount し、
embedding-service の 2 段目（`EMBEDDING_FINE_BACKEND=chest-profile`）が読む。

```bash
python scripts/export_chest_head.py \
  --head tmp/chest/ec2-full/head_pool-chest_nihcc-unet-local-v1.npz \
  --train-json tmp/chest/ec2-full/train.json \
  --backbone-revision <RAD-DINO の snapshot> \
  --output infra/chest/head-pool-chest.npz
```

成果物は**特徴の作り方**（backbone の revision、pooling の解像度、胸部野 mask の
作り方）を記録して持ち、embedding-service が起動時に自分の構成と照合する。
ずれていれば起動を止める。分類器は特徴の作り方が 1 つでも違うと、例外を出さずに
確率だけがずれるためである（`nihcc_rag/chest_head.py`）。

作り方は `docs/runbooks/chest-classifier.md`、配置して構築する手順は `docs/runbooks/chest-profile-setup.md`。

2 段目を使わない構成（`EMBEDDING_FINE_BACKEND=none`）では参照されない。
