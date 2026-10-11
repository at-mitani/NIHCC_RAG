# Location of the chest-field classifier
English | [日本語](README.md)

Place the trained classifier (`head-pool-chest.npz`) written by `scripts/export_chest_head.py`
here. `docker-compose.yml` mounts it **read-only** at `/models/chest`, and
the second stage of embedding-service (`EMBEDDING_FINE_BACKEND=chest-profile`) reads it.

```bash
python scripts/export_chest_head.py \
  --head tmp/chest/ec2-full/head_pool-chest_nihcc-unet-local-v1.npz \
  --train-json tmp/chest/ec2-full/train.json \
  --backbone-revision <RAD-DINO snapshot> \
  --output infra/chest/head-pool-chest.npz
```

The artifact records and carries its **feature recipe** (backbone revision, pooling resolution, how the chest-field mask
is made), and embedding-service checks it against its own configuration at startup.
If they differ, startup is stopped. This is because if even one aspect of the feature recipe differs, the classifier
raises no exception and only its probabilities drift (`nihcc_rag/chest_head.py`).

How to build it is in `docs/runbooks/chest-classifier_EN.md`; the procedure for deploying and building is in `docs/runbooks/chest-profile-setup_EN.md`.

It is not referenced in configurations that do not use the second stage (`EMBEDDING_FINE_BACKEND=none`).
