# 肺野分割モデル（U-Net）の置き場所
[English](README_EN.md) | 日本語

`scripts/train_lung_unet.py` が書き出した `lung-unet.onnx` をここへ置く。
compose が `/models/lung-unet` へ read-only で mount する。

    cp models/lung-unet/lung-unet.onnx infra/lung-unet/

**必須。** 既定の `ROI_PROFILE=nihcc-unet-local-v1` は U-Net を使うので、ここにモデルが
無いと roi-service が起動しない。

手順は `../../docs/runbooks/lung-unet.md`。モデル自体は git に入れない（`.gitignore`）。
