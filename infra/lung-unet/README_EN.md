# Location of the lung-field segmentation model (U-Net)
English | [日本語](README.md)

Place `lung-unet.onnx` written by `scripts/train_lung_unet.py` here.
compose mounts it read-only at `/models/lung-unet`.

    cp models/lung-unet/lung-unet.onnx infra/lung-unet/

**Required.** The default `ROI_PROFILE=nihcc-unet-local-v1` uses the U-Net, so if the model is
not here, roi-service does not start.

The procedure is in `../../docs/runbooks/lung-unet_EN.md`. The model itself is not committed to git (`.gitignore`).
