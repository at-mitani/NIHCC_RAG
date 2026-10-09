"""肺野・心臓の分割モデル（U-Net）を学習し、ONNX へ書き出す。

    # 動作確認（数十秒）
    python scripts/train_lung_unet.py --smoke

    # 本番（EC2 の GPU で 30 分ほど）
    python scripts/train_lung_unet.py --epochs 40 --batch 16

入力は `scripts/build_lung_dataset.py` が作った配列。出力は

    models/lung-unet/lung-unet.onnx    推論に使う（roi-service は onnxruntime で読む）
    models/lung-unet/best.pt           学習の再開・追加学習用
    models/lung-unet/metrics.json      epoch ごとの検証 Dice

## なぜ ONNX で出すのか

推論側（roi-service）は OpenCV + NumPy だけの軽い image（python:3.11-slim）である。
ここへ torch を入れると image が 2 GB 以上になり、起動も遅くなる。onnxruntime なら
50 MB ほどで CPU 推論も速い。**学習と推論で依存を分ける**のが要点で、学習は既存の
torch 入り embedding-service image を借りて行う（手順は docs/OPERATIONS.md から辿る）。

## 左右反転の扱い

水平反転は有効な水増しだが、**反転すると右肺と左肺が入れ替わる。** ラベルの
channel も一緒に入れ替えないと、左右を取り違えたモデルになる。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:                                            # noqa: BLE001
    print("torch が無い。docs/runbooks/lung-unet.md の手順で torch 入りの環境から実行する",
          file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "lung-dataset"
MODELS = ROOT / "models" / "lung-unet"

#: build_lung_dataset.py の bit と対応させる
CHANNELS = ("right_lung", "left_lung", "heart")
BITS = (1, 2, 4)


# --- モデル -----------------------------------------------------------------


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """素直な U-Net。肺野分割は易しい課題なので、深さ 4・基幅 32 で足りる。"""

    def __init__(self, out_ch: int = len(CHANNELS), base: int = 32) -> None:
        super().__init__()
        widths = [base, base * 2, base * 4, base * 8]
        self.downs = nn.ModuleList()
        in_ch = 1
        for width in widths:
            self.downs.append(conv_block(in_ch, width))
            in_ch = width
        self.bottom = conv_block(widths[-1], widths[-1] * 2)
        self.ups = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for width in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(width * 2, width, 2, stride=2))
            self.up_convs.append(conv_block(width * 2, width))
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block in self.downs:
            x = block(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottom(x)
        for up, block, skip in zip(self.ups, self.up_convs, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = block(torch.cat([skip, x], dim=1))
        return self.head(x)


# --- 損失・指標 --------------------------------------------------------------


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = (0, 2, 3)
    intersection = (probs * target).sum(dims)
    total = probs.sum(dims) + target.sum(dims)
    return 1.0 - ((2.0 * intersection + eps) / (total + eps)).mean()


@torch.no_grad()
def dice_per_channel(logits: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    pred = (torch.sigmoid(logits) > 0.5).float()
    dims = (2, 3)
    intersection = (pred * target).sum(dims)
    total = pred.sum(dims) + target.sum(dims)
    # 両方空なら 1.0 とする（心臓が写っていない画像など）
    score = torch.where(total > 0, 2.0 * intersection / total.clamp(min=1.0),
                        torch.ones_like(total))
    return score.mean(0).cpu().numpy()


# --- データ -----------------------------------------------------------------


class LungDataset(torch.utils.data.Dataset):
    """`build_lung_dataset.py` の配列を読む。水増しは学習側だけに掛ける。"""

    def __init__(self, directory: Path, part: str, augment: bool) -> None:
        self.images = np.load(directory / f"{part}-images.npy", mmap_mode="r")
        self.labels = np.load(directory / f"{part}-labels.npy", mmap_mode="r")
        self.augment = augment
        if len(self.images) != len(self.labels):
            raise ValueError("画像とラベルの枚数が違う")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        import cv2

        image = np.asarray(self.images[index], dtype=np.float32)
        packed = np.asarray(self.labels[index])
        masks = np.stack([(packed & bit) > 0 for bit in BITS]).astype(np.float32)

        if self.augment:
            # 回転・拡大・平行移動。胸部X線は向きが揃っているので控えめにする
            angle = np.random.uniform(-10, 10)
            scale = np.random.uniform(0.9, 1.1)
            shift = np.random.uniform(-0.05, 0.05, 2) * image.shape[0]
            centre = (image.shape[1] / 2, image.shape[0] / 2)
            matrix = cv2.getRotationMatrix2D(centre, angle, scale)
            matrix[:, 2] += shift
            size = (image.shape[1], image.shape[0])
            image = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            masks = np.stack([
                cv2.warpAffine(m, matrix, size, flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                for m in masks])
            if np.random.rand() < 0.5:
                # **反転したら右肺と左肺を入れ替える。** 心臓はそのまま
                image = image[:, ::-1].copy()
                masks = masks[:, :, ::-1].copy()
                masks = masks[[1, 0, 2]]
            # 明るさ・コントラスト。portable 撮影の幅を想定する
            image = image * np.random.uniform(0.85, 1.15) + np.random.uniform(-20, 20)

        image = np.clip(image, 0, 255) / 255.0
        return torch.from_numpy(image[None].astype(np.float32)), torch.from_numpy(masks)


# --- 学習 -------------------------------------------------------------------


def run_epoch(model, loader, device, optimiser=None, scaler=None):
    train = optimiser is not None
    model.train(train)
    losses, dices = [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logits = model(images)
            loss = F.binary_cross_entropy_with_logits(logits, targets) + dice_loss(logits, targets)
        if train:
            optimiser.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss.backward()
                optimiser.step()
        losses.append(float(loss.detach()))
        dices.append(dice_per_channel(logits.float(), targets))
    return float(np.mean(losses)), np.mean(dices, axis=0)


def export_onnx(model, size: int, path: Path, device) -> None:
    """ONNX へ書き出す。**重みを含めた 1 ファイル**にまとめる。

    torch の書き出しは既定で重みを `.onnx.data` へ外出しする。2 ファイルになると
    model volume への配置や SHA256 の突き合わせが面倒なので、onnx 側で読み直して
    1 ファイルに詰め直す。この手順は torch の版に依存しない。
    """
    import onnx

    model.eval()
    dummy = torch.zeros(1, 1, size, size, device=device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
    )
    loaded = onnx.load(str(path))                      # 外部データも一緒に読む
    external = path.with_suffix(".onnx.data")
    onnx.save_model(loaded, str(path), save_as_external_data=False)
    if external.exists():
        external.unlink()
    print(f"ONNX を書き出した: {path}（{path.stat().st_size / (1 << 20):.1f} MB）")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--out", type=Path, default=MODELS)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base", type=int, default=32, help="U-Net の基幅")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0,
                        help="CPU 学習のスレッド数。0 で論理コア数を使う")
    parser.add_argument("--smoke", action="store_true",
                        help="2 epoch・少数だけ回して配管を確かめる")
    args = parser.parse_args()

    if not (args.dataset / "meta.json").exists():
        print(f"学習データが無い: {args.dataset}\n"
              f"先に python scripts/build_lung_dataset.py を実行する", file=sys.stderr)
        return 1
    meta = json.loads((args.dataset / "meta.json").read_text(encoding="utf-8"))
    size = int(meta["size"])

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        # torch の既定は論理コアの半分。CPU 学習では全部使った方が速い。
        import os

        torch.set_num_threads(args.threads or os.cpu_count() or 1)
        print(f"CPU スレッド {torch.get_num_threads()}")
    print(f"device={device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    train_set = LungDataset(args.dataset, "train", augment=True)
    val_set = LungDataset(args.dataset, "val", augment=False)
    if args.smoke:
        args.epochs = 2
        train_set = torch.utils.data.Subset(train_set, range(min(16, len(train_set))))
        val_set = torch.utils.data.Subset(val_set, range(min(8, len(val_set))))
        args.workers = 0
    print(f"学習 {len(train_set)} 枚 / 検証 {len(val_set)} 枚、{size}x{size}")

    common = dict(batch_size=args.batch, num_workers=args.workers,
                  pin_memory=(device.type == "cuda"))
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, drop_last=False, **common)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **common)

    model = UNet(base=args.base).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数 {params / 1e6:.1f} M")
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    args.out.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    began = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, device, optimiser, scaler)
        val_loss, val_dice = run_epoch(model, val_loader, device)
        scheduler.step()
        lung_dice = float(np.mean(val_dice[:2]))
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_dice": {name: float(v) for name, v in zip(CHANNELS, val_dice)},
            "val_lung_dice": lung_dice,
        })
        marker = ""
        if lung_dice > best:
            best = lung_dice
            torch.save({"model": model.state_dict(), "base": args.base,
                        "size": size, "channels": CHANNELS}, args.out / "best.pt")
            marker = "  <- 最良"
        print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}/{val_loss:.4f}  "
              f"Dice 右 {val_dice[0]:.4f} 左 {val_dice[1]:.4f} 心 {val_dice[2]:.4f}  "
              f"肺平均 {lung_dice:.4f}{marker}", flush=True)
        (args.out / "metrics.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n最良の肺 Dice {best:.4f}（{(time.time() - began) / 60:.1f} 分）")
    checkpoint = torch.load(args.out / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model"])
    export_onnx(model, size, args.out / "lung-unet.onnx", device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
