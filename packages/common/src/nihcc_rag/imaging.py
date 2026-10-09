"""画像入力の検証と表示用派生画像（設計書 6.2 / 7.1.1 / 9.6 / 13）。

OpenCV には依存しない（ROI 抽出は roi-service 側の image-pipeline が担当）。
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

# 展開時の decompression bomb 対策
Image.MAX_IMAGE_PIXELS = 4096 * 4096 * 4

ALLOWED_MIME = ("image/png", "image/jpeg")
MIN_SIDE = 512
MAX_SIDE = 4096

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"


class ImageValidationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ImageFacts:
    sha256: str
    mime_type: str
    byte_size: int
    width: int
    height: int


def sniff_mime(data: bytes) -> str:
    if data.startswith(_PNG_MAGIC):
        return "image/png"
    if data.startswith(_JPEG_MAGIC):
        return "image/jpeg"
    raise ImageValidationError(
        "IMAGE_UNSUPPORTED_FORMAT", "Only PNG and JPEG images are accepted."
    )


def inspect(data: bytes, *, max_bytes: int = 20 * 1024 * 1024, strict_size: bool = True) -> ImageFacts:
    """magic bytes・デコード可否・解像度を検証し、SHA-256 を計算する。"""
    if not data:
        raise ImageValidationError("IMAGE_EMPTY", "Image content is empty.")
    if len(data) > max_bytes:
        raise ImageValidationError(
            "IMAGE_TOO_LARGE", f"Image exceeds the {max_bytes} byte limit."
        )
    mime = sniff_mime(data)
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            declared = Image.MIME.get(image.format or "", "")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("IMAGE_DECODE_FAILED", "Image could not be decoded.") from exc

    if declared and declared != mime:
        raise ImageValidationError(
            "IMAGE_MIME_MISMATCH", "Magic bytes and decoded format do not match."
        )
    if strict_size and (min(width, height) < MIN_SIDE or max(width, height) > MAX_SIDE):
        raise ImageValidationError(
            "IMAGE_RESOLUTION_OUT_OF_RANGE",
            f"Image resolution must be between {MIN_SIDE} and {MAX_SIDE} pixels per side.",
        )
    return ImageFacts(
        sha256=hashlib.sha256(data).hexdigest(),
        mime_type=mime,
        byte_size=len(data),
        width=width,
        height=height,
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_thumbnail(data: bytes, max_size: int = 512) -> bytes:
    max_size = max(64, min(1024, max_size))
    with Image.open(io.BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("L")
        image.thumbnail((max_size, max_size), Image.LANCZOS)
        out = io.BytesIO()
        image.save(out, format="PNG", optimize=True)
        return out.getvalue()


#: 表示用 region crop で mask 外に残す明るさの割合
#: （image_pipeline.pipeline.DISPLAY_DIM_FACTOR と同値にすること）
REGION_DIM_FACTOR = 0.12
#: mask を重畳する際の混合率（全体ランキングの overlay で使用）
REGION_TINT_ALPHA = 0.35


def region_display_crop(
    parent_png: bytes,
    mask_png: bytes,
    bbox: list[int] | None,
    *,
    dim: float = REGION_DIM_FACTOR,
    max_size: int = 768,
) -> bytes:
    """当該部位だけを切り出し、mask 外を減光した表示用画像を返す。

    部位 mask は肺輪郭に沿った帯状のため、外接矩形で切り出しただけでは
    肺のほぼ全体が写り、部位を判別できない。mask 外を減光して部位を明示する。

    部位別ランキングは同一部位どうしの比較が目的のため着色は行わない。
    一致部位の着色は全体ランキング（overlay_mask）で行う。
    """
    with Image.open(io.BytesIO(parent_png)) as parent_img:
        parent = parent_img.convert("L")
    with Image.open(io.BytesIO(mask_png)) as mask_img:
        mask = mask_img.convert("L")

    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(parent.width, x1), min(parent.height, y1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("invalid region bbox")
        cropped = parent.crop((x0, y0, x1, y1))
    else:
        cropped = parent

    if mask.size != cropped.size:
        mask = mask.resize(cropped.size, Image.NEAREST)
    binary = mask.point(lambda value: 255 if value > 0 else 0)

    dimmed = cropped.point(lambda value: int(value * dim))
    composed = Image.composite(cropped, dimmed, binary)
    composed.thumbnail((max_size, max_size), Image.LANCZOS)

    out = io.BytesIO()
    composed.save(out, format="PNG", optimize=True)
    return out.getvalue()


def overlay_mask(
    parent_png: bytes,
    mask_png: bytes,
    bbox: list[int] | None = None,
    *,
    color: tuple[int, int, int] = (0, 176, 255),
    alpha: float = REGION_TINT_ALPHA,
    max_size: int = 768,
) -> bytes:
    """原画像へ部位 mask を半透明合成した確認用画像を返す。"""
    with Image.open(io.BytesIO(parent_png)) as parent_img:
        parent = parent_img.convert("RGB")
    with Image.open(io.BytesIO(mask_png)) as mask_img:
        mask = mask_img.convert("L")

    canvas = Image.new("L", parent.size, 0)
    if bbox and len(bbox) == 4:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        box_size = (max(1, x1 - x0), max(1, y1 - y0))
        if mask.size != box_size:
            mask = mask.resize(box_size, Image.NEAREST)
        canvas.paste(mask, (x0, y0))
    else:
        canvas = mask.resize(parent.size, Image.NEAREST)

    tint = Image.new("RGB", parent.size, color)
    blended = Image.composite(
        Image.blend(parent, tint, alpha), parent, canvas.point(lambda v: 255 if v > 0 else 0)
    )
    blended.thumbnail((max_size, max_size), Image.LANCZOS)
    out = io.BytesIO()
    blended.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _paste_to_frame(mask_png: bytes, bbox: list[int] | None, size: tuple[int, int]):
    """bbox の大きさで保存された mask を、原画像の枠へ戻した 2 値配列にする。"""
    import numpy as np

    canvas = Image.new("L", size, 0)
    with Image.open(io.BytesIO(mask_png)) as raw:
        mask = raw.convert("L")
        if bbox and len(bbox) == 4:
            x0, y0, x1, y1 = (int(v) for v in bbox)
            box = (max(1, x1 - x0), max(1, y1 - y0))
            if mask.size != box:
                mask = mask.resize(box, Image.NEAREST)
            canvas.paste(mask, (x0, y0))
        else:
            canvas = mask.resize(size, Image.NEAREST)
    return np.asarray(canvas) > 0


def lung_outline(
    parent_png: bytes,
    masks: list[tuple[bytes, list[int] | None]],
    *,
    color: tuple[int, int, int] = (255, 214, 0),
    thickness: int = 3,
    fill_alpha: float = 0.12,
    max_size: int = 768,
) -> bytes:
    """原画像へ**肺野全体の外周**を描いた確認用画像を返す。

    肺野形状の検索（`shape.describe_lung`）が何を「肺野」として扱ったかを目で
    確かめるための表示である。脊柱・心陰影・横隔膜より下が輪郭の外側にあるかを
    見れば、ROI が正しく抜けているかが判る。

    外周は**和集合の境界**として描く。部位ごとの線を重ねると部位間の継ぎ目まで
    描かれ、どこが肺野の外縁なのかが判らなくなる。
    """
    import numpy as np

    with Image.open(io.BytesIO(parent_png)) as parent_img:
        parent = parent_img.convert("RGB")

    union = np.zeros((parent.size[1], parent.size[0]), dtype=bool)
    for mask_png, bbox in masks:
        union |= _paste_to_frame(mask_png, bbox, parent.size)
    if not union.any():
        raise ValueError("肺野 mask が空です")

    # 4 近傍の収縮との差が境界。scipy を持ち込まずに求める。
    eroded = union.copy()
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        eroded &= np.roll(union, shift, axis=axis)
    edge = union & ~eroded
    # 線を太らせる。原寸で 1px だと縮小表示で消える。
    thick = edge.copy()
    for _ in range(max(0, thickness - 1)):
        grown = thick.copy()
        for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
            grown |= np.roll(thick, shift, axis=axis)
        thick = grown

    base = np.asarray(parent).astype(np.float32)
    tint = np.array(color, dtype=np.float32)
    # 肺野の内側をごく薄く着色し、外周をはっきり描く
    inside = union[:, :, None]
    base = np.where(inside, base * (1.0 - fill_alpha) + tint * fill_alpha, base)
    base = np.where(thick[:, :, None], tint, base)

    out_img = Image.fromarray(base.clip(0, 255).astype(np.uint8), mode="RGB")
    out_img.thumbnail((max_size, max_size), Image.LANCZOS)
    out = io.BytesIO()
    out_img.save(out, format="PNG", optimize=True)
    return out.getvalue()
