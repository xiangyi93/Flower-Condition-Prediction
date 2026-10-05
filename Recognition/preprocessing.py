"""Shared, position-independent preprocessing for black-background plant regions."""

from dataclasses import dataclass
from math import ceil

import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as T


def rgb_on_black(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if "A" in image.getbands() or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


@dataclass
class ForegroundLetterbox:
    """Crop the union of ALL non-black regions, retaining their spatial arrangement.

    Exact black is the background convention for the current lossless PNG dataset.
    This does not identify plants or decide whether there is enough bloom evidence.
    """

    size: int = 224
    margin: float = 0.05

    def __post_init__(self):
        if self.size <= 0 or self.margin < 0:
            raise ValueError("size must be positive and margin must be nonnegative")

    def __call__(self, image: Image.Image) -> Image.Image:
        image = rgb_on_black(image)
        foreground = np.any(np.asarray(image) != 0, axis=2)
        rows = np.flatnonzero(foreground.any(axis=1))
        columns = np.flatnonzero(foreground.any(axis=0))
        if not len(rows):
            raise ValueError("No foreground: all-black image cannot be used for bloom classification")
        left, top = int(columns[0]), int(rows[0])
        right, bottom = int(columns[-1]) + 1, int(rows[-1]) + 1
        margin_x = ceil((right - left) * self.margin)
        margin_y = ceil((bottom - top) * self.margin)
        # Out-of-image bounds are padded black by PIL, preserving the same margin at edges.
        cropped = image.crop((left - margin_x, top - margin_y, right + margin_x, bottom + margin_y))
        resized = ImageOps.contain(cropped, (self.size, self.size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size))
        canvas.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
        return canvas


def bloom_transforms(size: int, augment: bool = False, hflip: float = 0.5) -> T.Compose:
    """Use identical geometry and [0, 1] RGB scaling in every phase.

    Only horizontal flipping is added during training. Do not erase flowers,
    randomly crop regions, or change the colors used to determine bloom stage.
    """
    transforms = [ForegroundLetterbox(size)]
    if augment:
        transforms.append(T.RandomHorizontalFlip(p=hflip))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)
