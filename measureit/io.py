"""Image loading shared by calibration and dataset preparation.

Phone JPEGs carry an EXIF orientation tag and a resolution that is not the
working resolution. Both are normalised here, and a wrong orientation is an
error rather than a silent resize: squashing a portrait frame into a landscape
working size would corrupt every intrinsic.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

ASPECT_TOLERANCE = 0.01


class ImageLoadError(RuntimeError):
    """An image cannot be read, or does not match the working frame."""


def load_image(path: str | Path, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load a BGR uint8 image, honouring EXIF orientation.

    ``target_size`` is (width, height). The source must already have that
    aspect ratio; a mismatch means the phone was rotated and is reported
    instead of being stretched.
    """
    source = Path(path)
    try:
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            array = np.asarray(image)
    except Exception as exc:
        raise ImageLoadError(f"cannot read {source.name}: {exc}") from exc

    height, width = array.shape[:2]
    if target_size is not None:
        target_width, target_height = (int(v) for v in target_size)
        if target_width <= 0 or target_height <= 0:
            raise ValueError("target_size must be positive")

        source_ratio = width / height
        target_ratio = target_width / target_height
        if abs(source_ratio - target_ratio) / target_ratio > ASPECT_TOLERANCE:
            raise ImageLoadError(
                f"{source.name} is {width}x{height} after EXIF rotation, which does not "
                f"match the {target_width}x{target_height} working frame. Keep the phone "
                "in one orientation for every photo, then recapture."
            )

        if (width, height) != (target_width, target_height):
            array = cv2.resize(
                array, (target_width, target_height), interpolation=cv2.INTER_AREA
            )

    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
