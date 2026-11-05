"""Image quality checks with configurable thresholds.

Thresholds live in ``dataset/config/quality.json`` and are committed before the
first real capture. They are advisory: a flagged frame is reported for review,
never silently deleted. The initial values are a starting point to be
re-tuned against the first real session and that change must be recorded in the
config file's ``notes``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

DEFAULT_CONFIG = Path("dataset/config/quality.json")


class QualityError(RuntimeError):
    """The quality configuration is missing or invalid."""


@dataclass(frozen=True)
class QualityConfig:
    working_size: tuple[int, int]
    blur_min: float
    mean_min: float
    mean_max: float
    dark_clip_pct_max: float
    bright_clip_pct_max: float
    hamming_max: int
    mask_area_pct_min: float
    mask_area_pct_max: float

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "QualityConfig":
        source = Path(path)
        if not source.is_file():
            raise QualityError(f"quality config not found: {source}")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise QualityError(f"{source} is not valid JSON: {exc}") from exc
        if payload.get("schema") != "measureit.quality/1":
            raise QualityError(
                f"unsupported quality schema {payload.get('schema')!r}"
            )

        try:
            width, height = (int(v) for v in payload["working_size"])
            blur = payload["blur"]
            exposure = payload["exposure"]
            duplicates = payload["duplicates"]
            mask = payload["mask"]
            config = cls(
                working_size=(width, height),
                blur_min=float(blur["laplacian_variance_min"]),
                mean_min=float(exposure["mean_luma_min"]),
                mean_max=float(exposure["mean_luma_max"]),
                dark_clip_pct_max=float(exposure["clipped_dark_pct_max"]),
                bright_clip_pct_max=float(exposure["clipped_bright_pct_max"]),
                hamming_max=int(duplicates["dhash_hamming_max"]),
                mask_area_pct_min=float(mask["area_pct_min"]),
                mask_area_pct_max=float(mask["area_pct_max"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise QualityError(f"{source} is missing or has an invalid key: {exc}") from exc

        if width <= 0 or height <= 0:
            raise QualityError("working_size must be positive")
        if config.mean_min >= config.mean_max:
            raise QualityError("mean_luma_min must be below mean_luma_max")
        if config.blur_min < 0:
            raise QualityError("laplacian_variance_min must be >= 0")
        if not 0 <= config.hamming_max <= 64:
            raise QualityError("dhash_hamming_max must be within 0..64")
        return config


@dataclass(frozen=True)
class QualityMetrics:
    blur_variance: float
    mean_luma: float
    dark_pct: float
    bright_pct: float
    flags: tuple[str, ...]

    @property
    def flagged(self) -> bool:
        return bool(self.flags)


def assess(image: np.ndarray, config: QualityConfig) -> QualityMetrics:
    """Measure blur and exposure on a BGR or grayscale working image."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise QualityError(f"expected a 2D or 3D image, got shape {image.shape}")

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean = float(gray.mean())
    dark = float(np.mean(gray <= 5) * 100.0)
    bright = float(np.mean(gray >= 250) * 100.0)

    flags: list[str] = []
    if blur < config.blur_min:
        flags.append("blurry")
    if mean < config.mean_min or dark > config.dark_clip_pct_max:
        flags.append("underexposed")
    if mean > config.mean_max or bright > config.bright_clip_pct_max:
        flags.append("overexposed")

    return QualityMetrics(
        blur_variance=blur,
        mean_luma=mean,
        dark_pct=dark,
        bright_pct=bright,
        flags=tuple(flags),
    )


def dhash64(image: np.ndarray) -> int:
    """64-bit difference hash: robust to resize and mild exposure change."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = 0
    for value in (small[:, 1:] > small[:, :-1]).ravel():
        bits = (bits << 1) | int(value)
    return bits


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def content_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def probe_size(path: str | Path) -> tuple[int, int]:
    """EXIF-corrected source size, i.e. what the phone actually captured."""
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
    return int(width), int(height)


def canonical_suffix(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".jpeg":
        return ".jpg"
    if suffix in (".jpg", ".png"):
        return suffix
    raise QualityError(f"unsupported image type: {suffix or '(none)'}")


def format_flags(flags: tuple[str, ...]) -> str:
    return ";".join(flags)


def parse_flags(text: str) -> tuple[str, ...]:
    return tuple(part for part in text.split(";") if part)
