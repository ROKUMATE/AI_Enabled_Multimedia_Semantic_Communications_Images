"""Crop-and-compress appearance encoder (default).

Crops the object's bounding box and compresses it (JPEG/WebP) at a quality tier
chosen by the object's mode: ``preserve`` objects get high quality so their
appearance survives intact, ``regenerate`` objects get low/medium quality since
a similar-looking version is acceptable.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

from ..types import ObjectMode, SceneObject
from .base import AppearanceEncoder


logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"JPEG", "WEBP"}


class CropCompressor(AppearanceEncoder):
    """Compress object crops at a mode-dependent quality tier."""

    def __init__(
        self,
        image_format: str = "JPEG",
        preserve_quality: int = 95,
        regenerate_quality: int = 35,
    ) -> None:
        """Configure the container format and per-mode quality tiers."""
        image_format = image_format.upper()
        if image_format == "JPG":
            image_format = "JPEG"
        if image_format not in _SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported appearance format: {image_format}")
        self.image_format = image_format
        self.preserve_quality = preserve_quality
        self.regenerate_quality = regenerate_quality

    def quality_for(self, mode: ObjectMode) -> int:
        """Return the encoder quality for an object mode."""
        return self.preserve_quality if mode == ObjectMode.PRESERVE else self.regenerate_quality

    def encode(self, obj: SceneObject, image_rgb: np.ndarray) -> bytes:
        """Crop the object and compress it at its mode's quality tier."""
        height, width = image_rgb.shape[:2]
        x1, y1, x2, y2 = obj.bbox
        left = max(0, int(round(x1)))
        top = max(0, int(round(y1)))
        right = min(width, int(round(x2)))
        bottom = min(height, int(round(y2)))
        if right <= left or bottom <= top:
            # Degenerate box: emit a 1x1 pixel so decode stays well-defined.
            crop = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            crop = image_rgb[top:bottom, left:right]

        buffer = io.BytesIO()
        Image.fromarray(crop).save(
            buffer, format=self.image_format, quality=self.quality_for(obj.mode)
        )
        data = buffer.getvalue()
        logger.debug(
            "Encoded crop for %s (%s, q=%d): %d bytes",
            obj.object_id,
            obj.mode.value,
            self.quality_for(obj.mode),
            len(data),
        )
        return data

    def decode(self, data: bytes) -> np.ndarray:
        """Decode compressed crop bytes back into an RGB array."""
        with Image.open(io.BytesIO(data)) as image:
            return np.asarray(image.convert("RGB"))
