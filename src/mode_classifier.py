"""Assign a transmission mode (regenerate / preserve) to each object.

An object is forced to ``preserve`` when either:

* its class is in the configured ``preserve_classes`` (e.g. ``person``, faces,
  logos), or
* OCR finds legible text inside its crop (text/documents) — the recovered text
  is stored on the object so the receiver can re-render it crisply.

Everything else defaults to ``regenerate``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from .ocr import OcrBackend
from .types import ObjectMode, SceneObject


logger = logging.getLogger(__name__)


class ObjectModeClassifier:
    """Classify each object as ``preserve`` or ``regenerate``."""

    def __init__(
        self,
        preserve_classes: Sequence[str] | None = None,
        ocr_backend: OcrBackend | None = None,
        min_ocr_chars: int = 3,
    ) -> None:
        """Configure forced-preserve classes and the optional OCR backend."""
        self.preserve_classes = {name.lower() for name in (preserve_classes or [])}
        self.ocr_backend = ocr_backend
        self.min_ocr_chars = min_ocr_chars

    def classify(self, objects: list[SceneObject], image_rgb: np.ndarray) -> list[SceneObject]:
        """Set ``mode`` (and ``ocr_text`` for text regions) on each object."""
        height, width = image_rgb.shape[:2]
        for obj in objects:
            if obj.name.lower() in self.preserve_classes:
                obj.mode = ObjectMode.PRESERVE
                continue

            text = self._read_text(obj, image_rgb, width, height)
            if text and len(text) >= self.min_ocr_chars:
                obj.mode = ObjectMode.PRESERVE
                obj.ocr_text = text
                logger.debug("Object %s flagged preserve via OCR: %r", obj.object_id, text)
            else:
                obj.mode = ObjectMode.REGENERATE

        preserved = sum(1 for obj in objects if obj.mode == ObjectMode.PRESERVE)
        logger.info("Mode classification: %d preserve, %d regenerate.", preserved, len(objects) - preserved)
        return objects

    def _read_text(
        self, obj: SceneObject, image_rgb: np.ndarray, width: int, height: int
    ) -> str:
        """Run OCR on the object's crop, returning recognized text (or '')."""
        if self.ocr_backend is None:
            return ""

        x1, y1, x2, y2 = obj.bbox
        left = max(0, int(round(x1)))
        top = max(0, int(round(y1)))
        right = min(width, int(round(x2)))
        bottom = min(height, int(round(y2)))
        if right - left < 2 or bottom - top < 2:
            return ""

        crop = image_rgb[top:bottom, left:right]
        try:
            return self.ocr_backend.read(crop)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            logger.debug("OCR failed on %s: %s", obj.object_id, exc)
            return ""
