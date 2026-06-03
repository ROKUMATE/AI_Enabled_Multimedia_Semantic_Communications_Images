"""CPU compositional reconstructor (default).

Builds a simple base canvas (a solid scene-colored background) and composites
the received object crops back at their bounding boxes. Objects sent as text
only are drawn as labeled rectangles. For ``preserve`` text objects the
high-quality crop is pasted and, optionally, the OCR text is re-rendered crisply
on top. Runs entirely on CPU with no GPU dependency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..appearance.base import AppearanceEncoder
from ..payload import SemanticPayload
from ..types import ObjectMode
from .base import Reconstructor, ReconstructionResult
from .text import describe_scene


logger = logging.getLogger(__name__)


def _clamp_box(bbox: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp a bbox to the canvas and return integer ``(l, t, r, b)``."""
    x1, y1, x2, y2 = bbox
    left = max(0, min(width - 1, int(round(x1))))
    top = max(0, min(height - 1, int(round(y1))))
    right = max(left + 1, min(width, int(round(x2))))
    bottom = max(top + 1, min(height, int(round(y2))))
    return left, top, right, bottom


class CompositionalReconstructor(Reconstructor):
    """Composite received crops onto a solid background canvas."""

    def __init__(
        self,
        background_color: tuple[int, int, int] = (127, 127, 127),
        rerender_text: bool = True,
        draw_labels: bool = True,
    ) -> None:
        """Configure background color and label/text rendering behavior."""
        self.background_color = tuple(int(c) for c in background_color)
        self.rerender_text = rerender_text
        self.draw_labels = draw_labels
        self._font = ImageFont.load_default()

    def make_background(self, image_size: tuple[int, int]) -> np.ndarray:
        """Return a solid-color background canvas (RGB uint8)."""
        width, height = image_size
        width = max(1, width)
        height = max(1, height)
        canvas = np.empty((height, width, 3), dtype=np.uint8)
        canvas[:, :] = self.background_color
        return canvas

    def composite(
        self,
        background: np.ndarray,
        payload: SemanticPayload,
        appearance_decoder: AppearanceEncoder,
    ) -> np.ndarray:
        """Paste crops / draw labels for every object onto ``background``."""
        canvas = Image.fromarray(background.copy())
        draw = ImageDraw.Draw(canvas)
        height, width = background.shape[:2]

        for obj in payload.structure.get("objects", []):
            object_id = str(obj.get("object_id"))
            left, top, right, bottom = _clamp_box(obj.get("bbox", [0, 0, 0, 0]), width, height)
            box_w, box_h = right - left, bottom - top
            mode = ObjectMode(str(obj.get("mode", ObjectMode.REGENERATE.value)))

            crop_bytes = payload.crops.get(object_id)
            if crop_bytes is not None:
                crop = appearance_decoder.decode(crop_bytes)
                resized = Image.fromarray(crop).resize((box_w, box_h), Image.BILINEAR)
                canvas.paste(resized, (left, top))
                if mode == ObjectMode.PRESERVE and self.rerender_text:
                    self._rerender_text(draw, obj, (left, top, right, bottom))
            elif self.draw_labels:
                self._draw_label(draw, obj, (left, top, right, bottom))

        return np.asarray(canvas)

    def reconstruct(
        self, payload: SemanticPayload, appearance_decoder: AppearanceEncoder
    ) -> ReconstructionResult:
        """Build the background, composite crops, and describe the scene."""
        background = self.make_background(payload.image_size)
        image = self.composite(background, payload, appearance_decoder)
        text = describe_scene(payload.structure)
        logger.info(
            "Compositional reconstruction: %dx%d canvas, %d crop(s) composited.",
            payload.image_size[0],
            payload.image_size[1],
            len(payload.crops),
        )
        return ReconstructionResult(image=image, text=text)

    def _draw_label(
        self, draw: ImageDraw.ImageDraw, obj: dict[str, Any], box: tuple[int, int, int, int]
    ) -> None:
        """Draw an outlined rectangle with the object's class name."""
        left, top, right, bottom = box
        draw.rectangle([left, top, right - 1, bottom - 1], outline=(255, 255, 255), width=2)
        label = str(obj.get("name", "object"))
        draw.text((left + 2, top + 2), label, fill=(255, 255, 0), font=self._font)

    def _rerender_text(
        self, draw: ImageDraw.ImageDraw, obj: dict[str, Any], box: tuple[int, int, int, int]
    ) -> None:
        """Overlay crisp OCR text on a preserve-text object's region."""
        ocr_text = obj.get("ocr_text")
        if not ocr_text:
            return
        left, top, right, bottom = box
        draw.rectangle([left, top, right - 1, bottom - 1], fill=(255, 255, 255))
        draw.text((left + 2, top + 2), str(ocr_text), fill=(0, 0, 0), font=self._font)
