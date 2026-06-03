"""Baselines for comparison against the semantic pipeline.

* ``jpeg_baseline`` — compress the whole original image to JPEG at the quality
  that best matches a target payload size (apples-to-apples on bytes).
* ``text_only_payload`` — drop the appearance stream so reconstruction uses
  labels only; shows that transmitting crops actually helps.
"""

from __future__ import annotations

import copy
import io
import logging

import numpy as np
from PIL import Image

from .payload import SemanticPayload


logger = logging.getLogger(__name__)


def _encode_jpeg(image_rgb: np.ndarray, quality: int) -> bytes:
    """Encode an RGB array to JPEG bytes at the given quality."""
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=int(quality))
    return buffer.getvalue()


def jpeg_baseline(image_rgb: np.ndarray, target_bytes: int) -> tuple[np.ndarray, int]:
    """Return ``(decoded_image, actual_bytes)`` for a JPEG matched to a byte budget.

    Binary-searches the JPEG quality (1..95) for the largest encoding that does
    not exceed ``target_bytes``; if even quality 1 is larger, that is used.
    """
    low, high = 1, 95
    best_quality = 1
    best_bytes = _encode_jpeg(image_rgb, low)
    while low <= high:
        mid = (low + high) // 2
        encoded = _encode_jpeg(image_rgb, mid)
        if len(encoded) <= max(1, target_bytes):
            best_quality, best_bytes = mid, encoded
            low = mid + 1
        else:
            high = mid - 1

    with Image.open(io.BytesIO(best_bytes)) as image:
        decoded = np.asarray(image.convert("RGB"))
    logger.info(
        "JPEG baseline matched target %dB -> quality=%d, %dB.",
        target_bytes,
        best_quality,
        len(best_bytes),
    )
    return decoded, len(best_bytes)


def text_only_payload(payload: SemanticPayload) -> SemanticPayload:
    """Return a copy of the payload with the appearance stream removed."""
    return SemanticPayload(
        structure=copy.deepcopy(payload.structure),
        crops={},
        structure_priority=payload.structure_priority,
        appearance_priority=payload.appearance_priority,
    )
