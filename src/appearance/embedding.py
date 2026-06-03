"""CLIP appearance embedding encoder.

STUB (P5). The future appearance stream will carry compact learned embeddings
(e.g. CLIP image features) instead of raw compressed crops, and the receiver
will use them to condition a generator. This stub fixes the interface now; it
is not used by default and raises if invoked without the optional deps.
"""

from __future__ import annotations

import logging

import numpy as np

from ..types import SceneObject
from .base import AppearanceEncoder


logger = logging.getLogger(__name__)


class EmbeddingEncoder(AppearanceEncoder):
    """Encode object appearance as a CLIP image embedding (not yet implemented)."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32") -> None:
        """Record the intended model; defer loading until implemented."""
        self.model_id = model_id
        logger.warning(
            "EmbeddingEncoder is a stub (model=%s); CropCompressor is used by "
            "default. See PLAN.md TODO backlog.",
            model_id,
        )

    def encode(self, obj: SceneObject, image_rgb: np.ndarray) -> bytes:
        """Encode the crop into an embedding byte string."""
        # TODO(P5): run the crop through a frozen CLIP image encoder and
        # serialize the resulting float vector (e.g. float16 bytes).
        raise NotImplementedError("EmbeddingEncoder.encode is not implemented yet.")

    def decode(self, data: bytes) -> np.ndarray:
        """Decode an embedding byte string."""
        # TODO(P5): deserialize the embedding vector for the generator to consume.
        raise NotImplementedError("EmbeddingEncoder.decode is not implemented yet.")
