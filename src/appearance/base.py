"""Base interface for appearance encoders.

An appearance encoder turns an object's pixels into the bytes carried by the
payload's appearance stream. v1 sends compressed crops (:class:`CropCompressor`);
a future version will send learned embeddings (:class:`EmbeddingEncoder`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..types import SceneObject


class AppearanceEncoder(ABC):
    """Encode/decode an object's appearance to/from transmittable bytes."""

    @abstractmethod
    def encode(self, obj: SceneObject, image_rgb: np.ndarray) -> bytes:
        """Encode the object's appearance into bytes for transmission."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, data: bytes) -> np.ndarray:
        """Decode transmitted bytes back into an RGB image array."""
        raise NotImplementedError
