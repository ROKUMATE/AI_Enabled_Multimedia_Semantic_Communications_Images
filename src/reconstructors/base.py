"""Base interface for receivers that reconstruct an image from a payload."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..appearance.base import AppearanceEncoder
from ..payload import SemanticPayload


@dataclass
class ReconstructionResult:
    """The receiver's output: a reconstructed image and a text description."""

    image: np.ndarray  # RGB uint8, shape (H, W, 3)
    text: str


class Reconstructor(ABC):
    """Rebuild an image (and text) from a received :class:`SemanticPayload`."""

    @abstractmethod
    def reconstruct(
        self, payload: SemanticPayload, appearance_decoder: AppearanceEncoder
    ) -> ReconstructionResult:
        """Reconstruct the image and text from a received payload."""
        raise NotImplementedError
