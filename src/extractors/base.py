"""Base interface for object extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..types import SceneObject


class ObjectExtractor(ABC):
    """Detect objects in an image and return them as :class:`SceneObject`.

    Concrete extractors (YOLO, a learned detector, ...) implement
    :meth:`extract`. Keeping this behind an abstract base lets a learned model
    replace the default detector without changing any callers.
    """

    @abstractmethod
    def extract(self, image_path: str | Path) -> list[SceneObject]:
        """Run detection on one image and return detected scene objects."""
        raise NotImplementedError
