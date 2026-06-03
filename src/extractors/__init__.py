"""Object extractor implementations behind a common base class."""

from __future__ import annotations

from .base import ObjectExtractor
from .learned import LearnedObjectExtractor
from .yolo import YoloExtractor

__all__ = ["ObjectExtractor", "YoloExtractor", "LearnedObjectExtractor"]
