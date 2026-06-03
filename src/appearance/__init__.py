"""Appearance encoders behind a common base class."""

from __future__ import annotations

from .base import AppearanceEncoder
from .crop import CropCompressor
from .embedding import EmbeddingEncoder

__all__ = ["AppearanceEncoder", "CropCompressor", "EmbeddingEncoder"]
