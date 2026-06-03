"""Transmission channels behind a common base class."""

from __future__ import annotations

from .base import Channel
from .identity import IdentityChannel

__all__ = ["Channel", "IdentityChannel"]
