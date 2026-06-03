"""Base interface for relation builders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from ..types import Relation


@runtime_checkable
class HasGeometry(Protocol):
    """Minimal object surface a relation builder needs."""

    object_id: str
    name: str

    def center(self) -> tuple[float, float]:
        """Return the bounding-box center."""
        ...


class RelationBuilder(ABC):
    """Infer semantic relations between detected objects.

    Concrete builders implement :meth:`build`. The rule-based default uses
    geometry heuristics; a learned scene-graph model can replace it later
    behind this same interface.
    """

    @abstractmethod
    def build(self, objects: list[HasGeometry]) -> list[Relation]:
        """Return the list of relations inferred from the objects."""
        raise NotImplementedError
