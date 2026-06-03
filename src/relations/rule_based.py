"""Rule-based spatial relation builder (default).

This preserves the original geometry heuristics: objects whose centers are
within ``near_distance_threshold`` pixels are ``near`` each other, and a
``person`` near a non-person object is ``interacting_with`` it.
"""

from __future__ import annotations

import math

from ..types import Relation
from .base import HasGeometry, RelationBuilder


class RuleBasedRelationBuilder(RelationBuilder):
    """Create relations from object geometry and classes."""

    def __init__(self, near_distance_threshold: float = 120.0) -> None:
        """Set the spatial threshold used for relation extraction."""
        self.near_distance_threshold = near_distance_threshold

    def build(self, objects: list[HasGeometry]) -> list[Relation]:
        """Infer ``near`` and ``interacting_with`` relations."""
        relations: list[Relation] = []
        seen: set[tuple[str, str, str]] = set()

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                left = objects[i]
                right = objects[j]
                distance = math.dist(left.center(), right.center())
                if distance > self.near_distance_threshold:
                    continue

                self._add(relations, seen, left.object_id, "near", right.object_id)
                self._add(relations, seen, right.object_id, "near", left.object_id)

                if left.name == "person" and right.name != "person":
                    self._add(relations, seen, left.object_id, "interacting_with", right.object_id)
                if right.name == "person" and left.name != "person":
                    self._add(relations, seen, right.object_id, "interacting_with", left.object_id)

        return relations

    @staticmethod
    def _add(
        relations: list[Relation],
        seen: set[tuple[str, str, str]],
        subject_id: str,
        predicate: str,
        object_id: str,
    ) -> None:
        """Insert a unique relation into the relation list."""
        key = (subject_id, predicate, object_id)
        if key in seen:
            return
        seen.add(key)
        relations.append(Relation(subject_id=subject_id, predicate=predicate, object_id=object_id))
