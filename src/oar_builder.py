"""OAR builder from detected object lists."""

from __future__ import annotations

import math

from .types import DetectedObject, OARRepresentation, Relation


class OARBuilder:
    """Create object-attribute-relation representations from detections."""

    def __init__(self, near_distance_threshold: float = 120.0) -> None:
        """Set spatial threshold used for relation extraction."""
        self.near_distance_threshold = near_distance_threshold

    def build(self, objects: list[DetectedObject]) -> OARRepresentation:
        """Build OAR using rule-based relation heuristics."""
        attributes = {obj.object_id: {} for obj in objects}
        relations = self._build_relations(objects)
        return OARRepresentation(objects=objects, attributes=attributes, relations=relations)

    def _build_relations(self, objects: list[DetectedObject]) -> list[Relation]:
        """Infer near and interacting_with relations from object geometry and classes."""
        relations: list[Relation] = []
        seen: set[tuple[str, str, str]] = set()

        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                left = objects[i]
                right = objects[j]
                distance = self._distance(left.center(), right.center())

                if distance <= self.near_distance_threshold:
                    self._add_relation(relations, seen, left.object_id, "near", right.object_id)
                    self._add_relation(relations, seen, right.object_id, "near", left.object_id)

                    if left.name == "person" and right.name != "person":
                        self._add_relation(
                            relations,
                            seen,
                            left.object_id,
                            "interacting_with",
                            right.object_id,
                        )
                    if right.name == "person" and left.name != "person":
                        self._add_relation(
                            relations,
                            seen,
                            right.object_id,
                            "interacting_with",
                            left.object_id,
                        )

        return relations

    def _add_relation(
        self,
        relations: list[Relation],
        seen: set[tuple[str, str, str]],
        subject_id: str,
        predicate: str,
        object_id: str,
    ) -> None:
        """Insert unique relation into relation list."""
        key = (subject_id, predicate, object_id)
        if key in seen:
            return
        seen.add(key)
        relations.append(Relation(subject_id=subject_id, predicate=predicate, object_id=object_id))

    def _distance(self, point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
        """Compute Euclidean distance between two 2D points."""
        return math.dist(point_a, point_b)
