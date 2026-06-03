"""Text description of a received scene graph.

Reuses the existing template generator (:class:`SemanticReconstructor`) by
rebuilding an :class:`OARRepresentation` from the payload's structure stream,
so the receiver still emits a human-readable description alongside the image.
"""

from __future__ import annotations

from typing import Any

from ..reconstruct import SemanticReconstructor
from ..types import DetectedObject, OARRepresentation, Relation


_text_reconstructor = SemanticReconstructor()


def describe_scene(structure: dict[str, Any]) -> str:
    """Produce a natural-language description from a scene-graph dict."""
    objects = [
        DetectedObject(
            object_id=str(item.get("object_id", f"obj_{idx}")),
            name=str(item.get("name", "unknown")),
            bbox=tuple(float(value) for value in item.get("bbox", [0, 0, 0, 0])),  # type: ignore[arg-type]
            confidence=float(item.get("confidence", 0.0)),
        )
        for idx, item in enumerate(structure.get("objects", []))
    ]
    relations = [
        Relation(
            subject_id=str(item.get("subject_id", "obj_0")),
            predicate=str(item.get("predicate", "related_to")),
            object_id=str(item.get("object_id", "obj_0")),
        )
        for item in structure.get("relations", [])
    ]
    oar = OARRepresentation(objects=objects, relations=relations)
    return _text_reconstructor.reconstruct_text(oar)
