"""Reconstruction module that converts semantics into text output."""

from __future__ import annotations

from .types import OARRepresentation


class SemanticReconstructor:
    """Generate human-readable reconstructions from OAR semantics."""

    def reconstruct_text(self, oar: OARRepresentation) -> str:
        """Create a deterministic textual description from OAR content."""
        if not oar.objects:
            return "No recognizable objects were transmitted through the semantic channel."

        object_tokens = [f"{obj.name} ({obj.object_id})" for obj in oar.objects]
        object_part = "Detected objects: " + ", ".join(object_tokens) + "."

        if not oar.relations:
            relation_part = "No explicit relations are available in the reconstructed semantics."
        else:
            relation_tokens = [
                f"{rel.subject_id} {rel.predicate} {rel.object_id}" for rel in oar.relations
            ]
            relation_part = "Relations: " + "; ".join(relation_tokens) + "."

        return f"{object_part} {relation_part}"

    def image_generation_placeholder(self, oar: OARRepresentation) -> str:
        """Return a placeholder statement for future image regeneration integration."""
        return (
            "Image generation is not enabled in this baseline. "
            f"Received {len(oar.objects)} objects and {len(oar.relations)} relations."
        )
