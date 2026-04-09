"""Reconstruction module that converts semantics into text output."""

from __future__ import annotations

import logging

from .types import OARRepresentation


logger = logging.getLogger(__name__)


class SemanticReconstructor:
    """Generate human-readable reconstructions from OAR semantics."""

    def reconstruct_text(self, oar: OARRepresentation) -> str:
        """Create a deterministic textual description from OAR content."""
        if not oar.objects:
            return "The semantic channel preserved no recognizable objects, so the scene is only partially recoverable."

        object_tokens = [f"{obj.name} ({obj.object_id})" for obj in oar.objects]
        scene_context = self._scene_context(oar)
        object_part = f"This reconstructed scene appears to contain {len(oar.objects)} objects: " + ", ".join(object_tokens) + "."

        if not oar.relations:
            relation_part = "No explicit relations survived channel corruption, so spatial and interaction structure is limited."
        else:
            relation_tokens = [self._relation_to_phrase(rel, oar) for rel in oar.relations]
            relation_part = "Observed relations include " + "; ".join(relation_tokens) + "."

        text = f"{scene_context} {object_part} {relation_part}"
        logger.debug("Reconstructed semantic description with %d objects and %d relations.", len(oar.objects), len(oar.relations))
        return text.strip()

    def image_generation_placeholder(self, oar: OARRepresentation) -> str:
        """Return a placeholder statement for future image regeneration integration."""
        return (
            "Image generation is not enabled in this baseline. "
            f"Received {len(oar.objects)} objects and {len(oar.relations)} relations."
        )

    def _scene_context(self, oar: OARRepresentation) -> str:
        """Generate a short scene-level context sentence."""
        object_names = [obj.name for obj in oar.objects]
        if any(name == "person" for name in object_names):
            return "The scene likely centers on people or human activity."
        if any(name in {"car", "bus", "truck", "bicycle", "motorcycle"} for name in object_names):
            return "The scene likely involves mobility or transport activity."
        if len(oar.objects) >= 3:
            return "The scene appears multi-object and moderately structured."
        return "The scene appears simple or partially observed."

    def _relation_to_phrase(self, relation, oar: OARRepresentation) -> str:
        """Turn a semantic relation into a natural language phrase."""
        object_lookup = {item.object_id: item.name for item in oar.objects}
        subject = object_lookup.get(relation.subject_id, relation.subject_id)
        object_name = object_lookup.get(relation.object_id, relation.object_id)

        if relation.predicate == "near":
            return f"{subject} is near {object_name}"
        if relation.predicate == "interacting_with":
            return f"{subject} is interacting with {object_name}"
        return f"{subject} {relation.predicate} {object_name}"
