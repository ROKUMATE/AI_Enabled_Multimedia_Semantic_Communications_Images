"""Learned relation builder.

STUB (P5). A learned scene-graph generation model will replace the rule-based
geometry heuristics here. For now this falls back to the rule-based builder so
the pipeline stays runnable, and logs that the learned model is not trained.
"""

from __future__ import annotations

import logging

from ..types import Relation
from .base import HasGeometry, RelationBuilder
from .rule_based import RuleBasedRelationBuilder


logger = logging.getLogger(__name__)


class LearnedRelationBuilder(RelationBuilder):
    """Placeholder for a trained scene-graph relation model."""

    def __init__(self, near_distance_threshold: float = 120.0) -> None:
        """Configure the fallback rule-based builder."""
        logger.warning(
            "LearnedRelationBuilder is a stub; using rule-based relations until "
            "a relation model is trained (see PLAN.md TODO backlog)."
        )
        # TODO(P5): load a trained relation/scene-graph model and predict
        # relations from object features instead of delegating to geometry.
        self._fallback = RuleBasedRelationBuilder(near_distance_threshold)

    def build(self, objects: list[HasGeometry]) -> list[Relation]:
        """Return relations from the fallback rule-based builder."""
        return self._fallback.build(objects)
