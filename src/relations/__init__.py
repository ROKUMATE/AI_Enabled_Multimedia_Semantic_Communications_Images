"""Relation builder implementations behind a common base class."""

from __future__ import annotations

from .base import RelationBuilder
from .learned import LearnedRelationBuilder
from .rule_based import RuleBasedRelationBuilder

__all__ = ["RelationBuilder", "RuleBasedRelationBuilder", "LearnedRelationBuilder"]
