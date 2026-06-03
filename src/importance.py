"""Importance scoring and budgeted selection of objects.

The transmitter cannot afford a crop for every object. The importance scorer
ranks objects so the most salient ones get their appearance transmitted while
the rest are sent as text only.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod

from .types import SceneObject


logger = logging.getLogger(__name__)


class ImportanceScorer(ABC):
    """Rank objects by importance and select a budgeted subset.

    Concrete scorers implement :meth:`score_one`. Selection logic
    (:meth:`rank`, :meth:`select`) is shared and swappable via the base.
    """

    @abstractmethod
    def score_one(self, obj: SceneObject, image_size: tuple[int, int]) -> float:
        """Return the importance score for a single object."""
        raise NotImplementedError

    def rank(
        self, objects: list[SceneObject], image_size: tuple[int, int]
    ) -> list[SceneObject]:
        """Set each object's ``importance`` and return objects sorted desc."""
        for obj in objects:
            obj.importance = self.score_one(obj, image_size)
        return sorted(objects, key=lambda item: item.importance, reverse=True)

    def select(
        self,
        objects: list[SceneObject],
        image_size: tuple[int, int],
        budget: int,
    ) -> list[SceneObject]:
        """Rank objects and mark the top ``budget`` (plus all preserve) selected.

        Preserve-mode objects are always selected regardless of the budget,
        because their exact appearance must survive. Returns the ranked list.
        """
        ranked = self.rank(objects, image_size)
        from .types import ObjectMode  # local import avoids a cycle at module load

        chosen = 0
        for obj in ranked:
            force = obj.mode == ObjectMode.PRESERVE
            if force or chosen < budget:
                obj.selected = True
                if not force:
                    chosen += 1
            else:
                obj.selected = False

        logger.info(
            "Selected %d/%d object(s) for appearance transmission (budget=%d).",
            sum(1 for obj in ranked if obj.selected),
            len(ranked),
            budget,
        )
        return ranked


class HeuristicImportanceScorer(ImportanceScorer):
    """Default scorer: ``normalized_area * confidence * centrality``.

    ``centrality`` is ``1`` at the image center and falls toward ``0`` at the
    corners. All three factors are in ``[0, 1]`` so the score is too.
    """

    def score_one(self, obj: SceneObject, image_size: tuple[int, int]) -> float:
        """Combine area, confidence, and centrality into one score."""
        width, height = image_size
        if width <= 0 or height <= 0:
            return obj.confidence

        norm_area = min(1.0, obj.area() / float(width * height))

        cx, cy = obj.center()
        image_cx, image_cy = width / 2.0, height / 2.0
        max_dist = 0.5 * math.hypot(width, height)
        dist = math.hypot(cx - image_cx, cy - image_cy)
        centrality = max(0.0, 1.0 - dist / max_dist) if max_dist > 0 else 1.0

        return norm_area * obj.confidence * centrality
