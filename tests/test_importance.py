"""Tests for importance scoring and budgeted selection."""

from __future__ import annotations

import unittest

from src.importance import HeuristicImportanceScorer
from src.types import ObjectMode, SceneObject


class ImportanceScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = HeuristicImportanceScorer()
        self.image_size = (100, 100)

    def test_central_large_beats_corner_small(self) -> None:
        central = SceneObject("a", "x", (30, 30, 70, 70), 0.9)
        corner = SceneObject("b", "y", (0, 0, 10, 10), 0.5)
        central_score = self.scorer.score_one(central, self.image_size)
        corner_score = self.scorer.score_one(corner, self.image_size)
        self.assertGreater(central_score, corner_score)

    def test_score_in_unit_range(self) -> None:
        obj = SceneObject("a", "x", (10, 10, 90, 90), 1.0)
        score = self.scorer.score_one(obj, self.image_size)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_rank_sets_importance_and_orders(self) -> None:
        objects = [
            SceneObject("small", "x", (0, 0, 5, 5), 0.3),
            SceneObject("big", "y", (20, 20, 80, 80), 0.95),
        ]
        ranked = self.scorer.rank(objects, self.image_size)
        self.assertEqual(ranked[0].object_id, "big")
        self.assertGreater(ranked[0].importance, ranked[1].importance)

    def test_budget_selects_top_k(self) -> None:
        objects = [
            SceneObject("c1", "x", (40, 40, 60, 60), 0.9),
            SceneObject("c2", "x", (35, 35, 65, 65), 0.8),
            SceneObject("c3", "x", (0, 0, 4, 4), 0.2),
        ]
        self.scorer.select(objects, self.image_size, budget=2)
        selected = [obj.object_id for obj in objects if obj.selected]
        self.assertEqual(len(selected), 2)
        self.assertNotIn("c3", selected)

    def test_preserve_always_selected_beyond_budget(self) -> None:
        objects = [
            SceneObject("big", "x", (40, 40, 60, 60), 0.9),
            SceneObject("tiny_text", "x", (0, 0, 2, 2), 0.1, mode=ObjectMode.PRESERVE),
        ]
        self.scorer.select(objects, self.image_size, budget=1)
        by_id = {obj.object_id: obj for obj in objects}
        self.assertTrue(by_id["tiny_text"].selected)
        self.assertTrue(by_id["big"].selected)


if __name__ == "__main__":
    unittest.main()
