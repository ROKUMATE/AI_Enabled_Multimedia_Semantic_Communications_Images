"""Tests for object mode classification (preserve vs regenerate)."""

from __future__ import annotations

import unittest

import numpy as np

from src.mode_classifier import ObjectModeClassifier
from src.ocr import OcrBackend
from src.types import ObjectMode, SceneObject


class _StubOcr(OcrBackend):
    """OCR backend that returns a fixed string for any crop."""

    name = "stub"

    def __init__(self, text: str) -> None:
        self.text = text

    def read(self, image_rgb: np.ndarray) -> str:  # noqa: D102
        return self.text


class ModeClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_forced_class_is_preserve(self) -> None:
        classifier = ObjectModeClassifier(preserve_classes=["person"], ocr_backend=None)
        objects = [SceneObject("a", "person", (0, 0, 50, 50), 0.9)]
        classifier.classify(objects, self.image)
        self.assertEqual(objects[0].mode, ObjectMode.PRESERVE)

    def test_non_forced_no_ocr_is_regenerate(self) -> None:
        classifier = ObjectModeClassifier(preserve_classes=["person"], ocr_backend=None)
        objects = [SceneObject("a", "car", (0, 0, 50, 50), 0.9)]
        classifier.classify(objects, self.image)
        self.assertEqual(objects[0].mode, ObjectMode.REGENERATE)
        self.assertIsNone(objects[0].ocr_text)

    def test_ocr_text_triggers_preserve_and_stores_text(self) -> None:
        classifier = ObjectModeClassifier(
            preserve_classes=[], ocr_backend=_StubOcr("INVOICE 2026"), min_ocr_chars=3
        )
        objects = [SceneObject("a", "book", (0, 0, 50, 50), 0.9)]
        classifier.classify(objects, self.image)
        self.assertEqual(objects[0].mode, ObjectMode.PRESERVE)
        self.assertEqual(objects[0].ocr_text, "INVOICE 2026")

    def test_short_ocr_below_threshold_stays_regenerate(self) -> None:
        classifier = ObjectModeClassifier(
            preserve_classes=[], ocr_backend=_StubOcr("a"), min_ocr_chars=3
        )
        objects = [SceneObject("a", "book", (0, 0, 50, 50), 0.9)]
        classifier.classify(objects, self.image)
        self.assertEqual(objects[0].mode, ObjectMode.REGENERATE)


if __name__ == "__main__":
    unittest.main()
