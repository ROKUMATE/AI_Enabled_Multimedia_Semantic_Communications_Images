"""Tests for reconstruction metrics."""

from __future__ import annotations

import unittest

import numpy as np

from src.metrics import Metrics, psnr
from src.types import ObjectMode, SceneObject


class PsnrTests(unittest.TestCase):
    def test_identical_images_high_psnr(self) -> None:
        image = np.full((16, 16, 3), 100, dtype=np.uint8)
        self.assertGreaterEqual(psnr(image, image), 99.0)

    def test_different_images_lower_psnr(self) -> None:
        a = np.zeros((16, 16, 3), dtype=np.uint8)
        b = np.full((16, 16, 3), 255, dtype=np.uint8)
        self.assertLess(psnr(a, b), 10.0)

    def test_shape_mismatch_is_resized(self) -> None:
        a = np.full((32, 32, 3), 50, dtype=np.uint8)
        b = np.full((16, 16, 3), 50, dtype=np.uint8)
        # Should not raise and should be high (same constant color).
        self.assertGreaterEqual(psnr(a, b), 99.0)


class MetricsComputeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = Metrics(downstream_extractor=None, ocr_backend=None,
                               deep_features=False, use_lpips=False)
        self.image = np.full((20, 20, 3), 80, dtype=np.uint8)

    def test_compression_ratio_and_optional_none(self) -> None:
        objects: list[SceneObject] = []
        result = self.metrics.compute(self.image, self.image, payload_bytes=100,
                                      raw_image_bytes=1000, original_objects=objects)
        self.assertAlmostEqual(result.compression_ratio, 10.0)
        self.assertIsNone(result.deep_feature_distance)
        self.assertIsNone(result.lpips)
        self.assertIsNone(result.ocr_legibility)

    def test_recall_without_extractor(self) -> None:
        # No objects -> trivially perfect recall.
        empty = self.metrics.compute(self.image, self.image, 100, 1000, [])
        self.assertEqual(empty.downstream_class_recall, 1.0)
        self.assertIsNone(empty.downstream_center_error)
        # Objects present but no extractor -> recall 0.
        objs = [SceneObject("a", "cat", (1, 1, 5, 5), 0.9)]
        withobj = self.metrics.compute(self.image, self.image, 100, 1000, objs)
        self.assertEqual(withobj.downstream_class_recall, 0.0)

    def test_preserve_text_count(self) -> None:
        objs = [
            SceneObject("a", "book", (1, 1, 5, 5), 0.9, mode=ObjectMode.PRESERVE, ocr_text="HI"),
            SceneObject("b", "cat", (6, 6, 9, 9), 0.9),
        ]
        result = self.metrics.compute(self.image, self.image, 100, 1000, objs)
        self.assertEqual(result.num_preserve_text, 1)

    def test_zero_payload_bytes_safe(self) -> None:
        result = self.metrics.compute(self.image, self.image, 0, 1000, [])
        self.assertEqual(result.compression_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
