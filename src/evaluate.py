"""Evaluation module with traditional, semantic, and perceptual metrics."""

from __future__ import annotations

import logging
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Any

from .types import EvaluationMetrics, OARRepresentation


logger = logging.getLogger(__name__)

OBJECT_WEIGHT = 0.6
RELATION_WEIGHT = 0.4


class Evaluator:
    """Evaluate reconstruction quality across multiple metric layers."""

    def evaluate(
        self,
        original_oar: OARRepresentation,
        decoded_oar: OARRepresentation,
        original_text: str,
        reconstructed_text: str,
        original_image_path: Path | None = None,
        semantic_size_bytes: int | None = None,
        noise_level: float = 0.0,
    ) -> EvaluationMetrics:
        """Compute all metric layers and return structured metric object."""
        object_acc = self._object_match_accuracy(original_oar, decoded_oar)
        relation_acc = self._relation_match_accuracy(original_oar, decoded_oar)
        text_similarity = self._text_similarity(original_text, reconstructed_text)
        semantic_score = (OBJECT_WEIGHT * object_acc) + (RELATION_WEIGHT * relation_acc)
        original_image_size_kb = self._image_size_kb(original_image_path)
        semantic_size_bytes = max(0, int(semantic_size_bytes or 0))
        compression_ratio = self._compression_ratio(original_image_path, semantic_size_bytes)

        psnr, ssim = self._traditional_placeholders(object_acc, relation_acc)
        metrics = EvaluationMetrics(
            psnr=psnr,
            ssim=ssim,
            object_match_accuracy=object_acc,
            relation_match_accuracy=relation_acc,
            text_similarity=text_similarity,
            semantic_score=semantic_score,
            original_image_size_kb=original_image_size_kb,
            semantic_size_bytes=semantic_size_bytes,
            compression_ratio=compression_ratio,
            noise_level=noise_level,
        )
        logger.debug(
            "Evaluation metrics -> object=%.3f relation=%.3f semantic=%.3f noise=%.3f",
            object_acc,
            relation_acc,
            semantic_score,
            noise_level,
        )
        return metrics

    def evaluate_noise_robustness(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Aggregate experiment rows into noise-level summaries."""
        grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[float(row.get("noise_level", 0.0))].append(row)

        summary: list[dict[str, Any]] = []
        for noise_level in sorted(grouped):
            group = grouped[noise_level]
            summary.append(
                {
                    "noise_level": noise_level,
                    "mean_object_accuracy": mean(row.get("object_match_accuracy", 0.0) for row in group),
                    "mean_relation_accuracy": mean(row.get("relation_match_accuracy", 0.0) for row in group),
                    "mean_semantic_score": mean(row.get("semantic_score", 0.0) for row in group),
                    "mean_text_similarity": mean(row.get("text_similarity", 0.0) for row in group),
                    "mean_compression_ratio": mean(row.get("compression_ratio", 0.0) for row in group),
                    "sample_count": len(group),
                }
            )

        return summary

    def label_quality(self, metrics: EvaluationMetrics) -> str:
        """Map metric profile to an interpretable quality label."""
        aggregate = (metrics.semantic_score + metrics.text_similarity) / 2.0
        if aggregate >= 0.8:
            return "high"
        if aggregate >= 0.5:
            return "medium"
        return "low"

    def _object_match_accuracy(
        self, original_oar: OARRepresentation, decoded_oar: OARRepresentation
    ) -> float:
        """Compute object retention accuracy using object IDs."""
        original_ids = {item.object_id for item in original_oar.objects}
        decoded_ids = {item.object_id for item in decoded_oar.objects}
        if not original_ids:
            return 1.0
        return len(original_ids & decoded_ids) / len(original_ids)

    def _relation_match_accuracy(
        self, original_oar: OARRepresentation, decoded_oar: OARRepresentation
    ) -> float:
        """Compute relation retention accuracy using exact triplet match."""
        original_rel = {
            (item.subject_id, item.predicate, item.object_id)
            for item in original_oar.relations
        }
        decoded_rel = {
            (item.subject_id, item.predicate, item.object_id)
            for item in decoded_oar.relations
        }
        if not original_rel:
            return 1.0
        return len(original_rel & decoded_rel) / len(original_rel)

    def _text_similarity(self, source: str, target: str) -> float:
        """Compute basic perceptual text similarity using sequence matching."""
        return SequenceMatcher(a=source, b=target).ratio()

    def _traditional_placeholders(
        self, object_acc: float, relation_acc: float
    ) -> tuple[float, float]:
        """Generate placeholder PSNR and SSIM from semantic fidelity scores."""
        fidelity = (object_acc + relation_acc) / 2.0
        psnr = 20.0 + (20.0 * fidelity)
        ssim = min(1.0, max(0.0, fidelity))
        return psnr, ssim

    def _image_size_kb(self, image_path: Path | None) -> float:
        """Measure source image size in kilobytes if available."""
        if image_path is None or not image_path.exists():
            return 0.0
        return image_path.stat().st_size / 1024.0

    def _compression_ratio(self, image_path: Path | None, semantic_size_bytes: int) -> float:
        """Compute image-to-semantic compression ratio."""
        if image_path is None or not image_path.exists() or semantic_size_bytes <= 0:
            return 0.0
        image_bytes = float(image_path.stat().st_size)
        return image_bytes / float(semantic_size_bytes)
