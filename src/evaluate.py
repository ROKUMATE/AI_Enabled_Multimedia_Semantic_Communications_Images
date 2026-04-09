"""Evaluation module with traditional, semantic, and perceptual metrics."""

from __future__ import annotations

from difflib import SequenceMatcher

from .types import EvaluationMetrics, OARRepresentation


class Evaluator:
    """Evaluate reconstruction quality across multiple metric layers."""

    def evaluate(
        self,
        original_oar: OARRepresentation,
        decoded_oar: OARRepresentation,
        original_text: str,
        reconstructed_text: str,
    ) -> EvaluationMetrics:
        """Compute all metric layers and return structured metric object."""
        object_acc = self._object_match_accuracy(original_oar, decoded_oar)
        relation_acc = self._relation_match_accuracy(original_oar, decoded_oar)
        text_similarity = self._text_similarity(original_text, reconstructed_text)

        psnr, ssim = self._traditional_placeholders(object_acc, relation_acc)
        return EvaluationMetrics(
            psnr=psnr,
            ssim=ssim,
            object_match_accuracy=object_acc,
            relation_match_accuracy=relation_acc,
            text_similarity=text_similarity,
        )

    def label_quality(self, metrics: EvaluationMetrics) -> str:
        """Map metric profile to an interpretable quality label."""
        semantic_score = (metrics.object_match_accuracy + metrics.relation_match_accuracy) / 2.0
        aggregate = (semantic_score + metrics.text_similarity) / 2.0
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
