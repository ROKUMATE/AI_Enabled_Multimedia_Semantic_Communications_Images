"""Reconstruction quality metrics.

Always computed (light, no extra deps): payload size, compression ratio vs the
raw image, PSNR, and a downstream detector match (re-run the detector on the
reconstruction and compare detected classes/positions to the original).

Optional (graceful ``None`` when the dep/model is unavailable): LPIPS, a
deep-feature cosine distance (torchvision VGG features), and an OCR legibility
check for ``preserve`` text objects.
"""

from __future__ import annotations

import logging
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .extractors.base import ObjectExtractor
from .ocr import OcrBackend
from .types import ObjectMode, SceneObject


logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """All metrics for one reconstruction (optional ones may be ``None``)."""

    payload_bytes: int
    raw_image_bytes: int
    compression_ratio: float
    psnr: float
    downstream_class_recall: float
    downstream_center_error: float | None
    deep_feature_distance: float | None = None
    lpips: float | None = None
    ocr_legibility: float | None = None
    num_preserve_text: int = 0

    def to_dict(self) -> dict[str, float | int | None]:
        """Return a flat dict for CSV/JSON tables."""
        return asdict(self)


def psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB (reconstructed resized to match)."""
    recon = _match_shape(original, reconstructed)
    mse = float(np.mean((original.astype(np.float64) - recon.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10((255.0**2) / mse))


def _match_shape(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Resize ``image`` to ``reference``'s height/width if they differ."""
    if image.shape[:2] == reference.shape[:2]:
        return image
    height, width = reference.shape[:2]
    return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))


class Metrics:
    """Compute reconstruction metrics for the experiment runner."""

    def __init__(
        self,
        downstream_extractor: ObjectExtractor | None = None,
        ocr_backend: OcrBackend | None = None,
        deep_features: bool = False,
        use_lpips: bool = False,
    ) -> None:
        """Configure which optional metrics to attempt."""
        self.downstream_extractor = downstream_extractor
        self.ocr_backend = ocr_backend
        self.deep_features = deep_features
        self.use_lpips = use_lpips
        self._vgg = None
        self._lpips_model = None

    def compute(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        payload_bytes: int,
        raw_image_bytes: int,
        original_objects: list[SceneObject],
    ) -> MetricsResult:
        """Compute the full metrics bundle for one reconstruction."""
        recon = _match_shape(original, reconstructed)
        recall, center_error = self._downstream_match(original_objects, recon, original.shape[:2])
        preserve_text = [
            obj for obj in original_objects
            if obj.mode == ObjectMode.PRESERVE and obj.ocr_text
        ]
        return MetricsResult(
            payload_bytes=payload_bytes,
            raw_image_bytes=raw_image_bytes,
            compression_ratio=(raw_image_bytes / payload_bytes) if payload_bytes > 0 else 0.0,
            psnr=psnr(original, recon),
            downstream_class_recall=recall,
            downstream_center_error=center_error,
            deep_feature_distance=self._deep_feature_distance(original, recon),
            lpips=self._lpips(original, recon),
            ocr_legibility=self._ocr_legibility(preserve_text, recon),
            num_preserve_text=len(preserve_text),
        )

    def _downstream_match(
        self,
        original_objects: list[SceneObject],
        reconstructed: np.ndarray,
        shape: tuple[int, int],
    ) -> tuple[float, float | None]:
        """Re-detect on the reconstruction and match classes/positions."""
        if self.downstream_extractor is None or not original_objects:
            return (1.0 if not original_objects else 0.0), None

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
            temp_path = Path(handle.name)
        try:
            Image.fromarray(reconstructed).save(temp_path)
            detected = self.downstream_extractor.extract(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

        diagonal = math.hypot(shape[0], shape[1]) or 1.0
        remaining = list(detected)
        matched = 0
        errors: list[float] = []
        for obj in original_objects:
            best_idx, best_dist = None, None
            ocx, ocy = obj.center()
            for idx, candidate in enumerate(remaining):
                if candidate.name != obj.name:
                    continue
                ccx, ccy = candidate.center()
                dist = math.hypot(ocx - ccx, ocy - ccy)
                if best_dist is None or dist < best_dist:
                    best_idx, best_dist = idx, dist
            if best_idx is not None:
                matched += 1
                errors.append(best_dist / diagonal)
                remaining.pop(best_idx)

        recall = matched / len(original_objects)
        center_error = float(np.mean(errors)) if errors else None
        return recall, center_error

    def _deep_feature_distance(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> float | None:
        """Cosine distance between VGG features (optional)."""
        if not self.deep_features:
            return None
        try:  # pragma: no cover - heavy/optional path
            import torch
            from torchvision import models, transforms

            if self._vgg is None:
                self._vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
            with torch.no_grad():
                feat_a = self._vgg(preprocess(Image.fromarray(original)).unsqueeze(0)).flatten()
                feat_b = self._vgg(preprocess(Image.fromarray(reconstructed)).unsqueeze(0)).flatten()
                cosine = torch.nn.functional.cosine_similarity(feat_a, feat_b, dim=0).item()
            return float(1.0 - cosine)
        except Exception as exc:  # pragma: no cover
            logger.warning("Deep-feature distance unavailable: %s", exc)
            return None

    def _lpips(self, original: np.ndarray, reconstructed: np.ndarray) -> float | None:
        """LPIPS perceptual distance (optional; needs the ``lpips`` package)."""
        if not self.use_lpips:
            return None
        try:  # pragma: no cover - optional dependency
            import lpips
            import torch

            if self._lpips_model is None:
                self._lpips_model = lpips.LPIPS(net="alex")

            def to_tensor(arr: np.ndarray) -> "torch.Tensor":
                resized = _match_shape(original, arr)
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 127.5 - 1.0
                return tensor.unsqueeze(0)

            with torch.no_grad():
                value = self._lpips_model(to_tensor(original), to_tensor(reconstructed))
            return float(value.item())
        except Exception as exc:  # pragma: no cover
            logger.warning("LPIPS unavailable: %s", exc)
            return None

    def _ocr_legibility(
        self, preserve_text: list[SceneObject], reconstructed: np.ndarray
    ) -> float | None:
        """Fraction of preserve-text objects whose OCR still reads the same."""
        if self.ocr_backend is None or not preserve_text:
            return None
        height, width = reconstructed.shape[:2]
        matches = 0
        for obj in preserve_text:
            x1, y1, x2, y2 = obj.bbox
            left, top = max(0, int(x1)), max(0, int(y1))
            right, bottom = min(width, int(x2)), min(height, int(y2))
            if right - left < 2 or bottom - top < 2:
                continue
            crop = reconstructed[top:bottom, left:right]
            try:
                read = self.ocr_backend.read(crop)
            except Exception:  # pragma: no cover
                read = ""
            if _normalize(read) == _normalize(obj.ocr_text or ""):
                matches += 1
        return matches / len(preserve_text)


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for OCR string comparison."""
    return " ".join(text.lower().split())
