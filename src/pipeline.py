"""End-to-end semantic image communication pipeline (v1).

Transmitter: detect -> relations -> importance -> mode -> crop -> payload.
Channel:     IdentityChannel (pass-through, only channel in v1).
Receiver:    reconstruct a similar image + composite crops + text description.

Every model-bearing step is selected from :class:`PipelineSettings` and lives
behind a base class, so a learned/alternative implementation can be swapped in
via config without changing this orchestration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .appearance import CropCompressor
from .channels import IdentityChannel
from .extractors import LearnedObjectExtractor, YoloExtractor
from .extractors.base import ObjectExtractor
from .importance import HeuristicImportanceScorer
from .mode_classifier import ObjectModeClassifier
from .ocr import get_ocr_backend
from .payload import SemanticPayload
from .reconstructors import CompositionalReconstructor, DiffusionReconstructor
from .reconstructors.base import ReconstructionResult, Reconstructor
from .relations import LearnedRelationBuilder, RuleBasedRelationBuilder
from .relations.base import RelationBuilder
from .types import Relation, SceneObject


logger = logging.getLogger(__name__)


@dataclass
class PipelineSettings:
    """Configuration for the v1 image pipeline (built from YAML/CLI)."""

    # detection
    extractor: str = "yolo"
    model_path: str = "yolov8n.pt"
    checkpoint_path: str = "checkpoints/detector.pt"
    conf_threshold: float = 0.25
    max_objects: int = 20
    # relations
    relation_builder: str = "rule_based"
    near_distance_threshold: float = 120.0
    # importance
    importance_budget: int = 3
    # mode classification
    preserve_classes: list[str] = field(default_factory=lambda: ["person"])
    ocr_enabled: bool = True
    ocr_backend: str = "auto"
    # appearance
    appearance_format: str = "JPEG"
    preserve_quality: int = 95
    regenerate_quality: int = 35
    # channel / streams
    channel: str = "identity"
    structure_priority: int = 0
    appearance_priority: int = 1
    # reconstruction
    reconstructor: str = "compositional"
    background_color: tuple[int, int, int] = (127, 127, 127)
    diffusion_enabled: bool = False
    diffusion_model_id: str = "stabilityai/sd-turbo"
    # misc
    seed: int | None = 42

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PipelineSettings":
        """Build settings from a merged config dict (extra keys ignored)."""
        importance = config.get("importance", {}) or {}
        appearance = config.get("appearance", {}) or {}
        ocr = config.get("ocr", {}) or {}
        diffusion = config.get("diffusion", {}) or {}
        streams = config.get("streams", {}) or {}
        background = config.get("background_color", [127, 127, 127])
        return cls(
            extractor=str(config.get("extractor", "yolo")),
            model_path=str(config.get("model_path", "yolov8n.pt")),
            checkpoint_path=str(config.get("checkpoint_path", "checkpoints/detector.pt")),
            conf_threshold=float(config.get("conf_threshold", 0.25)),
            max_objects=int(config.get("max_objects", 20)),
            relation_builder=str(config.get("relation_builder", "rule_based")),
            near_distance_threshold=float(config.get("near_distance_threshold", 120.0)),
            importance_budget=int(importance.get("budget", 3)),
            preserve_classes=list(config.get("preserve_classes", ["person"])),
            ocr_enabled=bool(ocr.get("enabled", True)),
            ocr_backend=str(ocr.get("backend", "auto")),
            appearance_format=str(appearance.get("format", "JPEG")),
            preserve_quality=int(appearance.get("preserve_quality", 95)),
            regenerate_quality=int(appearance.get("regenerate_quality", 35)),
            channel=str(config.get("channel", "identity")),
            structure_priority=int(streams.get("structure_priority", 0)),
            appearance_priority=int(streams.get("appearance_priority", 1)),
            reconstructor=str(config.get("reconstructor", "compositional")),
            background_color=tuple(int(c) for c in background),  # type: ignore[arg-type]
            diffusion_enabled=bool(diffusion.get("enabled", False)),
            diffusion_model_id=str(diffusion.get("model_id", "stabilityai/sd-turbo")),
            seed=config.get("seed", 42),
        )


@dataclass
class PipelineOutput:
    """Everything produced for one image, for downstream metrics/saving."""

    image_id: str
    original_image: np.ndarray
    objects: list[SceneObject]
    relations: list[Relation]
    payload: SemanticPayload
    received_payload: SemanticPayload
    size_report: dict[str, int]
    reconstruction: ReconstructionResult

    def scene_graph(self) -> dict[str, Any]:
        """Return the JSON-serializable scene graph that was transmitted."""
        return self.payload.structure


def load_image_rgb(image_path: str | Path) -> np.ndarray:
    """Load an image from disk as an RGB uint8 array."""
    with Image.open(image_path) as image:
        return np.asarray(image.convert("RGB"))


class SemanticPipeline:
    """Wire the transmitter, channel, and receiver together."""

    def __init__(self, settings: PipelineSettings) -> None:
        """Construct all pipeline components from settings."""
        self.settings = settings
        self.extractor: ObjectExtractor = self._build_extractor(settings)
        self.relation_builder: RelationBuilder = self._build_relation_builder(settings)
        self.scorer = HeuristicImportanceScorer()
        ocr_backend = (
            get_ocr_backend(settings.ocr_backend) if settings.ocr_enabled else None
        )
        self.mode_classifier = ObjectModeClassifier(
            preserve_classes=settings.preserve_classes, ocr_backend=ocr_backend
        )
        self.appearance_encoder = CropCompressor(
            image_format=settings.appearance_format,
            preserve_quality=settings.preserve_quality,
            regenerate_quality=settings.regenerate_quality,
        )
        self.channel = IdentityChannel()
        self.reconstructor: Reconstructor = self._build_reconstructor(settings)

    @staticmethod
    def _build_extractor(settings: PipelineSettings) -> ObjectExtractor:
        if settings.extractor == "learned":
            return LearnedObjectExtractor(
                checkpoint_path=settings.checkpoint_path,
                base_model_path=settings.model_path,
                conf_threshold=settings.conf_threshold,
                max_objects=settings.max_objects,
            )
        return YoloExtractor(
            model_path=settings.model_path,
            conf_threshold=settings.conf_threshold,
            max_objects=settings.max_objects,
        )

    @staticmethod
    def _build_relation_builder(settings: PipelineSettings) -> RelationBuilder:
        if settings.relation_builder == "learned":
            return LearnedRelationBuilder(settings.near_distance_threshold)
        return RuleBasedRelationBuilder(settings.near_distance_threshold)

    @staticmethod
    def _build_reconstructor(settings: PipelineSettings) -> Reconstructor:
        if settings.reconstructor == "diffusion" or settings.diffusion_enabled:
            return DiffusionReconstructor(
                model_id=settings.diffusion_model_id,
                background_color=settings.background_color,
                force_cpu_fallback=not settings.diffusion_enabled,
            )
        return CompositionalReconstructor(background_color=settings.background_color)

    def build_payload(
        self, image_rgb: np.ndarray, objects: list[SceneObject], relations: list[Relation]
    ) -> SemanticPayload:
        """Assemble the two-stream payload from the analysed scene."""
        height, width = image_rgb.shape[:2]
        structure = {
            "image_size": [int(width), int(height)],
            "objects": [obj.to_dict() for obj in objects],
            "relations": [rel.to_dict() for rel in relations],
        }
        crops = {
            obj.object_id: self.appearance_encoder.encode(obj, image_rgb)
            for obj in objects
            if obj.selected
        }
        return SemanticPayload(
            structure=structure,
            crops=crops,
            structure_priority=self.settings.structure_priority,
            appearance_priority=self.settings.appearance_priority,
        )

    def run(self, image_path: str | Path) -> PipelineOutput:
        """Run one image fully through transmitter, channel, and receiver."""
        image_path = Path(image_path)
        image_rgb = load_image_rgb(image_path)
        height, width = image_rgb.shape[:2]
        image_size = (width, height)

        objects = self.extractor.extract(image_path)
        relations = self.relation_builder.build(objects)
        # Mode must be decided before selection: preserve objects are always sent.
        self.mode_classifier.classify(objects, image_rgb)
        self.scorer.select(objects, image_size, self.settings.importance_budget)

        payload = self.build_payload(image_rgb, objects, relations)
        size_report = payload.size_report()
        logger.info(
            "Payload for %s: structure=%dB appearance=%dB total=%dB (%d crops).",
            image_path.name,
            size_report["structure_bytes"],
            size_report["appearance_bytes"],
            size_report["total_bytes"],
            size_report["num_crops"],
        )

        received = self.channel.transmit(payload)
        reconstruction = self.reconstructor.reconstruct(received, self.appearance_encoder)

        return PipelineOutput(
            image_id=image_path.stem,
            original_image=image_rgb,
            objects=objects,
            relations=relations,
            payload=payload,
            received_payload=received,
            size_report=size_report,
            reconstruction=reconstruction,
        )
