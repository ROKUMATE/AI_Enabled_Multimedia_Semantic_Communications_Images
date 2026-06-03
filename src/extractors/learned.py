"""Our own (fine-tuned) object extractor.

SCAFFOLD (P5). This is the seam where we plug in a detector we train ourselves
instead of relying on the off-the-shelf YOLO weights. For v1 it is a thin
fine-tuning wrapper around an existing detector backbone: if a trained
checkpoint is present it is loaded, otherwise we fall back to the pretrained
YOLO model with a warning so the pipeline keeps running.

See ``scripts/train_detector.py`` for the training entry point and
``PLAN.md`` (TODO backlog) for the full plan.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ..types import SceneObject
from .base import ObjectExtractor
from .yolo import YoloExtractor


logger = logging.getLogger(__name__)


class LearnedObjectExtractor(ObjectExtractor):
    """Detector fine-tuned on our own data, with graceful YOLO fallback.

    Selectable from config (``extractor: learned``) and runnable side by side
    with :class:`YoloExtractor` for comparison.
    """

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/detector.pt",
        base_model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        max_objects: int = 20,
    ) -> None:
        """Load our checkpoint if it exists, else fall back to base YOLO."""
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            logger.info("Loading learned detector checkpoint: %s", checkpoint)
            model_path = str(checkpoint)
            self.using_checkpoint = True
        else:
            logger.warning(
                "Learned detector checkpoint not found at %s; falling back to "
                "pretrained YOLO (%s). Train one via scripts/train_detector.py.",
                checkpoint,
                base_model_path,
            )
            model_path = base_model_path
            self.using_checkpoint = False

        # TODO(P5): replace this YOLO delegate with our own model class once
        # scripts/train_detector.py produces a checkpoint in our format.
        self._delegate = YoloExtractor(
            model_path=model_path,
            conf_threshold=conf_threshold,
            max_objects=max_objects,
        )

    def extract(self, image_path: str | Path) -> list[SceneObject]:
        """Run detection through the learned model (or fallback)."""
        return self._delegate.extract(image_path)
