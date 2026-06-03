"""YOLOv8-based object extractor (default)."""

from __future__ import annotations

import logging
from pathlib import Path

from ..types import SceneObject
from .base import ObjectExtractor


logger = logging.getLogger(__name__)


class YoloExtractor(ObjectExtractor):
    """Extract objects from an image with a pretrained YOLOv8 model."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        max_objects: int = 20,
    ) -> None:
        """Initialize the extractor and load the YOLO model from ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError(
                "ultralytics is required. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.max_objects = max_objects

    def extract(self, image_path: str | Path) -> list[SceneObject]:
        """Run detection on one image and return detected scene objects."""
        image_path = Path(image_path)
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            verbose=False,
        )

        detected: list[SceneObject] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue

            for idx in range(len(boxes)):
                if len(detected) >= self.max_objects:
                    logger.debug("Reached max_objects=%d; truncating.", self.max_objects)
                    return detected

                cls_id = int(boxes.cls[idx].item())
                confidence = float(boxes.conf[idx].item())
                x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
                detected.append(
                    SceneObject(
                        object_id=f"obj_{len(detected)}",
                        name=str(names[cls_id]),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=confidence,
                    )
                )

        logger.info("YOLO detected %d object(s) in %s", len(detected), image_path.name)
        return detected
