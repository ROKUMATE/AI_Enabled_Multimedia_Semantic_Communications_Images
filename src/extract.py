"""Object extraction module using YOLOv8."""

from __future__ import annotations

from pathlib import Path

from .types import DetectedObject


class ObjectExtractor:
    """Extract objects from an image with YOLOv8."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        max_objects: int = 20,
    ) -> None:
        """Initialize extractor and load YOLO model lazily from ultralytics."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required. Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.max_objects = max_objects

    def extract(self, image_path: str | Path) -> list[DetectedObject]:
        """Run detection on one image and return detected objects with bounding boxes."""
        image_path = Path(image_path)
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            verbose=False,
        )

        detected: list[DetectedObject] = []
        for result in results:
            names = result.names
            boxes = result.boxes
            if boxes is None:
                continue

            for idx in range(len(boxes)):
                if len(detected) >= self.max_objects:
                    return detected

                cls_id = int(boxes.cls[idx].item())
                confidence = float(boxes.conf[idx].item())
                x1, y1, x2, y2 = boxes.xyxy[idx].tolist()
                detected.append(
                    DetectedObject(
                        object_id=f"obj_{len(detected)}",
                        name=str(names[cls_id]),
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=confidence,
                    )
                )

        return detected
