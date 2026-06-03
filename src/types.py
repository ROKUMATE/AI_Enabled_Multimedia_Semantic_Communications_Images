"""Typed data models shared across semantic communication modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


BBox = tuple[float, float, float, float]


class ObjectMode(str, Enum):
    """Transmission mode for an object's appearance.

    ``REGENERATE`` objects may be replaced by a similar-looking version on the
    receiver. ``PRESERVE`` objects (text/documents, faces, logos) must keep
    their exact appearance, so they are sent as a higher-quality crop and are
    never produced by generative reconstruction.
    """

    REGENERATE = "regenerate"
    PRESERVE = "preserve"


@dataclass
class DetectedObject:
    """Represents one detected object and its geometric properties."""

    object_id: str
    name: str
    bbox: BBox
    confidence: float

    def center(self) -> tuple[float, float]:
        """Return the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert object to a JSON-serializable dictionary."""
        return {
            "object_id": self.object_id,
            "name": self.name,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DetectedObject":
        """Create an instance from serialized dictionary data."""
        bbox_data = data.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bbox: BBox = (
            float(bbox_data[0]),
            float(bbox_data[1]),
            float(bbox_data[2]),
            float(bbox_data[3]),
        )
        return cls(
            object_id=str(data.get("object_id", "obj_0")),
            name=str(data.get("name", "unknown")),
            bbox=bbox,
            confidence=float(data.get("confidence", 0.0)),
        )


@dataclass
class Relation:
    """Represents semantic relation between two objects."""

    subject_id: str
    predicate: str
    object_id: str

    def to_dict(self) -> dict[str, str]:
        """Convert relation to dictionary representation."""
        return {
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> "Relation":
        """Create relation from dictionary data."""
        return cls(
            subject_id=str(data.get("subject_id", "obj_0")),
            predicate=str(data.get("predicate", "related_to")),
            object_id=str(data.get("object_id", "obj_0")),
        )


@dataclass
class OARRepresentation:
    """Container for object-attribute-relation semantics."""

    objects: list[DetectedObject] = field(default_factory=list)
    attributes: dict[str, dict[str, Any]] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize OAR representation to dictionary."""
        return {
            "objects": [obj.to_dict() for obj in self.objects],
            "attributes": self.attributes,
            "relations": [rel.to_dict() for rel in self.relations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OARRepresentation":
        """Deserialize OAR representation from dictionary."""
        objects = [DetectedObject.from_dict(item) for item in data.get("objects", [])]
        relations = [Relation.from_dict(item) for item in data.get("relations", [])]
        attributes = data.get("attributes", {})
        return cls(objects=objects, attributes=attributes, relations=relations)


@dataclass
class SceneObject:
    """A detected object enriched with importance, mode and OCR metadata.

    This is the working object for the v1 image pipeline. It carries everything
    the transmitter decides about an object: how important it is, whether its
    appearance must be preserved, whether a crop is sent for it, and any OCR
    text recovered from it. It exposes the same ``object_id``/``name``/
    ``center()`` surface as :class:`DetectedObject` so relation builders can
    operate on either type.
    """

    object_id: str
    name: str
    bbox: BBox
    confidence: float
    mode: ObjectMode = ObjectMode.REGENERATE
    importance: float = 0.0
    selected: bool = False
    ocr_text: str | None = None

    def center(self) -> tuple[float, float]:
        """Return the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def area(self) -> float:
        """Return the bounding-box area in pixels."""
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @classmethod
    def from_detected(cls, detected: DetectedObject) -> "SceneObject":
        """Build a scene object from a bare detection."""
        return cls(
            object_id=detected.object_id,
            name=detected.name,
            bbox=detected.bbox,
            confidence=detected.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert object to a JSON-serializable dictionary."""
        return {
            "object_id": self.object_id,
            "name": self.name,
            "bbox": [float(value) for value in self.bbox],
            "confidence": self.confidence,
            "mode": self.mode.value,
            "importance": self.importance,
            "selected": self.selected,
            "ocr_text": self.ocr_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SceneObject":
        """Create an instance from serialized dictionary data."""
        bbox_data = data.get("bbox", [0.0, 0.0, 0.0, 0.0])
        bbox: BBox = (
            float(bbox_data[0]),
            float(bbox_data[1]),
            float(bbox_data[2]),
            float(bbox_data[3]),
        )
        return cls(
            object_id=str(data.get("object_id", "obj_0")),
            name=str(data.get("name", "unknown")),
            bbox=bbox,
            confidence=float(data.get("confidence", 0.0)),
            mode=ObjectMode(str(data.get("mode", ObjectMode.REGENERATE.value))),
            importance=float(data.get("importance", 0.0)),
            selected=bool(data.get("selected", False)),
            ocr_text=data.get("ocr_text"),
        )


@dataclass
class Stream:
    """One independently-degradable payload stream tagged with a priority.

    ``priority`` is an integer where ``0`` denotes the highest protection. A
    future :class:`~src.channels.base.Channel` will corrupt streams
    independently and apply unequal error protection based on this value.
    """

    name: str
    priority: int
    content: bytes

    @property
    def size_bytes(self) -> int:
        """Return the raw on-wire size of the stream content in bytes."""
        return len(self.content)
