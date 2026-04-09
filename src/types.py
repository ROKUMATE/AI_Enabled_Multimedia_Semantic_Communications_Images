"""Typed data models shared across semantic communication modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


BBox = tuple[float, float, float, float]


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
class EncodedPacket:
    """Represents compressed data payload and size estimate."""

    payload: str
    bit_estimate: int
    encoding: str = "semantic-token"
    semantic_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert encoded packet to dictionary."""
        return {
            "payload": self.payload,
            "bit_estimate": self.bit_estimate,
            "encoding": self.encoding,
            "semantic_size_bytes": self.semantic_size_bytes,
        }


@dataclass
class EvaluationMetrics:
    """Stores traditional, semantic, and perceptual quality metrics."""

    psnr: float
    ssim: float
    object_match_accuracy: float
    relation_match_accuracy: float
    text_similarity: float
    semantic_score: float = 0.0
    original_image_size_kb: float = 0.0
    semantic_size_bytes: int = 0
    compression_ratio: float = 0.0
    noise_level: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert metrics to dictionary format."""
        return {
            "psnr": self.psnr,
            "ssim": self.ssim,
            "object_match_accuracy": self.object_match_accuracy,
            "relation_match_accuracy": self.relation_match_accuracy,
            "text_similarity": self.text_similarity,
            "semantic_score": self.semantic_score,
            "original_image_size_kb": self.original_image_size_kb,
            "semantic_size_bytes": self.semantic_size_bytes,
            "compression_ratio": self.compression_ratio,
            "noise_level": self.noise_level,
        }
