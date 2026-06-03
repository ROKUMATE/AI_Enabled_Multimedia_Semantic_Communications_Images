"""Two-stream serializable semantic payload.

The payload carries two **independently-degradable** streams, each tagged with
a priority (0 = highest protection):

1. **structure stream** — the compact scene graph (object ids, classes, boxes,
   relations, per-object mode, OCR text, original image size). Serialized JSON.
2. **appearance stream** — ``object_id -> compressed crop bytes`` for the
   objects whose appearance is transmitted.

``to_bytes()``/``from_bytes()`` use a length-prefixed binary container (no
base64), so the reported byte sizes are the true on-wire sizes. A future
:class:`~src.channels.base.Channel` corrupts the two streams independently;
keeping them separate and priority-tagged is what makes unequal error
protection possible later without touching callers.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

from .types import Stream


_MAGIC = b"SPL1"
_VERSION = 1


def _pack_crops(crops: dict[str, bytes]) -> bytes:
    """Pack the appearance crop map into a length-prefixed binary blob."""
    out = bytearray()
    out += struct.pack(">I", len(crops))
    for object_id, data in crops.items():
        id_bytes = object_id.encode("utf-8")
        out += struct.pack(">H", len(id_bytes))
        out += id_bytes
        out += struct.pack(">I", len(data))
        out += data
    return bytes(out)


def _unpack_crops(blob: bytes) -> dict[str, bytes]:
    """Inverse of :func:`_pack_crops`."""
    crops: dict[str, bytes] = {}
    offset = 0
    (count,) = struct.unpack_from(">I", blob, offset)
    offset += 4
    for _ in range(count):
        (id_len,) = struct.unpack_from(">H", blob, offset)
        offset += 2
        object_id = blob[offset : offset + id_len].decode("utf-8")
        offset += id_len
        (data_len,) = struct.unpack_from(">I", blob, offset)
        offset += 4
        crops[object_id] = bytes(blob[offset : offset + data_len])
        offset += data_len
    return crops


@dataclass
class SemanticPayload:
    """Serializable payload with a structure stream and an appearance stream."""

    structure: dict[str, Any]
    crops: dict[str, bytes] = field(default_factory=dict)
    structure_priority: int = 0
    appearance_priority: int = 1

    @property
    def image_size(self) -> tuple[int, int]:
        """Return the original image ``(width, height)`` from the scene graph."""
        size = self.structure.get("image_size", [0, 0])
        return (int(size[0]), int(size[1]))

    def _structure_bytes(self) -> bytes:
        """Serialize the scene graph to canonical JSON bytes."""
        return json.dumps(self.structure, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def streams(self) -> list[Stream]:
        """Return the priority-tagged streams that make up this payload."""
        return [
            Stream("structure", self.structure_priority, self._structure_bytes()),
            Stream("appearance", self.appearance_priority, _pack_crops(self.crops)),
        ]

    @classmethod
    def from_streams(cls, streams: list[Stream]) -> "SemanticPayload":
        """Reassemble a payload from its (possibly degraded) streams."""
        by_name = {stream.name: stream for stream in streams}
        structure_stream = by_name["structure"]
        appearance_stream = by_name.get("appearance")
        structure = json.loads(structure_stream.content.decode("utf-8"))
        crops = _unpack_crops(appearance_stream.content) if appearance_stream else {}
        return cls(
            structure=structure,
            crops=crops,
            structure_priority=structure_stream.priority,
            appearance_priority=appearance_stream.priority if appearance_stream else 1,
        )

    def to_bytes(self) -> bytes:
        """Serialize the whole payload into one binary container."""
        structure_bytes = self._structure_bytes()
        out = bytearray()
        out += _MAGIC
        out += struct.pack(">B", _VERSION)
        out += struct.pack(">B", self.structure_priority & 0xFF)
        out += struct.pack(">B", self.appearance_priority & 0xFF)
        out += struct.pack(">I", len(structure_bytes))
        out += structure_bytes
        out += _pack_crops(self.crops)
        return bytes(out)

    @classmethod
    def from_bytes(cls, blob: bytes) -> "SemanticPayload":
        """Inverse of :meth:`to_bytes`."""
        if blob[:4] != _MAGIC:
            raise ValueError("Invalid payload: bad magic header.")
        offset = 4
        (version,) = struct.unpack_from(">B", blob, offset)
        offset += 1
        if version != _VERSION:
            raise ValueError(f"Unsupported payload version: {version}")
        (structure_priority,) = struct.unpack_from(">B", blob, offset)
        offset += 1
        (appearance_priority,) = struct.unpack_from(">B", blob, offset)
        offset += 1
        (structure_len,) = struct.unpack_from(">I", blob, offset)
        offset += 4
        structure = json.loads(blob[offset : offset + structure_len].decode("utf-8"))
        offset += structure_len
        crops = _unpack_crops(blob[offset:])
        return cls(
            structure=structure,
            crops=crops,
            structure_priority=structure_priority,
            appearance_priority=appearance_priority,
        )

    def size_report(self) -> dict[str, int]:
        """Report the byte size of each stream and the total container."""
        structure_bytes = len(self._structure_bytes())
        appearance_bytes = len(_pack_crops(self.crops))
        crops_raw_bytes = sum(len(data) for data in self.crops.values())
        return {
            "structure_bytes": structure_bytes,
            "appearance_bytes": appearance_bytes,
            "crops_raw_bytes": crops_raw_bytes,
            "num_crops": len(self.crops),
            "total_bytes": len(self.to_bytes()),
        }
