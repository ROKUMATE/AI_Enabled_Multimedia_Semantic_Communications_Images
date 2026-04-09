"""Decoder for recovering OAR data from encoded packets."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from .semantic_codec import decode_oar_compact, normalize_oar_dict
from .types import EncodedPacket, OARRepresentation


logger = logging.getLogger(__name__)


class OARDecoder:
    """Decode compressed packets back into OAR structures."""

    def decode(self, packet: EncodedPacket | Mapping[str, Any] | str) -> OARRepresentation:
        """Decode packet payload and reconstruct OAR representation."""
        if isinstance(packet, EncodedPacket):
            payload = packet.payload
        else:
            payload = packet

        data = decode_oar_compact(payload)
        data = self._repair_partial_graph(normalize_oar_dict(data))
        logger.debug(
            "Decoded semantic payload with %d objects and %d relations.",
            len(data.get("objects", [])),
            len(data.get("relations", [])),
        )
        return OARRepresentation.from_dict(data)

    def _repair_partial_graph(self, data: dict[str, Any]) -> dict[str, Any]:
        """Add placeholder nodes for relation endpoints missing from the decoded objects."""
        objects = list(data.get("objects", []))
        relations = list(data.get("relations", []))
        object_lookup = {
            str(item.get("object_id"))
            for item in objects
            if isinstance(item, dict) and item.get("object_id") is not None
        }

        for relation in relations:
            if not isinstance(relation, dict):
                continue

            for role in ("subject_id", "object_id"):
                object_id = str(relation.get(role, ""))
                if not object_id or object_id in object_lookup:
                    continue
                object_lookup.add(object_id)
                objects.append(
                    {
                        "object_id": object_id,
                        "name": "missing_object",
                        "bbox": [0.0, 0.0, 0.0, 0.0],
                        "confidence": 0.0,
                    }
                )

        data["objects"] = objects
        data["relations"] = relations
        return data
