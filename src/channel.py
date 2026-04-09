"""Noisy channel simulator for semantic transmission experiments."""

from __future__ import annotations

import base64
import json
import random
import zlib
from typing import Any

from .types import EncodedPacket


class NoisyChannel:
    """Simulate semantic transmission loss by dropping objects and relations."""

    def __init__(self, noise_level: float = 0.1, seed: int | None = None) -> None:
        """Configure drop probability and RNG seed for reproducibility."""
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be in [0.0, 1.0].")
        self.noise_level = noise_level
        self.random = random.Random(seed)

    def transmit(self, packet: EncodedPacket) -> EncodedPacket:
        """Apply random semantic corruption and return a new encoded packet."""
        data = self._decode_payload(packet.payload)
        noisy = self._apply_noise(data)
        return self._encode_payload(noisy)

    def _apply_noise(self, data: dict[str, Any]) -> dict[str, Any]:
        """Drop objects and relations based on configured noise level."""
        objects = data.get("objects", [])
        kept_objects = [
            obj for obj in objects if self.random.random() > self.noise_level
        ]

        if objects and not kept_objects:
            kept_objects = [self.random.choice(objects)]

        kept_ids = {item.get("object_id") for item in kept_objects}

        relations = data.get("relations", [])
        kept_relations = []
        for rel in relations:
            subject_id = rel.get("subject_id")
            object_id = rel.get("object_id")
            if subject_id not in kept_ids or object_id not in kept_ids:
                continue
            if self.random.random() <= self.noise_level:
                continue
            kept_relations.append(rel)

        attributes = data.get("attributes", {})
        kept_attributes = {
            key: value for key, value in attributes.items() if key in kept_ids
        }

        return {
            "objects": kept_objects,
            "attributes": kept_attributes,
            "relations": kept_relations,
        }

    def _decode_payload(self, payload: str) -> dict[str, Any]:
        """Decode and decompress payload into dictionary form."""
        compressed = base64.b64decode(payload.encode("utf-8"))
        raw = zlib.decompress(compressed)
        return json.loads(raw.decode("utf-8"))

    def _encode_payload(self, data: dict[str, Any]) -> EncodedPacket:
        """Compress dictionary into an encoded packet."""
        raw_bytes = json.dumps(data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        compressed = zlib.compress(raw_bytes, level=9)
        payload = base64.b64encode(compressed).decode("utf-8")
        return EncodedPacket(payload=payload, bit_estimate=len(compressed) * 8)
