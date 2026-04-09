"""Noisy channel simulator for semantic transmission experiments."""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

from .semantic_codec import decode_oar_compact, encode_oar_compact
from .types import EncodedPacket


def apply_channel_noise(oar_data: dict[str, Any], drop_prob: float = 0.2, seed: int | None = None) -> dict[str, Any]:
    """Apply reproducible semantic noise without mutating the original input."""
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError("drop_prob must be in [0.0, 1.0].")

    rng = random.Random(seed)
    data = deepcopy(oar_data) if isinstance(oar_data, dict) else {}
    objects = list(data.get("objects", []))
    relations = list(data.get("relations", []))
    attributes = data.get("attributes", {}) if isinstance(data.get("attributes", {}), dict) else {}

    kept_objects: list[dict[str, Any]] = [
        obj for obj in objects if rng.random() > drop_prob and isinstance(obj, dict)
    ]

    if objects and not kept_objects:
        fallback_objects = [obj for obj in objects if isinstance(obj, dict)]
        if fallback_objects:
            kept_objects = [rng.choice(fallback_objects)]

    kept_ids = {
        str(item.get("object_id"))
        for item in kept_objects
        if isinstance(item, dict) and item.get("object_id") is not None
    }

    kept_relations: list[dict[str, Any]] = []
    for relation in relations:
        if not isinstance(relation, dict):
            continue

        subject_id = str(relation.get("subject_id", ""))
        object_id = str(relation.get("object_id", ""))
        if subject_id not in kept_ids or object_id not in kept_ids:
            continue
        if rng.random() <= drop_prob:
            continue
        kept_relations.append(deepcopy(relation))

    kept_attributes = {
        key: deepcopy(value)
        for key, value in attributes.items()
        if key in kept_ids
    }

    return {
        "objects": deepcopy(kept_objects),
        "attributes": kept_attributes,
        "relations": kept_relations,
    }


class NoisyChannel:
    """Simulate semantic transmission loss by dropping objects and relations."""

    def __init__(self, noise_level: float = 0.1, seed: int | None = None) -> None:
        """Configure drop probability and RNG seed for reproducibility."""
        if not 0.0 <= noise_level <= 1.0:
            raise ValueError("noise_level must be in [0.0, 1.0].")
        self.noise_level = noise_level
        self.seed = seed
        self.random = random.Random(seed)

    def transmit(self, packet: EncodedPacket) -> EncodedPacket:
        """Apply random semantic corruption and return a new encoded packet."""
        data = decode_oar_compact(packet.payload)
        noisy = self._apply_noise(data)
        encoded_data, bit_size = encode_oar_compact(noisy)
        return EncodedPacket(
            payload=encoded_data,
            bit_estimate=bit_size,
            encoding="semantic-token",
            semantic_size_bytes=len(encoded_data.encode("utf-8")),
        )

    def _apply_noise(self, data: dict[str, Any]) -> dict[str, Any]:
        """Drop objects and relations based on configured noise level."""
        noisy = apply_channel_noise(data, drop_prob=self.noise_level, seed=self.random.randint(0, 2**31 - 1))
        return noisy
