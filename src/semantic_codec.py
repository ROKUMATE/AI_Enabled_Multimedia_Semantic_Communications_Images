"""Compact semantic token codec for OAR payloads."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any


def normalize_oar_dict(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize arbitrary OAR-like content into a safe dictionary structure."""
    if not isinstance(data, Mapping):
        return {"objects": [], "attributes": {}, "relations": []}

    objects: list[dict[str, Any]] = []
    for index, item in enumerate(data.get("objects", [])):
        if not isinstance(item, Mapping):
            continue

        bbox_data = item.get("bbox", [0.0, 0.0, 0.0, 0.0])
        if not isinstance(bbox_data, (list, tuple)) or len(bbox_data) != 4:
            bbox_data = [0.0, 0.0, 0.0, 0.0]

        objects.append(
            {
                "object_id": str(item.get("object_id", f"obj_{index}")),
                "name": str(item.get("name", "unknown")),
                "bbox": [float(value) for value in bbox_data],
                "confidence": float(item.get("confidence", 0.0)),
            }
        )

    relations: list[dict[str, str]] = []
    for item in data.get("relations", []):
        if not isinstance(item, Mapping):
            continue

        relations.append(
            {
                "subject_id": str(item.get("subject_id", "obj_0")),
                "predicate": str(item.get("predicate", "related_to")),
                "object_id": str(item.get("object_id", "obj_0")),
            }
        )

    attributes = data.get("attributes", {})
    if not isinstance(attributes, Mapping):
        attributes = {}

    return {
        "objects": objects,
        "attributes": copy.deepcopy(dict(attributes)),
        "relations": relations,
    }


def encode_oar_compact(data: Mapping[str, Any] | None) -> tuple[str, int]:
    """Encode OAR data into a compact semantic token string and its bit size."""
    normalized = normalize_oar_dict(data)
    tokens: list[str] = []

    for item in normalized["objects"]:
        tokens.append(f"{item['object_id']}={item['name']}")

    for item in normalized["relations"]:
        tokens.append(f"{item['subject_id']}-{item['predicate']}-{item['object_id']}")

    encoded_data = "|".join(tokens)
    bit_size = len(encoded_data.encode("utf-8")) * 8
    return encoded_data, bit_size


def decode_oar_compact(payload: str | bytes | Mapping[str, Any] | None) -> dict[str, Any]:
    """Decode compact semantic tokens or legacy JSON into an OAR dictionary."""
    if payload is None:
        return {"objects": [], "attributes": {}, "relations": []}

    if isinstance(payload, Mapping):
        return normalize_oar_dict(payload)

    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="ignore")

    payload_text = str(payload).strip()
    if not payload_text:
        return {"objects": [], "attributes": {}, "relations": []}

    if payload_text.startswith("{") or payload_text.startswith("["):
        try:
            parsed = json.loads(payload_text)
        except json.JSONDecodeError:
            parsed = None
        else:
            if isinstance(parsed, Mapping):
                return normalize_oar_dict(parsed)

    objects: list[dict[str, Any]] = []
    relations: list[dict[str, str]] = []
    attributes: dict[str, Any] = {}
    object_lookup: set[str] = set()

    for token in payload_text.split("|"):
        token = token.strip()
        if not token:
            continue

        if token.startswith("attributes="):
            try:
                attributes_payload = token.split("=", 1)[1]
                parsed_attributes = json.loads(attributes_payload)
                if isinstance(parsed_attributes, Mapping):
                    attributes.update(dict(parsed_attributes))
            except json.JSONDecodeError:
                continue
            continue

        if token.count("=") == 1 and token.count("-") < 2:
            object_id, name = token.split("=", 1)
            object_id = object_id.strip() or f"obj_{len(objects)}"
            if object_id in object_lookup:
                continue
            object_lookup.add(object_id)
            objects.append(
                {
                    "object_id": object_id,
                    "name": name.strip() or "unknown",
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "confidence": 0.0,
                }
            )
            continue

        if token.count("-") >= 2:
            subject_id, predicate, object_id = token.rsplit("-", 2)
            relations.append(
                {
                    "subject_id": subject_id.strip() or "obj_0",
                    "predicate": predicate.strip() or "related_to",
                    "object_id": object_id.strip() or "obj_0",
                }
            )
            continue

        object_id = f"obj_{len(objects)}"
        if object_id in object_lookup:
            continue
        object_lookup.add(object_id)
        objects.append(
            {
                "object_id": object_id,
                "name": token,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "confidence": 0.0,
            }
        )

    return {"objects": objects, "attributes": attributes, "relations": relations}


def estimate_semantic_bits(encoded_data: str) -> int:
    """Estimate semantic bitrate from UTF-8 token length."""
    return len(encoded_data.encode("utf-8")) * 8