"""Tests for two-stream payload (de)serialization."""

from __future__ import annotations

import unittest

from src.payload import SemanticPayload


def _sample_payload() -> SemanticPayload:
    structure = {
        "image_size": [640, 480],
        "objects": [
            {"object_id": "obj_0", "name": "person", "bbox": [1, 2, 3, 4],
             "confidence": 0.9, "mode": "preserve", "ocr_text": "HELLO"},
            {"object_id": "obj_1", "name": "car", "bbox": [5, 6, 7, 8],
             "confidence": 0.5, "mode": "regenerate", "ocr_text": None},
        ],
        "relations": [{"subject_id": "obj_0", "predicate": "near", "object_id": "obj_1"}],
    }
    crops = {"obj_0": b"\x00\x01\x02crop-bytes", "obj_1": b"\xff\xfe"}
    return SemanticPayload(structure=structure, crops=crops,
                           structure_priority=0, appearance_priority=1)


class PayloadRoundTripTests(unittest.TestCase):
    def test_to_from_bytes_roundtrip(self) -> None:
        payload = _sample_payload()
        restored = SemanticPayload.from_bytes(payload.to_bytes())
        self.assertEqual(restored.structure, payload.structure)
        self.assertEqual(restored.crops, payload.crops)
        self.assertEqual(restored.structure_priority, 0)
        self.assertEqual(restored.appearance_priority, 1)

    def test_streams_roundtrip(self) -> None:
        payload = _sample_payload()
        streams = payload.streams()
        names = {stream.name for stream in streams}
        self.assertEqual(names, {"structure", "appearance"})
        restored = SemanticPayload.from_streams(streams)
        self.assertEqual(restored.structure, payload.structure)
        self.assertEqual(restored.crops, payload.crops)

    def test_stream_priorities_tagged(self) -> None:
        payload = _sample_payload()
        by_name = {stream.name: stream for stream in payload.streams()}
        self.assertEqual(by_name["structure"].priority, 0)
        self.assertEqual(by_name["appearance"].priority, 1)

    def test_size_report(self) -> None:
        payload = _sample_payload()
        report = payload.size_report()
        self.assertEqual(report["num_crops"], 2)
        self.assertEqual(report["crops_raw_bytes"], len(b"\x00\x01\x02crop-bytes") + len(b"\xff\xfe"))
        self.assertGreater(report["structure_bytes"], 0)
        self.assertGreaterEqual(report["total_bytes"], report["structure_bytes"] + report["appearance_bytes"])

    def test_image_size_property(self) -> None:
        self.assertEqual(_sample_payload().image_size, (640, 480))

    def test_empty_crops_roundtrip(self) -> None:
        payload = SemanticPayload(structure={"image_size": [10, 10], "objects": [], "relations": []})
        restored = SemanticPayload.from_bytes(payload.to_bytes())
        self.assertEqual(restored.crops, {})
        self.assertEqual(restored.image_size, (10, 10))

    def test_bad_magic_raises(self) -> None:
        with self.assertRaises(ValueError):
            SemanticPayload.from_bytes(b"XXXX garbage")


if __name__ == "__main__":
    unittest.main()
