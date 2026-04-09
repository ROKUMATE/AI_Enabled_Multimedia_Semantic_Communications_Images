"""OAR encoder module for compact transmission payload generation."""

from __future__ import annotations

import base64
import json
import zlib

from .types import EncodedPacket, OARRepresentation


class OAREncoder:
    """Encode OAR representations into compressed base64 payloads."""

    def __init__(self, compression_level: int = 9) -> None:
        """Set zlib compression level used for payload generation."""
        self.compression_level = compression_level

    def encode(self, oar: OARRepresentation) -> EncodedPacket:
        """Serialize and compress OAR data and return packet plus bit estimate."""
        raw_bytes = json.dumps(oar.to_dict(), separators=(",", ":"), sort_keys=True).encode("utf-8")
        compressed = zlib.compress(raw_bytes, level=self.compression_level)
        payload = base64.b64encode(compressed).decode("utf-8")
        bit_estimate = len(compressed) * 8
        return EncodedPacket(payload=payload, bit_estimate=bit_estimate)
