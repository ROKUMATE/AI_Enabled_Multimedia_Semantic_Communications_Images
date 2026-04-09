"""Decoder for recovering OAR data from encoded packets."""

from __future__ import annotations

import base64
import json
import zlib

from .types import EncodedPacket, OARRepresentation


class OARDecoder:
    """Decode compressed packets back into OAR structures."""

    def decode(self, packet: EncodedPacket) -> OARRepresentation:
        """Decode packet payload and reconstruct OAR representation."""
        compressed = base64.b64decode(packet.payload.encode("utf-8"))
        raw = zlib.decompress(compressed)
        data = json.loads(raw.decode("utf-8"))
        return OARRepresentation.from_dict(data)
