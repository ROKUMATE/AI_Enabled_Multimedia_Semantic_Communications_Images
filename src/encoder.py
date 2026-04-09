"""OAR encoder module for compact transmission payload generation."""

from __future__ import annotations

import logging
from collections.abc import Mapping

from .semantic_codec import encode_oar_compact
from .types import EncodedPacket, OARRepresentation


logger = logging.getLogger(__name__)


class OAREncoder:
    """Encode OAR representations into compact semantic token payloads."""

    def encode_semantic(self, oar: OARRepresentation | Mapping[str, object]) -> tuple[str, int]:
        """Return the compact semantic payload and estimated bit size."""
        if isinstance(oar, OARRepresentation):
            data = oar.to_dict()
        else:
            data = dict(oar)

        encoded_data, bit_size = encode_oar_compact(data)
        logger.debug("Encoded semantic payload with %d tokens and %d bits.", encoded_data.count("|") + int(bool(encoded_data)), bit_size)
        return encoded_data, bit_size

    def encode(self, oar: OARRepresentation | Mapping[str, object]) -> EncodedPacket:
        """Serialize OAR data into a compact semantic packet and return size metadata."""
        encoded_data, bit_size = self.encode_semantic(oar)
        semantic_size_bytes = len(encoded_data.encode("utf-8"))
        return EncodedPacket(
            payload=encoded_data,
            bit_estimate=bit_size,
            encoding="semantic-token",
            semantic_size_bytes=semantic_size_bytes,
        )
