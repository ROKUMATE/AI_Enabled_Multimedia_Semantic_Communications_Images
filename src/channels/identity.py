"""Pass-through channel (the only channel in v1)."""

from __future__ import annotations

import logging

from ..payload import SemanticPayload
from ..types import Stream
from .base import Channel


logger = logging.getLogger(__name__)


class IdentityChannel(Channel):
    """Lossless pass-through channel.

    Streams are returned unchanged, but they still round-trip through the
    payload's binary (de)serialization so the on-wire format is exercised
    end-to-end today.

    TODO(P5): add ``AWGNChannel`` / ``RayleighChannel`` subclasses that corrupt
    each stream in :meth:`degrade_stream` according to ``stream.priority`` and a
    configurable SNR / signal strength (unequal error protection). No caller
    changes will be needed — only new ``Channel`` subclasses.
    """

    def transmit(self, payload: SemanticPayload) -> SemanticPayload:
        """Round-trip the payload through serialization without degradation."""
        received = SemanticPayload.from_bytes(payload.to_bytes())
        logger.info(
            "IdentityChannel transmitted %d byte(s) across %d stream(s).",
            payload.size_report()["total_bytes"],
            len(payload.streams()),
        )
        return received

    def degrade_stream(self, stream: Stream) -> Stream:
        """Return the stream unchanged (pass-through)."""
        return stream
