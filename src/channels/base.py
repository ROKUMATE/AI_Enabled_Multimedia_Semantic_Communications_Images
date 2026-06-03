"""Base interface for transmission channels.

The channel sits between the transmitter's :class:`~src.payload.SemanticPayload`
and the receiver. v1 ships only :class:`~src.channels.identity.IdentityChannel`
(pass-through), but the base class is shaped so future degrading channels
(AWGN, Rayleigh) with *per-stream unequal error protection* slot in without
changing any caller.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..payload import SemanticPayload
from ..types import Stream


class Channel(ABC):
    """Transport a payload from transmitter to receiver.

    Subclasses corrupt the payload's streams independently. The default
    :meth:`transmit` decomposes the payload into its priority-tagged streams,
    runs each through :meth:`degrade_stream`, and reassembles the payload — so a
    future channel only needs to override :meth:`degrade_stream` to apply
    unequal protection based on ``stream.priority`` and a signal-strength /
    SNR setting.
    """

    def transmit(self, payload: SemanticPayload) -> SemanticPayload:
        """Send a payload through the channel and return what was received."""
        degraded = [self.degrade_stream(stream) for stream in payload.streams()]
        return SemanticPayload.from_streams(degraded)

    @abstractmethod
    def degrade_stream(self, stream: Stream) -> Stream:
        """Apply this channel's effect to a single stream.

        Implementations may use ``stream.priority`` to apply unequal error
        protection (lower priority value = stronger protection).
        """
        raise NotImplementedError
