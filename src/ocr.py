"""Optional OCR backend abstraction.

Text regions must be transmitted as ``preserve`` objects and may be re-rendered
crisply on the receiver, so we need to read text from crops. OCR libraries
(``easyocr``, ``pytesseract``) are optional and may be absent on the CPU-light
target environment; this module resolves whichever backend is available and
degrades gracefully (returning ``None``) when none is installed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np


logger = logging.getLogger(__name__)


class OcrBackend(ABC):
    """Read text from an RGB image array."""

    name: str = "base"

    @abstractmethod
    def read(self, image_rgb: np.ndarray) -> str:
        """Return concatenated recognized text (empty string if none)."""
        raise NotImplementedError


class EasyOcrBackend(OcrBackend):
    """OCR via the ``easyocr`` package (lazy model load)."""

    name = "easyocr"

    def __init__(self) -> None:
        import easyocr  # noqa: F401  (import guarded by caller)

        self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, image_rgb: np.ndarray) -> str:
        results = self._reader.readtext(image_rgb, detail=0)
        return " ".join(str(item) for item in results).strip()


class PytesseractBackend(OcrBackend):
    """OCR via the ``pytesseract`` package."""

    name = "pytesseract"

    def __init__(self) -> None:
        import pytesseract

        self._pytesseract = pytesseract
        # Probe the tesseract binary early so we fail during construction.
        self._pytesseract.get_tesseract_version()

    def read(self, image_rgb: np.ndarray) -> str:
        from PIL import Image

        return str(self._pytesseract.image_to_string(Image.fromarray(image_rgb))).strip()


def get_ocr_backend(name: str = "auto") -> OcrBackend | None:
    """Return an available OCR backend, or ``None`` if none can be loaded.

    ``name`` is ``auto`` (try easyocr then pytesseract), ``easyocr``,
    ``pytesseract``, or ``none``.
    """
    name = (name or "auto").lower()
    if name == "none":
        return None

    candidates: list[type[OcrBackend]]
    if name == "easyocr":
        candidates = [EasyOcrBackend]
    elif name == "pytesseract":
        candidates = [PytesseractBackend]
    else:  # auto
        candidates = [EasyOcrBackend, PytesseractBackend]

    for backend_cls in candidates:
        try:
            backend = backend_cls()
            logger.info("Using OCR backend: %s", backend.name)
            return backend
        except Exception as exc:  # pragma: no cover - depends on optional deps
            logger.debug("OCR backend %s unavailable: %s", backend_cls.__name__, exc)

    logger.warning(
        "No OCR backend available (install easyocr or pytesseract). Text-region "
        "detection is disabled; mode classification falls back to forced classes."
    )
    return None
