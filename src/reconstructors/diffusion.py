"""Diffusion-based reconstructor (flag-gated, GPU-optional).

Generates the background/scene from the scene text via a frozen diffusion model
(``diffusers``), then composites the received crops on top — text regions are
NEVER regenerated (diffusion produces unreadable text), they are always
composited from their preserved crops. Falls back to the compositional
reconstructor when diffusion is disabled, the deps are missing, or no GPU is
available.
"""

from __future__ import annotations

import logging

import numpy as np

from ..appearance.base import AppearanceEncoder
from ..payload import SemanticPayload
from .base import Reconstructor, ReconstructionResult
from .compositional import CompositionalReconstructor
from .text import describe_scene


logger = logging.getLogger(__name__)


def _gpu_available() -> bool:
    """Return True if a CUDA/MPS device is usable."""
    try:
        import torch

        return bool(torch.cuda.is_available()) or bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
    except Exception:  # pragma: no cover - torch always present in this repo
        return False


class DiffusionReconstructor(Reconstructor):
    """Synthesize the background with diffusion, then composite crops."""

    def __init__(
        self,
        model_id: str = "stabilityai/sd-turbo",
        background_color: tuple[int, int, int] = (127, 127, 127),
        rerender_text: bool = True,
        force_cpu_fallback: bool = False,
    ) -> None:
        """Try to load the diffusion pipeline; fall back to compositional."""
        self.model_id = model_id
        self._compositional = CompositionalReconstructor(
            background_color=background_color, rerender_text=rerender_text
        )
        self._pipe = None

        if force_cpu_fallback or not _gpu_available():
            logger.warning(
                "DiffusionReconstructor: no GPU available; using compositional "
                "background instead."
            )
            return

        try:  # pragma: no cover - exercised only with diffusers + GPU
            import torch
            from diffusers import AutoPipelineForText2Image

            device = "cuda" if torch.cuda.is_available() else "mps"
            self._pipe = AutoPipelineForText2Image.from_pretrained(
                model_id, torch_dtype=torch.float16
            ).to(device)
            self._device = device
            logger.info("Loaded diffusion pipeline %s on %s.", model_id, device)
        except Exception as exc:  # pragma: no cover - depends on optional deps
            logger.warning(
                "DiffusionReconstructor: could not load %s (%s); falling back to "
                "compositional background.",
                model_id,
                exc,
            )
            self._pipe = None

    def _generate_background(self, payload: SemanticPayload) -> np.ndarray:
        """Generate a background image from the scene text, or fall back."""
        if self._pipe is None:
            return self._compositional.make_background(payload.image_size)

        # pragma: no cover - requires diffusers + GPU
        prompt = describe_scene(payload.structure)
        width, height = payload.image_size
        result = self._pipe(
            prompt=prompt,
            width=max(8, width - width % 8),
            height=max(8, height - height % 8),
            num_inference_steps=2,
            guidance_scale=0.0,
        )
        from PIL import Image

        generated = result.images[0].resize((max(1, width), max(1, height)), Image.BILINEAR)
        return np.asarray(Image.fromarray(np.asarray(generated)).convert("RGB"))

    def reconstruct(
        self, payload: SemanticPayload, appearance_decoder: AppearanceEncoder
    ) -> ReconstructionResult:
        """Generate the background then composite crops (text never generated)."""
        background = self._generate_background(payload)
        image = self._compositional.composite(background, payload, appearance_decoder)
        text = describe_scene(payload.structure)
        return ReconstructionResult(image=image, text=text)
