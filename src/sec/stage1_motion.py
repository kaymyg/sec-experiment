"""Stage 1 — motion-only latent transmission (SCAFFOLD, not yet implemented).

Intent (see docs/EXPERIMENT_DESIGN.md): transmit a small per-frame latent
z_t = {head pose, mouth shape, expression} and let a generative decoder,
conditioned on D and z_t, reconstruct each frame with no pixel residual.
This measures R_dynamics on top of Stage 0's R_appearance.

Why this is a stub here: a real implementation needs (a) a landmark/pose
extractor (e.g. mediapipe, dlib, or a learned face-mesh model) to *encode*
z_t from the real video, and (b) a generative decoder capable of rendering a
face from (D, z_t) to actually reconstruct frames. Both need pretrained
weights this sandbox can't download (no huggingface.co / storage.googleapis.com
in the network allowlist) and (b) needs a GPU to run at any reasonable speed.

To make this real on your own machine:
  1. `pip install mediapipe` (or dlib / a face-mesh model of your choice)
  2. Implement `extract_latent(frame) -> MotionLatent` below using it
  3. Implement a `GenerativeDecoder` (e.g. a first-order-motion-model style
     network, or a diffusion model conditioned on landmarks) and wire it
     into `render_from_latent`
  4. Re-run the same DistortionReport / RateBreakdown machinery from
     stage0_reference.py — it doesn't need to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MotionLatent:
    """Placeholder shape for z_t = {head pose, mouth, expression}."""

    head_pose: np.ndarray  # e.g. 6-DoF (rotation + translation)
    mouth_shape: np.ndarray  # e.g. blendshape coefficients
    expression: np.ndarray  # e.g. remaining facial blendshape coefficients


def extract_latent(frame: np.ndarray) -> MotionLatent:
    raise NotImplementedError(
        "Stage 1 needs a landmark/pose backend (e.g. mediapipe) that isn't "
        "installed in this environment. See the module docstring for how to "
        "wire one up locally."
    )


def render_from_latent(reference: np.ndarray, latent: MotionLatent) -> np.ndarray:
    raise NotImplementedError(
        "Stage 1 needs a generative decoder backend that isn't available in "
        "this environment. See the module docstring for how to wire one up "
        "locally (requires GPU + pretrained weights)."
    )
