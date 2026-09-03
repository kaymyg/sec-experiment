"""Distortion metrics for the SEC experiment.

Design note
-----------
Only PSNR and SSIM are implemented with hard dependencies (numpy +
scikit-image), so Stage 0 runs with no GPU and no downloaded model weights.

LPIPS (texture/perceptual), an ArcFace-style identity embedding distance,
and a SyncNet-style audio-visual alignment score are represented as optional
hooks (`OptionalMetric`) that raise a clear error if used without their
backend installed, rather than silently no-opping or faking a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def psnr(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """PSNR in dB between two HxWxC uint8 (or float [0,1]) frames."""
    data_range = 255 if frame_a.dtype == np.uint8 else 1.0
    return float(_psnr(frame_a, frame_b, data_range=data_range))


def ssim(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """Structural similarity (mean over channels) between two frames."""
    data_range = 255 if frame_a.dtype == np.uint8 else 1.0
    channel_axis = -1 if frame_a.ndim == 3 else None
    return float(_ssim(frame_a, frame_b, data_range=data_range, channel_axis=channel_axis))


class BackendUnavailableError(RuntimeError):
    """Raised when an optional metric is called without its backend installed."""


@dataclass
class OptionalMetric:
    """Wrapper for a metric that needs extra deps/weights not bundled here.

    Calling `.compute(...)` before `.backend` is set raises a clear error
    instead of returning a fabricated number.
    """

    name: str
    install_hint: str
    backend: Optional[Callable[..., float]] = field(default=None, repr=False)

    def compute(self, *args, **kwargs) -> float:
        if self.backend is None:
            raise BackendUnavailableError(
                f"'{self.name}' has no backend configured in this environment.\n"
                f"To use it: {self.install_hint}\n"
                f"Then set sec.metrics.{self.name}.backend = <your callable>."
            )
        return float(self.backend(*args, **kwargs))


# Hooks for the composite metric described in docs/EXPERIMENT_DESIGN.md.
# Wire these up locally once you have torch + the relevant pretrained weights.
lpips_texture = OptionalMetric(
    name="lpips_texture",
    install_hint=(
        "pip install lpips torch, then: "
        "import lpips; net = lpips.LPIPS(net='alex'); "
        "sec.metrics.lpips_texture.backend = lambda a, b: net(a, b).item()"
    ),
)

identity_distance = OptionalMetric(
    name="identity_distance",
    install_hint=(
        "Install an ArcFace-style face-embedding model (e.g. `insightface`), "
        "then wire a callable that returns 1 - cosine_similarity(embed(a), embed(b))."
    ),
)

lip_sync_score = OptionalMetric(
    name="lip_sync_score",
    install_hint=(
        "Install a SyncNet-style audio-visual model and wire a callable that "
        "returns the audio/video offset confidence for a clip."
    ),
)


@dataclass
class DistortionReport:
    """Per-frame distortion values plus simple aggregates."""

    psnr_values: list[float] = field(default_factory=list)
    ssim_values: list[float] = field(default_factory=list)

    def add(self, frame_a: np.ndarray, frame_b: np.ndarray) -> None:
        self.psnr_values.append(psnr(frame_a, frame_b))
        self.ssim_values.append(ssim(frame_a, frame_b))

    def summary(self) -> dict:
        return {
            "n_frames": len(self.psnr_values),
            "psnr_mean": float(np.mean(self.psnr_values)) if self.psnr_values else None,
            "psnr_min": float(np.min(self.psnr_values)) if self.psnr_values else None,
            "ssim_mean": float(np.mean(self.ssim_values)) if self.ssim_values else None,
            "ssim_min": float(np.min(self.ssim_values)) if self.ssim_values else None,
        }
