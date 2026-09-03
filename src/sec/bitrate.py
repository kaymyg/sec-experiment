"""Rate accounting helpers: R_G, R_R, eta_G, and encoded-size-in-bits utilities.

See docs/EXPERIMENT_DESIGN.md for the eta_G caveat: it is only meaningful
when R_G and R_R are compared at the same fixed distortion target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def file_size_bits(path: str | Path) -> int:
    """Actual encoded size of a file on disk, in bits.

    Using the real encoded byte count (not raw entropy estimates) keeps the
    rate numbers operational and honest, per the design doc's point about
    reporting R_G/R_R directly rather than H(r)/H(V).
    """
    return Path(path).stat().st_size * 8


def bits_per_second(total_bits: float, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    return total_bits / duration_seconds


@dataclass
class RateBreakdown:
    r_generative_bits: float  # R_G: cost of the generative latent / reference
    r_residual_bits: float = 0.0  # R_R: cost of any residual/correction layer
    duration_seconds: float = 1.0

    @property
    def r_total_bits(self) -> float:
        return self.r_generative_bits + self.r_residual_bits

    @property
    def eta_g(self) -> float:
        """Generative explanation fraction: 1 - R_R / R_total."""
        if self.r_total_bits == 0:
            return 0.0
        return 1.0 - (self.r_residual_bits / self.r_total_bits)

    def bps(self) -> dict:
        return {
            "r_generative_bps": bits_per_second(self.r_generative_bits, self.duration_seconds),
            "r_residual_bps": bits_per_second(self.r_residual_bits, self.duration_seconds),
            "r_total_bps": bits_per_second(self.r_total_bits, self.duration_seconds),
            "eta_g": self.eta_g,
        }
