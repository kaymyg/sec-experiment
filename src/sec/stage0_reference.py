"""Stage 0 — static reference reconstruction (the zero-bit / D0-D2 test).

This implements the *corrected* zero-bit test from docs/EXPERIMENT_DESIGN.md:

    V_hat_0 = f(D)          # f is fixed with NO access to V
    d_0(D)  = d(V, f(D))    # evaluated only afterward

Concretely: `D` is one still reference image (e.g. a photo of the speaker in
the room). `f(D)` is that image, held static for the whole clip — nothing
about the actual video content is used to pick or adjust it. We then measure
how far that static image is from every real frame of the clip.

This measures whether the prior can reproduce *appearance* at all, and gives
the R*(D, ε) point for zero transmitted motion/residual bits (plus the
one-time cost of sending the reference image itself).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import video_io
from .bitrate import RateBreakdown, file_size_bits
from .metrics import DistortionReport


@dataclass
class Stage0Result:
    reference_path: str
    video_path: str
    n_frames: int
    fps: float
    duration_seconds: float
    distortion: dict
    rate: dict

    def to_dict(self) -> dict:
        return {
            "stage": "stage0_reference_reconstruction",
            "reference_path": self.reference_path,
            "video_path": self.video_path,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "duration_seconds": self.duration_seconds,
            "distortion": self.distortion,
            "rate": self.rate,
        }


def run_stage0(
    video_path: str | Path,
    reference_path: str | Path,
    max_frames: int | None = None,
) -> Stage0Result:
    """Run the Stage 0 zero-bit test.

    Parameters
    ----------
    video_path: the real clip V.
    reference_path: the still image used as D (e.g. speaker+room reference).
        IMPORTANT: this should be an image not sampled from the clip being
        evaluated, or the test silently becomes non-blind. Using a frame
        from the same clip is fine for a sanity check but should be labeled
        as such, not reported as R*(D, epsilon).
    max_frames: optional cap, useful for quick local sanity checks.
    """
    video_path = Path(video_path)
    reference_path = Path(reference_path)

    fps = video_io.video_fps(video_path)
    reference = video_io.load_image(reference_path)

    report = DistortionReport()
    n = 0
    target_hw = None
    for frame in video_io.iter_frames(video_path, max_frames=max_frames):
        if target_hw is None:
            target_hw = frame.shape[:2]
            f_d = video_io.resize_to_match(reference, target_hw)  # f(D), fixed once
        report.add(frame, f_d)
        n += 1

    if n == 0:
        raise ValueError(f"No frames read from {video_path}")

    duration_seconds = n / fps

    # R_G here is the one-time cost of transmitting the reference image,
    # amortized over the clip duration. R_R is zero: no residual is sent.
    r_generative_bits = file_size_bits(reference_path)
    rate = RateBreakdown(
        r_generative_bits=r_generative_bits,
        r_residual_bits=0.0,
        duration_seconds=duration_seconds,
    )

    return Stage0Result(
        reference_path=str(reference_path),
        video_path=str(video_path),
        n_frames=n,
        fps=fps,
        duration_seconds=duration_seconds,
        distortion=report.summary(),
        rate=rate.bps(),
    )
