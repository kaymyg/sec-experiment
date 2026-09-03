"""End-to-end test: generate a tiny synthetic clip + reference image, run
Stage 0, and check the result shape and sanity of the numbers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from sec.stage0_reference import run_stage0


@pytest.fixture()
def tiny_clip_and_reference(tmp_path: Path):
    h, w = 64, 64
    fps = 10
    n_frames = 15

    # A static "room + speaker" scene: a smooth gradient (stand-in for a
    # photo-like image, not pixel noise) with tiny per-frame jitter, so a
    # static reference should score well but not perfectly.
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack(
        [
            (xx * 255 / w).astype(np.uint8),
            (yy * 255 / h).astype(np.uint8),
            np.full((h, w), 128, dtype=np.uint8),
        ],
        axis=-1,
    )

    video_path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
    for i in range(n_frames):
        jitter = rng.integers(-3, 3, base.shape)
        frame = np.clip(base.astype(int) + jitter, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    reference_path = tmp_path / "reference.png"
    cv2.imwrite(str(reference_path), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))

    return video_path, reference_path, n_frames, fps


def test_stage0_runs_end_to_end(tiny_clip_and_reference):
    video_path, reference_path, n_frames, fps = tiny_clip_and_reference
    result = run_stage0(video_path, reference_path)

    assert result.n_frames == n_frames
    assert result.fps == pytest.approx(fps, rel=0.1)
    assert result.distortion["psnr_mean"] > 15  # sanity: not garbage
    assert 0 <= result.distortion["ssim_mean"] <= 1
    assert result.rate["r_generative_bps"] > 0
    assert result.rate["r_residual_bps"] == 0
    assert result.rate["eta_g"] == pytest.approx(1.0)  # no residual sent
