import numpy as np
import pytest

from sec.metrics import psnr, ssim, BackendUnavailableError, lpips_texture
from sec.bitrate import RateBreakdown, bits_per_second


def test_psnr_identical_frames_is_inf_or_very_high():
    frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    value = psnr(frame, frame)
    assert value == float("inf") or value > 60


def test_psnr_decreases_with_noise():
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    noisy = np.clip(frame.astype(int) + rng.integers(-40, 40, frame.shape), 0, 255).astype(
        np.uint8
    )
    identical_psnr = psnr(frame, frame)
    noisy_psnr = psnr(frame, noisy)
    assert noisy_psnr < identical_psnr


def test_ssim_identical_frames_is_one():
    frame = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    assert ssim(frame, frame) == pytest.approx(1.0, abs=1e-6)


def test_optional_metric_raises_without_backend():
    with pytest.raises(BackendUnavailableError):
        lpips_texture.compute(None, None)


def test_bits_per_second():
    assert bits_per_second(8000, 10) == 800


def test_rate_breakdown_eta_g():
    rb = RateBreakdown(r_generative_bits=8000, r_residual_bits=2000, duration_seconds=10)
    bps = rb.bps()
    assert bps["r_total_bps"] == pytest.approx(1000.0)
    assert bps["eta_g"] == pytest.approx(0.8)


def test_rate_breakdown_zero_residual_gives_eta_g_one():
    rb = RateBreakdown(r_generative_bits=500, r_residual_bits=0, duration_seconds=5)
    assert rb.eta_g == pytest.approx(1.0)
