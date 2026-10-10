# SEC Experiment — Where Does Video's Information Actually Live?

[![CI](https://github.com/kaymyg/sec-experiment/actions/workflows/ci.yml/badge.svg)](https://github.com/kaymyg/sec-experiment/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kaymyg/sec-experiment/actions/workflows/codeql.yml/badge.svg)](https://github.com/kaymyg/sec-experiment/actions/workflows/codeql.yml)

This repo implements the **D0 → D1 → D2 prior ladder** experiment for measuring
how much of a talking-head video's bitrate can be explained by decoder-side
priors (generic model, speaker identity, scene/room) before any pixel data is
transmitted at all.

It does **not** claim to be a working "1MB / 2-hour video codec." It is a
measurement harness for the underlying scientific question:

> Given a fixed perceptual distortion budget ε, how does the minimum required
> rate R\*(D, ε) change as the decoder's prior D grows from "nothing" to
> "knows the speaker" to "knows the speaker and the room"?

See [`docs/EXPERIMENT_DESIGN.md`](docs/EXPERIMENT_DESIGN.md) for the full
methodology, including the corrected zero-bit test, the audio-conditioning
caveat, and the generative-explanation-fraction (η_G) metric.

## What's actually implemented right now

| Stage | Status | Notes |
|---|---|---|
| **Stage 0** — static reference reconstruction (zero-bit / D0-D2 test) | ✅ Working | Pure classical CV, no GPU/pretrained weights required. This is `f(D)` committed before seeing `V`. |
| **Stage 1** — motion-only latent (head pose / landmarks) | 🚧 Scaffolded | Interface defined in `src/sec/stage1_motion.py`; needs a landmark backend (e.g. mediapipe/dlib) installed and network access to fetch its model weights, which this sandbox doesn't have. Runs as a stub that raises `NotImplementedError` with instructions. |
| **Stage 2** — residual coding on top of Stage 1 | 🚧 Scaffolded | Depends on Stage 1. Interface only. |
| Rate accounting (R_G, R_R, η_G) | ✅ Working | `src/sec/bitrate.py` |
| Distortion metrics (PSNR, SSIM) | ✅ Working | `src/sec/metrics.py`. LPIPS/ArcFace/SyncNet hooks are present but optional (require `torch` + pretrained weights — not installed here). |

**Why Stage 0 is fully real and Stage 1/2 are not:** Stage 0 only needs a
still reference image and classical image metrics — no pretrained generative
model. Stage 1/2 need real generative/landmark models, which need GPU and
weight downloads this sandbox can't reach. The code is structured so you can
drop in real backends without touching the experiment/metrics/rate-accounting
logic.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Extract frames from a clip and run the Stage 0 (D0) zero-bit test
python -m sec.cli stage0 \
  --video data/my_clip.mp4 \
  --reference data/speaker_room_reference.png \
  --out experiments/results/stage0_d0.json
```

Output is a JSON report with per-frame and aggregate PSNR/SSIM, the one-time
bit cost of transmitting the reference image, and the resulting bits/sec
figure — i.e. one point on the R\*(D, ε) curve.

## Project layout

```
src/sec/
  metrics.py          # distortion metrics: PSNR, SSIM, (optional) LPIPS hook
  video_io.py          # frame extraction (ffmpeg/opencv)
  stage0_reference.py  # zero-bit test: f(D) fixed before seeing V
  stage1_motion.py     # scaffold: landmark/pose latent transmission
  stage2_residual.py   # scaffold: residual coder on top of Stage 1
  bitrate.py           # R_G / R_R / eta_G accounting
  cli.py               # `python -m sec.cli ...` entry points
docs/EXPERIMENT_DESIGN.md  # full methodology write-up
experiments/configs/        # example run configs
tests/                       # unit tests for metrics + bitrate math
```

## Status / honesty note

This is a research scaffold, not a paper's worth of results. Nothing here
has been run against a real talking-head dataset yet — that's the next step
once you plug in a video clip. See the "What's actually implemented" table
above before citing any numbers from this repo.
