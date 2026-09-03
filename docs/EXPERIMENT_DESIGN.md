# Experiment Design

## The question

Video compression research usually asks "how few bits can we use?" This
project asks a more diagnostic question first:

> **Where does the information in a talking-head video actually live** —
> in the speaker's identity, the room, the motion dynamics, the exact
> timing, or an irreducible residual?

The compression ratio is a downstream consequence of answering this
correctly. Treating "compress to X kbps" as the goal risks quietly
redefining compression as *performance synthesis from a script* (transcript
+ timing + voice + face reference) rather than reconstruction of a specific
video. This design tries to keep those two things measurably distinct.

## The prior ladder

| Level | Prior `D` | Adds |
|---|---|---|
| D0 | Generic | Nothing speaker/scene-specific — only a generic model of faces/scenes/motion |
| D1 | + Identity | One reference image of the speaker |
| D2 | + Scene | One reference image of the room |
| D2+A | + Audio | The real audio track, used to condition visual prediction |
| D4 | + Transcript | Full text of what was said |
| D5 | + Timing | Word-level timing alignment |

D0/D1/D2 use **identical codec machinery** so the only variable is what the
decoder already knows. D2+A, D4, and D5 are treated as **information-boundary
experiments, not headline compression results** — see the audio caveat below.

For each level we want:

```
R*(D0, ε), R*(D1, ε), R*(D2, ε), ...
ΔR_identity = R*(D0, ε) − R*(D1, ε)
ΔR_scene    = R*(D1, ε) − R*(D2, ε)
```

## The zero-bit test (corrected)

A naive "zero-bit reconstruction" lets the decoder pick whichever sample
`V̂_1 … V̂_N ~ P(V|D)` scores best against the target — but that secretly
turns the evaluator into an encoder (it used `V` to choose).

The legitimate version fixes the reconstruction function **before** looking
at the target:

```
V̂_0 = f(D)          # f is fixed with no access to V
d_0(D) = d(V, f(D))  # evaluated only afterward
```

`Stage 0` in this repo implements exactly this: `f(D)` is the single
reference image, held static for the whole clip, chosen with no knowledge of
the actual video content beyond what's in the reference itself.

## Stage progression

1. **Stage 0 — appearance only.** Zero-bit test above. Tells us whether the
   prior can reproduce *appearance* at all.
2. **Stage 1 — motion-only latent.** Transmit a small per-frame latent
   (head pose, mouth, expression) — no pixel residual. Tells us the cost of
   *dynamics* on top of appearance.
3. **Stage 2 — add residual.** Only once Stage 1's generative prediction is
   insufficient at the target distortion do we invoke a conventional
   residual coder. This isolates `R_dynamics` from `R_dynamics+residual`.

## Rate accounting: η_G

Report actual operational bit costs rather than raw entropy ratios:

```
R_G      = bits required by the generative latent
R_R      = bits required by the residual coder
R_total  = R_G + R_R
η_G      = 1 − R_R / R_total     # "generative explanation fraction"
```

**Caveat:** `η_G` is only meaningful when `R_G` and `R_R` are compared at the
*same fixed distortion ε*. A generative predictor that is cheap but
overconfident can produce a high η_G simply because the distortion metric
failed to notice what it got wrong (background detail, hands, text/logos are
known blind spots for PSNR/SSIM/LPIPS/ArcFace). Any η_G result should be
paired with a held-out human check on exactly the content the residual was
supposed to be explaining, not just the aggregate metric.

## The audio-conditioning caveat

Adding the real audio track (D2+A) can produce a large apparent bitrate drop
because **lip and jaw motion are already largely predictable from phonemes**
— this is a known, separate result (audio-driven face reenactment /
lip-sync models), not a validation of this codec's architecture. A dramatic
D2 → D2+A drop confirms known audio-visual redundancy; it does not by itself
mean the codec found a new efficient representation. D2+A, D4, and D5 results
should always be reported with this caveat attached, and should not be
blended into the headline D0/D1/D2 comparison table.

## Distortion measure

Composite metric (as implemented incrementally — see `src/sec/metrics.py`):

```
d(V, V̂) = λ1·d_identity + λ2·d_lip_sync + λ3·d_temporal + λ4·d_texture
```

Currently implemented: PSNR, SSIM (no external weights needed).
Hooks present but **not wired to pretrained weights** in this environment:
LPIPS (`d_texture`), ArcFace-style embedding distance (`d_identity`),
SyncNet-style alignment (`d_lip_sync`). These need `torch` and downloaded
checkpoints — install locally with GPU + full network access and point
`metrics.py` at them.

## Baseline

Every reported operating point should be compared against a conventional
codec (e.g. H.265/VVC) at the same perceptual quality:

```
R_generative(ε) < R_conventional(ε) ?
```

That comparison — not whether the generative reconstruction "looks nice" —
is the actual result of interest.

## What this repo does not yet claim

- No trained generative predictor is included (Stage 1/2 are scaffolds).
- No human evaluation pipeline is implemented yet.
- No numbers in this repo should be treated as experimental results until
  Stage 0 has actually been run against a real clip + reference image.
