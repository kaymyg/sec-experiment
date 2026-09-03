# data/

No data is committed to this repo. Put your own local clip + reference
image here to run Stage 0:

```
data/
  my_clip.mp4                     # the real video V
  speaker_room_reference.png      # the still reference image D (f(D))
```

Then:

```bash
python -m sec.cli stage0 \
  --video data/my_clip.mp4 \
  --reference data/speaker_room_reference.png \
  --out experiments/results/stage0_d0.json
```

**Blindness note:** for a legitimate D0 zero-bit measurement, the reference
image should *not* be a frame extracted from the same clip you're evaluating
against — otherwise the "prior" has secretly seen the target. Use a separate
photo of the same speaker/room, or clearly label the result as a sanity
check rather than a real R\*(D0, ε) measurement if you do use a same-clip
frame.
