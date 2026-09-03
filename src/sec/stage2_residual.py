"""Stage 2 — residual coding on top of Stage 1 (SCAFFOLD, not yet implemented).

Intent (see docs/EXPERIMENT_DESIGN.md): only once Stage 1's generative
prediction V_hat_G = G(D, z_t) fails to meet the target distortion do we
compute and encode a correction:

    r_t = V_t - V_hat_G
    (encode r_t with a conventional codec, e.g. AV1/VVC)

This isolates R_dynamics (Stage 1 alone) from R_dynamics+residual (Stage 2),
which is one of the three key curves in the design doc.

This depends on Stage 1 being real first — see stage1_motion.py.
"""

from __future__ import annotations

import numpy as np


def compute_residual(frame: np.ndarray, predicted_frame: np.ndarray) -> np.ndarray:
    """r_t = V_t - V_hat_G. This part needs no external model — it's just
    a subtraction — but it's only meaningful once predicted_frame comes from
    a real Stage 1 decoder."""
    return frame.astype(np.int16) - predicted_frame.astype(np.int16)


def encode_residual(residual: np.ndarray) -> bytes:
    raise NotImplementedError(
        "Wire this to a real residual codec (e.g. shell out to an AV1/VVC "
        "encoder, or use a learned residual coder) once Stage 1 produces "
        "real predicted frames. Not implemented in this scaffold."
    )
