"""lab_utils.eval.hide — MIL patch-hiding mask construction (torch-free).

Builds the boolean *hide mask* that `fetch.repool_hidden` feeds to the MIL pool
(True = hide this patch from the pool readout).  This is deliberately separate
from:
  - zoom.py        — bbox geometry / crop / place-back (what the *zoom* sees),
  - fetch.py       — the model-touching re-pool itself (I2),
so the "which patches count as region 1, and how much margin around them" policy
lives in one place.  The deferred backbone hides (bool_masked_pos / attn-bias)
will consume the same hide mask, so they get a home here too.

Why this is not just the bbox: the zoom crop is a *padded* box, but the hide must
suppress the attention's actual hot patches — if we only hid the tight hot set,
the renormalised re-pool keeps leaning on that same blob's immediate fringe.  So
the hide is built a touch more generously than the bbox (lower threshold) and
dilated by a ring of 8-connected neighbours.

All bbox/threshold primitives are reused from zoom.py; this module only adds the
hide-specific composition (lower threshold + neighbour dilation).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from lab_utils.eval.zoom import attention_hot_mask, peak_hot_component

# Hide a little more eagerly than the bbox does.  attention_to_bbox thresholds at
# otsu*0.70; the hide uses a lower multiplier so the "positive" set that gets
# suppressed is slightly broader (the re-pool should look genuinely elsewhere).
HIDE_THRESH_MULT: float = 0.55


def dilate_mask_8(mask: np.ndarray, iters: int = 1) -> np.ndarray:
    """Dilate a boolean grid mask by `iters` rings of 8-connected neighbours.

    Pure-numpy 3x3 dilation (includes diagonals).  Used to add a margin around
    the hidden patches so the MIL re-pool can't immediately re-attend to the
    blob's fringe.  iters<=0 returns a copy unchanged.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f'dilate_mask_8 expects a 2-D mask, got shape {m.shape}')
    for _ in range(max(0, int(iters))):
        out = m.copy()
        out[:-1, :] |= m[1:, :]      # N
        out[1:, :]  |= m[:-1, :]     # S
        out[:, :-1] |= m[:, 1:]      # W
        out[:, 1:]  |= m[:, :-1]     # E
        out[:-1, :-1] |= m[1:, 1:]   # NW
        out[:-1, 1:]  |= m[1:, :-1]  # NE
        out[1:, :-1]  |= m[:-1, 1:]  # SW
        out[1:, 1:]   |= m[:-1, :-1] # SE
        m = out
    return m


def build_hide_mask(
    attention: np.ndarray,
    grid_hw: Tuple[int, int],
    *,
    mode: str = 'component',
    percentile: float | str = 'otsu',
    thresh_mult: float = HIDE_THRESH_MULT,
    dilate: int = 1,
) -> np.ndarray:
    """Flat (N,) boolean hide mask for MIL patch hiding (True = hide).

    Args:
        attention:   (N,) per-patch attention from pass 1.
        grid_hw:     (n_rows, n_cols) patch grid.
        mode:        'component' — only the peak's connected hot blob (hide
                     region 1, leave other real regions visible for the re-pool);
                     'hot' — the whole thresholded hot set (redirect entirely).
        percentile:  threshold method passed to the hot-mask primitives.
        thresh_mult: otsu/gap multiplier for the 'positive' set.  Lower than the
                     bbox's 0.70 so the hide is a touch broader (HIDE_THRESH_MULT).
        dilate:      rings of 8-connected neighbours to also hide (padding).

    Returns:
        Flat (N,) bool array ready for `fetch.repool_hidden`.
    """
    if mode == 'hot':
        hot = attention_hot_mask(attention, grid_hw, percentile=percentile,
                                 thresh_mult=thresh_mult)
    elif mode == 'component':
        hot = peak_hot_component(attention, grid_hw, percentile=percentile,
                                 thresh_mult=thresh_mult)
    else:
        raise ValueError(f"build_hide_mask: unknown mode {mode!r} (component|hot)")

    if dilate > 0:
        hot = dilate_mask_8(hot, iters=dilate)
    return hot.reshape(-1)
