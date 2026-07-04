"""experiments.configs.zoom — THE attention-zoom operating point.

Single source of truth for the zoom crop parameters. Before this file
existed there were four divergent default sets: eval.py / eval_robustness.py
defaulted to peak/0.08 while attention_zoom_single / predict.py defaulted to
otsu/1.0 — so scored evals and the GT-free predict pipeline silently zoomed
with different thresholds, and per-epoch val_zoom used yet another (the
function defaults). Every consumer now reads DEFAULT_ZOOM:

- experiments/labs/attention_zoom.py   (attention_zoom_single signature)
- experiments/scripts/eval.py          (CLI defaults)
- experiments/scripts/eval_robustness.py (CLI defaults)
- experiments/scripts/predict.py       (function + CLI defaults)
- lab_utils/train/loop.py val_zoom     (inherits via attention_zoom_single;
  cfg.val_zoom_pad_frac/min_area are passed only when explicitly set)

Change the operating point HERE, in one commit — never by drifting one
script's default.

The canonical point is the recall-first 'peak' threshold at 0.08 x max
attention with the 0.06 per-side pad floor — the operating point the scored
eval surfaces were tuned to.
"""

import dataclasses
from typing import Optional, Union


@dataclasses.dataclass(frozen=True)
class ZoomParams:
    """Attention-zoom crop parameters (see attention_zoom_single for docs)."""
    attn_percentile:   Union[float, str] = 'peak'  # 'peak' | 'otsu' | 'gap' | numeric percentile
    attn_thresh_mult:  float = 0.08   # for 'peak': fraction of max attention
    attn_pad_frac:     float = 0.10   # patch-based crop padding
    min_box_size:      int   = 8      # minimum crop size in patches
    attn_min_pad_frac: float = 0.06   # per-side pad floor (0 = legacy)
    pad_side_frac:     Optional[float] = None  # area-based padding; None = patch-based
    min_area_frac:     float = 0.0    # with pad_side_frac: floor crop to this frame frac
    min_crop_frac:     float = 0.25   # bbox >= this frame frac => whole-frame fallback


DEFAULT_ZOOM = ZoomParams()
