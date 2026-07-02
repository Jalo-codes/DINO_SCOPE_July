"""lab_utils.eval.record — EvalRecord: immutable output of metric().

Produced by metric(), consumed by aggregate(), robustness(), and labs.
All per-image scores are pre-computed at creation time; nothing downstream
recomputes them.
"""

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class EvalRecord:
    """Packaged per-image eval result.

    gt_mask and pred_mask are at PIXEL resolution (GT's native size; the square
    input frame for real items).  All scoring is per-pixel — the patch grid is
    never used for IOU/F1.  Scores are pre-computed in metric(); aggregate()
    operates on records only.
    """
    # provenance
    item_id:     str
    is_real:     bool
    source:      str
    decoder:     str

    # packaged signal (everything downstream needs without re-running model)
    gt_mask:     np.ndarray            # (H, W) bool — GT at native pixel resolution
    pred_mask:   np.ndarray            # (H, W) bool — pred upsampled to pixel res
    attention:   Optional[np.ndarray]  # (N,) pool attention, or None
    image_score: float                 # sigmoid(image_logit), or NaN if disabled

    # per-image scores
    f1:        float
    iou:       float
    precision: float
    recall:    float
    accuracy:  float

    # reporting dims derived in metric() from Item.mask_area() — I5
    mask_area: float    # fraction of pixels manipulated (0.0 for real items)
    bucket:    str      # 'tiny' | 'small' | 'medium' | 'large'

    # optional, GT-free reporting dim — an arbitrary subgroup label the CALLER
    # chooses from Item.meta (e.g. a TGIF (model|type|family) cell).  metric()
    # stores whatever string it is handed; aggregate.by_subgroup() groups on it.
    subgroup:  Optional[str] = None
