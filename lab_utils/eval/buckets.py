"""lab_utils.eval.buckets — splice area fraction → reporting bucket (eval-only, I5).

Ported from legacy/lab_utils/data/area_tiers.py.

Buckets are an EVAL REPORTING concept only — never used at training time and
never stored on Items.  metric() calls area_to_bucket() to label EvalRecords.
Datasets own mask area via Item.mask_area().
"""

import math
from typing import Tuple


BUCKET_LABELS: Tuple[str, ...] = ('tiny', 'small', 'medium', 'large')
BUCKET_EDGES:  Tuple[float, float, float] = (0.05, 0.15, 0.30)


def area_to_bucket(area_frac: float) -> str:
    """Map a splice mask-area fraction to a reporting bucket label.

    Thresholds: area ≤ 0.05 → 'tiny', ≤ 0.15 → 'small', ≤ 0.30 → 'medium',
    else 'large'.  Real items (area=0.0) land in 'tiny'; the aggregate layer
    gates on is_real separately before breaking down by bucket.
    """
    try:
        a = float(area_frac)
    except (TypeError, ValueError):
        a = 0.0
    if not math.isfinite(a) or a <= BUCKET_EDGES[0]:
        return 'tiny'
    if a < BUCKET_EDGES[1]:
        return 'small'
    if a < BUCKET_EDGES[2]:
        return 'medium'
    return 'large'
