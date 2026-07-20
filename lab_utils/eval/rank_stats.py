"""lab_utils.eval.rank_stats — shared AUC/percentile helpers for lab report scripts.

Extracted out of analysis/probe_contrasts.py so analysis/
full_fakes_report.py (and any future records-CSV report) doesn't fork its own
copy of the rank-based AUC computation — scripts may not import each other
(C-script invariant, see lab_utils/eval/val_sources.py), so shared logic
lives in lab_utils.

TORCH-FREE — numpy only.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg) + 0.5 * P(equal)."""
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    all_scores = np.concatenate([pos, neg])
    order = all_scores.argsort(kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def stats(vals: Sequence[float]) -> dict:
    """Median-led stat block: {n, median, mean, p25, p75}. NaNs are dropped."""
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if len(a) == 0:
        return {'n': 0, 'median': float('nan'), 'mean': float('nan'),
                'p25': float('nan'), 'p75': float('nan')}
    return {'n': len(a), 'median': float(np.median(a)), 'mean': float(a.mean()),
            'p25': float(np.percentile(a, 25)), 'p75': float(np.percentile(a, 75))}
