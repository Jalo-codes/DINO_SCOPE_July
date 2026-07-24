"""analysis.image_auroc_by_size — detection AUROC broken down by manipulation size.

Image-level AUROC (fake vs real, from the AttentionPool image score) split by the
FAKE's mask-size bucket: for each bucket, {fakes in that bucket} vs {all reals}.
Answers "does the detector catch SMALL manipulations as well as large ones?" —
and, across two checkpoints, whether a diet change moved small-manip detection.

image_score = sigmoid(image_logit) is the full-frame pass-1 score: it does NOT
depend on the localization decoder OR on zoom (both only touch the mask). So any
one decoder's records CSV gives the same AUROC, and a zoom run's AUROC equals the
flat run's. Reals carry no size (area 0) — they are the shared negative pool.

Run:
    python -m analysis.image_auroc_by_size --records results/<run>/probe_zoom/threshold_records.csv --by_source
"""

import argparse
import csv
from collections import defaultdict
from typing import List

import numpy as np

from lab_utils.eval.buckets import BUCKET_LABELS, area_to_bucket


def _auroc(pos: List[float], neg: List[float]) -> float:
    """Rank-based (Mann-Whitney) AUROC, tie-corrected. pos=fake scores, neg=real."""
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    allv = np.asarray(pos + neg, dtype=np.float64)
    uniq, inv, counts = np.unique(allv, return_counts=True, return_inverse=True)
    csum = np.cumsum(counts)
    start = csum - counts
    avg_rank = (start + 1 + csum) / 2.0          # average rank of each unique value
    ranks = avg_rank[inv]
    r_pos = ranks[:n_pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _rows(path: str):
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            try:
                yield (r['source'], int(r['is_real']),
                       float(r['image_score']), float(r['mask_area']))
            except (KeyError, ValueError):
                continue  # NaN image_score (head disabled) or malformed row


def _table(title: str, fakes_by_bucket, reals) -> None:
    print(f'=== {title}  (n_real={len(reals)}) ===')
    print(f'{"bucket":<9}{"n_fake":>8}{"auroc":>10}')
    all_fakes = []
    for b in BUCKET_LABELS:
        fb = fakes_by_bucket.get(b, [])
        all_fakes += fb
        auc = _auroc(fb, reals)
        print(f'{b:<9}{len(fb):>8}{("   -  " if auc != auc else f"{auc:8.4f}"):>10}')
    auc_all = _auroc(all_fakes, reals)
    print(f'{"OVERALL":<9}{len(all_fakes):>8}{("   -  " if auc_all != auc_all else f"{auc_all:8.4f}"):>10}')
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--records', required=True,
                    help='any decoder records CSV (image_score is decoder-independent)')
    ap.add_argument('--by_source', action='store_true',
                    help='also break down per source, with same-source reals as the negatives')
    args = ap.parse_args()

    # pooled
    fakes_by_bucket = defaultdict(list)
    reals: List[float] = []
    # per source
    src_fakes = defaultdict(lambda: defaultdict(list))
    src_reals = defaultdict(list)

    for source, is_real, score, area in _rows(args.records):
        if score != score:  # NaN guard
            continue
        if is_real:
            reals.append(score)
            src_reals[source].append(score)
        else:
            b = area_to_bucket(area)
            fakes_by_bucket[b].append(score)
            src_fakes[source][b].append(score)

    _table('ALL SOURCES POOLED', fakes_by_bucket, reals)

    if args.by_source:
        for source in sorted(src_fakes):
            _table(f'source={source}', src_fakes[source], src_reals.get(source, []))


if __name__ == '__main__':
    main()
