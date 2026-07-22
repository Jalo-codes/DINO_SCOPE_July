"""analysis.audit_patch_budget — pre-run gate for equal-budget patch BCE (D6).

Model-free: computes each FAKE item's BANDED fake-patch count (kp) exactly
the way lab_utils.model.losses.bce.equal_budget_patch_bce_loss and
lab_utils.eval.patch_scores.collect_patch_scores will see it, and reports its
distribution overall and per area bucket (lab_utils.eval.buckets). The point
is to pick --patch_k_min from data instead of vibes, and to catch — BEFORE
spending GPU time — a k_min that mutes the exact stratum an experiment means
to measure (small splices).

Audits the UNION of a source's train + val split: kp distribution is a
property of the corpus's items, not of the train/val partition. This script
does NOT check train/eval leakage or disjointness — that is a separate,
existing concern (analysis.audit_openfake_split_overlap for full_fakes;
tgif2's held-out coco_id split is enforced elsewhere). Real items and items
with meta['gt_mask_reliable'] is False are skipped (no meaningful kp).

Usage:
    python -m analysis.audit_patch_budget --tgif2_root /data/tgif2 \\
        --tgif_types sp --band 0.2 0.8 --k_min 4

GATE (see CLAUDE.md-style hard rule in docs/equal_budget_bce_spec.md C0): if
more than 20% of the 'small'-bucket items have kp below --k_min, k_min is
muting the exact stratum the experiment measures — do not train, fix k_min
or the band first.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from lab_utils.data.datasets.registry import REGISTRY
from lab_utils.data.resolution import Resolution
from lab_utils.eval.buckets import BUCKET_LABELS, area_to_bucket
from lab_utils.eval.val_sources import SOURCE_ROOT_ARGS, add_source_root_args
from lab_utils.logging.text import log_line


def _percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {k: float('nan') for k in ('min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max')}
    arr = np.asarray(values, dtype=np.float64)
    qs = np.quantile(arr, [0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0])
    return dict(zip(('min', 'p10', 'p25', 'p50', 'p75', 'p90', 'max'), (float(q) for q in qs)))


def _fmt_pct(d: Dict[str, float]) -> str:
    return ' '.join(f'{k}={v:.1f}' for k, v in d.items())


def collect_kp(
    root: Path, source: str, *, res: Resolution, tgif_types: Optional[set],
    band: tuple, max_items: Optional[int],
) -> List[Dict]:
    """[{item_id, bucket, kp, mask_area}, ...] for every FAKE, reliable-mask item."""
    import torch  # noqa: F401 (mask_to_patch_labels_soft is torch-based)
    from PIL import Image

    from lab_utils.data.resolution import mask_to_patch_labels_soft

    kw = {}
    if source == 'tgif2':
        kw['types'] = tgif_types
    train_ds, val_ds = REGISTRY[source](root, res=res, **kw)
    items = list(train_ds.items) + list(val_ds.items)
    if max_items:
        items = items[:max_items]

    low, high = band
    rows = []
    n_real = n_unreliable = n_failed = 0
    for item in items:
        if item.is_real:
            n_real += 1
            continue
        if item.meta.get('gt_mask_reliable') is False:
            n_unreliable += 1
            continue
        try:
            mask_pil = (
                Image.open(item.mask).convert('L')
                .resize((res.image_size, res.image_size), Image.NEAREST)
            )
            labels_t, weights_t = mask_to_patch_labels_soft(mask_pil, res, low=low, high=high)
            kp = float((labels_t.float() * weights_t).sum().item())
        except Exception as exc:
            log_line(f'[patch-budget] WARN failed on {item.item_id}: {exc}')
            n_failed += 1
            continue
        rows.append({
            'item_id': item.item_id,
            'bucket': area_to_bucket(item.mask_area(res)),
            'kp': kp,
        })
    log_line(f'[patch-budget] {source}: {len(rows)} fake items scored '
             f'(skipped real={n_real} unreliable_mask={n_unreliable} failed={n_failed})')
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog='audit_patch_budget',
        description="Audit a splice/inpaint source's banded fake-patch-count (kp) "
                    "distribution before running an equal-budget patch-BCE experiment.",
    )
    g = p.add_argument_group('datasets')
    add_source_root_args(g)
    g.add_argument('--sources', nargs='*', default=None,
                   help='Restrict to these source names (default: every configured root).')
    g.add_argument('--tgif_types', nargs='+', default=None, choices=['sp', 'fr'],
                   help='Restrict TGIF manipulation types (tgif2 only).')
    g.add_argument('--max_items', type=int, default=None, help='Cap items per source (smoke test).')

    g2 = p.add_argument_group('band / k_min')
    g2.add_argument('--image_size', type=int, default=448)
    g2.add_argument('--patch_size', type=int, default=16)
    g2.add_argument('--band', type=float, nargs=2, default=(0.2, 0.8), metavar=('LOW', 'HIGH'))
    g2.add_argument('--k_min', type=float, default=4.0)
    args = p.parse_args(argv)

    if not (0.0 < args.band[0] < args.band[1] <= 1.0):
        p.error(f'--band needs 0 < LOW < HIGH <= 1, got {args.band}')

    res = Resolution(image_size=args.image_size, patch_size=args.patch_size)
    restrict = set(args.sources) if args.sources else None

    any_source = False
    gate_failed = False
    for source, attr in SOURCE_ROOT_ARGS.items():
        if restrict and source not in restrict:
            continue
        root_str = getattr(args, attr, None)
        if not root_str:
            continue
        root = Path(root_str)
        if not root.is_dir():
            log_line(f'[patch-budget] WARN: root not found for {source}: {root}')
            continue
        any_source = True

        rows = collect_kp(
            root, source, res=res,
            tgif_types=set(args.tgif_types) if args.tgif_types else None,
            band=tuple(args.band), max_items=args.max_items,
        )
        if not rows:
            log_line(f'[patch-budget] {source}: no fake items to audit — skipping report')
            continue

        all_kp = [r['kp'] for r in rows]
        log_line(f'[patch-budget] {source} — kp overall (n={len(all_kp)}): {_fmt_pct(_percentiles(all_kp))}')
        n_zero = sum(1 for k in all_kp if k <= 0.0)
        n_below = sum(1 for k in all_kp if k < args.k_min)
        log_line(f'[patch-budget] {source} — kp==0 (fully banded out): {n_zero}/{len(all_kp)} '
                 f'({100.0*n_zero/len(all_kp):.1f}%); kp<k_min={args.k_min}: {n_below}/{len(all_kp)} '
                 f'({100.0*n_below/len(all_kp):.1f}%)')

        for b in BUCKET_LABELS:
            b_kp = [r['kp'] for r in rows if r['bucket'] == b]
            if not b_kp:
                log_line(f'[patch-budget] {source} — bucket={b}: n=0')
                continue
            n_b_below = sum(1 for k in b_kp if k < args.k_min)
            pct_below = 100.0 * n_b_below / len(b_kp)
            log_line(f'[patch-budget] {source} — bucket={b} (n={len(b_kp)}): {_fmt_pct(_percentiles(b_kp))} '
                     f'| kp<k_min: {n_b_below}/{len(b_kp)} ({pct_below:.1f}%)')
            if b == 'small' and pct_below > 20.0:
                gate_failed = True
                log_line(f'[patch-budget] {source} !! GATE FAILED: {pct_below:.1f}% of bucket=small items '
                         f'have kp < k_min={args.k_min} (>20% threshold) — k_min is muting the '
                         f'exact stratum this experiment measures. Lower k_min, widen the band, '
                         f'or pick a different corpus. Do NOT train yet.')

    if not any_source:
        p.error('no dataset roots configured/found — pass at least one --<source>_root')

    if gate_failed:
        log_line('[patch-budget] GATE: FAILED — see above. Do not proceed to training.')
        return 1
    log_line('[patch-budget] GATE: passed for all sources/buckets checked.')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
