"""experiments.scripts.eval_threshold_sweep — threshold-decoder calibration sweep.

Model-free, GPU-free: replays decode_threshold(t) for a grid of thresholds over
a frozen ModelInfo cache (built once with eval.py --cache_dir), scoring each
(item, t) through the canonical metric() path. Produces the full F1/IoU-vs-
threshold curve per probe condition, and the ORACLE-BEST row per condition.

WHY: the BCE↔contrastive localization comparison is decoder-confounded — the
k-means decoder re-clusters per crop (self-calibrating on OOD inputs) while the
threshold decoder commits to t=0.5. The sweep separates "the BCE features don't
rank fake patches above real ones" from "the fixed threshold is miscalibrated
OOD": if contrastive's fixed k-means number beats even BCE's best-t envelope,
the gap is features; if the envelope catches up, it was calibration.

FAIRNESS: the per-condition best-t is an ORACLE quantity (picking t by test F1
leaks GT) — report it as an upper envelope next to the production t=0.5 number,
never as the headline. This mirrors eval_oracle.py's isolation convention.

Usage (probe sweep over a cache built by eval.py)::

    $PY -m experiments.scripts.eval ... --cache_dir $CACHE      # once, GPU
    $PY -m experiments.scripts.eval_threshold_sweep \
        --cache_dir $CACHE \
        --ai_interior_root $SAGID --ai_boundary_root $SAGID \
        --real_crop_root $SAGID \
        --sp_interior_root $IMD --sp_boundary_root $IMD \
        --fr_bg_matched_root $TGIF2 \
        --ai_interior_tgif_root $TGIF2 --ai_boundary_tgif_root $TGIF2 \
        --real_crop_tgif_root $TGIF2 \
        --out_dir results/<cond>/threshold_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from lab_utils.errors import DataError
from lab_utils.eval.cache import load_cache
from lab_utils.eval.decode.threshold import decode_threshold
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.val_sources import add_source_root_args, collect_val_items_by_source
from lab_utils.logging.text import install_log, log_line


# Default grid: dense in the calibrated middle, with tails — the OOD-collapse
# hypothesis predicts the sp_boundary optimum for inpaint-trained BCE sits far
# from 0.5, so the tails matter.
DEFAULT_THRESHOLDS = (
    [0.01, 0.02, 0.05]
    + [round(0.10 + 0.05 * i, 2) for i in range(17)]   # 0.10 … 0.90
    + [0.95, 0.98, 0.99]
)

_CSV_COLS = ['item_id', 'source', 'is_real', 'threshold',
             'f1', 'iou', 'precision', 'recall', 'accuracy',
             'image_score', 'mask_area']


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='eval_threshold_sweep',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--cache_dir', required=True,
                   help='ModelInfo cache written by eval.py --cache_dir (one '
                        'checkpoint = one cache; never mix conditions).')
    p.add_argument('--out_dir', required=True,
                   help='Directory for sweep_records.csv, sweep_summary.json, sweep.log')
    p.add_argument('--thresholds', type=float, nargs='+', default=None,
                   help=f'Threshold grid (default {len(DEFAULT_THRESHOLDS)} values, '
                        '0.01–0.99, dense in 0.10–0.90)')
    g = p.add_argument_group('dataset roots (same flags as eval.py)')
    add_source_root_args(g)
    g = p.add_argument_group('eval control')
    g.add_argument('--max_items', type=int, default=None)
    g.add_argument('--sources', nargs='*', default=None,
                   help='Restrict to these source names (default: all configured)')
    return p


def main() -> None:
    args = _build_parser().parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    install_log(str(out_dir / 'sweep.log'))

    thresholds = sorted(args.thresholds or DEFAULT_THRESHOLDS)
    cache_dir = Path(args.cache_dir)
    index_path = cache_dir / 'index.json'
    if not index_path.exists():
        raise FileNotFoundError(f'cache index not found: {index_path}')

    # Resolution comes from the cache itself — items must be built with the
    # same res the cached forward pass used (window floors depend on it).
    with open(index_path) as f:
        index_ids = json.load(f)
    if not index_ids:
        raise RuntimeError(f'cache index is empty: {index_path}')
    probe_info = load_cache(cache_dir, item_ids=index_ids[:1])
    if not probe_info:
        raise RuntimeError(f'cache index lists items but none loadable: {cache_dir}')
    res = next(iter(probe_info.values())).res
    log_line(f'[sweep] cache={cache_dir} ({len(index_ids)} items), res={res.image_size}/{res.patch_size}')
    log_line(f'[sweep] thresholds ({len(thresholds)}): {thresholds}')

    val_items_by_source = collect_val_items_by_source(args, res, log_tag='[sweep]')
    if not val_items_by_source:
        raise RuntimeError('eval_threshold_sweep: no dataset roots configured')
    all_items = [it for items in val_items_by_source.values() for it in items]

    # source → threshold → list of f1 (fakes only, the localization stratum)
    f1_by: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    iou_by: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))

    n_missing = 0
    n_done = 0
    with open(out_dir / 'sweep_records.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_COLS)
        for item in all_items:
            cached = load_cache(cache_dir, item_ids=[item.item_id])
            info = cached.get(item.item_id)
            if info is None:
                n_missing += 1
                continue
            if info.patch_logits is None:
                raise RuntimeError(
                    'eval_threshold_sweep: cached ModelInfo has no patch_logits '
                    f'(item {item.item_id}) — this cache is from a checkpoint '
                    'without the patch-BCE head; the sweep only applies to bce_* '
                    'conditions.'
                )
            try:
                for t in thresholds:
                    patch_mask = decode_threshold(info, t=t)
                    rec = eval_metric(patch_mask, info, item, decoder='threshold')
                    writer.writerow([
                        rec.item_id, item.source, int(rec.is_real), f'{t:.2f}',
                        f'{rec.f1:.6f}', f'{rec.iou:.6f}', f'{rec.precision:.6f}',
                        f'{rec.recall:.6f}', f'{rec.accuracy:.6f}',
                        f'{rec.image_score:.6f}', f'{rec.mask_area:.6f}',
                    ])
                    if not rec.is_real:
                        f1_by[item.source][t].append(rec.f1)
                        iou_by[item.source][t].append(rec.iou)
            except DataError:
                raise  # alignment/pairing bug — abort, never skip
            n_done += 1
            if n_done % 200 == 0:
                log_line(f'[sweep] {n_done}/{len(all_items)} items')

    if n_missing:
        log_line(f'[sweep] WARN: {n_missing} items had no cache entry')
    log_line(f'[sweep] scored {n_done} items x {len(thresholds)} thresholds')

    # Summary: mean F1/IoU per (source, t); production t=0.5 and ORACLE best-t.
    summary: Dict[str, dict] = {}
    for source in sorted(f1_by):
        per_t = {t: float(np.mean(v)) for t, v in sorted(f1_by[source].items())}
        per_t_iou = {t: float(np.mean(v)) for t, v in sorted(iou_by[source].items())}
        best_t = max(per_t, key=per_t.get)
        summary[source] = {
            'n_fakes': len(f1_by[source][best_t]),
            'f1_by_threshold': {f'{t:.2f}': round(v, 4) for t, v in per_t.items()},
            'iou_by_threshold': {f'{t:.2f}': round(v, 4) for t, v in per_t_iou.items()},
            'f1_at_0.50': round(per_t.get(0.5, float('nan')), 4),
            'ORACLE_best_t': best_t,
            'ORACLE_best_f1': round(per_t[best_t], 4),
            'ORACLE_best_iou': round(per_t_iou[best_t], 4),
        }
        log_line(
            f'[sweep] {source}: F1@0.50={summary[source]["f1_at_0.50"]:.4f}  '
            f'ORACLE best t={best_t:.2f} F1={per_t[best_t]:.4f} '
            f'(n={summary[source]["n_fakes"]})'
        )

    with open(out_dir / 'sweep_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    log_line(f'[sweep] wrote {out_dir / "sweep_records.csv"} and sweep_summary.json')


if __name__ == '__main__':
    main()
