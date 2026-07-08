"""experiments.labs.full_fakes_report — AUROC + localization distribution for
whole-image ("full fake") generations, per generator, across the 6
BCE-emergence cells.

Consumes per-cell records CSVs (experiments/scripts/eval.py --out_dir →
<decoder>_records.csv), filtered to source == 'full_fakes'
(lab_utils/data/datasets/full_fakes.py: root/real/ vs root/<generator>/, no
splice boundary, no real GT mask). Because every fake item carries a
synthetic full-frame sentinel mask, recall and iou are mechanically equal to
the predicted-positive pixel fraction — that IS the localization
distribution this report exists to show: how much of a wholly-fake frame the
model's patch head lights up (and, for reals via 1 - accuracy, how much of a
wholly-real frame it falsely lights up). image_score AUROC is the actual
real/fake separability number, reported both per generator and pooled across
all generators.

TORCH-FREE — csv + numpy only; runs anywhere the records CSVs live.

Usage::

    $PY -m experiments.labs.full_fakes_report \\
        --records bce_inpaint_s0=/runs/bce/bce_inpaint_s0/full_fakes_eval/threshold_records.csv \\
        --records cont_inpaint_s0=/runs/bce/cont_inpaint_s0/full_fakes_eval/kmeans_records.csv \\
        --out_csv /runs/bce/full_fakes_report.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from lab_utils.eval.rank_stats import rank_auc, stats
from lab_utils.logging.text import log_line


def _log(msg: str) -> None:
    log_line(f'[full_fakes] {msg}')


def _read_records(path: Path) -> List[dict]:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('f1', 'iou', 'precision', 'recall', 'accuracy',
                  'image_score', 'mask_area'):
            r[k] = float(r[k]) if r.get(k) not in (None, '') else float('nan')
        r['is_real'] = bool(int(r['is_real']))
    return [r for r in rows if r['source'] == 'full_fakes']


def _pred_pos_frac(row: dict) -> float:
    """Fraction of the frame flagged manipulated — the localization
    distribution. GT is all-true for fakes (recall == frac) and all-false
    for reals (1 - accuracy == frac); see module docstring."""
    return (1.0 - row['accuracy']) if row['is_real'] else row['recall']


def _fmt(v: float) -> str:
    return '   nan' if not np.isfinite(v) else f'{v:6.3f}'


def report_cell(cell: str, rows: List[dict], out_rows: List[dict]) -> None:
    reals = [r for r in rows if r['is_real']]
    fakes = [r for r in rows if not r['is_real']]
    _log(f'── {cell} ──────────────────────────────────────────────')
    _log(f'n_real={len(reals)}  n_fake={len(fakes)}')
    if not reals and not fakes:
        _log('no full_fakes records in this file — skipping')
        return

    real_ppf = stats([_pred_pos_frac(r) for r in reals])
    _log(f'{"reals":<24} n={len(reals):>5}  ppf med={_fmt(real_ppf["median"])} '
         f'mean={_fmt(real_ppf["mean"])}')
    out_rows.append({'cell': cell, 'kind': 'reals', 'name': 'real',
                     'metric': 'pred_pos_frac', 'stratum': 'all', **real_ppf})

    real_scores = np.asarray([r['image_score'] for r in reals])
    fake_scores_all = np.asarray([r['image_score'] for r in fakes])
    auc_all = rank_auc(fake_scores_all, real_scores)
    _log(f'{"ALL generators":<24} n={len(fakes):>5}  auroc(real vs fake)={_fmt(auc_all)}')
    out_rows.append({'cell': cell, 'kind': 'overall', 'name': 'all_generators',
                     'metric': 'auroc', 'stratum': 'all',
                     'n': len(fakes) + len(reals), 'median': auc_all, 'mean': auc_all,
                     'p25': float('nan'), 'p75': float('nan')})

    by_gen: Dict[str, List[dict]] = defaultdict(list)
    for r in fakes:
        by_gen[r['subgroup'] or 'unknown'].append(r)

    _log(f'{"generator":<24} {"n":>5}  {"ppf med":>8} {"ppf mean":>8}  {"auroc":>6}')
    for gen in sorted(by_gen):
        gr = by_gen[gen]
        ppf = stats([_pred_pos_frac(r) for r in gr])
        auc = rank_auc(np.asarray([r['image_score'] for r in gr]), real_scores)
        _log(f'{gen:<24} {len(gr):>5}  {_fmt(ppf["median"]):>8} {_fmt(ppf["mean"]):>8}  '
             f'{_fmt(auc):>6}')
        out_rows.append({'cell': cell, 'kind': 'generator', 'name': gen,
                         'metric': 'pred_pos_frac', 'stratum': 'all', **ppf})
        out_rows.append({'cell': cell, 'kind': 'generator', 'name': gen,
                         'metric': 'auroc', 'stratum': 'all',
                         'n': len(gr) + len(reals), 'median': auc, 'mean': auc,
                         'p25': float('nan'), 'p75': float('nan')})
    _log('')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--records', action='append', required=True, metavar='CELL=CSV',
                   help='cell_name=path/to/<decoder>_records.csv (repeatable).')
    p.add_argument('--out_csv', default=None,
                   help='Long-format output CSV of every reported number.')
    args = p.parse_args()

    out_rows: List[dict] = []
    for spec in args.records:
        cell, _, path = spec.partition('=')
        if not path:
            raise SystemExit(f'--records expects CELL=CSV, got {spec!r}')
        report_cell(cell, _read_records(Path(path)), out_rows)

    if args.out_csv:
        cols = ['cell', 'kind', 'name', 'metric', 'stratum', 'n',
                'median', 'mean', 'p25', 'p75']
        out = Path(args.out_csv)
        if out.parent != Path(''):
            out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(out_rows)
        _log(f'wrote {len(out_rows)} rows -> {out}')


if __name__ == '__main__':
    main()
