#!/usr/bin/env python3
"""Roll up ablation eval results into per-cell and headline tables.

Reads the orchestrator.log files written under an eval run_root and pulls out
the per-subgroup F1 lines (which are only logged, never serialized into
eval_summary.json). Emits two CSVs you can eyeball or drop into the paper:

  - <out_prefix>_percell.csv : rows = TGIF cell, cols = each run's f1 mean
  - <out_prefix>_overall.csv : one row per run with overall f1/iou/etc from
                               eval_summary.json (the headline curve)

Usage (on the box):
  PY=/home/fri-team-4/dino_venv/bin/python
  $PY -m experiments.scripts.rollup_ablation_eval \
      --run_root /media/ssd/runs/ablation_eval/lora_rank_sweep \
      --metric mean \
      --out_prefix /media/ssd/runs/ablation_eval/lora_rank_sweep/rollup

Run filtering:
  --only_suffix _tgif   restrict per-cell table to *_tgif runs (default: all)
"""
import argparse
import csv
import glob
import json
import os
import re

# Matches e.g.
# [eval] kmeans subgroup=flux1dev|fr|random   f1: n= 500  med=0.2156  mean=0.3351±0.3304 ...
_SUBGROUP_RE = re.compile(
    r'subgroup=(?P<cell>\S+)\s+f1:\s+n=\s*(?P<n>\d+)\s+'
    r'med=(?P<med>[-\d.]+)\s+mean=(?P<mean>[-\d.]+)'
)


def parse_log_percell(log_path, metric='mean'):
    """Return {cell: value} for the chosen metric ('mean' or 'med')."""
    out = {}
    with open(log_path, 'r', errors='replace') as f:
        for line in f:
            m = _SUBGROUP_RE.search(line)
            if not m:
                continue
            cell = m.group('cell')
            # Skip the header-only "subgroup=real" cell that carries no splice f1.
            try:
                out[cell] = float(m.group(metric if metric in ('med',) else 'mean'))
            except ValueError:
                continue
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_root', required=True,
                    help='eval run_root holding <entry>/orchestrator.log dirs')
    ap.add_argument('--metric', default='mean', choices=['mean', 'med'],
                    help="per-cell f1 statistic to tabulate (default mean)")
    ap.add_argument('--only_suffix', default=None,
                    help="only include runs whose name ends with this (e.g. _tgif)")
    ap.add_argument('--out_prefix', default=None,
                    help='write CSVs to <out_prefix>_percell.csv / _overall.csv')
    args = ap.parse_args()

    entry_dirs = sorted(
        d for d in glob.glob(os.path.join(args.run_root, '*'))
        if os.path.isdir(d)
    )

    percell = {}   # run -> {cell: val}
    overall = {}   # run -> flat eval_summary dict
    for d in entry_dirs:
        run = os.path.basename(d)
        if args.only_suffix and not run.endswith(args.only_suffix):
            continue
        log_path = os.path.join(d, 'orchestrator.log')
        if os.path.exists(log_path):
            cells = parse_log_percell(log_path, args.metric)
            if cells:
                percell[run] = cells
        js = os.path.join(d, 'eval_summary.json')
        if os.path.exists(js):
            with open(js) as f:
                overall[run] = json.load(f)

    runs = sorted(percell)
    all_cells = sorted({c for v in percell.values() for c in v})

    # ---- print per-cell table ----
    if runs:
        w = max((len(c) for c in all_cells), default=12)
        print(f'\n=== per-cell f1 {args.metric} ===')
        print('cell'.ljust(w) + '  ' + '  '.join(r.rjust(10) for r in runs))
        for c in all_cells:
            row = c.ljust(w)
            for r in runs:
                v = percell[r].get(c)
                row += '  ' + (f'{v:.4f}'.rjust(10) if v is not None else '—'.rjust(10))
            print(row)
        # mean over cells per run
        row = 'MEAN_over_cells'.ljust(w)
        for r in runs:
            vals = [percell[r][c] for c in all_cells if c in percell[r]]
            row += '  ' + (f'{sum(vals)/len(vals):.4f}'.rjust(10) if vals else '—'.rjust(10))
        print(row)

    # ---- write CSVs ----
    if args.out_prefix:
        pc_path = f'{args.out_prefix}_percell.csv'
        with open(pc_path, 'w', newline='') as f:
            wri = csv.writer(f)
            wri.writerow(['cell'] + runs)
            for c in all_cells:
                wri.writerow([c] + [percell[r].get(c, '') for r in runs])
            mean_row = ['MEAN_over_cells']
            for r in runs:
                vals = [percell[r][c] for c in all_cells if c in percell[r]]
                mean_row.append(sum(vals) / len(vals) if vals else '')
            wri.writerow(mean_row)
        print(f'\nwrote {pc_path}')

        if overall:
            keys = sorted({k for v in overall.values() for k in v})
            ov_path = f'{args.out_prefix}_overall.csv'
            with open(ov_path, 'w', newline='') as f:
                wri = csv.writer(f)
                wri.writerow(['run'] + keys)
                for r in sorted(overall):
                    wri.writerow([r] + [overall[r].get(k, '') for k in keys])
            print(f'wrote {ov_path}')


if __name__ == '__main__':
    main()
