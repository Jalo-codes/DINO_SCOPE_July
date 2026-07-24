"""analysis.otsu_vs_threshold — adaptive-vs-fixed localization decode, nailed down.

Lines up, per source and size bucket, the four localization decodes for a
BCE-head checkpoint:

    thr@0.5     fixed production threshold, sigmoid(logit) >= 0.5
    thr@oracle  best SINGLE global threshold per source (the fixed-threshold
                ceiling with post-hoc tuning), F1 broken down by bucket
    otsu        kmeans_logit — exact 1-D two-means on the logits, i.e. an
                adaptive per-image threshold on the SAME axis as thr@*
    feats       kmeans_feats — spherical k-means on the whole 1280-d patch
                vector (reference: what the representation holds unsupervised)

The point of the table: does the adaptive split (otsu) beat the best fixed cut
(thr@oracle), and where (which size bucket)? If otsu > thr@oracle the per-image
calibration is doing real work a fixed threshold cannot; if otsu ~ thr@oracle a
recalibrated fixed threshold suffices. thr@0.5 -> thr@oracle isolates plain
miscalibration; otsu -> feats isolates learned-axis vs whole-vector.

All numbers are MEAN F1 over fakes only (reals carry no localization target).

Consumes:
  --eval_dir   dir with the decoder-bench records written by
               `eval.py --decoder threshold kmeans_logit kmeans_feats --bench`:
               threshold_records.csv / kmeans_logit_records.csv /
               kmeans_feats_records.csv  (any subset; missing -> column skipped)
  --sweep_dir  dir with sweep_records.csv from `eval_threshold_sweep.py`
               (optional; omit -> no thr@oracle column)

Run:
    python -m analysis.otsu_vs_threshold \
        --eval_dir  results/<run>/probe_clustering \
        --sweep_dir results/<run>/threshold_sweep
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

from lab_utils.eval.buckets import BUCKET_LABELS, area_to_bucket

# decode label -> bench records filename
_BENCH_FILES = [
    ('thr@0.5', 'threshold_records.csv'),
    ('otsu',    'kmeans_logit_records.csv'),
    ('feats',   'kmeans_feats_records.csv'),
]


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x == x]  # drop nan
    return sum(xs) / len(xs) if xs else float('nan')


def _fmt(x: float) -> str:
    return '  -   ' if x != x else f'{x:6.4f}'


def _load_bench(path: str):
    """Bench decoder CSV -> ({(source,bucket): [f1]}, {source: [f1]}), fakes only."""
    by_sb: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    by_s:  Dict[str, List[float]] = defaultdict(list)
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            if int(row['is_real']):
                continue
            f1 = float(row['f1'])
            src, bkt = row['source'], row['bucket']
            by_sb[(src, bkt)].append(f1)
            by_s[src].append(f1)
    return by_sb, by_s


def _load_sweep(path: str):
    """sweep_records.csv -> (src -> t -> bucket -> [f1], src -> t -> [f1], sorted_ts).

    Bucket derived from mask_area via area_to_bucket (sweep CSV has no bucket col).
    Fakes only.
    """
    per_bkt: Dict[str, Dict[float, Dict[str, List[float]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_all: Dict[str, Dict[float, List[float]]] = defaultdict(lambda: defaultdict(list))
    ts = set()
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            if int(row['is_real']):
                continue
            t = float(row['threshold'])
            ts.add(t)
            src, f1 = row['source'], float(row['f1'])
            bkt = area_to_bucket(float(row['mask_area']))
            per_bkt[src][t][bkt].append(f1)
            per_all[src][t].append(f1)
    return per_bkt, per_all, sorted(ts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--eval_dir', required=True,
                    help='dir with the decoder-bench *_records.csv files')
    ap.add_argument('--sweep_dir', default=None,
                    help='dir with sweep_records.csv (for the thr@oracle column)')
    ap.add_argument('--out_csv', default=None,
                    help='write the flat table here (default: <eval_dir>/otsu_vs_threshold.csv)')
    args = ap.parse_args()

    # ── load bench decoders that are present ──────────────────────────────────
    bench_sb: Dict[str, Dict] = {}
    bench_s:  Dict[str, Dict] = {}
    cols: List[str] = []
    for label, fname in _BENCH_FILES:
        p = os.path.join(args.eval_dir, fname)
        if os.path.exists(p):
            bench_sb[label], bench_s[label] = _load_bench(p)
            cols.append(label)
        else:
            print(f'# note: {fname} not found in eval_dir — skipping {label} column')
    if not cols:
        raise SystemExit(f'no decoder-bench *_records.csv found in {args.eval_dir}')

    # ── load sweep and pick the oracle global-t per source ────────────────────
    oracle_t: Dict[str, float] = {}
    sweep_bkt = sweep_all = None
    if args.sweep_dir:
        sp = os.path.join(args.sweep_dir, 'sweep_records.csv')
        if os.path.exists(sp):
            sweep_bkt, sweep_all, _ = _load_sweep(sp)
            for src, per_t in sweep_all.items():
                oracle_t[src] = max(per_t, key=lambda t: _mean(per_t[t]))
        else:
            print(f'# note: sweep_records.csv not found in {args.sweep_dir} — no thr@oracle')

    have_oracle = bool(oracle_t)

    # ordering: sources sorted, canonical n from the first present bench decoder
    n_ref_label = cols[0]
    sources = sorted(bench_s[n_ref_label].keys())

    # ── consistency check: bench thr@0.5 vs sweep t=0.50 (same crops?) ────────
    if 'thr@0.5' in cols and sweep_all is not None:
        diffs = []
        for src in sources:
            b = _mean(bench_s['thr@0.5'].get(src, []))
            s_t = sweep_all.get(src, {})
            # nearest available t to 0.50
            if s_t:
                t50 = min(s_t, key=lambda t: abs(t - 0.50))
                s = _mean(s_t[t50])
                if b == b and s == s:
                    diffs.append(abs(b - s))
        if diffs:
            md = max(diffs)
            flag = 'OK' if md < 5e-3 else 'WARN (>5e-3 — different crops?)'
            print(f'# validity: max |bench thr@0.5 - sweep t=0.50| = {md:.4g}  [{flag}]\n')

    # ── header ────────────────────────────────────────────────────────────────
    header_cols = ['thr@0.5'] + (['thr@oracle'] if have_oracle else []) + \
                  [c for c in cols if c != 'thr@0.5']
    flat_rows: List[List] = []

    def col_val(label: str, src: str, bkt) -> float:
        """mean F1 for a decode column at (src, bucket) or (src, overall if bkt None)."""
        if label == 'thr@oracle':
            t = oracle_t.get(src)
            if t is None:
                return float('nan')
            if bkt is None:
                return _mean(sweep_all[src][t])
            return _mean(sweep_bkt[src][t].get(bkt, []))
        if bkt is None:
            return _mean(bench_s.get(label, {}).get(src, []))
        return _mean(bench_sb.get(label, {}).get((src, bkt), []))

    def n_at(src: str, bkt) -> int:
        if bkt is None:
            return len(bench_s[n_ref_label].get(src, []))
        return len(bench_sb[n_ref_label].get((src, bkt), []))

    colw = '  '.join(f'{c:>10}' for c in header_cols)
    for src in sources:
        ot = f'  (oracle global t*={oracle_t[src]:.2f})' if have_oracle else ''
        print(f'=== {src}{ot} ===')
        print(f'{"bucket":<9}{"n":>6}   {colw}')
        for bkt in list(BUCKET_LABELS) + [None]:
            name = bkt if bkt is not None else 'OVERALL'
            n = n_at(src, bkt)
            if n == 0 and bkt is not None:
                continue
            vals = [col_val(c, src, bkt) for c in header_cols]
            print(f'{name:<9}{n:>6}   ' + '  '.join(f'{_fmt(v):>10}' for v in vals))
            flat_rows.append([src, name, n] + vals)
        print()

    # ── grand overall (all sources pooled), fakes only ────────────────────────
    def pooled(label: str, bkt) -> float:
        if label == 'thr@oracle':
            acc = []
            for src in sources:
                t = oracle_t.get(src)
                if t is None:
                    continue
                acc += (sweep_all[src][t] if bkt is None else sweep_bkt[src][t].get(bkt, []))
            return _mean(acc)
        if bkt is None:
            return _mean([f1 for src in sources for f1 in bench_s.get(label, {}).get(src, [])])
        return _mean([f1 for src in sources for f1 in bench_sb.get(label, {}).get((src, bkt), [])])

    print(f'=== ALL SOURCES POOLED ===')
    print(f'{"bucket":<9}{"n":>6}   {colw}')
    for bkt in list(BUCKET_LABELS) + [None]:
        name = bkt if bkt is not None else 'OVERALL'
        n = sum(n_at(src, bkt) for src in sources)
        if n == 0 and bkt is not None:
            continue
        vals = [pooled(c, bkt) for c in header_cols]
        print(f'{name:<9}{n:>6}   ' + '  '.join(f'{_fmt(v):>10}' for v in vals))
        flat_rows.append(['ALL', name, n] + vals)

    # ── flat CSV out ──────────────────────────────────────────────────────────
    out_csv = args.out_csv or os.path.join(args.eval_dir, 'otsu_vs_threshold.csv')
    with open(out_csv, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['source', 'bucket', 'n'] + header_cols)
        for r in flat_rows:
            w.writerow(r[:3] + [('' if v != v else f'{v:.6f}') for v in r[3:]])
    print(f'\n# wrote {out_csv}')


if __name__ == '__main__':
    main()
