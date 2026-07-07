"""experiments.labs.probe_contrasts — the BCE-emergence result tables.

Consumes per-item records CSVs (experiments/scripts/eval.py --out_dir →
<decoder>_records.csv) from one or more trained cells, plus the probe
manifest (experiments/labs/probe_manifest.py), and prints RAW numbers per
cell × condition — no deltas, no normalization; analysis happens off the pure
numbers:

  * localization: median / mean / p25 / p75 of F1 and IoU per condition;
  * image head:   median / mean image_score per condition (raw score table);
  * pred-positive fraction per condition — a patch-level raw score derived
    from the CSV columns (interior fakes have all-white GT so recall == the
    predicted-positive fraction; mask-less conditions have all-zero GT so it
    is 1 - accuracy);
  * pairwise AUC (image_score) for the study's contrasts — raw AUC values,
    Mann-Whitney rank formulation;
  * matched pairs (ai_interior ↔ real_crop joined on pair_stem via the
    manifest): both raw means, the mean per-pair difference and the win rate;
  * contrast AUCs stratified by parent splice bucket.

TORCH-FREE — csv + numpy only; runs anywhere the records CSVs live.

Usage::

    $PY -m experiments.labs.probe_contrasts \
        --records bce_inpaint_s0=/runs/bce/bce_inpaint_s0/eval/threshold_records.csv \
        --records cont_inpaint_s0=/runs/bce/cont_inpaint_s0/eval/kmeans_records.csv \
        --manifest /runs/probe_manifest.csv \
        --out_csv /runs/probe_contrasts.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from lab_utils.logging.text import log_line


def _log(msg: str) -> None:
    log_line(f'[probe] {msg}')

PROBE_CONDITIONS = ('ai_interior', 'ai_boundary', 'sp_interior', 'sp_boundary',
                    'fr_bg', 'real_crop')

# (positive source, negative source) — AUC of image_score separating pos from neg.
DEFAULT_CONTRASTS: Tuple[Tuple[str, str], ...] = (
    ('ai_interior', 'real_crop'),    # 1  the "thing", matched null
    ('ai_interior', 'sp_interior'),  # 2  AI vs splice content, boundary-free
    ('sp_interior', 'real_crop'),    # 3  provenance-shortcut meter (expect ~0.5)
    ('ai_interior', 'fr_bg'),        # 4  semantic AI-ness beyond fingerprint
    ('ai_boundary', 'sp_boundary'),  # 5  boundary evidence by content type
    ('ai_boundary', 'ai_interior'),  # 6a how much the boundary adds (AI)
    ('sp_boundary', 'sp_interior'),  # 6b how much the boundary adds (splice)
)


# ── IO ───────────────────────────────────────────────────────────────────────

def _read_records(path: Path) -> List[dict]:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ('f1', 'iou', 'precision', 'recall', 'accuracy',
                  'image_score', 'mask_area'):
            r[k] = float(r[k]) if r.get(k) not in (None, '') else float('nan')
        r['is_real'] = bool(int(r['is_real']))
    return rows


def _read_manifest(path: Optional[Path]) -> Dict[str, dict]:
    """item_id → manifest row (pair_stem, upsample_factor, window geometry)."""
    if path is None:
        return {}
    with open(path, newline='') as f:
        return {r['item_id']: r for r in csv.DictReader(f)}


# ── stats ────────────────────────────────────────────────────────────────────

def _rank_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(score_pos > score_neg) + 0.5 * P(equal)."""
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    all_scores = np.concatenate([pos, neg])
    order = all_scores.argsort(kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    # average ranks for ties
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


def _stats(vals: List[float]) -> dict:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if len(a) == 0:
        return {'n': 0, 'median': float('nan'), 'mean': float('nan'),
                'p25': float('nan'), 'p75': float('nan')}
    return {'n': len(a), 'median': float(np.median(a)), 'mean': float(a.mean()),
            'p25': float(np.percentile(a, 25)), 'p75': float(np.percentile(a, 75))}


def _pred_pos_frac(row: dict) -> float:
    """Predicted-positive pixel fraction, derived from the CSV columns.

    Fake probes (interior: all-white GT) → recall.  Mask-less probes
    (real_crop / fr_bg: all-zero GT) → 1 - accuracy.  Boundary crops have
    genuine partial GT, so no exact derivation exists — NaN there.
    """
    if row['is_real']:
        return 1.0 - row['accuracy']
    if row['source'].endswith('_interior'):
        return row['recall']
    return float('nan')


# ── report ───────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return '   nan' if not np.isfinite(v) else f'{v:6.3f}'


def report_cell(cell: str, rows: List[dict], manifest: Dict[str, dict],
                contrasts, out_rows: List[dict]) -> None:
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_source[r['source']].append(r)

    _log(f'── {cell} ──────────────────────────────────────────────')

    # Raw per-condition tables ------------------------------------------------
    _log(f'{"source":<14} {"n":>4}  {"f1 med":>7} {"f1 mean":>7} '
             f'{"iou med":>7}  {"score med":>9} {"score mean":>10} {"ppos med":>8}')
    for source in sorted(by_source):
        rs = by_source[source]
        loc = _stats([r['f1'] for r in rs if not r['is_real']])
        iou = _stats([r['iou'] for r in rs if not r['is_real']])
        sco = _stats([r['image_score'] for r in rs])
        ppf = _stats([_pred_pos_frac(r) for r in rs])
        _log(f'{source:<14} {len(rs):>4}  {_fmt(loc["median"]):>7} '
                 f'{_fmt(loc["mean"]):>7} {_fmt(iou["median"]):>7}  '
                 f'{_fmt(sco["median"]):>9} {_fmt(sco["mean"]):>10} '
                 f'{_fmt(ppf["median"]):>8}')
        for metric, st in (('f1', loc), ('iou', iou),
                           ('image_score', sco), ('pred_pos_frac', ppf)):
            out_rows.append({'cell': cell, 'kind': 'condition', 'name': source,
                             'metric': metric, 'stratum': 'all', **st})

    # Contrast AUCs (raw values) ----------------------------------------------
    _log(f'{"contrast":<32} {"AUC":>6} {"n_pos":>6} {"n_neg":>6}')
    for pos_src, neg_src in contrasts:
        pos = np.asarray([r['image_score'] for r in by_source.get(pos_src, [])])
        neg = np.asarray([r['image_score'] for r in by_source.get(neg_src, [])])
        auc = _rank_auc(pos, neg)
        _log(f'{pos_src + " vs " + neg_src:<32} {_fmt(auc):>6} '
                 f'{len(pos):>6} {len(neg):>6}')
        out_rows.append({'cell': cell, 'kind': 'contrast',
                         'name': f'{pos_src}|{neg_src}', 'metric': 'auc',
                         'stratum': 'all', 'n': min(len(pos), len(neg)),
                         'median': auc, 'mean': auc,
                         'p25': float('nan'), 'p75': float('nan')})

        # Stratify the headline contrasts by parent splice bucket.
        if (pos_src, neg_src) in (('ai_interior', 'real_crop'),
                                  ('ai_interior', 'sp_interior')):
            pos_rows = by_source.get(pos_src, [])
            neg_rows = by_source.get(neg_src, [])
            for bucket in ('medium', 'large'):
                p = np.asarray([r['image_score'] for r in pos_rows
                                if r.get('bucket') == bucket])
                n = np.asarray([r['image_score'] for r in neg_rows
                                if r.get('bucket') == bucket or neg_src == 'real_crop'])
                a = _rank_auc(p, n)
                _log(f'  [{bucket:<6}] {pos_src} vs {neg_src:<20} {_fmt(a)}')
                out_rows.append({'cell': cell, 'kind': 'contrast',
                                 'name': f'{pos_src}|{neg_src}', 'metric': 'auc',
                                 'stratum': f'bucket={bucket}',
                                 'n': min(len(p), len(n)), 'median': a, 'mean': a,
                                 'p25': float('nan'), 'p75': float('nan')})

    # Matched pairs (needs manifest) -------------------------------------------
    if manifest:
        stem = lambda r: manifest.get(r['item_id'], {}).get('pair_stem')
        ai = {stem(r): r['image_score'] for r in by_source.get('ai_interior', []) if stem(r)}
        rc = {stem(r): r['image_score'] for r in by_source.get('real_crop', []) if stem(r)}
        common = sorted(set(ai) & set(rc))
        if common:
            a = np.asarray([ai[s] for s in common])
            b = np.asarray([rc[s] for s in common])
            ok = np.isfinite(a) & np.isfinite(b)
            a, b = a[ok], b[ok]
            _log(f'matched pairs (n={len(a)}): ai_interior mean={_fmt(a.mean())} '
                     f'real_crop mean={_fmt(b.mean())} '
                     f'pair diff mean={_fmt((a - b).mean())} '
                     f'win rate={_fmt((a > b).mean())}')
            out_rows.append({'cell': cell, 'kind': 'matched_pairs',
                             'name': 'ai_interior|real_crop', 'metric': 'win_rate',
                             'stratum': 'all', 'n': len(a),
                             'median': float((a > b).mean()),
                             'mean': float((a - b).mean()),
                             'p25': float(a.mean()), 'p75': float(b.mean())})
        else:
            _log('matched pairs: no shared pair_stems between ai_interior and '
                     'real_crop records — check the manifest matches this eval')
    _log('')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--records', action='append', required=True, metavar='CELL=CSV',
                   help='cell_name=path/to/<decoder>_records.csv (repeatable).')
    p.add_argument('--manifest', default=None,
                   help='probe_manifest.csv (enables matched-pairs analysis).')
    p.add_argument('--contrast', action='append', default=[], metavar='POS:NEG',
                   help='Extra contrast, e.g. tgif2:indoor (repeatable; added to defaults).')
    p.add_argument('--out_csv', default=None,
                   help='Long-format output CSV of every reported number.')
    args = p.parse_args()

    contrasts = list(DEFAULT_CONTRASTS)
    for c in args.contrast:
        pos_src, neg_src = c.split(':', 1)
        contrasts.append((pos_src, neg_src))

    manifest = _read_manifest(Path(args.manifest) if args.manifest else None)

    out_rows: List[dict] = []
    for spec in args.records:
        cell, _, path = spec.partition('=')
        if not path:
            raise SystemExit(f'--records expects CELL=CSV, got {spec!r}')
        report_cell(cell, _read_records(Path(path)), manifest, contrasts, out_rows)

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
