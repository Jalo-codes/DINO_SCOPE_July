"""lab_utils.eval.aggregate — EvalRecord → reports.

Pure aggregation layer: List[EvalRecord] → report dicts / printed tables.
No model, no GT beyond what the records already carry.

Reporting style (fixed preference):
    median-led with mean alongside, full percentiles (p25/p75),
    reals pooled and reported separately from splices, legible aligned output.
"""

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lab_utils.compat import trapz
from lab_utils.eval.record import EvalRecord
from lab_utils.eval.buckets import BUCKET_LABELS
from lab_utils.logging.text import log_line


# ── Stat helpers ───────────────────────────────────────────────────────────────

def _stats(vals: Sequence[float]) -> dict:
    """Median-led stat block from a sequence of floats."""
    if not vals:
        return dict(n=0, median=float('nan'), mean=float('nan'),
                    std=float('nan'), p25=float('nan'), p75=float('nan'))
    a = np.array(vals, dtype=np.float64)
    return dict(
        n=len(a),
        median=float(np.median(a)),
        mean=float(np.mean(a)),
        std=float(np.std(a)),
        p25=float(np.percentile(a, 25)),
        p75=float(np.percentile(a, 75)),
    )


def _image_auc(records: List[EvalRecord]) -> float:
    """AUC from image_score over splices+reals (NaN if any score is NaN)."""
    if not records:
        return float('nan')
    scores = np.array([r.image_score for r in records], dtype=np.float64)
    labels = np.array([0 if r.is_real else 1 for r in records], dtype=np.int32)
    if np.any(np.isnan(scores)):
        return float('nan')
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order  = np.argsort(-scores)
    sl     = labels[order]
    tpr_pts = np.cumsum(sl) / n_pos
    fpr_pts = np.cumsum(1 - sl) / n_neg
    auc = float(trapz(tpr_pts, fpr_pts))
    return 1.0 + auc if auc < 0 else auc


def _splice_f1_stats(splices: List[EvalRecord]) -> dict:
    """F1 / IoU / prec / recall stats over splice records."""
    return {
        'f1':        _stats([r.f1        for r in splices]),
        'iou':       _stats([r.iou       for r in splices]),
        'precision': _stats([r.precision for r in splices]),
        'recall':    _stats([r.recall    for r in splices]),
    }


# ── Formatting helpers ─────────────────────────────────────────────────────────

def _fmt_stat(s: dict) -> str:
    """'median  mean ± std  [p25, p75]'."""
    if s['n'] == 0:
        return f'n=0  {"—":>6}'
    return (
        f'n={s["n"]:>4d}  '
        f'med={s["median"]:.4f}  '
        f'mean={s["mean"]:.4f}±{s["std"]:.4f}  '
        f'[p25={s["p25"]:.4f}, p75={s["p75"]:.4f}]'
    )


def _log_splice_block(
    records: List[EvalRecord],
    *,
    tag: str,
    log_tag: str = '[eval]',
) -> None:
    if not records:
        log_line(f'{log_tag} {tag}: n=0 (no splice records)')
        return
    st = _splice_f1_stats(records)
    for metric_name, s in st.items():
        log_line(f'{log_tag} {tag} {metric_name:>10}: {_fmt_stat(s)}')


# ── Public API ─────────────────────────────────────────────────────────────────

def by_bucket(records: List[EvalRecord]) -> Dict[str, List[EvalRecord]]:
    """Group splice records by bucket label."""
    out: Dict[str, List[EvalRecord]] = {b: [] for b in BUCKET_LABELS}
    for r in records:
        if not r.is_real and r.bucket in out:
            out[r.bucket].append(r)
    return out


def by_source(records: List[EvalRecord]) -> Dict[str, List[EvalRecord]]:
    """Group records by source name."""
    out: Dict[str, List[EvalRecord]] = defaultdict(list)
    for r in records:
        out[r.source].append(r)
    return dict(out)


def by_decoder(records: List[EvalRecord]) -> Dict[str, List[EvalRecord]]:
    """Group records by decoder name."""
    out: Dict[str, List[EvalRecord]] = defaultdict(list)
    for r in records:
        out[r.decoder].append(r)
    return dict(out)


def by_subgroup(records: List[EvalRecord]) -> Dict[str, List[EvalRecord]]:
    """Group records by their caller-assigned ``subgroup`` label.

    Records with ``subgroup is None`` are dropped (they opted out of the
    partition).  Pure aggregation — no GT, no model.
    """
    out: Dict[str, List[EvalRecord]] = defaultdict(list)
    for r in records:
        if r.subgroup is not None:
            out[r.subgroup].append(r)
    return dict(out)


def summarize(
    records: List[EvalRecord],
    *,
    log_tag: str = '[eval]',
    tag: str = '',
    include_sources: bool = False,
) -> Dict:
    """Overall + per-bucket + optional per-source summary.

    Reporting style: median-led with mean alongside, full percentiles,
    reals pooled separately from splices, legible aligned output.

    Returns a nested dict; also prints via log_line.
    """
    prefix = f'{log_tag} {tag}' if tag else log_tag

    splices = [r for r in records if not r.is_real]
    reals   = [r for r in records if r.is_real]

    log_line(f'{prefix} ─── summary: n_splice={len(splices)} n_real={len(reals)} ───')

    # ── Image-level AUC (splices + reals together)
    auc = _image_auc(records)
    if not np.isnan(auc):
        log_line(f'{prefix} image_auc: {auc:.4f}')

    # ── Splice localization: overall
    log_line(f'{prefix} splices (all):')
    _log_splice_block(splices, tag='all', log_tag=prefix)

    # ── Splice localization: per bucket
    buckets = by_bucket(records)
    for b in BUCKET_LABELS:
        bs = buckets[b]
        log_line(f'{prefix} splices bucket={b} (n={len(bs)}):')
        _log_splice_block(bs, tag=f'bucket={b}', log_tag=prefix)

    # ── Real accuracy
    if reals:
        accs = [r.accuracy for r in reals]
        s    = _stats(accs)
        log_line(f'{prefix} reals accuracy: {_fmt_stat(s)}')

    # ── Per-source breakdown (optional)
    if include_sources:
        sources = by_source(records)
        for src, src_records in sorted(sources.items()):
            src_splices = [r for r in src_records if not r.is_real]
            src_reals   = [r for r in src_records if r.is_real]
            log_line(
                f'{prefix} source={src}: '
                f'n_splice={len(src_splices)} n_real={len(src_reals)}'
            )
            _log_splice_block(src_splices, tag=f'source={src}', log_tag=prefix)

    # ── Build return dict
    result: Dict = {
        'n_splice':  len(splices),
        'n_real':    len(reals),
        'image_auc': auc,
        'splices':   _splice_f1_stats(splices) if splices else {},
        'reals':     {'accuracy': _stats([r.accuracy for r in reals])} if reals else {},
        'by_bucket': {b: _splice_f1_stats(buckets[b]) for b in BUCKET_LABELS},
    }
    if include_sources:
        result['by_source'] = {
            src: {
                'n_splice': len([r for r in rs if not r.is_real]),
                'n_real':   len([r for r in rs if r.is_real]),
                'splices':  _splice_f1_stats([r for r in rs if not r.is_real]),
            }
            for src, rs in sorted(by_source(records).items())
        }
    return result


def summarize_by_subgroup(
    records: List[EvalRecord],
    *,
    log_tag: str = '[eval]',
    tag: str = '',
    metric: str = 'f1',
) -> Dict[str, Dict]:
    """Per-subgroup splice breakdown (median-led, reals pooled separately).

    Groups records by their ``subgroup`` label (e.g. a TGIF (model|type|family)
    cell) and prints one median-led splice block per subgroup, plus a pooled
    reals-accuracy line.  Same reporting style as ``summarize`` — just sliced by
    the caller-chosen subgroup instead of the area bucket.

    Returns {subgroup: {'splices': {...}, 'n_real': int, 'reals_acc': {...}}}.
    """
    prefix = f'{log_tag} {tag}' if tag else log_tag
    groups = by_subgroup(records)

    if not groups:
        log_line(f'{prefix} subgroup breakdown: no records carry a subgroup label')
        return {}

    log_line(f'{prefix} ─── by subgroup ({len(groups)} cells) ───')
    results: Dict[str, Dict] = {}
    for sub in sorted(groups):
        recs    = groups[sub]
        splices = [r for r in recs if not r.is_real]
        reals   = [r for r in recs if r.is_real]
        log_line(
            f'{prefix} subgroup={sub} '
            f'(n_splice={len(splices)} n_real={len(reals)}):'
        )
        _log_splice_block(splices, tag=f'subgroup={sub}', log_tag=prefix)
        reals_acc = _stats([r.accuracy for r in reals]) if reals else _stats([])
        if reals:
            log_line(f'{prefix} subgroup={sub} reals accuracy: {_fmt_stat(reals_acc)}')
        results[sub] = {
            'n_splice':  len(splices),
            'n_real':    len(reals),
            'splices':   _splice_f1_stats(splices) if splices else {},
            'reals_acc': reals_acc,
        }
    return results


def decoder_bench(
    records_by_decoder: Dict[str, List[EvalRecord]],
    *,
    log_tag: str = '[eval]',
    tag: str = '',
    metric: str = 'f1',
) -> Dict[str, Dict]:
    """Side-by-side comparison across decoders.

    Prints a compact table (median metric per decoder, overall + per bucket).
    Returns {decoder: summary_dict}.
    """
    prefix = f'{log_tag} {tag}' if tag else log_tag
    results: Dict[str, Dict] = {}

    # Header
    pad = max((len(d) for d in records_by_decoder), default=8)
    header = f'{"decoder".ljust(pad)}  {"overall":>8}'
    for b in BUCKET_LABELS:
        header += f'  {b:>8}'
    log_line(f'{prefix} decoder_bench ({metric}):')
    log_line(f'{prefix}   {header}')

    for decoder, recs in sorted(records_by_decoder.items()):
        splices = [r for r in recs if not r.is_real]
        overall = _stats([getattr(r, metric, float('nan')) for r in splices])
        row     = f'{decoder.ljust(pad)}  {overall["median"]:>8.4f}'
        buckets = by_bucket(recs)
        for b in BUCKET_LABELS:
            bs  = buckets[b]
            bst = _stats([getattr(r, metric, float('nan')) for r in bs]) if bs else _stats([])
            row += f'  {bst["median"]:>8.4f}' if bst['n'] > 0 else f'  {"—":>8}'
        log_line(f'{prefix}   {row}')
        results[decoder] = {'splices': overall, 'by_bucket': {b: _stats([getattr(r, metric, float('nan')) for r in buckets[b]]) for b in BUCKET_LABELS}}

    return results


def save_summary_json(path: str, summaries: Dict[str, dict]) -> None:
    """Flatten and write overall/bucket metrics for all decoders to a JSON file."""
    flat = {}
    for decoder_name, summary in summaries.items():
        prefix = f"{decoder_name}_" if len(summaries) > 1 else ""
        
        # image AUC
        auc = summary.get('image_auc')
        if auc is not None and not np.isnan(auc):
            flat[f"{prefix}image_auc"] = float(auc)
            
        # splice stats
        splices = summary.get('splices', {})
        for metric in ['f1', 'iou', 'precision', 'recall']:
            m_stats = splices.get(metric, {})
            if 'median' in m_stats and not np.isnan(m_stats['median']):
                flat[f"{prefix}{metric}_median"] = float(m_stats['median'])
            if 'mean' in m_stats and not np.isnan(m_stats['mean']):
                flat[f"{prefix}{metric}_mean"] = float(m_stats['mean'])
                
        # reals stats
        reals = summary.get('reals', {})
        acc_stats = reals.get('accuracy', {})
        if 'mean' in acc_stats and not np.isnan(acc_stats['mean']):
            flat[f"{prefix}reals_acc"] = float(acc_stats['mean'])
            
        # per-bucket stats (F1 median)
        by_bucket = summary.get('by_bucket', {})
        for bucket, b_stats in by_bucket.items():
            f1_stats = b_stats.get('f1', {})
            if 'median' in f1_stats and not np.isnan(f1_stats['median']):
                flat[f"{prefix}bucket_{bucket}_f1_median"] = float(f1_stats['median'])
                
    import json
    import os
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(flat, f, indent=2)


def write_records_csv(records: List[EvalRecord], path: str) -> None:
    """Dump per-item scalar fields of EvalRecords to a CSV (one row per record).

    Arrays (gt_mask/pred_mask/attention) are NOT written — this is the
    spreadsheet-side view: per-item scores for sorting, per-source pivots,
    and cross-run diffs. Columns are stable and explicit.
    """
    import csv
    import os

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cols = ['item_id', 'source', 'decoder', 'subgroup', 'is_real',
            'f1', 'iou', 'precision', 'recall', 'accuracy',
            'image_score', 'mask_area', 'bucket']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([
                r.item_id, r.source, r.decoder, r.subgroup or '', int(r.is_real),
                f'{r.f1:.6f}', f'{r.iou:.6f}', f'{r.precision:.6f}',
                f'{r.recall:.6f}', f'{r.accuracy:.6f}',
                f'{r.image_score:.6f}', f'{r.mask_area:.6f}', r.bucket,
            ])
    log_line(f'[eval] wrote {len(records)} records -> {path}')
