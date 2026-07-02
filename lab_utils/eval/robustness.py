"""lab_utils.eval.robustness — robustness sweep over augmentation conditions.

Operates on pre-computed EvalRecord lists, not on models or loaders directly.
The caller is responsible for running fetch → decode → metric under each
augmentation condition and collecting the resulting EvalRecord lists.

Typical usage::

    records_clean = [metric(decode_kmeans(model_info(model, img)), info, item)
                     for item in val_items]
    records_jpeg  = [...]  # same pipeline after applying jpeg aug

    report = robustness_sweep({'clean': records_clean, 'jpeg': records_jpeg})
    format_robustness_table(report, log_tag='[robust]')
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lab_utils.compat import trapz
from lab_utils.eval.record import EvalRecord
from lab_utils.eval.aggregate import summarize, by_bucket, BUCKET_LABELS
from lab_utils.logging.text import log_line


# ── Condition-level stat extraction ───────────────────────────────────────────

def _condition_stats(records: List[EvalRecord], *, metric: str = 'f1') -> Dict[str, Any]:
    """Derive a flat stats dict from a list of EvalRecords for one condition."""
    splices = [r for r in records if not r.is_real]
    reals   = [r for r in records if r.is_real]

    def _med(rs: List[EvalRecord]) -> float:
        vals = [getattr(r, metric, float('nan')) for r in rs]
        return float(np.median(vals)) if vals else float('nan')

    def _mean(rs: List[EvalRecord]) -> float:
        vals = [getattr(r, metric, float('nan')) for r in rs]
        return float(np.mean(vals)) if vals else float('nan')

    image_scores  = np.array([r.image_score for r in records], dtype=np.float64)
    image_labels  = np.array([0 if r.is_real else 1 for r in records], dtype=np.int32)
    image_auc     = float('nan')
    if not np.any(np.isnan(image_scores)) and image_labels.sum() > 0 and (image_labels == 0).sum() > 0:
        order    = np.argsort(-image_scores)
        sl       = image_labels[order]
        n_pos    = int(image_labels.sum())
        n_neg    = int((image_labels == 0).sum())
        tpr_pts  = np.cumsum(sl) / n_pos
        fpr_pts  = np.cumsum(1 - sl) / n_neg
        auc      = float(trapz(tpr_pts, fpr_pts))
        image_auc = 1.0 + auc if auc < 0 else auc

    buckets = by_bucket(records)
    return {
        'n_splice':  len(splices),
        'n_real':    len(reals),
        'image_auc': image_auc,
        metric:      _med(splices),
        f'{metric}_mean': _mean(splices),
        **{f'{metric}_{b}': _med(buckets[b]) for b in BUCKET_LABELS},
    }


# ── Sweep ──────────────────────────────────────────────────────────────────────

def robustness_sweep(
    records_by_condition: Dict[str, List[EvalRecord]],
    *,
    metric: str = 'f1',
    baseline_name: Optional[str] = None,
    log_tag: str = '[robust]',
    tag: str = '',
) -> Dict[str, Dict[str, Any]]:
    """Compare eval records across augmentation conditions.

    Args:
        records_by_condition: {condition_name: List[EvalRecord]}.
        metric:               Which EvalRecord field to summarise.
        baseline_name:        If given, print Δ vs this condition.
        log_tag:              Log tag (must be in ALLOWED_TAGS).
        tag:                  Optional sub-tag e.g. 'imd_val'.

    Returns:
        {condition_name: stats_dict}
    """
    prefix = f'{log_tag} {tag}' if tag else log_tag
    log_line(
        f'{prefix} robustness_sweep: {len(records_by_condition)} conditions '
        f'metric={metric}'
    )

    results: Dict[str, Dict[str, Any]] = {}
    for name, recs in records_by_condition.items():
        results[name] = _condition_stats(recs, metric=metric)

    format_robustness_table(
        results,
        metric=metric,
        baseline_name=baseline_name,
        log_tag=log_tag,
        tag=tag,
    )
    return results


# ── Table formatting ───────────────────────────────────────────────────────────

def format_robustness_table(
    results: Dict[str, Dict[str, Any]],
    *,
    metric: str = 'f1',
    baseline_name: Optional[str] = None,
    log_tag: str = '[robust]',
    tag: str = '',
) -> None:
    """Pretty-print a robustness results dict as a side-by-side table.

    Columns: condition | overall median | per-bucket medians | Δ vs baseline.
    """
    prefix = f'{log_tag} {tag}' if tag else log_tag
    if not results:
        log_line(f'{prefix} format_robustness_table: no results')
        return

    baseline = results.get(baseline_name) if baseline_name else None
    name_w   = max(10, max(len(n) for n in results))

    # Header
    header = (
        f'{"condition".ljust(name_w)}  '
        f'{"overall":>8}  '
        + '  '.join(f'{b:>8}' for b in BUCKET_LABELS)
        + f'  {"image_auc":>9}'
        + (f'  {"Δ(base)":>8}' if baseline is not None else '')
    )
    log_line(f'{prefix} {header}')
    log_line(f'{prefix} ' + '─' * len(header))

    for name, m in results.items():
        overall = m.get(metric, float('nan'))
        buckets = [m.get(f'{metric}_{b}', float('nan')) for b in BUCKET_LABELS]
        iauc    = m.get('image_auc', float('nan'))

        def _f(v: float) -> str:
            return f'{v:>8.4f}' if not (v != v) else f'{"—":>8}'

        row = (
            f'{name.ljust(name_w)}  '
            f'{_f(overall)}  '
            + '  '.join(_f(v) for v in buckets)
            + f'  {_f(iauc)}'
        )
        if baseline is not None:
            base_val = baseline.get(metric, float('nan'))
            try:
                delta = float(overall) - float(base_val)
                row += f'  {delta:>+8.4f}'
            except (TypeError, ValueError):
                row += f'  {"—":>8}'
        log_line(f'{prefix} {row}')
