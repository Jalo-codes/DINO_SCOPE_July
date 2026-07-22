"""lab_utils.eval.patch_scores — threshold-free per-patch separability readout.

Every existing eval metric (lab_utils.eval.metric.metric) scores a COMMITTED,
thresholded mask. That is the wrong comparison for the equal-budget patch-BCE
redesign (lab_utils.model.losses.bce.equal_budget_patch_bce_loss): per-image
budget balancing changes what a patch's sigmoid output MEANS (a rarity-
suppressed posterior under the old 'global' loss becomes an appearance
likelihood-ratio under 'per_image' — CLAUDE.md rule 1's calibration
confound). Comparing the two at any fixed decode threshold measures the
calibration shift, not localization quality.

This module reports raw per-patch sigmoid scores against per-patch GT labels
with NO threshold anywhere: AUROC, both pooled and stratified. It is decoder-
free and model-agnostic — useful for any patch-BCE checkpoint, not just the
equal-budget A/B.

Three background strata, kept separate because they are different
distributions (CLAUDE.md rule 4's real_crop vs fr_bg lesson, generalized):
    fake       — GT-fake patches on splice/inpaint items
    splice_bg  — GT-not-fake patches on a PARTLY-fake item (the untouched
                 region around a splice)
    real_bg    — every patch of a wholly-real item

``real_bg`` p99 is the sprinkle canary: if equal-budget training pushes it up
materially relative to a 'global' checkpoint, the symmetric false-alarm pool
failed in practice, not just in theory.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lab_utils.eval.buckets import BUCKET_LABELS, area_to_bucket
from lab_utils.logging.text import log_line


def weighted_auroc(
    scores: Sequence[float],
    labels: Sequence[float],
    weights: Optional[Sequence[float]] = None,
) -> float:
    """Sample-weighted AUROC via the weighted Mann-Whitney formulation.

    AUROC = P(score_pos > score_neg) + 0.5 * P(score_pos == score_neg), with
    every pairwise comparison weighted by w_i * w_j. Computed in O(n log n)
    by grouping on unique score values (no O(n^2) pairwise loop) — ties get
    exactly 0.5 credit, matching the textbook (unweighted) AUROC definition
    in the limit weights=None.

    A weight of exactly 2 on one sample is mathematically identical to two
    weight-1 duplicates at that (score, label) — the formula is linear in
    weight and treats them the same.

    Returns NaN if either class has zero total weight (undefined AUROC).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    n = scores.shape[0]
    if n == 0:
        return float('nan')
    if weights is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)

    pos = (labels > 0.5).astype(np.float64)
    neg = 1.0 - pos
    w_pos_total = float((w * pos).sum())
    w_neg_total = float((w * neg).sum())
    if w_pos_total <= 0.0 or w_neg_total <= 0.0:
        return float('nan')

    uniq_vals, inverse = np.unique(scores, return_inverse=True)
    n_groups = uniq_vals.shape[0]
    pos_w_per_group = np.bincount(inverse, weights=w * pos, minlength=n_groups)
    neg_w_per_group = np.bincount(inverse, weights=w * neg, minlength=n_groups)
    # exclusive cumsum: negative weight strictly below each group's score
    # (np.unique sorts ascending, so this cumsum walks scores low -> high)
    cum_neg_less = np.cumsum(neg_w_per_group) - neg_w_per_group
    numerator = float(np.sum(pos_w_per_group * (cum_neg_less + 0.5 * neg_w_per_group)))
    return numerator / (w_pos_total * w_neg_total)


def _quantiles(values: List[np.ndarray]) -> Dict[str, float]:
    if not values:
        return {'p50': float('nan'), 'p90': float('nan'), 'p99': float('nan')}
    arr = np.concatenate(values)
    if arr.size == 0:
        return {'p50': float('nan'), 'p90': float('nan'), 'p99': float('nan')}
    p50, p90, p99 = np.quantile(arr, [0.5, 0.9, 0.99])
    return {'p50': float(p50), 'p90': float(p90), 'p99': float(p99)}


def collect_patch_scores(
    model,
    items: List,
    res,
    *,
    device,
    use_amp: bool = True,
    amp_dtype: str = 'bfloat16',
    band: Tuple[float, float] = (0.2, 0.8),
    log_tag: str = '[patch-auroc]',
) -> Dict:
    """Run one forward per item, score raw patch sigmoids against GT — no decode.

    Skips (and counts):
      * items with meta['gt_mask_reliable'] is False — geometry-free sentinel
        masks (full_fakes, pseudo-mask sources); patch-level GT there is not
        meaningful (CLAUDE.md rule 2).
      * items with meta['crop_window'] set — region-probe items whose scored
        region is a fractional sub-window, not the model's full input frame;
        handling that geometry is out of scope here (use eval.metric for
        those sources instead).

    Returns a dict (see module docstring for the strata semantics):
        {
          'n_items': int, 'n_skipped_unreliable': int, 'n_skipped_cropwin': int,
          'auroc_pooled': float,       # all fake patches vs (splice_bg + real_bg)
          'auroc_vs_splice_bg': float, # all fake patches vs splice_bg only
          'auroc_vs_real_bg': float,   # all fake patches vs real_bg only
          'auroc_by_bucket': {bucket: float},  # that bucket's fakes vs ALL bg
          'score_quantiles': {stratum: {'p50':.., 'p90':.., 'p99':..}},
          'per_image': [{'item_id':.., 'bucket':.., 'n_fake':.., 'n_bg':..,
                         'scores_fake_mean':.., 'scores_bg_mean':..}, ...],
        }
    """
    import torch
    from PIL import Image

    from lab_utils.data.resolution import mask_to_patch_labels_soft
    from lab_utils.eval.fetch import model_info
    from lab_utils.eval.preprocess import load_image_tensor

    low, high = float(band[0]), float(band[1])

    strata_scores: Dict[str, List[np.ndarray]] = {'fake': [], 'splice_bg': [], 'real_bg': []}
    strata_weights: Dict[str, List[np.ndarray]] = {'fake': [], 'splice_bg': [], 'real_bg': []}
    bucket_fake_scores: Dict[str, List[np.ndarray]] = defaultdict(list)
    bucket_fake_weights: Dict[str, List[np.ndarray]] = defaultdict(list)
    per_image: List[Dict] = []

    n_skipped_unreliable = 0
    n_skipped_cropwin = 0
    n_fetch_failed = 0

    n = len(items)
    every = max(1, n // 10)
    with torch.no_grad():
        for i, item in enumerate(items):
            if item.meta.get('gt_mask_reliable') is False:
                n_skipped_unreliable += 1
                continue
            if item.meta.get('crop_window') is not None:
                n_skipped_cropwin += 1
                continue

            try:
                img_t = load_image_tensor(item, res, device=device)
                info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
            except Exception as exc:
                log_line(f'{log_tag} WARN fetch failed {item.item_id}: {exc}')
                n_fetch_failed += 1
                continue

            if info.patch_logits is None:
                log_line(
                    f'{log_tag} WARN: patch_logits is None for {item.item_id} '
                    f'(patch-BCE head disabled) — skipping'
                )
                n_fetch_failed += 1
                continue

            logits = np.asarray(info.patch_logits, dtype=np.float64).reshape(-1)
            probs = 1.0 / (1.0 + np.exp(-logits))
            n_side = info.grid_hw[0]
            n_patches = n_side * n_side

            if item.is_real:
                bucket = 'real'
                strata_scores['real_bg'].append(probs)
                strata_weights['real_bg'].append(np.ones(n_patches))
                n_fake_i, n_bg_i = 0, n_patches
                scores_fake_mean = float('nan')
                scores_bg_mean = float(probs.mean()) if n_patches else float('nan')
            else:
                mask_area = item.mask_area(res)
                bucket = area_to_bucket(mask_area)
                mask_pil = (
                    Image.open(item.mask).convert('L')
                    .resize((res.image_size, res.image_size), Image.NEAREST)
                )
                labels_t, weights_t = mask_to_patch_labels_soft(mask_pil, res, low=low, high=high)
                labels = labels_t.numpy().astype(np.float64).reshape(-1)
                weights = weights_t.numpy().astype(np.float64).reshape(-1)
                if labels.shape[0] != n_patches:
                    log_line(
                        f'{log_tag} WARN grid mismatch {item.item_id}: '
                        f'mask={labels.shape[0]} model={n_patches} — skipping'
                    )
                    n_fetch_failed += 1
                    continue

                fake_m = (labels > 0.5) & (weights > 0.0)
                bg_m = (labels <= 0.5) & (weights > 0.0)

                if fake_m.any():
                    strata_scores['fake'].append(probs[fake_m])
                    strata_weights['fake'].append(weights[fake_m])
                    bucket_fake_scores[bucket].append(probs[fake_m])
                    bucket_fake_weights[bucket].append(weights[fake_m])
                if bg_m.any():
                    strata_scores['splice_bg'].append(probs[bg_m])
                    strata_weights['splice_bg'].append(weights[bg_m])

                n_fake_i, n_bg_i = int(fake_m.sum()), int(bg_m.sum())
                scores_fake_mean = float(probs[fake_m].mean()) if fake_m.any() else float('nan')
                scores_bg_mean = float(probs[bg_m].mean()) if bg_m.any() else float('nan')

            per_image.append({
                'item_id': item.item_id, 'bucket': bucket,
                'n_fake': n_fake_i, 'n_bg': n_bg_i,
                'scores_fake_mean': scores_fake_mean, 'scores_bg_mean': scores_bg_mean,
            })

            if (i + 1) % every == 0 or (i + 1) == n:
                log_line(f'{log_tag} {i + 1}/{n} items '
                         f'(skipped unreliable={n_skipped_unreliable} '
                         f'crop_window={n_skipped_cropwin} fetch_fail={n_fetch_failed})')

    def _cat(key_lists: List[np.ndarray]) -> np.ndarray:
        return np.concatenate(key_lists) if key_lists else np.zeros(0, dtype=np.float64)

    fake_s, fake_w = _cat(strata_scores['fake']), _cat(strata_weights['fake'])
    splicebg_s, splicebg_w = _cat(strata_scores['splice_bg']), _cat(strata_weights['splice_bg'])
    realbg_s, realbg_w = _cat(strata_scores['real_bg']), _cat(strata_weights['real_bg'])

    all_bg_s = np.concatenate([splicebg_s, realbg_s])
    all_bg_w = np.concatenate([splicebg_w, realbg_w])

    def _auc(pos_s, pos_w, neg_s, neg_w) -> float:
        s = np.concatenate([pos_s, neg_s])
        y = np.concatenate([np.ones_like(pos_s), np.zeros_like(neg_s)])
        w = np.concatenate([pos_w, neg_w])
        return weighted_auroc(s, y, w)

    auroc_pooled = _auc(fake_s, fake_w, all_bg_s, all_bg_w)
    auroc_vs_splice_bg = _auc(fake_s, fake_w, splicebg_s, splicebg_w)
    auroc_vs_real_bg = _auc(fake_s, fake_w, realbg_s, realbg_w)

    auroc_by_bucket: Dict[str, float] = {}
    for b in BUCKET_LABELS:
        b_s = _cat(bucket_fake_scores.get(b, []))
        b_w = _cat(bucket_fake_weights.get(b, []))
        auroc_by_bucket[b] = _auc(b_s, b_w, all_bg_s, all_bg_w) if b_s.size else float('nan')

    score_quantiles = {
        'fake': _quantiles(strata_scores['fake']),
        'splice_bg': _quantiles(strata_scores['splice_bg']),
        'real_bg': _quantiles(strata_scores['real_bg']),
    }

    log_line(
        f'{log_tag} n_items={len(per_image)} skipped(unreliable={n_skipped_unreliable} '
        f'crop_window={n_skipped_cropwin} fetch_fail={n_fetch_failed})'
    )
    log_line(
        f'{log_tag} auroc_pooled={auroc_pooled:.4f} '
        f'vs_splice_bg={auroc_vs_splice_bg:.4f} vs_real_bg={auroc_vs_real_bg:.4f}'
    )
    for b in BUCKET_LABELS:
        log_line(f'{log_tag}   bucket={b}: auroc={auroc_by_bucket[b]:.4f}')
    log_line(
        f'{log_tag} real_bg score quantiles: '
        f'p50={score_quantiles["real_bg"]["p50"]:.4f} '
        f'p90={score_quantiles["real_bg"]["p90"]:.4f} '
        f'p99={score_quantiles["real_bg"]["p99"]:.4f}'
    )

    return {
        'n_items': len(per_image),
        'n_skipped_unreliable': n_skipped_unreliable,
        'n_skipped_cropwin': n_skipped_cropwin,
        'auroc_pooled': auroc_pooled,
        'auroc_vs_splice_bg': auroc_vs_splice_bg,
        'auroc_vs_real_bg': auroc_vs_real_bg,
        'auroc_by_bucket': auroc_by_bucket,
        'score_quantiles': score_quantiles,
        'per_image': per_image,
    }
