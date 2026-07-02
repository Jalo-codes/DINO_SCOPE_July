"""lab_utils.eval.multibox — efficient box cover + box gating (pure, reusable).

Given a binary ON/OFF patch grid (any source — thresholded attention, a decode
mask, …), `cover_bboxes` returns the cheapest *set* of boxes that covers the ON
patches, trading wasted background area against the number of boxes.  This is the
"how many boxes and where" policy, kept separate from:
  * thresholding         (lab_utils.eval.zoom.attention_hot_mask),
  * the zoom execution    (experiments.labs.attention_zoom.run_bbox_zoom), and
  * the per-box accept gate (gate_boxes_by_logit, below).

Pure numpy + the geometry primitives in zoom.py — no model call, no GT — so any
module can call these on any patch mask.
"""

from __future__ import annotations

from typing import List

import numpy as np

from lab_utils.eval.zoom import (
    BBox, _dilate8, _label_components, _pad_bbox, mask_components_bboxes,
)


# ── box geometry helpers ─────────────────────────────────────────────────────────

def _area(b: BBox) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _union(a: BBox, b: BBox) -> BBox:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _inter_area(a: BBox, b: BBox) -> float:
    iy0, ix0 = max(a[0], b[0]), max(a[1], b[1])
    iy1, ix1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, iy1 - iy0) * max(0.0, ix1 - ix0)


def suppress_contained_boxes(boxes: List[BBox], *, frac: float = 0.30) -> List[BBox]:
    """Drop a box when more than ``frac`` of its area lies inside a larger box.

    Greedy by descending area: keep the big boxes; a smaller box is killed when
    > ``frac`` of IT sits inside an already-kept (larger) box.  Removes the
    redundant inner/overlapping boxes that padding + squaring can create around a
    dominant region, so the zoom doesn't burn a window re-covering area a bigger
    crop already contains.  ``frac`` <= 0 disables (returns the boxes unchanged).
    """
    if frac is None or frac <= 0 or len(boxes) < 2:
        return list(boxes)
    kept: List[BBox] = []
    for b in sorted(boxes, key=_area, reverse=True):
        ab = _area(b)
        if ab <= 0:
            continue
        if any(_inter_area(b, k) / ab > frac for k in kept):
            continue
        kept.append(b)
    return kept


def _pad(b: BBox, pad_frac: float) -> BBox:
    ph = (b[2] - b[0]) * pad_frac
    pw = (b[3] - b[1]) * pad_frac
    return (max(0.0, b[0] - ph), max(0.0, b[1] - pw),
            min(1.0, b[2] + ph), min(1.0, b[3] + pw))


def _square(b: BBox, cap: float) -> BBox:
    """Expand the shorter side so the final aspect ratio is at most `cap`.

    Leaves already-square-ish boxes untouched; only partially squares elongated
    ones (bounded distortion when the crop is resized to a square), so we don't
    balloon a thin region into a huge square full of background.
    """
    y0, x0, y1, x1 = b
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return b
    long_, short_ = max(h, w), min(h, w)
    if long_ <= cap * short_:
        return b                                  # already within the cap
    grow = (long_ / cap - short_) / 2.0           # expand short side to hit the cap
    if h < w:
        y0, y1 = y0 - grow, y1 + grow
    else:
        x0, x1 = x0 - grow, x1 + grow
    return (max(0.0, y0), max(0.0, x0), min(1.0, y1), min(1.0, x1))


# ── efficient box cover ──────────────────────────────────────────────────────────

def _cost_merge(boxes: List[BBox], box_area_weight: float, max_regions: int) -> List[BBox]:
    """Greedy agglomerative cover: repeatedly merge the box pair whose union adds
    the least background area, while either (a) that added area is worth one fewer
    box (< box_area_weight) or (b) we still exceed max_regions.

    cost(S) = Σ area(b) + box_area_weight · |S|, so merging i,j → union u is worth
    it iff area(u) − area(i) − area(j) < box_area_weight.
    """
    boxes = list(boxes)
    while len(boxes) > 1:
        best = None  # (added_area, i, j)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                added = _area(_union(boxes[i], boxes[j])) - _area(boxes[i]) - _area(boxes[j])
                if best is None or added < best[0]:
                    best = (added, i, j)
        added, i, j = best
        if added < box_area_weight or len(boxes) > max_regions:
            u = _union(boxes[i], boxes[j])
            boxes = [b for k, b in enumerate(boxes) if k not in (i, j)] + [u]
        else:
            break
    return boxes


def cover_bboxes(
    hot_mask: np.ndarray,
    *,
    box_area_weight: float = 0.04,
    min_patches: int = 2,
    max_regions: int = 4,
    pad_frac: float = 0.08,
    square_cap: float = 1.4,
) -> List[BBox]:
    """Cheapest set of fractional boxes covering the ON patches of a binary mask.

    1. atoms  = one tight box per 4-connected ON component ≥ `min_patches`
       (drops obvious specks);
    2. merge  = greedily group atoms by the cost trade-off (`box_area_weight` =
       how much wasted background area one fewer box is worth — bigger ⇒ fewer,
       larger boxes; smaller ⇒ more, tighter boxes), capped at `max_regions`;
    3. shape  = pad each surviving box (`pad_frac`) and partially square it
       (`square_cap`), clipped to the frame.

    Mask-agnostic and model-free: feed it a thresholded-attention mask, a decode
    mask, anything.  Noise beyond the speck filter is left to the downstream MIL
    gate (gate_boxes_by_logit).

    Returns [] when nothing survives the speck filter.
    """
    m = np.asarray(hot_mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f'cover_bboxes expects a 2-D patch mask, got shape {m.shape}')

    atoms = mask_components_bboxes(m, pad_frac=0.0, min_size=min_patches, min_box_size=0)
    if not atoms:
        return []

    merged = _cost_merge(atoms, box_area_weight, max_regions)
    return [_square(_pad(b, pad_frac), square_cap) for b in merged]


# ── patch-space proximity cover (no hull-merge containment bug) ──────────────────

def proximity_bboxes(
    hot_mask: np.ndarray,
    *,
    dilate: int = 1,
    min_patches: int = 2,
    max_regions: int = 4,
    pad_frac: float = 0.08,
    min_box_size: int = 6,
    min_pad_frac: float = 0.06,
    small_base_pad: int = 1,
    square_cap: float = 1.4,
) -> List[BBox]:
    """Cheapest set of boxes covering ON patches, grouped in PATCH space.

    Unlike `cover_bboxes` (which agglomerates in bbox-HULL space and so
    unconditionally swallows a region whose tight box sits inside a larger one —
    `added = area(union) - area(A) - area(B) = -area(inner) < 0` always merges —
    and gives a sprawling/non-convex component a hull full of background), this
    groups on the mask itself:

      1. dilate ON by `dilate` patches (the proximity radius: cells within
         `dilate` of each other join one group);
      2. 4-connected components of the dilated mask;
      3. one box per component, tight around its *original* ON patches, padded
         via `_pad_bbox` and partially squared (`square_cap`).  Small splices get
         only `small_base_pad` (1) patch of margin; `min_box_size` is the real
         floor that stops a tiny splice from being over-magnified into a sliver.

    A sprawling region stays ONE box; a separate small blob keeps its own box
    unless it falls within `dilate` of a neighbour.  Components with fewer than
    `min_patches` true ON patches are dropped (specks); the largest
    `max_regions` survive.
    """
    m = np.asarray(hot_mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f'proximity_bboxes expects a 2-D patch mask, got shape {m.shape}')
    n_rows, n_cols = m.shape
    grouped = _dilate8(m, dilate) if dilate > 0 else m

    sized: List[tuple] = []  # (on_count, BBox)
    for cells in _label_components(grouped):
        on = [(r, c) for (r, c) in cells if m[r, c]]
        if len(on) < min_patches:
            continue
        rs = [r for r, _ in on]
        cs = [c for _, c in on]
        box = _pad_bbox(min(rs), min(cs), max(rs) + 1, max(cs) + 1,
                        n_rows, n_cols, pad_frac,
                        min_box_size=min_box_size, min_pad_frac=min_pad_frac,
                        small_base_pad=small_base_pad)
        sized.append((len(on), _square(box, square_cap)))

    sized.sort(key=lambda t: t[0], reverse=True)
    return [b for _, b in sized[:max_regions]]


# ── logit gating (relative to the full-image MIL score) ──────────────────────────

def gate_boxes_by_logit(
    box_logits: List[float],
    full_logit: float,
    *,
    margin: float = 0.0,
) -> List[int]:
    """Indices of the candidate boxes whose crop MIL logit clears the full image.

    A genuine splice crop *concentrates* the manipulation, so the MIL head should
    score its crop at least as high as the diluted full image.  A box whose crop
    logit falls below ``full_logit - margin`` is rejected as zoom-introduced noise
    (the head finds the crop *less* manipulated than the whole frame).

    Pure policy — no model call, no GT; `box_logits` / `full_logit` are computed
    upstream (fetch.model_info gives every crop an image_logit).

    Returns [] when `full_logit` is None or no box clears the bar.  The caller
    treats the empty result as "defer to the original unzoomed decode".
    """
    if full_logit is None:
        return []
    bar = float(full_logit) - float(margin)
    return [i for i, lg in enumerate(box_logits)
            if lg is not None and float(lg) >= bar]
