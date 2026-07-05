"""lab_utils.eval.zoom — decoder-agnostic zoom geometry.

The crop → re-decode → place-back geometry used to live (private) inside
labs/attention_zoom.py, coupled to the attention map.  But the geometry has
nothing to do with attention: given *any* fractional bbox you crop, run the
model on the crop, decode, and project the crop-resolution mask back into the
full-frame patch grid.  Attention-bbox and cluster-mask-bbox are just two ways
to *derive* the bbox.

All bboxes are `(y0, x0, y1, x1)` as fractions of the image in [0, 1].

Pure geometry + numpy/PIL; no model call, no GT.  The model entry stays in
fetch.model_info (I2); GT stays in metric (I3).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

BBox = Tuple[float, float, float, float]  # (y0, x0, y1, x1) fractions


# ── bbox derivation ────────────────────────────────────────────────────────────

def mask_to_bbox(
    patch_mask: np.ndarray,
    *,
    pad_frac: float = 0.10,
    min_box_size: int = 8,
) -> Optional[BBox]:
    """Tight fractional bbox around the True patches of a boolean patch mask.

    Works on any (n_rows, n_cols) boolean mask — a thresholded attention map,
    an HDBSCAN cluster mask, a threshold-decode mask, whatever.

    Returns None when the mask is empty (caller decides the fallback).
    """
    m = np.asarray(patch_mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f'mask_to_bbox expects a 2-D patch mask, got shape {m.shape}')
    n_rows, n_cols = m.shape
    rows, cols = np.where(m)
    if rows.size == 0:
        return None

    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    return _pad_bbox(r0, c0, r1, c1, n_rows, n_cols, pad_frac, min_box_size=min_box_size)


def grid_locked_box(
    patch_idx: int,
    h_frac: float,
    w_frac: float,
    grid_hw: Tuple[int, int],
) -> BBox:
    """Fractional bbox centered on a patch CELL with a given (h, w) extent.

    The center is locked to the patch's own grid center — never a continuous
    offset — which is the whole point of the learned box-policy: positions are
    read off the patch grid, only the extent is predicted.  `h_frac` / `w_frac`
    are fractions of the full frame; the box is clipped to [0, 1].
    """
    n_rows, n_cols = grid_hw
    r = int(patch_idx) // n_cols
    c = int(patch_idx) % n_cols
    yc = (r + 0.5) / n_rows
    xc = (c + 0.5) / n_cols
    hh = 0.5 * float(h_frac)
    hw = 0.5 * float(w_frac)
    return (max(0.0, yc - hh), max(0.0, xc - hw),
            min(1.0, yc + hh), min(1.0, xc + hw))


def compute_otsu_threshold(values: np.ndarray) -> float:
    """Otsu's threshold (1D 2-means): split values into two clusters maximizing
    between-class variance, return the midpoint at the best split.
    """
    flat = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = len(flat)
    if n < 3:
        return float(flat.max()) + 1.0
    csum = np.cumsum(flat)
    total = csum[-1]
    i = np.arange(1, n)
    m0 = csum[:-1] / i
    m1 = (total - csum[:-1]) / (n - i)
    var_between = i * (n - i) * (m0 - m1) ** 2
    gi = int(np.argmax(var_between))
    return float(0.5 * (flat[gi] + flat[gi + 1]))


def compute_gap_threshold(values: np.ndarray) -> float:
    """Find the largest gap in the sorted values and return its midpoint."""
    flat = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    if len(flat) < 3:
        return float(flat.max()) + 1.0
    diffs = np.diff(flat)
    gi = int(np.argmax(diffs))
    return float(0.5 * (flat[gi] + flat[gi + 1]))


def attention_hot_mask(
    attention: np.ndarray,
    grid_hw: Tuple[int, int],
    *,
    percentile: float | str = 'otsu',
    thresh_mult: float = 0.70,
) -> np.ndarray:
    """Boolean (n_rows, n_cols) mask of the 'hot' attention patches.

    When percentile is 'otsu' or 'gap', dynamically thresholds the attention map
    (applying `thresh_mult` — 0.70 by default — to be more generous and include
    all warm/spliced margins).  Otherwise, keeps patches at/above the top-
    `percentile` percentile (no multiplier, matching numeric-percentile usage).

    This is the shared thresholding step behind `attention_to_bbox` and
    `peak_hot_component`.
    """
    n_rows, n_cols = grid_hw
    n = n_rows * n_cols
    attn = np.asarray(attention, dtype=np.float64).reshape(-1)[:n]

    if isinstance(percentile, str):
        p = percentile.lower()
        if p == 'otsu':
            thresh = compute_otsu_threshold(attn) * thresh_mult
        elif p == 'gap':
            thresh = compute_gap_threshold(attn) * thresh_mult
        elif p == 'peak':
            # everything lit at all: any patch >= thresh_mult * the peak.  Use a
            # small thresh_mult to grab the full attended region incl. faint
            # margins (recall-first — excluded patches are guaranteed misses).
            thresh = float(attn.max()) * thresh_mult
        else:
            raise ValueError(f"Unknown threshold method: {percentile}")
    else:
        thresh = float(np.percentile(attn, percentile))

    return (attn >= thresh).reshape(n_rows, n_cols)


def attention_to_bbox(
    attention: np.ndarray,
    grid_hw: Tuple[int, int],
    *,
    percentile: float | str = 'otsu',
    thresh_mult: float = 0.70,
    pad_frac: float = 0.10,
    min_box_size: int = 8,
    min_pad_frac: float = 0.0,
    pad_side_frac: Optional[float] = None,
    min_area_frac: float = 0.0,
) -> BBox:
    """Fractional bbox around the active attention patches.

    When percentile is 'otsu'/'gap'/'peak', dynamically thresholds the attention
    map (scaled by `thresh_mult`).  'peak' with a small thresh_mult grabs the
    whole lit region incl. faint margins (recall-first single-box crop).
    Otherwise, uses the top-`percentile` percentile.

    `min_pad_frac` is a floor on the per-side padding fraction so the breathing
    room does not collapse to ~0 on medium/large boxes (see `_pad_bbox`).
    min_pad_frac=0.0 (default) reproduces the legacy padding.
    """
    n_rows, n_cols = grid_hw
    hot = attention_hot_mask(attention, grid_hw, percentile=percentile, thresh_mult=thresh_mult)
    rows, cols = np.where(hot)
    if rows.size == 0:
        return 0.0, 0.0, 1.0, 1.0
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    return _pad_bbox(r0, c0, r1, c1, n_rows, n_cols, pad_frac,
                     min_box_size=min_box_size, min_pad_frac=min_pad_frac,
                     pad_side_frac=pad_side_frac, min_area_frac=min_area_frac)


def _label_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """4-connected components of a boolean mask → list of cell-lists [(r, c), ...].

    Pure numpy flood-fill — no scipy dependency.  Components are returned in
    row-major discovery order (top-left scan), which keeps callers deterministic.
    """
    m = np.asarray(mask, dtype=bool)
    n_rows, n_cols = m.shape
    seen = np.zeros_like(m, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []

    for sr in range(n_rows):
        for sc in range(n_cols):
            if not m[sr, sc] or seen[sr, sc]:
                continue
            # BFS flood fill
            stack = [(sr, sc)]
            seen[sr, sc] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols and m[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def plug_holes(mask: np.ndarray) -> np.ndarray:
    """Fill fully-enclosed background holes in a boolean grid mask.

    Background is flood-filled 8-connected from the border (the topological
    dual of the 4-connected foreground in _label_components); any background
    cell the flood never reaches is inside a closed foreground ring — an
    enclosed hole — and is set True. Background regions touching the border
    are never filled. Pure numpy flood-fill, no scipy.
    """
    m = np.asarray(mask, dtype=bool)
    if not m.any() or m.all():
        return m.copy()
    n_rows, n_cols = m.shape
    outside = np.zeros_like(m)
    stack: List[Tuple[int, int]] = []
    for r in range(n_rows):
        for c in (0, n_cols - 1):
            if not m[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    for c in range(n_cols):
        for r in (0, n_rows - 1):
            if not m[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (0 <= nr < n_rows and 0 <= nc < n_cols
                        and not m[nr, nc] and not outside[nr, nc]):
                    outside[nr, nc] = True
                    stack.append((nr, nc))
    return m | ~outside


def mask_components_bboxes(
    patch_mask: np.ndarray,
    *,
    pad_frac: float = 0.0,
    min_size: int = 1,
    min_box_size: int = 0,
) -> List[BBox]:
    """One fractional bbox per 4-connected True component of a patch mask.

    Useful for drawing a box around each accepted HDBSCAN cluster region.
    """
    m = np.asarray(patch_mask, dtype=bool)
    n_rows, n_cols = m.shape
    boxes: List[BBox] = []

    for cells in _label_components(m):
        if len(cells) < min_size:
            continue
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        boxes.append(_pad_bbox(min(rs), min(cs), max(rs) + 1, max(cs) + 1,
                               n_rows, n_cols, pad_frac, min_box_size=min_box_size))
    return boxes


def peak_hot_component(
    attention: np.ndarray,
    grid_hw: Tuple[int, int],
    *,
    percentile: float | str = 'otsu',
    thresh_mult: float = 0.70,
) -> np.ndarray:
    """Boolean (n_rows, n_cols) mask of the single hot component holding the peak.

    Thresholds attention into a hot mask (`attention_hot_mask`), 4-connected-
    labels it, and returns ONLY the component containing the global attention
    argmax.  This is the hide-set for second-best search: hiding it suppresses
    region 1 entirely so a re-pool must look elsewhere.

    Returns an all-False mask when nothing is hot.  If the peak patch is not in
    the hot set (degenerate, e.g. flat attention), falls back to the component
    carrying the most attention mass.
    """
    n_rows, n_cols = grid_hw
    n = n_rows * n_cols
    attn = np.asarray(attention, dtype=np.float64).reshape(-1)[:n]
    hot = attention_hot_mask(attention, grid_hw, percentile=percentile, thresh_mult=thresh_mult)

    out = np.zeros((n_rows, n_cols), dtype=bool)
    if not hot.any():
        return out

    comps = _label_components(hot)
    peak_idx = int(np.argmax(attn))
    peak_rc = (peak_idx // n_cols, peak_idx % n_cols)

    chosen = next((cells for cells in comps if peak_rc in cells), None)
    if chosen is None:
        chosen = max(comps, key=lambda cells: sum(attn[r * n_cols + c] for r, c in cells))

    for r, c in chosen:
        out[r, c] = True
    return out


def _pad_bbox(r0, c0, r1, c1, n_rows, n_cols, pad_frac,
              min_box_size: int = 8, min_pad_frac: float = 0.0,
              small_base_pad: int = 2,
              pad_side_frac: Optional[float] = None,
              min_area_frac: float = 0.0) -> BBox:
    # ── Area-based mode (resolution-invariant) ──────────────────────────────────
    # When pad_side_frac is set, pad in frame-FRACTION space and skip the
    # patch-unit math entirely: `pad_side_frac` is the margin added to each side
    # as a fraction of the frame, so it means the same thing at any grid size.
    # `min_area_frac` floors the padded box to that fraction of the frame area,
    # grown symmetrically about the center (aspect-preserving).  The downstream
    # crop (crop_to_bbox / place_mask_in_frame_pixels) already consumes fractions,
    # so no patch alignment is needed here.
    if pad_side_frac is not None:
        y0 = r0 / n_rows; x0 = c0 / n_cols
        y1 = r1 / n_rows; x1 = c1 / n_cols
        p = float(pad_side_frac)
        y0 -= p; y1 += p; x0 -= p; x1 += p
        if min_area_frac > 0.0:
            h = max(1e-6, y1 - y0); w = max(1e-6, x1 - x0)
            cur = h * w
            if cur < min_area_frac:
                s  = (min_area_frac / cur) ** 0.5
                yc = 0.5 * (y0 + y1); xc = 0.5 * (x0 + x1)
                hh = 0.5 * h * s;     hw = 0.5 * w * s
                y0, y1 = yc - hh, yc + hh
                x0, x1 = xc - hw, xc + hw
        return max(0.0, y0), max(0.0, x0), min(1.0, y1), min(1.0, x1)

    if pad_frac <= 0 and min_box_size <= 0 and min_pad_frac <= 0:
        return r0 / n_rows, c0 / n_cols, r1 / n_rows, c1 / n_cols

    # Calculate bounding box area as a fraction of the full frame area.
    bbox_area_frac = ((r1 - r0) / n_rows) * ((c1 - c0) / n_cols)

    # Scale the padding fraction inversely with the bounding box area.
    # Large bounding boxes need almost no padding, while small boxes get more.
    # `min_pad_frac` floors this so the breathing room never fully collapses on
    # medium/large boxes — a slightly-tight box still gets a margin to recover
    # splice content the attention map under-lit at its border.
    scale = (1.0 - bbox_area_frac) ** 1.5
    eff_pad_frac = max(min_pad_frac, pad_frac * scale)

    # Determine base minimum padding depending on bbox size.  ``small_base_pad``
    # is the floor for small boxes (attention path keeps 2; the box-heatmap
    # read-off sets 1 so tiny splices get just one patch of margin and rely on
    # ``min_box_size`` for the don't-over-magnify floor instead).
    if bbox_area_frac < 0.08:
        base_pad = small_base_pad
    elif bbox_area_frac < 0.35:
        base_pad = 1
    else:
        base_pad = 0

    do_pad = pad_frac > 0 or min_pad_frac > 0
    pad_r = max(base_pad, int(round(eff_pad_frac * (r1 - r0)))) if do_pad else 0
    pad_c = max(base_pad, int(round(eff_pad_frac * (c1 - c0)))) if do_pad else 0

    r0 = max(0, r0 - pad_r); r1 = min(n_rows, r1 + pad_r)
    c0 = max(0, c0 - pad_c); c1 = min(n_cols, c1 + pad_c)

    # Enforce minimum box size symmetrically from the center
    if min_box_size > 0:
        r_cent = 0.5 * (r0 + r1)
        if (r1 - r0) < min_box_size:
            half = min_box_size / 2.0
            r0 = int(round(r_cent - half))
            r1 = r0 + min_box_size
            if r0 < 0:
                r0 = 0
                r1 = min(n_rows, min_box_size)
            elif r1 > n_rows:
                r1 = n_rows
                r0 = max(0, n_rows - min_box_size)

        c_cent = 0.5 * (c0 + c1)
        if (c1 - c0) < min_box_size:
            half = min_box_size / 2.0
            c0 = int(round(c_cent - half))
            c1 = c0 + min_box_size
            if c0 < 0:
                c0 = 0
                c1 = min(n_cols, min_box_size)
            elif c1 > n_cols:
                c1 = n_cols
                c0 = max(0, n_cols - min_box_size)

    return r0 / n_rows, c0 / n_cols, r1 / n_rows, c1 / n_cols


# Efficient multi-region box cover (cover_bboxes) + box gating live in
# lab_utils.eval.multibox, which imports the geometry primitives above.


# ── crop + place-back ──────────────────────────────────────────────────────────

def crop_to_bbox(img_pil, bbox: BBox):
    """Crop a PIL image to a fractional bbox (y0, x0, y1, x1)."""
    W, H = img_pil.size
    y0, x0, y1, x1 = bbox
    left  = int(round(x0 * W));  right = int(round(x1 * W))
    upper = int(round(y0 * H));  lower = int(round(y1 * H))
    left, right = min(left, right - 1), max(right, left + 1)
    upper, lower = min(upper, lower - 1), max(lower, upper + 1)
    return img_pil.crop((left, upper, right, lower))


def place_mask_in_frame(
    crop_mask: np.ndarray,
    bbox: BBox,
    full_grid_hw: Tuple[int, int],
) -> np.ndarray:
    """Project a crop-resolution patch mask back into the full-frame patch grid.

    Args:
        crop_mask:    Boolean (h, w) mask decoded from the cropped region.
        bbox:         The fractional bbox the crop was taken from.
        full_grid_hw: (n_rows, n_cols) of the full-frame patch grid.

    Returns:
        Boolean (n_rows, n_cols) full-frame patch mask.
    """
    from PIL import Image as PILImage

    n_rows, n_cols = full_grid_hw
    full = np.zeros((n_rows, n_cols), dtype=bool)

    y0, x0, y1, x1 = bbox
    r_start = int(round(y0 * n_rows)); r_end = int(round(y1 * n_rows))
    c_start = int(round(x0 * n_cols)); c_end = int(round(x1 * n_cols))
    r_h = max(1, r_end - r_start)
    c_w = max(1, c_end - c_start)

    crop_pil = PILImage.fromarray(np.asarray(crop_mask, dtype=np.uint8) * 255)
    placed   = np.array(crop_pil.resize((c_w, r_h), PILImage.NEAREST)) > 127

    r_end2 = min(n_rows, r_start + r_h)
    c_end2 = min(n_cols, c_start + c_w)
    full[r_start:r_end2, c_start:c_end2] = placed[:r_end2 - r_start, :c_end2 - c_start]
    return full


def place_mask_in_frame_pixels(
    crop_mask: np.ndarray,
    bbox: BBox,
    full_hw: Tuple[int, int],
) -> np.ndarray:
    """Project a crop patch mask back into the full frame at PIXEL resolution.

    Unlike place_mask_in_frame (which bins the crop mask down to the coarse
    full-frame patch grid and discards the resolution the zoom bought), this
    writes the crop's decode into a pixel-resolution canvas at the crop's true
    location.  A crop covering 30% of the frame keeps its full crop-grid detail
    instead of collapsing to ~8x8 patches — that is the whole point of zooming.

    Args:
        crop_mask: Boolean (h, w) mask decoded from the cropped region.
        bbox:      The fractional bbox the crop was taken from.
        full_hw:   (H, W) pixel size of the full-frame canvas.

    Returns:
        Boolean (H, W) full-frame pixel mask.
    """
    from PIL import Image as PILImage

    H, W = full_hw
    full = np.zeros((H, W), dtype=bool)

    y0, x0, y1, x1 = bbox
    r0 = int(round(y0 * H)); r1 = max(r0 + 1, int(round(y1 * H)))
    c0 = int(round(x0 * W)); c1 = max(c0 + 1, int(round(x1 * W)))
    r1 = min(H, r1); c1 = min(W, c1)
    r_h = r1 - r0; c_w = c1 - c0

    crop_pil = PILImage.fromarray(np.asarray(crop_mask, dtype=np.uint8) * 255)
    placed   = np.array(crop_pil.resize((c_w, r_h), PILImage.NEAREST)) > 127
    full[r0:r1, c0:c1] = placed[:r_h, :c_w]
    return full


def bbox_is_trivial(bbox: BBox, *, min_crop_frac: float = 0.25) -> bool:
    """True when the bbox covers ~the whole frame (zooming would not help)."""
    y0, x0, y1, x1 = bbox
    return (y1 - y0) * (x1 - x0) >= (1.0 - min_crop_frac)


# ── single-box heatmap geometry (supervised MVP) ─────────────────────────────────
#
# Grid-level box geometry for the supervised single-box head: build a binary box
# TARGET from a GT mask, and READ a box back off a predicted heatmap.  Pure numpy
# (+ lazy PIL); GT is passed in as an array (the load stays in metric, I3).

GridBBox = Tuple[int, int, int, int]   # (r0, c0, r1, c1) inclusive, grid coords


def gt_grid_mask(
    gt_pixels: Optional[np.ndarray],
    grid_hw: Tuple[int, int],
    *,
    patch_frac: float = 0.25,
) -> np.ndarray:
    """Downsample a native-resolution GT pixel mask to a (n_rows, n_cols) bool grid.

    A patch is GT when at least ``patch_frac`` of its pixel block is GT (area
    pooling via PIL BOX), so faint single-pixel leakage doesn't light a patch.
    """
    from PIL import Image as PILImage
    n_rows, n_cols = grid_hw
    if gt_pixels is None or not np.asarray(gt_pixels).any():
        return np.zeros((n_rows, n_cols), dtype=bool)
    pil = PILImage.fromarray((np.asarray(gt_pixels).astype(np.uint8) * 255), mode='L')
    small = pil.resize((n_cols, n_rows), PILImage.BOX)          # area-average downsample
    frac = np.asarray(small, dtype=np.float32) / 255.0
    return frac >= patch_frac


def all_component_bboxes(grid_mask: np.ndarray) -> List[Tuple[GridBBox, int]]:
    """All 4-connected ON components as [(inclusive bbox, patch_count), ...].

    Dependency-free flood fill (the grid is tiny, ≤ ~32×32).  Order is row-major
    by each component's first-seen patch.
    """
    m = np.asarray(grid_mask, dtype=bool)
    out: List[Tuple[GridBBox, int]] = []
    if not m.any():
        return out
    n_rows, n_cols = m.shape
    seen = np.zeros_like(m)
    for sr in range(n_rows):
        for sc in range(n_cols):
            if not m[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            r0 = r1 = sr
            c0 = c1 = sc
            size = 0
            while stack:
                r, c = stack.pop()
                size += 1
                r0, r1 = min(r0, r), max(r1, r)
                c0, c1 = min(c0, c), max(c1, c)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols and m[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(((r0, c0, r1, c1), size))
    return out


def largest_component_bbox(grid_mask: np.ndarray) -> Optional[GridBBox]:
    """Inclusive bbox of the largest 4-connected ON component, or None if empty."""
    comps = all_component_bboxes(grid_mask)
    if not comps:
        return None
    return max(comps, key=lambda cb: cb[1])[0]


def pad_grid_bbox(
    box: GridBBox,
    grid_hw: Tuple[int, int],
    pad_frac: float,
    *,
    pad_min_patches: int = 1,
) -> GridBBox:
    """Expand an inclusive grid bbox by ``pad_frac`` of its own size, clipped.

    ``pad_min_patches`` floors the per-side pad (default 1 ⇒ always ≥1 patch of
    context).  Set 0 to allow a fully tight box.
    """
    n_rows, n_cols = grid_hw
    r0, c0, r1, c1 = box
    pr = max(int(pad_min_patches), int(round(pad_frac * (r1 - r0 + 1))))
    pc = max(int(pad_min_patches), int(round(pad_frac * (c1 - c0 + 1))))
    return (max(0, r0 - pr), max(0, c0 - pc),
            min(n_rows - 1, r1 + pr), min(n_cols - 1, c1 + pc))


def grid_bbox_to_frac(box: GridBBox, grid_hw: Tuple[int, int]) -> BBox:
    """Inclusive grid bbox → fractional (y0, x0, y1, x1) covering those whole cells."""
    n_rows, n_cols = grid_hw
    r0, c0, r1, c1 = box
    return (r0 / n_rows, c0 / n_cols, (r1 + 1) / n_rows, (c1 + 1) / n_cols)


def single_box_target(
    gt_pixels: Optional[np.ndarray],
    grid_hw: Tuple[int, int],
    *,
    pad_frac: float = 0.08,
    large_thresh: float = 0.75,
    patch_frac: float = 0.25,
    pad_min_patches: int = 1,
) -> Tuple[np.ndarray, Optional[GridBBox], str]:
    """Binary per-patch box target (N,) for one item.

    Returns (target (N,) float in {0,1}, padded grid bbox or None, kind), where
    kind ∈ {'box', 'large', 'no_gt'}:
      'box'   — target is 1 inside the padded largest-component bbox,
      'large' — padded box ≥ ``large_thresh`` of the frame ⇒ all-0 (don't zoom),
      'no_gt' — no GT on the grid ⇒ all-0.
    Padding is baked into the target on purpose: the head learns to emit a box
    that already carries authentic context, so the crop's contrast decode stays
    well-posed.
    """
    n_rows, n_cols = grid_hw
    n = n_rows * n_cols
    gm = gt_grid_mask(gt_pixels, grid_hw, patch_frac=patch_frac)
    box = largest_component_bbox(gm)
    if box is None:
        return np.zeros(n, dtype=np.float32), None, 'no_gt'

    pbox = pad_grid_bbox(box, grid_hw, pad_frac, pad_min_patches=pad_min_patches)
    r0, c0, r1, c1 = pbox
    area_frac = ((r1 - r0 + 1) * (c1 - c0 + 1)) / float(n)
    if area_frac >= large_thresh:
        return np.zeros(n, dtype=np.float32), None, 'large'

    tgt = np.zeros((n_rows, n_cols), dtype=np.float32)
    tgt[r0:r1 + 1, c0:c1 + 1] = 1.0
    return tgt.reshape(-1), pbox, 'box'


def box_from_heatmap(
    prob: np.ndarray,
    grid_hw: Tuple[int, int],
    *,
    thresh: float = 0.5,
    min_patches: int = 2,
) -> Tuple[Optional[BBox], Optional[GridBBox]]:
    """Read THE zoom box off a heatmap: threshold → largest component → its bbox.

    Absolute threshold (default 0.5, matching the {0,1} BCE target) so a uniformly
    cold heatmap — the "don't zoom" prediction for a large splice — yields no box
    instead of a spurious split.  Returns (fractional bbox, grid bbox) or (None, None).
    """
    n_rows, n_cols = grid_hw
    p = np.asarray(prob, dtype=np.float32).reshape(n_rows, n_cols)
    on = p >= float(thresh)
    if int(on.sum()) < int(min_patches):
        return None, None
    gbox = largest_component_bbox(on)
    if gbox is None:
        return None, None
    return grid_bbox_to_frac(gbox, grid_hw), gbox


def multi_box_target(
    gt_pixels: Optional[np.ndarray],
    grid_hw: Tuple[int, int],
    *,
    pad_frac: float = 0.08,
    large_thresh: float = 0.75,
    patch_frac: float = 0.25,
    pad_min_patches: int = 1,
    min_component_patches: int = 1,
) -> Tuple[np.ndarray, List[GridBBox], str]:
    """Per-patch box target lighting up EVERY GT component (each padded), not just
    the largest.

    Returns (target (N,) float in {0,1}, list of padded grid bboxes, kind).
    A component is dropped (left as background) when its padded box would cover
    ≥ ``large_thresh`` of the frame (large splice — don't zoom it) or it is smaller
    than ``min_component_patches``.  kind ∈ {'box', 'large', 'no_gt'}: 'box' if any
    component survives, 'large' if components exist but all were dropped, 'no_gt'
    if there is no GT on the grid.
    """
    n_rows, n_cols = grid_hw
    n = n_rows * n_cols
    gm = gt_grid_mask(gt_pixels, grid_hw, patch_frac=patch_frac)
    comps = all_component_bboxes(gm)
    if not comps:
        return np.zeros(n, dtype=np.float32), [], 'no_gt'

    tgt = np.zeros((n_rows, n_cols), dtype=np.float32)
    boxes: List[GridBBox] = []
    for cbox, size in comps:
        if size < int(min_component_patches):
            continue
        pbox = pad_grid_bbox(cbox, grid_hw, pad_frac, pad_min_patches=pad_min_patches)
        r0, c0, r1, c1 = pbox
        if ((r1 - r0 + 1) * (c1 - c0 + 1)) / float(n) >= large_thresh:
            continue
        tgt[r0:r1 + 1, c0:c1 + 1] = 1.0
        boxes.append(pbox)

    kind = 'box' if boxes else 'large'
    return tgt.reshape(-1), boxes, kind


def _dilate8(mask: np.ndarray, k: int) -> np.ndarray:
    """8-connected binary dilation by ``k`` patches (numpy-only, no scipy)."""
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(k))):
        o = out.copy()
        o[1:, :] |= out[:-1, :]; o[:-1, :] |= out[1:, :]
        o[:, 1:] |= out[:, :-1]; o[:, :-1] |= out[:, 1:]
        o[1:, 1:] |= out[:-1, :-1]; o[:-1, :-1] |= out[1:, 1:]
        o[1:, :-1] |= out[:-1, 1:]; o[:-1, 1:] |= out[1:, :-1]
        out = o
    return out


def coverage_target(
    gt_pixels: Optional[np.ndarray],
    grid_hw: Tuple[int, int],
    *,
    pad_patches: int = 1,
    large_thresh: float = 0.75,
    patch_frac: float = 0.25,
) -> Tuple[np.ndarray, str]:
    """Per-patch coverage target (N,) = the GT splice SHAPE, grown for context.

    NOT one filled bbox per component — that quilt of abutting rectangles is the
    pathological "insane GT".  Instead: downsample GT → grow by ``pad_patches``
    (8-conn dilation, the learned context margin, following the splice shape) →
    drop any connected region whose bbox is ≥ ``large_thresh`` of the frame (large
    splice ⇒ don't zoom).  Grouping into boxes is left entirely to read-off.

    Returns (target (N,) float in {0,1}, kind ∈ {'box','large','no_gt'}).
    """
    n_rows, n_cols = grid_hw
    n = n_rows * n_cols
    gm = gt_grid_mask(gt_pixels, grid_hw, patch_frac=patch_frac)
    if not gm.any():
        return np.zeros(n, dtype=np.float32), 'no_gt'

    cov = _dilate8(gm, pad_patches)
    kept = np.zeros_like(cov)
    seen = np.zeros_like(cov)
    saw_large = False
    for sr in range(n_rows):
        for sc in range(n_cols):
            if not cov[sr, sc] or seen[sr, sc]:
                continue
            stack = [(sr, sc)]
            seen[sr, sc] = True
            coords = []
            r0 = r1 = sr
            c0 = c1 = sc
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                r0, r1 = min(r0, r), max(r1, r)
                c0, c1 = min(c0, c), max(c1, c)
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols and cov[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if ((r1 - r0 + 1) * (c1 - c0 + 1)) / float(n) >= large_thresh:
                saw_large = True
                continue
            for (r, c) in coords:
                kept[r, c] = True

    if not kept.any():
        return np.zeros(n, dtype=np.float32), ('large' if saw_large else 'no_gt')
    return kept.reshape(-1).astype(np.float32), 'box'


def boxes_from_heatmap(
    prob: np.ndarray,
    grid_hw: Tuple[int, int],
    *,
    thresh: float = 0.5,
    min_patches: int = 2,
) -> List[BBox]:
    """Read EVERY box off a heatmap: threshold → each component ≥ ``min_patches``
    → its bbox.  Returns a (possibly empty) list of fractional boxes.
    """
    n_rows, n_cols = grid_hw
    p = np.asarray(prob, dtype=np.float32).reshape(n_rows, n_cols)
    on = p >= float(thresh)
    out: List[BBox] = []
    for cbox, size in all_component_bboxes(on):
        if size < int(min_patches):
            continue
        out.append(grid_bbox_to_frac(cbox, grid_hw))
    return out
