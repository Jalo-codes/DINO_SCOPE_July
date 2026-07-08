"""lab_utils.data.crop_conditions — window sampling for region-probe conditions.

TORCH-FREE (GAMEPLAN C3). PIL + numpy + stdlib only.

Shared geometry core for the BCE-emergence probe conditions (ai_interior,
sp_interior, ai_boundary, sp_boundary, fr_bg, real_crop — see
datasets/region_probes.py) and for the train-time fr-background negative
sampler (Dataset.fr_bg_negative_prob).

Everything that makes the probe conditions comparable lives HERE, once:
erosion margins, the native-pixel size floor, the ratio band, the boundary
in/out band, and the deterministic per-item RNG.  Builders and Dataset call
these functions; they never roll their own windows.

Windows are fractional ``(y0, x0, y1, x1)`` in [0, 1] of the source image
frame (same convention as lab_utils/eval/zoom.py bboxes), so they survive any
later resize.  ``apply_crop_window`` converts to a pixel box and crops.

Determinism: window sampling is driven by ``rng_for(item_id, group)`` — an
``random.Random`` seeded from (WINDOW_SPEC.version, item_id, group).  The same
item always yields the same windows, on any machine, in any process.  The
paired real_crop condition re-derives the *interior* group windows of its
parent item, so fake crop and real crop share identical geometry by
construction.

Sizing reference: the sampling range for interior/boundary windows is anchored
to ``best_inscribed_side`` — the best achievable window side across the whole
aspect-ratio band, NOT a single inscribed square.  A pure square badly
underestimates the usable area of an elongated or irregular mask (e.g. a
horizontal strip fits a wide-short rectangle far larger than any square it
contains), which otherwise collapses every sampled window down to the native
floor regardless of the object's true extent, and starves the position-search
grid down to one or two cells (the ``max_overlap_frac`` de-dup below then has
nothing to pick from and hands back near-identical windows). ``_sample_distinct``
additionally rejects a candidate window that overlaps (IoU) an already-accepted
window for the same item beyond ``max_overlap_frac`` — if a mask genuinely
can't support ``windows_per_item`` sufficiently distinct crops, fewer are
returned rather than padding with duplicates.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import random
from typing import List, Optional, Sequence, Tuple, TypeVar

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class WindowSpec:
    """Versioned probe-window parameters. Change → bump version (results keyed on it).

    Sizes are expressed in NATIVE pixels of the source image — the anti-upsample
    floor is about real information content, not mask fraction (a 5% splice in a
    2 MP frame is croppable; the same 5% at 700 px is not).
    """
    version: str = 'v3'
    # Window side must be >= this multiple of the eval resolution so model
    # pixels are (approximately) not interpolated. 1.0 => side >= image_size.
    min_side_mult: float = 1.0
    # Interior windows: side sampled in [min_side_frac_of_max * best_inscribed,
    # best_inscribed] (subject to the native floor) — moderate minimum area,
    # never deterministically the max box (object-core salience bias).
    # best_inscribed = best_inscribed_side(), NOT a pure square (see module
    # docstring) — a square badly underestimates elongated/irregular masks.
    min_side_frac_of_max: float = 0.60
    # Aspect band shared by every condition (mirrors the train crop band).
    ratio_range: Tuple[float, float] = (0.60, 1.70)
    # Boundary windows: in-mask fill fraction band (~half in / half out).
    boundary_in_range: Tuple[float, float] = (0.35, 0.65)
    # Outside/background windows: side sampled in [floor, floor * this].
    outside_side_mult_range: Tuple[float, float] = (1.0, 1.6)
    # Windows drawn per parent item per condition group. Deliberately higher
    # than a "just get 2" default: the interior floor gate only lets a
    # fraction of any parent pool's items through at all, so hitting a
    # meaningful total crop count means asking each PASSING item for several
    # sub-crops. _sample_distinct's overlap dedup makes this safe — an item
    # whose eroded region can't support this many distinct crops just returns
    # fewer, never a padded duplicate.
    windows_per_item: int = 10
    # Size draws attempted before giving up on this window slot.
    size_tries: int = 16
    # Position re-draws per size draw before concluding this size can't yield
    # a sufficiently distinct window (see max_overlap_frac).
    position_tries: int = 8
    # Max allowed IoU (native pixel boxes) between a candidate window and any
    # already-accepted window for the same item/group — the de-dup guard.
    max_overlap_frac: float = 0.30


WINDOW_SPEC = WindowSpec()

# Eval-probe-only spec: the region-probe conditions (datasets/region_probes.py)
# are never used for training, so they can tolerate real upsampling in
# exchange for a much higher floor pass rate — real box numbers showed only
# ~9%/1%/0.4% of sagid/IMD2020/tgif2 items clearing a full-resolution
# (min_side_mult=1.0) floor. WINDOW_SPEC itself stays strict (min_side_mult=
# 1.0, true anti-upsample) because it's also the default for the train-time
# fr-background-negative sampler (Dataset.fr_bg_negative_prob) — crops that
# land in actual gradient updates should not be pushed further from native
# resolution than necessary. 256/448 ≈ a 256px floor at the study's current
# 448 eval resolution (raised from an initial 144px once max_probes capping
# made the resulting probe counts sane — 256 buys back some crop quality
# while still passing far more candidates than the strict 448px floor;
# per Jake, 2026-07-08).
PROBE_WINDOW_SPEC = dataclasses.replace(WINDOW_SPEC, version='v5', min_side_mult=256.0 / 448.0)


# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------

def rng_for(item_id: str, group: str, spec: WindowSpec = WINDOW_SPEC) -> random.Random:
    """Deterministic RNG for one (item, condition-group) pair.

    Groups: 'interior' (shared by *_interior AND real_crop — identical paired
    windows), 'boundary', 'outside'.
    """
    raw = f'{spec.version}|{item_id}|{group}'
    seed = int(hashlib.md5(raw.encode('utf-8')).hexdigest()[:12], 16)
    return random.Random(seed)


# ---------------------------------------------------------------------------
# Morphology (PIL — no scipy in the dependency set)
# ---------------------------------------------------------------------------

def erode_radius_px(img_wh: Tuple[int, int], res) -> int:
    """One model-patch width mapped back to native pixels of this image.

    The margin guards the interior labels against imprecise GT edges and
    blend halos: a window inside the eroded mask sits >= one patch width
    (at model scale) away from the annotated boundary.
    """
    W, H = int(img_wh[0]), int(img_wh[1])
    n_side = max(1, int(res.image_size) // int(res.patch_size))
    return max(1, int(math.ceil(min(W, H) / n_side)))


def _filter_mask(mask: np.ndarray, radius: int, *, maximum: bool) -> np.ndarray:
    """Binary erosion (maximum=False) / dilation (True) with a square kernel.

    Integral-image implementation — O(n) regardless of radius (PIL's rank
    filters are O(n * k^2), which at the radii erode_radius_px produces on
    multi-megapixel masks is far too slow for the train-time sampler).
    Borders are edge-replicated, so a mask touching the frame edge does not
    erode inward from the frame (the frame edge is not a splice boundary).
    """
    if radius <= 0:
        return mask.astype(bool)
    r = int(radius)
    k = 2 * r + 1
    padded = np.pad(mask.astype(bool), r, mode='edge')
    sums = _window_sums(_integral(padded), k, k)
    return (sums > 0) if maximum else (sums == k * k)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    return _filter_mask(mask, radius, maximum=False)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    return _filter_mask(mask, radius, maximum=True)


# ---------------------------------------------------------------------------
# Integral-image window sums (vectorized containment / fill checks)
# ---------------------------------------------------------------------------

def _integral(mask: np.ndarray) -> np.ndarray:
    """Zero-padded 2-D cumulative sum: ii[y, x] = sum(mask[:y, :x])."""
    m = mask.astype(np.int64)
    ii = np.zeros((m.shape[0] + 1, m.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = m.cumsum(0).cumsum(1)
    return ii


def _window_sums(ii: np.ndarray, h: int, w: int) -> np.ndarray:
    """Sum inside every h x w window; shape (H - h + 1, W - w + 1)."""
    return (ii[h:, w:] - ii[:-h, w:] - ii[h:, :-w] + ii[:-h, :-w])


def _largest_rect_side(ii: np.ndarray, H: int, W: int, ratio: float) -> int:
    """Largest h such that an (h, round(h*ratio)) window fits fully inside the
    mask described by integral image ``ii`` (0 if none fits)."""
    def fits(h: int) -> bool:
        if h <= 0 or h > H:
            return False
        w = max(1, int(round(h * ratio)))
        if w > W:
            return False
        return bool((_window_sums(ii, h, w) == h * w).any())

    lo, hi = 0, H
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def largest_square_side(mask: np.ndarray) -> int:
    """Side of the largest axis-aligned square fully inside ``mask`` (0 if none)."""
    if not mask.any():
        return 0
    H, W = mask.shape
    return _largest_rect_side(_integral(mask), H, W, 1.0)


def best_inscribed_side(
    mask: np.ndarray, ratio_range: Tuple[float, float] = (1.0, 1.0),
) -> int:
    """Largest achievable window side across the aspect-ratio band (0 if none).

    A pure inscribed SQUARE badly underestimates the usable area of an
    elongated or irregular mask — see module docstring. Probes a handful of
    ratios spanning ``ratio_range`` (plus 1.0) and returns the best.
    """
    if not mask.any():
        return 0
    H, W = mask.shape
    ii = _integral(mask)
    lo, hi = ratio_range
    ratios = sorted({lo, 1.0, hi} if lo <= 1.0 <= hi else {lo, hi})
    return max(_largest_rect_side(ii, H, W, r) for r in ratios)


def _pick_positions(
    valid: np.ndarray, rng: random.Random, n: int,
) -> List[Tuple[int, int]]:
    """Up to ``n`` DISTINCT True cells of a (Y, X) validity grid, no replacement.

    ``np.nonzero`` is the expensive O(H*W) part — call it once per size draw
    and hand every position try a slice of the same result, rather than
    re-scanning the grid on every retry.
    """
    ys, xs = np.nonzero(valid)
    n_pos = len(ys)
    if n_pos == 0:
        return []
    idxs = rng.sample(range(n_pos), min(n, n_pos))
    return [(int(ys[i]), int(xs[i])) for i in idxs]


def _iou_native(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """IoU of two native-pixel (y0, x0, y1, x1) boxes."""
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    ih, iw = max(0, iy1 - iy0), max(0, ix1 - ix0)
    inter = ih * iw
    if inter <= 0:
        return 0.0
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    denom = area_a + area_b - inter
    return (inter / denom) if denom > 0 else 0.0


def _sample_distinct(
    rng: random.Random,
    match_fn,
    side_lo: int,
    side_hi: int,
    shape_hw: Tuple[int, int],
    res,
    group: str,
    k: int,
    spec: WindowSpec,
) -> List[ProbeWindow]:
    """Shared accept loop for the three condition samplers (see module docstring
    for why de-dup is needed). Returns fewer than ``k`` windows rather than a
    near-duplicate if the mask can't support that many distinct crops."""
    H, W = shape_hw
    accepted: List[Tuple[int, int, int, int]] = []
    out: List[ProbeWindow] = []
    for i in range(k):
        chosen = None
        for _try in range(spec.size_tries):
            h, w = _draw_hw(rng, side_lo, side_hi, (H, W), spec)
            if h > H or w > W:
                continue
            valid = match_fn(h, w)
            for pos in _pick_positions(valid, rng, spec.position_tries):
                box = (pos[0], pos[1], pos[0] + h, pos[1] + w)
                if all(_iou_native(box, b) <= spec.max_overlap_frac for b in accepted):
                    chosen = (pos, h, w)
                    break
            if chosen is not None:
                break
        if chosen is None:
            break  # mask can't support another sufficiently distinct window
        pos, h, w = chosen
        accepted.append((pos[0], pos[1], pos[0] + h, pos[1] + w))
        out.append(_to_probe_window(pos[0], pos[1], h, w, (H, W), res, group, i))
    return out


# ---------------------------------------------------------------------------
# Window record
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ProbeWindow:
    """One sampled window on a source image."""
    window: Tuple[float, float, float, float]   # fractional (y0, x0, y1, x1)
    native_wh: Tuple[int, int]                  # (w, h) native pixels
    upsample_factor: float                      # image_size / min(w, h)
    group: str                                  # 'interior' | 'boundary' | 'outside'
    index: int                                  # draw index within the item

    def meta(self, spec: WindowSpec = WINDOW_SPEC) -> dict:
        return {
            'crop_window':     tuple(float(v) for v in self.window),
            'window_native_wh': tuple(int(v) for v in self.native_wh),
            'upsample_factor': float(self.upsample_factor),
            'window_group':    self.group,
            'window_index':    int(self.index),
            'window_spec':     spec.version,
        }


def _to_probe_window(
    y: int, x: int, h: int, w: int,
    shape_hw: Tuple[int, int],
    res,
    group: str,
    index: int,
) -> ProbeWindow:
    H, W = shape_hw
    return ProbeWindow(
        window=(y / H, x / W, (y + h) / H, (x + w) / W),
        native_wh=(w, h),
        upsample_factor=float(res.image_size) / float(min(w, h)),
        group=group,
        index=index,
    )


def _draw_hw(
    rng: random.Random,
    side_lo: int,
    side_hi: int,
    max_hw: Tuple[int, int],
    spec: WindowSpec,
) -> Tuple[int, int]:
    """Draw (h, w): h uniform in [side_lo, side_hi], w = h * ratio, both clamped."""
    H, W = max_hw
    h = int(round(rng.uniform(side_lo, min(side_hi, H))))
    h = max(1, min(h, H))
    ratio = rng.uniform(*spec.ratio_range)
    w = int(round(h * ratio))
    w = max(side_lo if side_lo <= W else W, min(w, W))
    w = max(1, w)
    return h, w


# ---------------------------------------------------------------------------
# Samplers (one per condition group)
# ---------------------------------------------------------------------------

def sample_interior_windows(
    mask: np.ndarray,
    res,
    *,
    item_id: str,
    spec: WindowSpec = WINDOW_SPEC,
    k: Optional[int] = None,
) -> List[ProbeWindow]:
    """Windows fully inside the eroded mask — zero background, zero boundary.

    Returns [] when the eroded region cannot fit a window at the native floor
    (this is the gate that excludes small/tiny splices and low-res sources).
    """
    H, W = mask.shape
    floor = int(round(spec.min_side_mult * res.image_size))
    radius = erode_radius_px((W, H), res)

    # Cheap pre-filter: erosion can only shrink extent, never grow it, so no
    # eroded rectangle >= floor can exist unless the RAW mask's own bounding
    # box (minus ~2*radius, one erosion margin per side) already clears it.
    # One np.nonzero pass vs. the ~35 array ops the full erode + multi-ratio
    # inscribed-rectangle search costs below — the overwhelming majority of
    # real candidates fail the floor outright, so this matters at scale
    # (searching thousands of parent items across a large source pool).
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    bbox_h = int(ys.max() - ys.min()) + 1
    bbox_w = int(xs.max() - xs.min()) + 1
    if bbox_h - 2 * radius < floor or bbox_w - 2 * radius < floor:
        return []

    eroded = erode_mask(mask.astype(bool), radius)
    if not eroded.any():
        return []

    max_sq = best_inscribed_side(eroded, spec.ratio_range)
    if max_sq < floor:
        return []

    rng = rng_for(item_id, 'interior', spec)
    ii = _integral(eroded)
    side_lo = max(floor, int(round(spec.min_side_frac_of_max * max_sq)))

    def match_fn(h: int, w: int) -> np.ndarray:
        return _window_sums(ii, h, w) == h * w      # full containment

    return _sample_distinct(
        rng, match_fn, side_lo, max_sq, (H, W), res, 'interior',
        k if k is not None else spec.windows_per_item, spec,
    )


def sample_boundary_windows(
    mask: np.ndarray,
    res,
    *,
    item_id: str,
    spec: WindowSpec = WINDOW_SPEC,
    k: Optional[int] = None,
) -> List[ProbeWindow]:
    """Windows straddling the mask boundary: in-mask fill in boundary_in_range."""
    H, W = mask.shape
    m = mask.astype(bool)
    if not m.any() or m.all():
        return []

    floor = int(round(spec.min_side_mult * res.image_size))
    if floor > min(H, W):
        return []
    # Scale anchor: the mask's own best inscribed side (across the ratio band)
    # when it beats the floor, else the floor — keeps boundary sizes kin to
    # interior sizes on the same item while still allowing boundary probes on
    # items too small for interior.
    max_sq = max(best_inscribed_side(m, spec.ratio_range), floor)
    side_hi = min(max_sq, min(H, W))

    rng = rng_for(item_id, 'boundary', spec)
    ii = _integral(m)
    lo_f, hi_f = spec.boundary_in_range

    def match_fn(h: int, w: int) -> np.ndarray:
        frac = _window_sums(ii, h, w) / float(h * w)
        return (frac >= lo_f) & (frac <= hi_f)

    return _sample_distinct(
        rng, match_fn, floor, side_hi, (H, W), res, 'boundary',
        k if k is not None else spec.windows_per_item, spec,
    )


def sample_outside_windows(
    mask: np.ndarray,
    res,
    *,
    item_id: str,
    spec: WindowSpec = WINDOW_SPEC,
    k: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[ProbeWindow]:
    """Windows fully outside the dilated mask — background only, no edit, no boundary.

    Works on an all-False mask too (whole frame is eligible) — used both for
    fr_bg probes (crop away from the edit) and the train-time fr-background
    negative sampler (which passes its own non-deterministic ``rng`` so each
    epoch sees fresh windows; probe builders leave rng unset for determinism).
    """
    H, W = mask.shape
    radius = erode_radius_px((W, H), res)
    dilated = dilate_mask(mask.astype(bool), radius) if mask.any() else mask.astype(bool)

    floor = int(round(spec.min_side_mult * res.image_size))
    if floor > min(H, W):
        return []

    rng = rng if rng is not None else rng_for(item_id, 'outside', spec)
    ii = _integral(dilated)
    lo_m, hi_m = spec.outside_side_mult_range
    side_lo = floor
    side_hi = max(side_lo, min(int(round(floor * hi_m)), min(H, W)))

    def match_fn(h: int, w: int) -> np.ndarray:
        return _window_sums(ii, h, w) == 0            # zero mask contact

    return _sample_distinct(
        rng, match_fn, side_lo, side_hi, (H, W), res, 'outside',
        k if k is not None else spec.windows_per_item, spec,
    )


# ---------------------------------------------------------------------------
# Output-size capping (breadth over depth)
# ---------------------------------------------------------------------------

_T = TypeVar('_T')


def breadth_first_cap(groups: Sequence[Sequence[_T]], max_total: int) -> List[_T]:
    """Flatten per-parent element lists breadth-first, capped at max_total.

    Round r takes each group's r-th element before any group's (r+1)-th, so
    when the cap is hit it thins per-group depth, never how many distinct
    groups (parent images) are represented. Used by
    datasets/region_probes.py to bound total emitted probes: PROBE_WINDOW_SPEC
    (see its docstring) passes a much larger fraction of candidates than the
    strict default, so a large parent pool times windows_per_item can
    otherwise emit far more probes than an eval run needs.

    Deterministic: iterates ``groups`` and each group's elements in the order
    given — same input, same output, every run.
    """
    out: List[_T] = []
    round_idx = 0
    while len(out) < max_total:
        progressed = False
        for g in groups:
            if round_idx >= len(g):
                continue
            progressed = True
            out.append(g[round_idx])
            if len(out) >= max_total:
                break
        if not progressed:
            break
        round_idx += 1
    return out


# ---------------------------------------------------------------------------
# Window application (the one crop implementation for every load path)
# ---------------------------------------------------------------------------

def apply_crop_window(pil: Image.Image, window: Tuple[float, float, float, float]) -> Image.Image:
    """Crop a PIL image to a fractional (y0, x0, y1, x1) window of ITS OWN frame.

    Used by Dataset._build_sample, eval/preprocess, and eval/metric GT loading —
    one rounding rule everywhere, so image and mask crops stay aligned even
    when the mask file is a different resolution than the image (fractions are
    resolution-invariant; both sides round identically relative to their frame).
    """
    y0, x0, y1, x1 = window
    W, H = pil.size

    def _half_up(v: float) -> int:
        # round-half-up (not banker's): fractional edges land on identical
        # relative pixels across image/mask resolutions.
        return int(math.floor(v + 0.5))

    px0 = max(0, min(W - 1, _half_up(x0 * W)))
    py0 = max(0, min(H - 1, _half_up(y0 * H)))
    px1 = max(px0 + 1, min(W, _half_up(x1 * W)))
    py1 = max(py0 + 1, min(H, _half_up(y1 * H)))
    return pil.crop((px0, py0, px1, py1))
