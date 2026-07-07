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
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import random
from typing import List, Optional, Tuple

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
    version: str = 'v1'
    # Window side must be >= this multiple of the eval resolution so model
    # pixels are (approximately) not interpolated. 1.0 => side >= image_size.
    min_side_mult: float = 1.0
    # Interior windows: side sampled in [min_side_frac_of_max * max_square,
    # max_square] (subject to the native floor) — moderate minimum area,
    # never deterministically the max box (object-core salience bias).
    min_side_frac_of_max: float = 0.60
    # Aspect band shared by every condition (mirrors the train crop band).
    ratio_range: Tuple[float, float] = (0.60, 1.70)
    # Boundary windows: in-mask fill fraction band (~half in / half out).
    boundary_in_range: Tuple[float, float] = (0.35, 0.65)
    # Outside/background windows: side sampled in [floor, floor * this].
    outside_side_mult_range: Tuple[float, float] = (1.0, 1.6)
    # Windows drawn per parent item per condition group.
    windows_per_item: int = 2
    # Position-draw attempts per window before giving up on a size draw.
    size_tries: int = 8


WINDOW_SPEC = WindowSpec()


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


def largest_square_side(mask: np.ndarray) -> int:
    """Side of the largest axis-aligned square fully inside ``mask`` (0 if none)."""
    H, W = mask.shape
    if not mask.any():
        return 0
    ii = _integral(mask)

    def fits(s: int) -> bool:
        if s > H or s > W:
            return False
        return bool((_window_sums(ii, s, s) == s * s).any())

    lo, hi = 0, min(H, W)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _pick_position(valid: np.ndarray, rng: random.Random) -> Optional[Tuple[int, int]]:
    """Uniform draw over True cells of a (Y, X) validity grid; None if empty."""
    ys, xs = np.nonzero(valid)
    if len(ys) == 0:
        return None
    i = rng.randrange(len(ys))
    return int(ys[i]), int(xs[i])


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
    radius = erode_radius_px((W, H), res)
    eroded = erode_mask(mask.astype(bool), radius)
    if not eroded.any():
        return []

    floor = int(round(spec.min_side_mult * res.image_size))
    max_sq = largest_square_side(eroded)
    if max_sq < floor:
        return []

    rng = rng_for(item_id, 'interior', spec)
    ii = _integral(eroded)
    side_lo = max(floor, int(round(spec.min_side_frac_of_max * max_sq)))

    out: List[ProbeWindow] = []
    for i in range(k if k is not None else spec.windows_per_item):
        for _try in range(spec.size_tries):
            h, w = _draw_hw(rng, side_lo, max_sq, (H, W), spec)
            if h > H or w > W:
                continue
            valid = _window_sums(ii, h, w) == h * w      # full containment
            pos = _pick_position(valid, rng)
            if pos is not None:
                out.append(_to_probe_window(pos[0], pos[1], h, w, (H, W), res, 'interior', i))
                break
    return out


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
    # Scale anchor: the mask's own largest square when it beats the floor,
    # else the floor — keeps boundary sizes kin to interior sizes on the same
    # item while still allowing boundary probes on items too small for interior.
    max_sq = max(largest_square_side(m), floor)
    side_hi = min(max_sq, min(H, W))

    rng = rng_for(item_id, 'boundary', spec)
    ii = _integral(m)
    lo_f, hi_f = spec.boundary_in_range

    out: List[ProbeWindow] = []
    for i in range(k if k is not None else spec.windows_per_item):
        for _try in range(spec.size_tries):
            h, w = _draw_hw(rng, floor, side_hi, (H, W), spec)
            if h > H or w > W:
                continue
            frac = _window_sums(ii, h, w) / float(h * w)
            valid = (frac >= lo_f) & (frac <= hi_f)
            pos = _pick_position(valid, rng)
            if pos is not None:
                out.append(_to_probe_window(pos[0], pos[1], h, w, (H, W), res, 'boundary', i))
                break
    return out


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
    side_hi = min(int(round(floor * hi_m)), min(H, W))

    out: List[ProbeWindow] = []
    for i in range(k if k is not None else spec.windows_per_item):
        for _try in range(spec.size_tries):
            h, w = _draw_hw(rng, side_lo, max(side_lo, side_hi), (H, W), spec)
            if h > H or w > W:
                continue
            valid = _window_sums(ii, h, w) == 0            # zero mask contact
            pos = _pick_position(valid, rng)
            if pos is not None:
                out.append(_to_probe_window(pos[0], pos[1], h, w, (H, W), res, 'outside', i))
                break
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
