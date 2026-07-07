"""Tests for lab_utils.data.crop_conditions — torch-free tier.

Label-correctness invariants for the BCE-emergence probe windows:
  * interior windows contain ONLY mask pixels (with the erosion margin);
  * boundary windows contain both classes within the fill band;
  * outside windows contain ZERO (dilated) mask pixels;
  * the native floor rejects regions too small to crop without upsampling;
  * sampling is deterministic across independent runs (same item_id → same
    windows), which is what makes real_crop pairing and cross-cell probe
    identity work with no exported files.
"""

import numpy as np
import pytest
from PIL import Image

from lab_utils.data.crop_conditions import (
    WINDOW_SPEC,
    WindowSpec,
    apply_crop_window,
    dilate_mask,
    erode_mask,
    erode_radius_px,
    largest_square_side,
    rng_for,
    sample_boundary_windows,
    sample_interior_windows,
    sample_outside_windows,
)


class _Res:
    """Minimal Resolution stand-in (image_size, patch_size)."""
    def __init__(self, image_size=64, patch_size=16):
        self.image_size = image_size
        self.patch_size = patch_size


RES = _Res(image_size=64, patch_size=16)   # floor = 64 px, n_side = 4


def _rect_mask(H, W, y0, x0, y1, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _px_box(window, shape_hw):
    H, W = shape_hw
    y0, x0, y1, x1 = window
    return (int(round(y0 * H)), int(round(x0 * W)),
            int(round(y1 * H)), int(round(x1 * W)))


# ── geometry primitives ──────────────────────────────────────────────────────

def test_largest_square_side():
    m = _rect_mask(200, 300, 50, 60, 150, 220)      # 100 x 160 rect
    assert largest_square_side(m) == 100
    assert largest_square_side(np.zeros((50, 50), dtype=bool)) == 0
    assert largest_square_side(np.ones((30, 40), dtype=bool)) == 30


def test_erode_dilate_roundtrip_margin():
    m = _rect_mask(100, 100, 20, 20, 80, 80)
    er = erode_mask(m, 5)
    assert er.sum() < m.sum() and er[25:75, 25:75].all()
    assert not er[20, 20]                            # corner gone
    dl = dilate_mask(m, 5)
    assert dl.sum() > m.sum() and dl[15:85, 15:85].all()


def test_erode_radius_scales_with_native_size():
    # one patch width at model res, mapped to native pixels: min_dim / n_side
    assert erode_radius_px((448, 448), _Res(448, 16)) == 16
    assert erode_radius_px((1120, 2000), _Res(448, 16)) == 40


# ── interior ─────────────────────────────────────────────────────────────────

def test_interior_windows_fully_inside_mask():
    m = _rect_mask(600, 600, 100, 100, 500, 500)    # 400px region >> floor 64
    wins = sample_interior_windows(m, RES, item_id='item-a')
    assert len(wins) == WINDOW_SPEC.windows_per_item
    for w in wins:
        y0, x0, y1, x1 = _px_box(w.window, m.shape)
        assert m[y0:y1, x0:x1].all(), 'interior window leaked outside the mask'
        assert w.native_wh[0] >= RES.image_size or w.native_wh[1] >= RES.image_size
        assert min(w.native_wh) >= int(WINDOW_SPEC.min_side_mult * RES.image_size)
        assert w.upsample_factor <= 1.0 + 1e-6


def test_interior_respects_erosion_margin():
    m = _rect_mask(600, 600, 100, 100, 500, 500)
    radius = erode_radius_px((600, 600), RES)
    eroded = erode_mask(m, radius)
    for w in sample_interior_windows(m, RES, item_id='item-b'):
        y0, x0, y1, x1 = _px_box(w.window, m.shape)
        assert eroded[y0:y1, x0:x1].all(), 'window touched the erosion margin'


def test_interior_floor_gate_rejects_small_regions():
    # 40px region < 64px floor → no interior windows (small-splice gate)
    m = _rect_mask(600, 600, 100, 100, 140, 140)
    assert sample_interior_windows(m, RES, item_id='item-c') == []


def test_interior_never_max_box_deterministically():
    # Across several items, windows must vary in size/position (not always the
    # max inscribed square == whole rect here).
    m = _rect_mask(600, 600, 100, 100, 500, 500)
    seen = set()
    for iid in ('i1', 'i2', 'i3', 'i4'):
        for w in sample_interior_windows(m, RES, item_id=iid):
            seen.add((w.native_wh, _px_box(w.window, m.shape)[:2]))
    assert len(seen) > 1, 'sampler collapsed to a single deterministic box'


# ── boundary ─────────────────────────────────────────────────────────────────

def test_boundary_windows_straddle():
    m = _rect_mask(600, 600, 100, 100, 500, 500)
    wins = sample_boundary_windows(m, RES, item_id='item-d')
    assert wins, 'no boundary windows on an easy mask'
    lo, hi = WINDOW_SPEC.boundary_in_range
    for w in wins:
        y0, x0, y1, x1 = _px_box(w.window, m.shape)
        frac = m[y0:y1, x0:x1].mean()
        assert lo - 0.02 <= frac <= hi + 0.02, f'fill {frac:.2f} outside band'


def test_boundary_empty_and_full_masks_yield_nothing():
    assert sample_boundary_windows(np.zeros((300, 300), bool), RES, item_id='x') == []
    assert sample_boundary_windows(np.ones((300, 300), bool), RES, item_id='x') == []


# ── outside ──────────────────────────────────────────────────────────────────

def test_outside_windows_avoid_dilated_mask():
    m = _rect_mask(600, 600, 200, 200, 320, 320)
    radius = erode_radius_px((600, 600), RES)
    dl = dilate_mask(m, radius)
    wins = sample_outside_windows(m, RES, item_id='item-e')
    assert wins
    for w in wins:
        y0, x0, y1, x1 = _px_box(w.window, m.shape)
        assert not dl[y0:y1, x0:x1].any(), 'outside window touched the dilated mask'


def test_outside_on_empty_mask_uses_whole_frame():
    wins = sample_outside_windows(np.zeros((300, 300), bool), RES, item_id='item-f')
    assert wins


# ── determinism / pairing ────────────────────────────────────────────────────

def test_windows_deterministic_across_runs():
    m = _rect_mask(600, 600, 100, 100, 500, 500)
    a = sample_interior_windows(m, RES, item_id='same-item')
    b = sample_interior_windows(m, RES, item_id='same-item')
    assert [w.window for w in a] == [w.window for w in b]
    c = sample_interior_windows(m, RES, item_id='other-item')
    assert [w.window for w in a] != [w.window for w in c]


def test_rng_group_isolation():
    assert rng_for('id', 'interior').random() != rng_for('id', 'boundary').random()


def test_spec_version_changes_windows():
    m = _rect_mask(600, 600, 100, 100, 500, 500)
    v1 = sample_interior_windows(m, RES, item_id='it')
    v2 = sample_interior_windows(m, RES, item_id='it', spec=WindowSpec(version='v2'))
    assert [w.window for w in v1] != [w.window for w in v2]


# ── window application ───────────────────────────────────────────────────────

def test_apply_crop_window_fractional_consistency():
    # Same fractional window on image and half-res mask → aligned crops.
    img = Image.new('RGB', (400, 300))
    msk = Image.new('L', (200, 150))
    win = (0.25, 0.25, 0.75, 0.75)
    ci, cm = apply_crop_window(img, win), apply_crop_window(msk, win)
    assert ci.size == (200, 150)
    assert cm.size == (100, 75)
    assert abs(ci.size[0] / cm.size[0] - 2.0) < 0.05


def test_apply_crop_window_clamps():
    img = Image.new('RGB', (100, 100))
    out = apply_crop_window(img, (-0.1, -0.1, 1.2, 1.2))
    assert out.size == (100, 100)


# ── ratio band ───────────────────────────────────────────────────────────────

def test_ratio_band_respected():
    m = _rect_mask(900, 900, 50, 50, 850, 850)
    lo, hi = WINDOW_SPEC.ratio_range
    for iid in ('r1', 'r2', 'r3'):
        for w in (sample_interior_windows(m, RES, item_id=iid)
                  + sample_outside_windows(_rect_mask(900, 900, 0, 0, 60, 60), RES, item_id=iid)):
            wpx, hpx = w.native_wh
            # clamping at frame/region edges can push ratio slightly out; allow slack
            assert lo * 0.8 <= wpx / hpx <= hi * 1.25
