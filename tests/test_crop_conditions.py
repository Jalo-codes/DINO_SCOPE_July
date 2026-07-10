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
    PROBE_WINDOW_SPEC,
    WINDOW_SPEC,
    WindowSpec,
    apply_crop_window,
    best_inscribed_side,
    breadth_first_cap,
    dilate_mask,
    erode_mask,
    erode_radius_px,
    largest_square_side,
    rng_for,
    sample_boundary_windows,
    sample_interior_windows,
    sample_outside_windows,
    sample_outside_windows_sized,
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


def test_best_inscribed_side_beats_square_on_elongated_mask():
    # A wedge, wide at the top (width 300) tapering to width 20 at the bottom.
    # A square starting at the top is capped by how far down it can go before
    # the taper narrows past its own side; a narrower-than-square rectangle
    # (low end of ratio_range: less width needed per row) tolerates more
    # taper and reaches further down — a real advantage within the actual
    # production ratio band, unlike a straight strip (which no in-band ratio
    # can exploit, since the short axis caps the window regardless of shape).
    H, W = 300, 300
    m = np.zeros((H, W), dtype=bool)
    for y in range(H):
        width_at_y = max(20, W - y)
        m[y, :width_at_y] = True
    assert best_inscribed_side(m, (0.6, 1.7)) > largest_square_side(m)


def test_best_inscribed_side_matches_square_on_square_mask():
    m = _rect_mask(200, 300, 50, 60, 150, 220)   # 100 x 160 rect
    assert best_inscribed_side(m, (1.0, 1.0)) == largest_square_side(m)


def test_erode_radius_scales_with_native_size():
    # one patch width at model res, mapped to native pixels: min_dim / n_side
    assert erode_radius_px((448, 448), _Res(448, 16)) == 16
    assert erode_radius_px((1120, 2000), _Res(448, 16)) == 40


# ── interior ─────────────────────────────────────────────────────────────────

def test_interior_windows_fully_inside_mask():
    m = _rect_mask(600, 600, 100, 100, 500, 500)    # 400px region >> floor 64
    wins = sample_interior_windows(m, RES, item_id='item-a')
    # De-dup (max_overlap_frac) means count is demand-driven, not guaranteed to
    # hit windows_per_item — see test_interior_windows_are_mutually_distinct
    # for a geometry with enough room to actually deliver windows_per_item.
    assert 1 <= len(wins) <= WINDOW_SPEC.windows_per_item
    for w in wins:
        y0, x0, y1, x1 = _px_box(w.window, m.shape)
        assert m[y0:y1, x0:x1].all(), 'interior window leaked outside the mask'
        assert w.native_wh[0] >= RES.image_size or w.native_wh[1] >= RES.image_size
        assert min(w.native_wh) >= int(WINDOW_SPEC.min_side_mult * RES.image_size)
        assert w.upsample_factor <= 1.0 + 1e-6


def test_interior_windows_are_mutually_distinct():
    # Generous mask (large relative to the erosion margin) plus a lower
    # min_side_frac_of_max (more size variety, so small-and-far-apart windows
    # are findable) — enough room for SEVERAL genuinely non-overlapping
    # crops. windows_per_item is a ceiling, not a guarantee (see
    # _sample_distinct): once the mask runs out of sufficiently distinct
    # room it stops rather than padding with near-duplicates, so a mask this
    # size finds several but not necessarily all windows_per_item of them.
    m = _rect_mask(3000, 3000, 50, 50, 2950, 2950)
    spec = WindowSpec(min_side_frac_of_max=0.3)
    wins = sample_interior_windows(m, RES, item_id='item-roomy', spec=spec)
    assert 2 <= len(wins) <= spec.windows_per_item
    boxes = [_px_box(w.window, m.shape) for w in wins]

    def iou(a, b):
        ay0, ax0, ay1, ax1 = a
        by0, bx0, by1, bx1 = b
        iy0, ix0 = max(ay0, by0), max(ax0, bx0)
        iy1, ix1 = min(ay1, by1), min(ax1, bx1)
        ih, iw = max(0, iy1 - iy0), max(0, ix1 - ix0)
        inter = ih * iw
        if inter == 0:
            return 0.0
        area_a = (ay1 - ay0) * (ax1 - ax0)
        area_b = (by1 - by0) * (bx1 - bx0)
        return inter / (area_a + area_b - inter)

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            assert iou(boxes[i], boxes[j]) <= spec.max_overlap_frac + 1e-6, \
                'sampler returned near-duplicate windows for a roomy mask'


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


def test_interior_bbox_prefilter_matches_full_gate_on_large_frame():
    # A tiny mask on a LARGE frame (the common real-world case: a small local
    # edit on a multi-megapixel photo) exercises the cheap raw-bbox pre-filter
    # (bbox - 2*radius < floor) rather than the frame-size-only paths the
    # other tests use. Must reject exactly like the full erode+search path
    # would, just without paying for it.
    m = _rect_mask(4000, 4000, 1000, 1000, 1030, 1030)   # 30px region
    assert sample_interior_windows(m, RES, item_id='item-tiny-on-huge') == []


def test_probe_window_spec_accepts_what_default_spec_rejects():
    # PROBE_WINDOW_SPEC's looser floor (256/448 of eval res, vs. the default
    # 1.0x) is eval-probe-only -- exercised here on a mask whose eroded
    # region lands strictly between the two floors: too small for the
    # strict default, large enough for the looser probe spec.
    from lab_utils.data.crop_conditions import PROBE_WINDOW_SPEC
    assert PROBE_WINDOW_SPEC.min_side_mult < WINDOW_SPEC.min_side_mult
    assert PROBE_WINDOW_SPEC.version != WINDOW_SPEC.version

    m = _rect_mask(400, 400, 75, 75, 325, 325)   # 250x250 region, eroded to ~50px
    assert sample_interior_windows(m, RES, item_id='split-spec', spec=WINDOW_SPEC) == []
    loose = sample_interior_windows(m, RES, item_id='split-spec', spec=PROBE_WINDOW_SPEC)
    assert loose, 'PROBE_WINDOW_SPEC should accept a region the strict default rejects'


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


def test_outside_sized_draws_only_pool_sizes_and_avoids_mask():
    m = _rect_mask(600, 600, 200, 200, 320, 320)
    radius = erode_radius_px((600, 600), RES)
    dl = dilate_mask(m, radius)
    pool = [(90, 110), (130, 100), (70, 70)]
    wins = sample_outside_windows_sized(m, RES, item_id='item-s', size_pool=pool)
    assert wins
    for w in wins:
        assert w.group == 'outside_matched'
        w_px, h_px = w.native_wh
        assert (h_px, w_px) in pool, 'emitted size not drawn from the pool'
        y0, x0, y1, x1 = _px_box(w.window, m.shape)
        assert not dl[y0:y1, x0:x1].any(), 'sized window touched the dilated mask'


def test_outside_sized_deterministic_and_item_keyed():
    m = _rect_mask(600, 600, 200, 200, 320, 320)
    pool = [(90, 110), (130, 100), (70, 70)]
    a = sample_outside_windows_sized(m, RES, item_id='same', size_pool=pool)
    b = sample_outside_windows_sized(m, RES, item_id='same', size_pool=pool)
    c = sample_outside_windows_sized(m, RES, item_id='other', size_pool=pool)
    assert [w.window for w in a] == [w.window for w in b]
    assert [w.window for w in a] != [w.window for w in c]


def test_outside_sized_degenerate_pools():
    m = _rect_mask(600, 600, 200, 200, 320, 320)
    assert sample_outside_windows_sized(m, RES, item_id='x', size_pool=[]) == []
    # entries that cannot fit the frame burn draws but never crash
    assert sample_outside_windows_sized(m, RES, item_id='x', size_pool=[(5000, 5000)]) == []


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
    spec_a = WindowSpec(version='test-a')
    spec_b = WindowSpec(version='test-b')
    a = sample_interior_windows(m, RES, item_id='it', spec=spec_a)
    b = sample_interior_windows(m, RES, item_id='it', spec=spec_b)
    assert [w.window for w in a] != [w.window for w in b]


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


# ── output-size capping ──────────────────────────────────────────────────────

class TestBreadthFirstCap:
    def test_under_cap_returns_everything(self):
        groups = [['a1', 'a2'], ['b1'], ['c1', 'c2', 'c3']]
        out = breadth_first_cap(groups, max_total=100)
        assert sorted(out) == ['a1', 'a2', 'b1', 'c1', 'c2', 'c3']

    def test_cap_prefers_breadth_over_depth(self):
        # 3 groups, cap=3: round 0 alone satisfies the cap, so every group
        # contributes exactly its FIRST element — none gets a second.
        groups = [['a1', 'a2', 'a3'], ['b1', 'b2'], ['c1']]
        out = breadth_first_cap(groups, max_total=3)
        assert out == ['a1', 'b1', 'c1']

    def test_cap_spills_into_second_round_when_needed(self):
        # cap=4 with only 3 groups: round 0 yields 3 (a1,b1,c1), round 1
        # yields 1 more from the first group that still has an element (a2).
        groups = [['a1', 'a2'], ['b1'], ['c1']]
        out = breadth_first_cap(groups, max_total=4)
        assert out == ['a1', 'b1', 'c1', 'a2']

    def test_exhausts_naturally_below_cap(self):
        groups = [['a1'], ['b1']]
        out = breadth_first_cap(groups, max_total=1000)
        assert out == ['a1', 'b1']

    def test_empty_groups_return_empty(self):
        assert breadth_first_cap([], max_total=10) == []
        assert breadth_first_cap([[], []], max_total=10) == []

    def test_deterministic_same_input_same_output(self):
        groups = [['a1', 'a2', 'a3'], ['b1', 'b2'], ['c1']]
        assert breadth_first_cap(groups, 4) == breadth_first_cap(groups, 4)
