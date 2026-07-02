"""tests.test_zoom_geometry — numpy-only tests for the zoom geometry helpers.

Covers the §1 additions to lab_utils/eval/zoom.py:
  - attention_hot_mask
  - attention_to_bbox  (identity-regression vs the original inline logic)
  - peak_hot_component
  - place-back union property (what run_bbox_zoom composes)

No torch, no GPU, no datasets.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_utils.eval.zoom import (  # noqa: E402
    attention_hot_mask,
    attention_to_bbox,
    compute_gap_threshold,
    compute_otsu_threshold,
    grid_locked_box,
    peak_hot_component,
    place_mask_in_frame_pixels,
    _pad_bbox,
)
from lab_utils.eval.multibox import (  # noqa: E402
    cover_bboxes,
    gate_boxes_by_logit,
)


# ── reference: the ORIGINAL attention_to_bbox body, pre-refactor ────────────────

def _attention_to_bbox_reference(attention, grid_hw, *, percentile='otsu',
                                 pad_frac=0.10, min_box_size=8):
    n_rows, n_cols = grid_hw
    n = n_rows * n_cols
    attn = np.asarray(attention, dtype=np.float64).reshape(-1)[:n]
    if isinstance(percentile, str):
        if percentile.lower() == 'otsu':
            thresh = compute_otsu_threshold(attn) * 0.70
        elif percentile.lower() == 'gap':
            thresh = compute_gap_threshold(attn) * 0.70
        else:
            raise ValueError(percentile)
    else:
        thresh = float(np.percentile(attn, percentile))
    rows, cols = np.where((attn >= thresh).reshape(n_rows, n_cols))
    if rows.size == 0:
        return 0.0, 0.0, 1.0, 1.0
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    return _pad_bbox(r0, c0, r1, c1, n_rows, n_cols, pad_frac, min_box_size=min_box_size)


# ── helpers ─────────────────────────────────────────────────────────────────────

def _blob_attention(grid, blobs):
    """grid=(h,w); blobs=list of (r_slice, c_slice, value). Returns flat (N,) attn."""
    h, w = grid
    a = np.zeros((h, w), dtype=np.float64)
    for rs, cs, v in blobs:
        a[rs, cs] = v
    return a.reshape(-1)


# ── attention_hot_mask ──────────────────────────────────────────────────────────

def test_hot_mask_selects_blob():
    grid = (6, 6)
    attn = _blob_attention(grid, [(slice(1, 3), slice(1, 3), 1.0)])
    hot = attention_hot_mask(attn, grid, percentile='otsu')
    assert hot.shape == grid
    assert hot[1:3, 1:3].all()
    # everything outside the blob is cold
    expected = np.zeros(grid, dtype=bool)
    expected[1:3, 1:3] = True
    assert np.array_equal(hot, expected)


def test_hot_mask_empty_path():
    # The empty hot-set path is reached via the n<3 otsu guard (returns max+1),
    # so a tiny grid yields no hot patches.  (Perfectly-uniform large attention,
    # by contrast, degenerates to all-hot — not a real case for learned softmax.)
    grid = (1, 2)
    attn = np.array([0.5, 0.5], dtype=np.float64)
    assert not attention_hot_mask(attn, grid, percentile='otsu').any()


def test_hot_mask_peak_grabs_faint_halo_at_low_mult():
    # bright core (1.0) + faint halo (0.1) + dead background (0).
    grid = (8, 8)
    attn = _blob_attention(grid, [(slice(2, 6), slice(2, 6), 0.1),
                                  (slice(3, 5), slice(3, 5), 1.0)])
    strict = attention_hot_mask(attn, grid, percentile='peak', thresh_mult=0.5)
    broad  = attention_hot_mask(attn, grid, percentile='peak', thresh_mult=0.05)
    assert strict.sum() < broad.sum()
    assert broad[2:6, 2:6].all()      # low mult grabs the faint halo too
    assert not broad[0, 0]            # dead background (0) still excluded


def test_hot_mask_numeric_percentile_no_multiplier():
    grid = (4, 4)
    attn = np.arange(16, dtype=np.float64)
    hot = attention_hot_mask(attn, grid, percentile=75.0)
    # 75th percentile of 0..15 is 11.25; patches >= 11.25 → values 12..15 (4 patches)
    assert hot.sum() == 4


# ── attention_to_bbox identity regression ────────────────────────────────────────

@pytest.mark.parametrize('percentile', ['otsu', 'gap', 90.0, 75.0])
def test_attention_to_bbox_matches_reference(percentile):
    rng = np.random.default_rng(0)
    grid = (12, 12)
    for _ in range(50):
        attn = rng.random(144)
        got = attention_to_bbox(attn, grid, percentile=percentile)
        ref = _attention_to_bbox_reference(attn, grid, percentile=percentile)
        assert np.allclose(got, ref), f'{percentile}: {got} != {ref}'


def test_attention_to_bbox_min_pad_floor_grows_larger_box():
    # a box covering ~44% of the frame: base_pad is 0 here and inverse-area
    # scaling drives legacy padding to ~0; min_pad_frac floors it so the box
    # still gains a margin to recover under-lit splice borders.
    grid = (12, 12)
    attn = _blob_attention(grid, [(slice(2, 10), slice(2, 10), 1.0)])
    no_floor = attention_to_bbox(attn, grid, percentile='otsu', pad_frac=0.10,
                                 min_box_size=0, min_pad_frac=0.0)
    floored = attention_to_bbox(attn, grid, percentile='otsu', pad_frac=0.10,
                                min_box_size=0, min_pad_frac=0.15)
    assert (floored[2] - floored[0]) > (no_floor[2] - no_floor[0])
    assert (floored[3] - floored[1]) > (no_floor[3] - no_floor[1])


def test_attention_to_bbox_empty_returns_full_frame():
    grid = (6, 6)
    attn = np.full(36, 1.0 / 36, dtype=np.float64)
    assert attention_to_bbox(attn, grid, percentile='otsu') == (0.0, 0.0, 1.0, 1.0)


# ── grid_locked_box (box policy: center pinned to a patch cell) ──────────────────

def test_grid_locked_box_centers_on_patch():
    # patch (r=2, c=3) on a 10x10 grid: center = (2.5/10, 3.5/10) = (0.25, 0.35).
    y0, x0, y1, x1 = grid_locked_box(2 * 10 + 3, 0.2, 0.2, (10, 10))
    assert abs(0.5 * (y0 + y1) - 0.25) < 1e-9
    assert abs(0.5 * (x0 + x1) - 0.35) < 1e-9
    assert abs((y1 - y0) - 0.2) < 1e-9 and abs((x1 - x0) - 0.2) < 1e-9


def test_grid_locked_box_clips_to_frame():
    # a corner patch with a big extent must clip into [0, 1], staying valid.
    y0, x0, y1, x1 = grid_locked_box(0, 0.8, 0.8, (10, 10))
    assert 0.0 <= y0 < y1 <= 1.0 and 0.0 <= x0 < x1 <= 1.0
    assert y0 == 0.0 and x0 == 0.0           # clamped at the top-left


def test_grid_locked_box_distinct_cells_distinct_centers():
    a = grid_locked_box(0, 0.1, 0.1, (8, 8))
    b = grid_locked_box(8 + 1, 0.1, 0.1, (8, 8))   # patch (1, 1)
    assert a != b


# ── peak_hot_component ───────────────────────────────────────────────────────────

def test_peak_component_isolates_peak_blob():
    grid = (8, 8)
    # blob A (earlier, hotter) and a spatially separate blob B
    attn = _blob_attention(grid, [
        (slice(1, 3), slice(1, 3), 1.0),   # A — contains the peak
        (slice(5, 7), slice(5, 7), 0.8),   # B — separate, cooler
    ])
    comp = peak_hot_component(attn, grid, percentile='otsu')
    # A fully returned
    assert comp[1:3, 1:3].all()
    # B fully excluded — this is the point: hide A, B survives for the re-pool
    assert not comp[5:7, 5:7].any()


def test_peak_component_empty_when_nothing_hot():
    grid = (1, 2)
    attn = np.array([0.5, 0.5], dtype=np.float64)
    assert not peak_hot_component(attn, grid, percentile='otsu').any()


def test_peak_component_single_blob_returns_it():
    grid = (6, 6)
    attn = _blob_attention(grid, [(slice(2, 4), slice(2, 4), 1.0)])
    comp = peak_hot_component(attn, grid, percentile='otsu')
    expected = np.zeros(grid, dtype=bool)
    expected[2:4, 2:4] = True
    assert np.array_equal(comp, expected)


# ── place-back union (the composition run_bbox_zoom performs) ────────────────────

def test_placed_mask_union_is_logical_or():
    pytest.importorskip('PIL')  # place_mask_in_frame_pixels uses PIL; skip if absent
    full_hw = (64, 64)
    # two crop masks placed at disjoint bboxes; union must equal logical_or
    crop_a = np.ones((8, 8), dtype=bool)
    crop_b = np.ones((8, 8), dtype=bool)
    bbox_a = (0.0, 0.0, 0.25, 0.25)
    bbox_b = (0.6, 0.6, 0.9, 0.9)
    pa = place_mask_in_frame_pixels(crop_a, bbox_a, full_hw)
    pb = place_mask_in_frame_pixels(crop_b, bbox_b, full_hw)
    union = np.logical_or.reduce([pa, pb])
    assert union.sum() == pa.sum() + pb.sum()      # disjoint → additive
    assert union[:16, :16].any() and union[38:58, 38:58].any()


# ── cover_bboxes (efficient box cover) ──────────────────────────────────────────

def _mask(grid, blobs):
    m = np.zeros(grid, dtype=bool)
    for rs, cs in blobs:
        m[rs, cs] = True
    return m


def test_cover_two_disjoint_blobs_stay_split():
    m = _mask((16, 16), [(slice(1, 4), slice(1, 4)), (slice(11, 14), slice(11, 14))])
    boxes = cover_bboxes(m, box_area_weight=0.01, pad_frac=0.0)
    assert len(boxes) == 2          # gap too costly to bridge at low weight


def test_cover_merges_blobs_at_high_weight():
    m = _mask((16, 16), [(slice(1, 4), slice(1, 4)), (slice(11, 14), slice(11, 14))])
    boxes = cover_bboxes(m, box_area_weight=1.0, pad_frac=0.0)
    assert len(boxes) == 1          # one fewer box worth a lot of area → merge


def test_cover_drops_speck_below_min_patches():
    m = _mask((12, 12), [(slice(2, 5), slice(2, 5))])   # 9-patch blob
    m[11, 11] = True                                    # 1-patch speck
    boxes = cover_bboxes(m, box_area_weight=0.01, min_patches=2, pad_frac=0.0)
    assert len(boxes) == 1


def test_cover_caps_at_max_regions():
    m = _mask((16, 16), [
        (slice(1, 3), slice(1, 3)), (slice(1, 3), slice(13, 15)),
        (slice(13, 15), slice(1, 3)), (slice(13, 15), slice(13, 15)),
    ])
    boxes = cover_bboxes(m, box_area_weight=0.001, max_regions=2, pad_frac=0.0)
    assert len(boxes) == 2          # forced merges down to the cap


def test_cover_empty_mask_returns_empty():
    assert cover_bboxes(np.zeros((8, 8), dtype=bool)) == []


def test_cover_squares_elongated_box():
    # a thin wide blob (aspect ~6:1) → square_cap bounds the final aspect.
    m = _mask((16, 16), [(slice(7, 9), slice(2, 14))])
    boxes = cover_bboxes(m, box_area_weight=0.01, pad_frac=0.0, square_cap=1.4)
    assert len(boxes) == 1
    y0, x0, y1, x1 = boxes[0]
    h, w = y1 - y0, x1 - x0
    assert max(h, w) / min(h, w) <= 1.45      # within the cap (+ε)


# ── gate_boxes_by_logit (relative-to-full-image gate) ────────────────────────────

def test_gate_keeps_boxes_at_or_above_full():
    # full logit 1.0; boxes at 1.5 (more confident) and 0.5 (less) → keep only #0
    keep = gate_boxes_by_logit([1.5, 0.5], full_logit=1.0)
    assert keep == [0]


def test_gate_margin_loosens_bar():
    # with margin 0.6 the 0.5 box clears 1.0-0.6=0.4 too
    assert gate_boxes_by_logit([1.5, 0.5], full_logit=1.0, margin=0.6) == [0, 1]


def test_gate_all_below_returns_empty():
    # every crop less confident than the full image → [] → caller defers to unzoomed
    assert gate_boxes_by_logit([-0.2, 0.1], full_logit=0.5) == []


def test_gate_none_full_logit_returns_empty():
    # image head disabled upstream → no reference → [] (caller skips gating)
    assert gate_boxes_by_logit([1.0, 2.0], full_logit=None) == []


def test_gate_skips_none_box_logits():
    keep = gate_boxes_by_logit([None, 2.0], full_logit=1.0)
    assert keep == [1]


# ── area-based padding mode (gated, resolution-invariant) ───────────────────────

def test_area_pad_adds_exact_frame_fraction():
    # A 2x2 hot block at the top-left of an 8x8 grid → tight box (0,0,.25,.25).
    # pad_side_frac=0.10 must add exactly 0.10 of the frame to each free side.
    attn = np.zeros((8, 8)); attn[0:2, 0:2] = 1.0
    y0, x0, y1, x1 = attention_to_bbox(
        attn, (8, 8), percentile='otsu', pad_side_frac=0.10, min_area_frac=0.0,
    )
    # top/left clip at 0.0; bottom/right grow by 0.10 from 0.25.
    assert (y0, x0) == (0.0, 0.0)
    assert abs(y1 - 0.35) < 1e-9 and abs(x1 - 0.35) < 1e-9


def test_area_pad_is_resolution_invariant():
    # The SAME quarter-frame tight box on two different grids must yield the
    # same fractional padded box — the whole point of leaving patch units behind.
    def box(grid):
        n = grid
        attn = np.zeros((n, n)); attn[: n // 2, : n // 2] = 1.0
        return attention_to_bbox(attn, (n, n), percentile='otsu', pad_side_frac=0.08)
    b32 = box(32)
    b44 = box(44)
    assert all(abs(a - b) < 1e-9 for a, b in zip(b32, b44))


def test_area_min_area_floor_grows_box():
    # A 1x1 hot patch on a 16x16 grid is tiny; min_area_frac=0.25 must grow the
    # padded box to >=25% of the frame area.
    attn = np.zeros((16, 16)); attn[8, 8] = 1.0
    y0, x0, y1, x1 = attention_to_bbox(
        attn, (16, 16), percentile='peak', thresh_mult=0.5,
        pad_side_frac=0.02, min_area_frac=0.25,
    )
    assert (y1 - y0) * (x1 - x0) >= 0.25 - 1e-6


def test_area_mode_off_matches_legacy():
    # pad_side_frac=None must be byte-for-byte the legacy patch path.
    attn = np.zeros((10, 10)); attn[3:6, 4:7] = 1.0
    legacy = attention_to_bbox(attn, (10, 10), percentile='otsu', pad_frac=0.10)
    explicit_none = attention_to_bbox(attn, (10, 10), percentile='otsu',
                                      pad_frac=0.10, pad_side_frac=None)
    assert legacy == explicit_none
