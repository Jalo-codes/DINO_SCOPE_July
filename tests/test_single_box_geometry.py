"""tests.test_single_box_geometry — numpy-only tests for the single-box helpers.

Covers the supervised-MVP geometry added to lab_utils/eval/zoom.py:
  - gt_grid_mask            (pixel GT → patch grid)
  - largest_component_bbox  (4-connected, picks the bigger blob)
  - single_box_target       (padded box / large-splice no-zoom / no-GT)
  - box_from_heatmap        (read-off + cold-heatmap → no box)
  - grid_bbox_to_frac

No torch, no GPU, no datasets.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_utils.eval.zoom import (  # noqa: E402
    all_component_bboxes,
    box_from_heatmap,
    boxes_from_heatmap,
    coverage_target,
    grid_bbox_to_frac,
    gt_grid_mask,
    largest_component_bbox,
    multi_box_target,
    single_box_target,
)
from lab_utils.eval.multibox import (  # noqa: E402
    cover_bboxes,
    proximity_bboxes,
    suppress_contained_boxes,
)


def test_largest_component_picks_bigger_blob():
    m = np.zeros((10, 10), dtype=bool)
    m[0:1, 0:1] = True          # 1-patch blob
    m[4:8, 4:8] = True          # 16-patch blob
    box = largest_component_bbox(m)
    assert box == (4, 4, 7, 7)


def test_largest_component_empty_is_none():
    assert largest_component_bbox(np.zeros((6, 6), dtype=bool)) is None


def test_largest_component_diagonal_not_connected():
    # 4-connectivity: a diagonal touch is two separate 1-patch blobs.
    m = np.zeros((5, 5), dtype=bool)
    m[1, 1] = True
    m[2, 2] = True
    box = largest_component_bbox(m)
    # Both size 1 → first found (row-major) wins: (1,1).
    assert box == (1, 1, 1, 1)


def test_gt_grid_mask_downsamples_center_block():
    # A centered pixel square → centered patch block on the grid.
    gt = np.zeros((64, 64), dtype=bool)
    gt[24:40, 24:40] = True          # central quarter-ish
    grid = gt_grid_mask(gt, (8, 8), patch_frac=0.25)
    assert grid.shape == (8, 8)
    assert grid.any()
    rows, cols = np.where(grid)
    # Activated patches are centered, not at the borders.
    assert rows.min() >= 2 and rows.max() <= 5
    assert cols.min() >= 2 and cols.max() <= 5


def test_gt_grid_mask_none_is_empty():
    assert not gt_grid_mask(None, (8, 8)).any()


def test_single_box_target_pads_and_labels_rectangle():
    gt = np.zeros((64, 64), dtype=bool)
    gt[24:40, 24:40] = True
    tgt, pbox, kind = single_box_target(gt, (8, 8), pad_frac=0.15, large_thresh=0.9)
    assert kind == 'box'
    assert pbox is not None
    r0, c0, r1, c1 = pbox
    # Target is exactly the padded rectangle, nothing else.
    grid = tgt.reshape(8, 8)
    on = np.zeros((8, 8), dtype=bool)
    on[r0:r1 + 1, c0:c1 + 1] = True
    assert np.array_equal(grid > 0.5, on)
    # Padding added at least one patch margin around the core blob.
    core = largest_component_bbox(gt_grid_mask(gt, (8, 8)))
    assert r0 < core[0] and c0 < core[1] and r1 > core[2] and c1 > core[3]


def test_single_box_target_large_splice_is_no_zoom():
    gt = np.ones((64, 64), dtype=bool)               # whole frame tampered
    tgt, pbox, kind = single_box_target(gt, (8, 8), large_thresh=0.75)
    assert kind == 'large'
    assert pbox is None
    assert not (tgt > 0.5).any()                     # all-zero target ⇒ don't zoom


def test_single_box_target_no_gt():
    tgt, pbox, kind = single_box_target(None, (8, 8))
    assert kind == 'no_gt'
    assert pbox is None
    assert not (tgt > 0.5).any()


def test_box_from_heatmap_reads_off_blob():
    grid_hw = (8, 8)
    prob = np.zeros((8, 8), dtype=np.float32)
    prob[2:5, 3:6] = 0.9
    fbox, gbox = box_from_heatmap(prob.reshape(-1), grid_hw, thresh=0.5)
    assert gbox == (2, 3, 4, 5)
    assert fbox == pytest.approx((2 / 8, 3 / 8, 5 / 8, 6 / 8))


def test_box_from_heatmap_cold_map_yields_no_box():
    prob = np.full((8, 8), 0.2, dtype=np.float32)    # uniformly below threshold
    fbox, gbox = box_from_heatmap(prob.reshape(-1), (8, 8), thresh=0.5)
    assert fbox is None and gbox is None


def test_box_from_heatmap_respects_min_patches():
    prob = np.zeros((8, 8), dtype=np.float32)
    prob[0, 0] = 0.9                                  # single hot patch
    fbox, gbox = box_from_heatmap(prob.reshape(-1), (8, 8), thresh=0.5, min_patches=2)
    assert fbox is None and gbox is None


def test_grid_bbox_to_frac_covers_whole_cells():
    assert grid_bbox_to_frac((0, 0, 0, 0), (4, 4)) == pytest.approx((0.0, 0.0, 0.25, 0.25))
    assert grid_bbox_to_frac((0, 0, 3, 3), (4, 4)) == pytest.approx((0.0, 0.0, 1.0, 1.0))


# ── multi-component ──────────────────────────────────────────────────────────────

def test_all_component_bboxes_finds_two_blobs():
    m = np.zeros((10, 10), dtype=bool)
    m[1:3, 1:3] = True          # 4 patches
    m[6:9, 6:9] = True          # 9 patches
    comps = all_component_bboxes(m)
    assert len(comps) == 2
    sizes = sorted(s for _, s in comps)
    assert sizes == [4, 9]
    boxes = {b for b, _ in comps}
    assert (1, 1, 2, 2) in boxes and (6, 6, 8, 8) in boxes


def test_multi_box_target_lights_all_components():
    gt = np.zeros((64, 64), dtype=bool)
    gt[8:16, 8:16] = True       # one splice
    gt[40:48, 40:48] = True     # a second, disjoint splice
    tgt, boxes, kind = multi_box_target(gt, (8, 8), pad_frac=0.0, pad_min_patches=1,
                                        large_thresh=0.9)
    assert kind == 'box'
    assert len(boxes) == 2      # both components kept, not collapsed to one
    grid = tgt.reshape(8, 8)
    # Both corners region active, center largely empty (two separate boxes).
    assert grid[0:3, 0:3].any() and grid[4:7, 4:7].any()


def test_multi_box_target_drops_large_component_keeps_small():
    gt = np.zeros((64, 64), dtype=bool)
    gt[:, :] = False
    gt[0:60, 0:60] = True        # huge component → dropped
    gt[62:64, 62:64] = False     # (nothing) — ensure only the big blob
    tgt, boxes, kind = multi_box_target(gt, (8, 8), large_thresh=0.75)
    assert kind == 'large'       # the only component is too big to zoom
    assert boxes == []
    assert not (tgt > 0.5).any()


def test_boxes_from_heatmap_reads_multiple():
    prob = np.zeros((10, 10), dtype=np.float32)
    prob[1:3, 1:3] = 0.9
    prob[6:9, 6:9] = 0.9
    boxes = boxes_from_heatmap(prob.reshape(-1), (10, 10), thresh=0.5, min_patches=2)
    assert len(boxes) == 2
    assert grid_bbox_to_frac((1, 1, 2, 2), (10, 10)) in boxes
    assert grid_bbox_to_frac((6, 6, 8, 8), (10, 10)) in boxes


def test_boxes_from_heatmap_cold_map_empty():
    prob = np.full((8, 8), 0.2, dtype=np.float32)
    assert boxes_from_heatmap(prob.reshape(-1), (8, 8), thresh=0.5) == []


# ── coverage target (the new GT) ─────────────────────────────────────────────────

def test_coverage_target_bridges_nearby_into_one_region():
    gt = np.zeros((64, 64), dtype=bool)
    gt[8:16, 8:16] = True        # patch (row1, col1)
    gt[8:16, 24:32] = True       # patch (row1, col3) — one empty cell gap
    cov, kind = coverage_target(gt, (8, 8), pad_patches=1, large_thresh=0.9)
    assert kind == 'box'
    cov2d = cov.reshape(8, 8) > 0.5
    # Dilation by 1 bridges the 1-cell gap ⇒ a single connected region, not two.
    assert len(all_component_bboxes(cov2d)) == 1


def test_coverage_target_follows_shape_not_filled_bbox():
    # Two far-apart patches: coverage must NOT fill the bounding rectangle between
    # them (that was the "insane GT").  With a 1-patch grow they stay separate.
    gt = np.zeros((80, 80), dtype=bool)
    gt[0:10, 0:10] = True        # top-left  (cell 0,0)
    gt[70:80, 70:80] = True      # bottom-right (cell 7,7)
    cov, kind = coverage_target(gt, (8, 8), pad_patches=1, large_thresh=0.95)
    cov2d = cov.reshape(8, 8) > 0.5
    assert kind == 'box'
    assert len(all_component_bboxes(cov2d)) == 2     # two regions, gap NOT filled
    assert not cov2d[3:5, 3:5].any()                 # center stays empty


def test_coverage_target_large_splice_is_no_zoom():
    gt = np.ones((64, 64), dtype=bool)
    cov, kind = coverage_target(gt, (8, 8), large_thresh=0.75)
    assert kind == 'large'
    assert not (cov > 0.5).any()


def test_coverage_target_no_gt():
    cov, kind = coverage_target(None, (8, 8))
    assert kind == 'no_gt'
    assert not (cov > 0.5).any()


# ── patch-space proximity read-off (the mask-in-mask fix) ─────────────────────────

def _ring_with_inner_blob():
    """Hollow ring (thick arms) with a small blob far inside the hole — the
    containment case that breaks hull-space cover_bboxes."""
    m = np.zeros((20, 20), dtype=bool)
    m[2:4, 2:18] = True; m[16:18, 2:18] = True      # top / bottom arms
    m[2:18, 2:4] = True; m[2:18, 16:18] = True      # left / right arms
    m[9:11, 9:11] = True                            # blob alone in the center
    return m


def test_cover_swallows_inner_blob_containment_bug():
    # Documents the bug proximity fixes: the inner blob's box sits inside the
    # ring's hull, so added = -area(inner) < 0 → unconditional merge → one box.
    boxes = cover_bboxes(_ring_with_inner_blob(), min_patches=1, max_regions=3)
    assert len(boxes) == 1


def test_proximity_keeps_inner_blob_separate():
    boxes = proximity_bboxes(_ring_with_inner_blob(), dilate=1, min_patches=1, max_regions=3)
    assert len(boxes) == 2
    # One box is the big ring; the other is the tight central blob.
    areas = sorted((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
    assert areas[0] < 0.2 < areas[1]


def test_proximity_groups_adjacent_into_one():
    # Two ON cells one empty patch apart → dilate=1 bridges them into a single box.
    m = np.zeros((10, 10), dtype=bool)
    m[5, 2] = True
    m[5, 4] = True
    assert len(proximity_bboxes(m, dilate=1, min_patches=1, max_regions=4)) == 1
    # With no dilation they are two separate boxes.
    assert len(proximity_bboxes(m, dilate=0, min_patches=1, max_regions=4)) == 2


def test_proximity_pads_small_box_off_the_floor():
    # A 2x2 blob must get real breathing room from the pad floor, not a sliver.
    m = np.zeros((28, 28), dtype=bool)
    m[13:15, 13:15] = True
    box = proximity_bboxes(m, dilate=0, min_patches=1, max_regions=1,
                           pad_frac=0.08, min_pad_frac=0.06)[0]
    side = box[2] - box[0]
    assert side > 2.0 / 28          # bigger than the raw 2-patch blob
    assert side < 0.5               # but not ballooned to the whole frame


def test_proximity_drops_specks_below_min_patches():
    m = np.zeros((10, 10), dtype=bool)
    m[1, 1] = True                  # lone 1-patch speck
    m[5:8, 5:8] = True              # a real 9-patch blob
    boxes = proximity_bboxes(m, dilate=0, min_patches=2, max_regions=4)
    assert len(boxes) == 1


def test_suppress_kills_smaller_box_mostly_inside_larger():
    big = (0.0, 0.0, 0.6, 0.6)
    small = (0.4, 0.4, 0.6, 0.6)          # fully inside big → 100% > 30% → killed
    assert suppress_contained_boxes([big, small], frac=0.30) == [big]


def test_suppress_keeps_lightly_overlapping_boxes():
    a = (0.0, 0.0, 0.5, 0.5)
    b = (0.45, 0.45, 0.95, 0.95)          # tiny corner overlap, far below 30% of b
    kept = suppress_contained_boxes([a, b], frac=0.30)
    assert len(kept) == 2


def test_suppress_threshold_is_fraction_of_smaller():
    big = (0.0, 0.0, 1.0, 1.0)
    # small box with exactly 25% of itself inside big → kept (below 0.30)
    small = (0.9, 0.0, 1.1, 1.0)          # half its height (0.1 of 0.2) inside → 50%
    assert suppress_contained_boxes([big, small], frac=0.30) == [big]   # 50% > 30% → killed
    # raise the threshold above the overlap → keep both
    assert len(suppress_contained_boxes([big, small], frac=0.60)) == 2


def test_suppress_disabled_passes_through():
    boxes = [(0.0, 0.0, 0.6, 0.6), (0.4, 0.4, 0.6, 0.6)]
    assert suppress_contained_boxes(boxes, frac=0.0) == boxes


def test_proximity_caps_at_max_regions_keeping_largest():
    m = np.zeros((20, 20), dtype=bool)
    m[1:5, 1:5] = True              # 16 patches (biggest)
    m[1:3, 15:17] = True           # 4 patches
    m[15:17, 1:3] = True           # 4 patches
    m[18, 18] = True               # 1 patch
    boxes = proximity_bboxes(m, dilate=0, min_patches=1, max_regions=2)
    assert len(boxes) == 2          # only the two largest survive the cap
