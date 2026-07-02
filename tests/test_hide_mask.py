"""tests.test_hide_mask — numpy-only tests for lab_utils/eval/hide.py.

Covers the MIL patch-hiding mask construction:
  - dilate_mask_8 (8-connected neighbour padding)
  - build_hide_mask (lower threshold + dilation, component vs hot)

No torch, no GPU, no datasets.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_utils.eval.hide import (  # noqa: E402
    HIDE_THRESH_MULT,
    build_hide_mask,
    dilate_mask_8,
)
from lab_utils.eval.zoom import peak_hot_component  # noqa: E402


def _blob(grid, blobs):
    h, w = grid
    a = np.zeros((h, w), dtype=np.float64)
    for rs, cs, v in blobs:
        a[rs, cs] = v
    return a.reshape(-1)


# ── dilate_mask_8 ────────────────────────────────────────────────────────────────

def test_dilate_adds_8_neighbours():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    out = dilate_mask_8(m, iters=1)
    # the full 3x3 block around the centre is now set (diagonals included)
    assert out[1:4, 1:4].all()
    assert out.sum() == 9


def test_dilate_zero_iters_is_noop():
    m = np.zeros((4, 4), dtype=bool)
    m[1, 1] = True
    assert np.array_equal(dilate_mask_8(m, iters=0), m)


def test_dilate_respects_borders():
    m = np.zeros((3, 3), dtype=bool)
    m[0, 0] = True                      # corner — only 3 in-grid neighbours
    out = dilate_mask_8(m, iters=1)
    assert out[0, 0] and out[0, 1] and out[1, 0] and out[1, 1]
    assert out.sum() == 4               # no wrap-around


def test_dilate_two_iters_grows_two_rings():
    m = np.zeros((7, 7), dtype=bool)
    m[3, 3] = True
    out = dilate_mask_8(m, iters=2)
    assert out[1:6, 1:6].all() and out.sum() == 25


# ── build_hide_mask ──────────────────────────────────────────────────────────────

def test_build_hide_mask_is_superset_of_undilated_component():
    grid = (10, 10)
    attn = _blob(grid, [(slice(3, 5), slice(3, 5), 1.0)])
    raw = peak_hot_component(attn, grid, percentile='otsu',
                             thresh_mult=HIDE_THRESH_MULT).reshape(-1)
    hide = build_hide_mask(attn, grid, mode='component', dilate=1)
    # dilation only adds patches → strict superset, never drops the core
    assert hide.sum() > raw.sum()
    assert np.all(hide[raw])


def test_build_hide_mask_lower_threshold_hides_more():
    # a hot core plus a warm shoulder; the lower hide threshold should pull the
    # shoulder into the positive set that gets hidden.
    grid = (10, 10)
    attn = _blob(grid, [(slice(4, 6), slice(4, 8), 0.5),   # warm shoulder
                        (slice(4, 6), slice(4, 6), 1.0)])  # hot core
    strict = build_hide_mask(attn, grid, mode='component',
                             thresh_mult=0.70, dilate=0)
    lenient = build_hide_mask(attn, grid, mode='component',
                              thresh_mult=0.40, dilate=0)
    assert lenient.sum() >= strict.sum()


def test_build_hide_mask_hot_vs_component():
    grid = (10, 10)
    attn = _blob(grid, [(slice(1, 3), slice(1, 3), 1.0),    # peak blob
                        (slice(7, 9), slice(7, 9), 1.0)])   # separate hot blob
    comp = build_hide_mask(attn, grid, mode='component', dilate=0)
    hot = build_hide_mask(attn, grid, mode='hot', dilate=0)
    # 'component' isolates the peak blob; 'hot' grabs both → hot hides strictly more
    assert hot.sum() > comp.sum()


def test_build_hide_mask_unknown_mode_raises():
    grid = (4, 4)
    attn = _blob(grid, [(slice(1, 3), slice(1, 3), 1.0)])
    with pytest.raises(ValueError):
        build_hide_mask(attn, grid, mode='bogus')
