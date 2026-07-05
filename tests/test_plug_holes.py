"""tests.test_plug_holes — enclosed-hole filling for pseudo-mask grids.

plug_holes (lab_utils/eval/zoom.py): background is flood-filled 8-connected
from the border (dual of the 4-connected foreground components); unreached
background cells are enclosed holes and get filled. No torch, numpy only.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_utils.eval.zoom import plug_holes  # noqa: E402


def _grid(rows):
    return np.array([[ch == '#' for ch in row] for row in rows], dtype=bool)


class TestPlugHoles:
    def test_enclosed_hole_is_filled(self):
        mask = _grid([
            '.....',
            '.###.',
            '.#.#.',
            '.###.',
            '.....',
        ])
        out = plug_holes(mask)
        assert out[2, 2]                      # the hole is plugged
        assert out.sum() == mask.sum() + 1    # and nothing else changed

    def test_border_touching_background_not_filled(self):
        # A "C" shape: the concavity opens to the border — not a hole.
        mask = _grid([
            '###',
            '#..',
            '###',
        ])
        out = plug_holes(mask)
        assert not out[1, 1] and not out[1, 2]
        assert (out == mask).all()

    def test_diagonal_gap_leaks_no_fill(self):
        # The enclosure has a diagonal gap; 8-connected background escapes
        # through it, so the pocket is NOT an enclosed hole.
        mask = _grid([
            '##..',
            '#.#.',
            '###.',
            '....',
        ])
        out = plug_holes(mask)
        assert not out[1, 1]

    def test_multiple_holes_all_filled(self):
        mask = _grid([
            '#####..#####',
            '#...#..#...#',
            '#.#.#..#.#.#',
            '#...#..#...#',
            '#####..#####',
        ])
        out = plug_holes(mask)
        left = out[1:4, 1:4]
        right = out[1:4, 8:11]
        assert left.all() and right.all()
        assert not out[:, 5:7].any()          # gap between the two rings stays

    def test_empty_and_full_masks_unchanged(self):
        empty = np.zeros((6, 6), dtype=bool)
        full = np.ones((6, 6), dtype=bool)
        assert not plug_holes(empty).any()
        assert plug_holes(full).all()

    def test_solid_blob_unchanged(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 3:7] = True
        assert (plug_holes(mask) == mask).all()

    def test_input_not_mutated(self):
        mask = _grid([
            '###',
            '#.#',
            '###',
        ])
        snapshot = mask.copy()
        plug_holes(mask)
        assert (mask == snapshot).all()
