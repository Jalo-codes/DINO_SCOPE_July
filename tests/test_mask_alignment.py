"""tests.test_mask_alignment — the image/mask alignment hard-check.

Alignment rule (shared by verify.py, dataset.py, eval/metric.py via
``verify.mask_alignment``):
  - identical sizes            → 'aligned'
  - same aspect, different res → 'resizable'  (data property: half-res masks,
                                 generator size snapping, CASIA off-by-one)
  - aspect mismatch            → 'misaligned' (pairing bug → hard DataError,
                                 never dropped or resized over silently)

No torch, no GPU; PIL + numpy only.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_utils.data.item import Item, make_item_id          # noqa: E402
from lab_utils.data.verify import (                          # noqa: E402
    mask_alignment,
    verify,
    verify_all,
)
from lab_utils.errors import DataError                       # noqa: E402


# ── mask_alignment classification ─────────────────────────────────────────────

class TestMaskAlignmentClassifier:
    def test_identical_sizes_aligned(self):
        assert mask_alignment((672, 1008), (672, 1008)) == 'aligned'

    def test_flux_size_snap_resizable(self):
        # The real TGIF/flux case: pre-generation source mask vs snapped output.
        assert mask_alignment((672, 1008), (680, 1023)) == 'resizable'

    def test_casia_off_by_one_resizable(self):
        assert mask_alignment((384, 256), (385, 256)) == 'resizable'

    def test_half_resolution_mask_resizable(self):
        assert mask_alignment((672, 1008), (336, 504)) == 'resizable'

    def test_aspect_mismatch_misaligned(self):
        # Square mask paired with a portrait image = wrong pairing.
        assert mask_alignment((672, 1008), (512, 512)) == 'misaligned'

    def test_transposed_pair_misaligned(self):
        # Same pixels, swapped axes — a classic wrong-orientation bug.
        assert mask_alignment((672, 1008), (1008, 672)) == 'misaligned'

    def test_zero_dimension_misaligned(self):
        assert mask_alignment((672, 1008), (0, 1008)) == 'misaligned'


# ── verify() / verify_all() behavior ──────────────────────────────────────────

def _noise_image(path: Path, size, mode='RGB'):
    rng = np.random.default_rng(0)
    w, h = size
    arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(arr, 'RGB').convert(mode).save(path)


def _mask_image(path: Path, size, fg_frac=0.25):
    w, h = size
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[: max(1, int(h * fg_frac)), :] = 255
    Image.fromarray(arr, 'L').save(path)


def _fake_item(img_path: Path, mask_path: Path) -> Item:
    return Item(
        image=img_path,
        authentic=None,
        mask=mask_path,
        source='test',
        item_id=make_item_id('test', img_path),
        meta={},
    )


class TestVerifyAlignment:
    def test_aligned_pair_passes(self, tmp_path):
        img, mask = tmp_path / 'a.png', tmp_path / 'a_mask.png'
        _noise_image(img, (90, 60))
        _mask_image(mask, (90, 60))
        assert verify(_fake_item(img, mask)) is None

    def test_same_aspect_resolution_diff_is_kept_with_warn(self, tmp_path):
        img, mask = tmp_path / 'b.png', tmp_path / 'b_mask.png'
        _noise_image(img, (90, 60))
        _mask_image(mask, (45, 30))  # half-res, same aspect
        item = _fake_item(img, mask)
        assert verify(item) == 'warn_mask_native_resize'

        kept, rejected = verify_all([item], log_tag='[verify]')
        assert kept == [item]          # warned items PASS — data property
        assert rejected == []

    def test_aspect_mismatch_raises_dataerror(self, tmp_path):
        img, mask = tmp_path / 'c.png', tmp_path / 'c_mask.png'
        _noise_image(img, (90, 60))
        _mask_image(mask, (60, 60))    # square mask on a landscape image
        with pytest.raises(DataError, match='misaligned'):
            verify(_fake_item(img, mask))

    def test_aspect_mismatch_raises_through_verify_all(self, tmp_path):
        img, mask = tmp_path / 'd.png', tmp_path / 'd_mask.png'
        _noise_image(img, (90, 60))
        _mask_image(mask, (60, 60))
        with pytest.raises(DataError, match='misaligned'):
            verify_all([_fake_item(img, mask)], log_tag='[verify]')

    def test_sentinel_mask_exempt_from_alignment(self, tmp_path):
        # pico_banana's synthetic full-frame sentinel: a tiny all-white square
        # (geometry-free by declaration, meta['gt_mask_reliable']=False) paired
        # with an arbitrarily-sized image must NOT trip the alignment check.
        from lab_utils.data.verify import VerifyPolicy

        img, mask = tmp_path / 'e.png', tmp_path / 'e_mask.png'
        _noise_image(img, (128, 96))                 # 4:3, like 1024x768
        _mask_image(mask, (32, 32), fg_frac=1.0)     # all-white 32x32 sentinel
        item = Item(
            image=img,
            authentic=None,
            mask=mask,
            source='pico_banana',
            item_id=make_item_id('pico_banana', img),
            meta={'gt_mask_reliable': False},
        )
        # max_mask_area=1.0 mirrors the pico_banana builder's policy (the
        # sentinel is 100% foreground on purpose).
        assert verify(item, policy=VerifyPolicy(max_mask_area=1.0)) is None
