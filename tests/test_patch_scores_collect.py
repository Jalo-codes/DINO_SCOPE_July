"""tests.test_patch_scores_collect — lab_utils.eval.patch_scores.collect_patch_scores.

Split out of tests/test_patch_scores.py: pytest imports a test file as one
module at collection time, so a mid-file pytest.importorskip('torch') aborts
collection of the WHOLE module (including tests defined above it) when torch
is missing — not just the tests below it. weighted_auroc (numpy-only) stays
in test_patch_scores.py so it still runs on a torch-less machine; everything
here needs the full data/model stack and is skipped as a unit instead.
"""

import numpy as np
import pytest

pytest.importorskip('torch')

from PIL import Image  # noqa: E402

from lab_utils.data.item import Item  # noqa: E402
from lab_utils.data.resolution import Resolution  # noqa: E402
from lab_utils.eval.patch_scores import collect_patch_scores  # noqa: E402


class _FakeModelInfo:
    def __init__(self, patch_logits, grid_hw):
        self.patch_logits = patch_logits
        self.attention = None
        self.embeddings = None
        self.image_logit = 0.0
        self.grid_hw = grid_hw
        self.res = None
        self.patch_feats = None


def _write_image(path, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, (100, 110, 120)).save(path)


def test_collect_patch_scores_strata_and_skip_logic(tmp_path, monkeypatch):
    patch_size = 16
    n_side = 4
    S = n_side * patch_size
    res = Resolution(image_size=S, patch_size=patch_size)
    n_patches = n_side * n_side

    # 1) a real item -> all patches real_bg
    real_img = tmp_path / 'real.png'
    _write_image(real_img, (S, S))
    real_item = Item(image=real_img, authentic=None, mask=None, source='t',
                     item_id='real_0', meta={})

    # 2) a fake item with a clean 50/50 split mask (top half fake, bottom half real)
    fake_img = tmp_path / 'fake.png'
    _write_image(fake_img, (S, S))
    mask_arr = np.zeros((S, S), dtype=np.uint8)
    mask_arr[: S // 2, :] = 255
    fake_mask = tmp_path / 'fake_mask.png'
    Image.fromarray(mask_arr, mode='L').save(fake_mask)
    fake_item = Item(image=fake_img, authentic=None, mask=fake_mask, source='t',
                     item_id='fake_0', meta={})

    # 3) an item that should be SKIPPED: sentinel / geometry-free mask
    unreliable_img = tmp_path / 'unreliable.png'
    _write_image(unreliable_img, (S, S))
    unreliable_item = Item(image=unreliable_img, authentic=None, mask=fake_mask,
                           source='t', item_id='unreliable_0',
                           meta={'gt_mask_reliable': False})

    # 4) an item that should be SKIPPED: crop_window set
    cropwin_item = Item(image=fake_img, authentic=None, mask=fake_mask, source='t',
                        item_id='cropwin_0', meta={'crop_window': (0.1, 0.1, 0.9, 0.9)})

    items = [real_item, fake_item, unreliable_item, cropwin_item]

    def fake_model_info(model, img_t, *, device, amp, amp_dtype):
        # Fake patches score high, real patches score low -- perfect separation
        # so the resulting AUROC values are unambiguous to assert on. img_t is
        # the literal 'REAL'/'FAKE' string load_image_tensor below returns.
        logits = np.full(n_patches, -5.0)
        if img_t == 'FAKE':
            logits[: n_patches // 2] = 5.0   # top half "fake"-scored
        return _FakeModelInfo(logits, (n_side, n_side))

    def fake_load_image_tensor(src, res, *, device=None):
        return 'REAL' if (isinstance(src, Item) and src is real_item) else 'FAKE'

    monkeypatch.setattr('lab_utils.eval.fetch.model_info', fake_model_info)
    monkeypatch.setattr('lab_utils.eval.preprocess.load_image_tensor', fake_load_image_tensor)

    out = collect_patch_scores(
        model=None, items=items, res=res, device='cpu', use_amp=False,
        amp_dtype='float32', band=(0.2, 0.8),
    )

    assert out['n_skipped_unreliable'] == 1
    assert out['n_skipped_cropwin'] == 1
    assert out['n_items'] == 2   # real + fake only
    assert not np.isnan(out['auroc_pooled'])
    # perfect separation by construction (fake patches score high, real/bg low)
    assert out['auroc_pooled'] == pytest.approx(1.0)
    assert out['auroc_vs_real_bg'] == pytest.approx(1.0)
    assert out['auroc_vs_splice_bg'] == pytest.approx(1.0)

    item_ids = {row['item_id'] for row in out['per_image']}
    assert item_ids == {'real_0', 'fake_0'}
