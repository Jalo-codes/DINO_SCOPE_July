"""tests.test_full_fakes — lab_utils.data.datasets.full_fakes.build().

Whole-image ("full fake") generation eval set: root/real/ vs
root/<generator>/, no splice boundary, no real GT mask (synthetic full-frame
sentinel keeps Item.is_real correct). No torch, no GPU; PIL + numpy only.
"""

import sys
from pathlib import Path

import pytest
pytest.importorskip('torch')  # full_fakes.py pulls in Dataset (lab_utils/data/dataset.py)

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_utils.data.datasets.full_fakes import build          # noqa: E402
from lab_utils.data.resolution import Resolution               # noqa: E402

RES = Resolution(image_size=64, patch_size=16)


def _write_image(path: Path, size=(96, 64), color=(120, 130, 140)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', size, color).save(path)


def _make_dataset(root: Path, *, n_real=3, generators=None):
    generators = generators if generators is not None else {'sdxl': 2, 'flux': 4}
    for i in range(n_real):
        _write_image(root / 'real' / f'{i:04d}.png', color=(10 + i, 20, 30))
    for gen, n in generators.items():
        for i in range(n):
            _write_image(root / gen / f'{i:04d}.png', color=(200 + i, 50, 60 + i))


class TestFullFakesBuild:
    def test_missing_root_returns_empty(self, tmp_path):
        train_ds, val_ds = build(tmp_path / 'nope', res=RES)
        assert len(train_ds.items) == 0
        assert len(val_ds.items) == 0

    def test_empty_root_returns_empty(self, tmp_path):
        train_ds, val_ds = build(tmp_path, res=RES)
        assert len(val_ds.items) == 0

    def test_counts_and_is_real(self, tmp_path):
        _make_dataset(tmp_path, n_real=3, generators={'sdxl': 2, 'flux': 4})
        train_ds, val_ds = build(tmp_path, res=RES)
        # Eval-only BY DEFAULT (val_split=1.0) — see TestValSplit for the
        # train-side modes.
        assert len(train_ds.items) == 0
        items = val_ds.items
        assert len(items) == 3 + 2 + 4

        reals = [it for it in items if it.is_real]
        fakes = [it for it in items if not it.is_real]
        assert len(reals) == 3
        assert len(fakes) == 6
        assert all(it.mask is None for it in reals)
        assert all(it.mask is not None for it in fakes)

    def test_generator_meta_and_source(self, tmp_path):
        _make_dataset(tmp_path, n_real=1, generators={'sdxl-juggernaut': 3})
        _, val_ds = build(tmp_path, res=RES)
        fakes = [it for it in val_ds.items if not it.is_real]
        assert len(fakes) == 3
        assert all(it.meta.get('generator') == 'sdxl-juggernaut' for it in fakes)
        assert all(it.meta.get('gt_mask_reliable') is False for it in fakes)
        assert all(it.source == 'full_fakes' for it in val_ds.items)

    def test_real_dir_case_insensitive_and_reals_alias(self, tmp_path):
        _write_image(tmp_path / 'REAL' / '0000.png')
        _write_image(tmp_path / 'sdxl' / '0000.png', color=(255, 0, 0))
        _, val_ds = build(tmp_path, res=RES)
        reals = [it for it in val_ds.items if it.is_real]
        assert len(reals) == 1

    def test_sentinel_mask_survives_default_verify_policy(self, tmp_path):
        # DEFAULT_POLICY.max_mask_area=0.99 would reject a 100%-white mask;
        # build() must relax it when the caller passes no explicit policy.
        _make_dataset(tmp_path, n_real=1, generators={'sdxl': 1})
        _, val_ds = build(tmp_path, res=RES)
        fakes = [it for it in val_ds.items if not it.is_real]
        assert len(fakes) == 1
        m = np.asarray(Image.open(fakes[0].mask).convert('L'))
        assert (m > 0).mean() == 1.0

    def test_mask_area_is_full_frame(self, tmp_path):
        _make_dataset(tmp_path, n_real=0, generators={'sdxl': 1})
        _, val_ds = build(tmp_path, res=RES)
        fake = val_ds.items[0]
        assert fake.mask_area(RES) == 1.0

    def test_non_image_files_ignored(self, tmp_path):
        _make_dataset(tmp_path, n_real=1, generators={'sdxl': 1})
        (tmp_path / 'sdxl' / 'readme.txt').write_text('not an image')
        _, val_ds = build(tmp_path, res=RES)
        assert len(val_ds.items) == 2


class TestValSplit:
    """val_split routing: 1.0 = eval-only (default), 0.0 = all train, else split.

    The 0.0 mode is what lets a whole-image-fakes root be TRAINED on. OpenFake's
    train / validation / test splits are separate downloads (validation = held-out
    images from the training generators; test = held-out GENERATORS), so the real
    split boundary is expressed as two roots, not a ratio — the intermediate mode
    exists only for single-root convenience.
    """

    def test_default_is_eval_only(self, tmp_path):
        _make_dataset(tmp_path, n_real=3, generators={'sdxl': 2, 'flux': 4})
        train_ds, val_ds = build(tmp_path, res=RES)
        assert (len(train_ds.items), len(val_ds.items)) == (0, 9)

    def test_zero_sends_everything_to_train(self, tmp_path):
        _make_dataset(tmp_path, n_real=3, generators={'sdxl': 2, 'flux': 4})
        train_ds, val_ds = build(tmp_path, res=RES, val_split=0.0)
        assert (len(train_ds.items), len(val_ds.items)) == (9, 0)
        # Train side must augment; val side must not.
        assert train_ds.augment is True

    def test_internal_split_is_disjoint_and_lossless(self, tmp_path):
        _make_dataset(tmp_path, n_real=10, generators={'sdxl': 10, 'flux': 10})
        train_ds, val_ds = build(tmp_path, res=RES, val_split=0.3)
        train_ids = {it.item_id for it in train_ds.items}
        val_ids = {it.item_id for it in val_ds.items}
        assert not (train_ids & val_ids), 'train/val overlap'
        assert len(train_ids) + len(val_ids) == 30, 'items lost in the split'
        assert len(val_ids) > 0 and len(train_ids) > 0

    def test_internal_split_is_deterministic(self, tmp_path):
        _make_dataset(tmp_path, n_real=10, generators={'sdxl': 10, 'flux': 10})
        a = build(tmp_path, res=RES, val_split=0.3)[1]
        b = build(tmp_path, res=RES, val_split=0.3)[1]
        assert [it.item_id for it in a.items] == [it.item_id for it in b.items]
        c = build(tmp_path, res=RES, val_split=0.3, split_seed=7)[1]
        assert [it.item_id for it in c.items] != [it.item_id for it in a.items], \
            'split_seed had no effect'

    def test_internal_split_is_stratified_by_generator(self, tmp_path):
        # A naive global shuffle can drop a small generator from val entirely;
        # stratifying per generator (reals are their own stratum) cannot.
        _make_dataset(tmp_path, n_real=20, generators={'sdxl': 20, 'tiny': 4})
        train_ds, val_ds = build(tmp_path, res=RES, val_split=0.5)
        strata = lambda ds: {(it.meta.get('generator') or 'real') for it in ds.items}
        assert strata(val_ds) == {'real', 'sdxl', 'tiny'}
        assert strata(train_ds) == {'real', 'sdxl', 'tiny'}


class TestValCaps:
    """val_per_pool / val_real_cap: a bounded, STABLE per-epoch eval set.

    Pools are wildly uneven (200 images down to 3), so an uncapped val is both
    slow and dominated by whichever generators happen to be large. The caps must
    also be stable across calls — a val set that moves between epochs makes
    epoch-to-epoch deltas unreadable.
    """

    def _uneven(self, root):
        pools = {'real': 40, 'sdxl': 20, 'flux': 20, 'tiny': 3}
        for name, n in pools.items():
            for i in range(n):
                _write_image(root / name / f'{i:04d}.png', color=(i % 256, 40, 90))
        return pools

    def test_caps_applied_per_pool(self, tmp_path):
        self._uneven(tmp_path)
        _, val_ds = build(tmp_path, res=RES, val_per_pool=5, val_real_cap=10)
        counts = {}
        for it in val_ds.items:
            k = it.meta.get('generator') or 'real'
            counts[k] = counts.get(k, 0) + 1
        assert counts == {'real': 10, 'sdxl': 5, 'flux': 5, 'tiny': 3}

    def test_pool_smaller_than_cap_is_taken_whole_not_padded(self, tmp_path):
        self._uneven(tmp_path)
        _, val_ds = build(tmp_path, res=RES, val_per_pool=5)
        tiny = [it for it in val_ds.items if it.meta.get('generator') == 'tiny']
        assert len(tiny) == 3
        assert len({it.item_id for it in tiny}) == 3, 'duplicated to fill the cap'

    def test_val_set_is_stable_across_calls(self, tmp_path):
        self._uneven(tmp_path)
        kw = dict(res=RES, val_per_pool=5, val_real_cap=10)
        a = build(tmp_path, **kw)[1]
        b = build(tmp_path, **kw)[1]
        assert [it.item_id for it in a.items] == [it.item_id for it in b.items]

    def test_split_seed_varies_the_draw(self, tmp_path):
        self._uneven(tmp_path)
        a = build(tmp_path, res=RES, val_per_pool=5, val_real_cap=10)[1]
        c = build(tmp_path, res=RES, val_per_pool=5, val_real_cap=10, split_seed=7)[1]
        assert [it.item_id for it in a.items] != [it.item_id for it in c.items]

    def test_caps_are_opt_in(self, tmp_path):
        self._uneven(tmp_path)
        _, val_ds = build(tmp_path, res=RES)
        assert len(val_ds.items) == 83  # 40 + 20 + 20 + 3, uncapped

    def test_only_real_cap_leaves_generators_uncapped(self, tmp_path):
        self._uneven(tmp_path)
        _, val_ds = build(tmp_path, res=RES, val_real_cap=10)
        counts = {}
        for it in val_ds.items:
            k = it.meta.get('generator') or 'real'
            counts[k] = counts.get(k, 0) + 1
        assert counts == {'real': 10, 'sdxl': 20, 'flux': 20, 'tiny': 3}
