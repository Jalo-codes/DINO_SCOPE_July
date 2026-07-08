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
        assert len(train_ds.items) == 0  # eval-only, like sagid/pico_banana/unpaired
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
