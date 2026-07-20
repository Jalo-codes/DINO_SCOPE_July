"""tests.test_build_ae_corpus_pipeline — torch-dependent tier of
experiments.scripts.build_ae_corpus (needs_gpu-adjacent: needs torch, but not
an actual GPU or diffusers — a fake identity "AE" stands in for the real
diffusers model so these exercise the real orchestration code
(_materialize_real/_materialize_ae/_pad_to_multiple) without a network call
or GPU. Split from tests/test_build_ae_corpus.py because a failed module-level
`pytest.importorskip('torch')` skips the WHOLE file at collection time — the
torch-free discovery/hash/manifest/hygiene tests live there so they still run
on a torch-free CPU dev box; this file is the part that needs torch."""

import csv
import sys
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip('torch')

from experiments.scripts.build_ae_corpus import (   # noqa: E402
    MANIFEST_FIELDS,
    REAL_NAME,
    _content_md5,
    _discover_sources,
    _load_manifest_state,
    _materialize_ae,
    _materialize_real,
    _stem_for,
    verify_container_hygiene,
)


def _write_image(path: Path, size=(40, 24), color=(120, 130, 140), mode='RGB', fmt=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', size, color)
    if mode != 'RGB':
        img = img.convert(mode)
    img.save(path, format=fmt)


class _FakeAEOutput:
    def __init__(self, sample):
        self.sample = sample


class _FakeIdentityAE(torch.nn.Module):
    """Stands in for a diffusers AutoencoderKL/Tiny/VQModel: same call
    signature/return shape (`model(x, return_dict=True).sample`), but skips
    the network+GPU weight download entirely — exercises the real
    _materialize_ae/_run_ae_batch/_pad_to_multiple code, not a reimplementation."""

    def forward(self, x, return_dict=True):
        return _FakeAEOutput(sample=x)


def _fake_loader(spec, dtype, device):
    return _FakeIdentityAE().to(device=device, dtype=dtype)


def _failing_loader(spec, dtype, device):
    raise OSError('simulated: repo not found / gated / no network')


class TestPadCrop:
    def test_round_trips_exact_shape_when_already_multiple(self):
        from experiments.scripts.build_ae_corpus import _crop_to, _pad_to_multiple
        x = torch.randn(1, 3, 16, 24)
        padded, hw = _pad_to_multiple(x, 8)
        assert padded.shape == x.shape
        assert _crop_to(padded, hw).shape == x.shape

    def test_pads_up_to_multiple_and_crops_back(self):
        from experiments.scripts.build_ae_corpus import _crop_to, _pad_to_multiple
        x = torch.randn(1, 3, 17, 30)  # not a multiple of 8
        padded, hw = _pad_to_multiple(x, 8)
        assert padded.shape[-2] % 8 == 0 and padded.shape[-1] % 8 == 0
        cropped = _crop_to(padded, hw)
        assert cropped.shape == x.shape
        assert torch.equal(cropped, x)

    def test_falls_back_to_replicate_when_pad_exceeds_extent(self):
        # a 3x3 image padding to a multiple of 8 needs pad >= the image's own
        # extent, where 'reflect' mode raises — must not crash.
        from experiments.scripts.build_ae_corpus import _crop_to, _pad_to_multiple
        x = torch.randn(1, 3, 3, 3)
        padded, hw = _pad_to_multiple(x, 8)
        assert padded.shape[-2:] == (8, 8)
        assert torch.equal(_crop_to(padded, hw), x)


class TestPipelineIntegration:
    """End-to-end through the real orchestration functions with a fake AE."""

    SPEC = dict(name='fake_ae', repo='n/a', kind='kl', divisor=8, input_range=(-1.0, 1.0))

    def _setup(self, tmp_path, sizes):
        source_dir = tmp_path / 'src'
        for i, size in enumerate(sizes):
            _write_image(source_dir / f'img{i}.png', size=size, color=(10 * i, 20, 200))
        return source_dir

    def _run_real_and_ae(self, tmp_path, source_dir, load_ae=_fake_loader):
        import csv as csv_mod

        sources = _discover_sources(source_dir)
        md5_by_path = {p: _content_md5(p) for p in sources}
        root = tmp_path / 'out'
        root.mkdir()
        manifest = root / 'manifest.csv'
        done = {}
        with manifest.open('w', newline='') as mfh:
            writer = csv_mod.DictWriter(mfh, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            _materialize_real(sources, md5_by_path, root, writer, done, mfh)
            ok = _materialize_ae(self.SPEC, sources, md5_by_path, root,
                                  batch_size=4, device='cpu', dtype=torch.float32,
                                  writer=writer, done=done, mfh=mfh, load_ae=load_ae)
        return root, sources, md5_by_path, done, ok

    def test_stem_pairing_across_real_and_ae_folders(self, tmp_path):
        # arbitrary, non-multiple-of-8 resolutions on purpose (pad/crop path)
        source_dir = self._setup(tmp_path, sizes=[(37, 51), (37, 51), (64, 48)])
        root, sources, md5_by_path, done, ok = self._run_real_and_ae(tmp_path, source_dir)
        assert ok

        real_stems = {p.stem for p in (root / REAL_NAME).iterdir()}
        ae_stems = {p.stem for p in (root / 'fake_ae').iterdir()}
        expected_stems = {_stem_for(md5) for md5 in md5_by_path.values()}
        assert real_stems == ae_stems == expected_stems

    def test_output_is_pixel_aligned_with_source(self, tmp_path):
        source_dir = self._setup(tmp_path, sizes=[(37, 51)])
        root, sources, md5_by_path, done, ok = self._run_real_and_ae(tmp_path, source_dir)
        stem = _stem_for(md5_by_path[sources[0]])
        with Image.open(root / REAL_NAME / f'{stem}.png') as real_img:
            assert real_img.size == (37, 51)
        with Image.open(root / 'fake_ae' / f'{stem}.png') as ae_img:
            assert ae_img.size == (37, 51)

    def test_identity_ae_reconstruction_error_is_near_zero(self, tmp_path):
        source_dir = self._setup(tmp_path, sizes=[(40, 24)])
        root, sources, md5_by_path, done, ok = self._run_real_and_ae(tmp_path, source_dir)
        row = done[(md5_by_path[sources[0]], 'fake_ae')]
        assert float(row['mse']) < 1e-3
        assert float(row['psnr']) > 30.0

    def test_container_hygiene_passes_on_real_pipeline_output(self, tmp_path):
        source_dir = self._setup(tmp_path, sizes=[(37, 51), (64, 48)])
        root, *_ = self._run_real_and_ae(tmp_path, source_dir)
        assert verify_container_hygiene(root) == []

    def test_non_rgb_source_is_normalized_to_rgb_everywhere(self, tmp_path):
        source_dir = tmp_path / 'src'
        _write_image(source_dir / 'gray.png', size=(20, 20), mode='L')
        root, sources, md5_by_path, done, ok = self._run_real_and_ae(tmp_path, source_dir)
        stem = _stem_for(md5_by_path[sources[0]])
        with Image.open(root / REAL_NAME / f'{stem}.png') as img:
            assert img.mode == 'RGB'
        with Image.open(root / 'fake_ae' / f'{stem}.png') as img:
            assert img.mode == 'RGB'

    def test_jpeg_source_is_resaved_as_png_not_copied(self, tmp_path):
        source_dir = tmp_path / 'src'
        _write_image(source_dir / 'photo.jpg', size=(20, 20), fmt='JPEG')
        sources = _discover_sources(source_dir)
        assert sources[0].suffix == '.jpg'
        root, _, md5_by_path, done, ok = self._run_real_and_ae(tmp_path, source_dir)
        stem = _stem_for(md5_by_path[sources[0]])
        out = root / REAL_NAME / f'{stem}.png'
        assert out.exists()
        with Image.open(out) as img:
            assert img.format == 'PNG'

    def test_resume_skips_already_done_pairs(self, tmp_path):
        source_dir = self._setup(tmp_path, sizes=[(20, 20)])
        sources = _discover_sources(source_dir)
        md5_by_path = {p: _content_md5(p) for p in sources}
        calls = {'n': 0}

        def counting_loader(spec, dtype, device):
            calls['n'] += 1
            return _fake_loader(spec, dtype, device)

        root = tmp_path / 'out'
        root.mkdir()
        manifest = root / 'manifest.csv'
        done = {}
        with manifest.open('w', newline='') as mfh:
            writer = csv.DictWriter(mfh, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            _materialize_real(sources, md5_by_path, root, writer, done, mfh)
            _materialize_ae(self.SPEC, sources, md5_by_path, root, 4, 'cpu',
                             torch.float32, writer, done, mfh, load_ae=counting_loader)
        assert calls['n'] == 1

        # re-run against the SAME `done` state: nothing pending -> loader never called
        with manifest.open('a', newline='') as mfh:
            writer = csv.DictWriter(mfh, fieldnames=MANIFEST_FIELDS)
            ok = _materialize_ae(self.SPEC, sources, md5_by_path, root, 4, 'cpu',
                                  torch.float32, writer, done, mfh, load_ae=counting_loader)
        assert ok
        assert calls['n'] == 1  # loader not called again — nothing pending

    def test_new_source_tops_up_without_redoing_existing(self, tmp_path):
        source_dir = self._setup(tmp_path, sizes=[(20, 20)])
        root, sources, md5_by_path, done, ok = self._run_real_and_ae(tmp_path, source_dir)
        first_stems = {p.stem for p in (root / 'fake_ae').iterdir()}

        _write_image(source_dir / 'new_image.png', size=(20, 20), color=(99, 5, 200))
        sources2 = _discover_sources(source_dir)
        md5_by_path2 = {p: _content_md5(p) for p in sources2}
        # reload manifest state (mirrors a fresh process resuming)
        done2 = _load_manifest_state(root / 'manifest.csv')
        with (root / 'manifest.csv').open('a', newline='') as mfh:
            writer = csv.DictWriter(mfh, fieldnames=MANIFEST_FIELDS)
            _materialize_real(sources2, md5_by_path2, root, writer, done2, mfh)
            _materialize_ae(self.SPEC, sources2, md5_by_path2, root, 4, 'cpu',
                            torch.float32, writer, done2, mfh, load_ae=_fake_loader)

        final_stems = {p.stem for p in (root / 'fake_ae').iterdir()}
        assert first_stems < final_stems  # strict superset: topped up, not redone
        assert len(final_stems) == 2

    def test_failed_ae_load_is_skipped_not_fatal(self, tmp_path):
        source_dir = self._setup(tmp_path, sizes=[(20, 20)])
        root, sources, md5_by_path, done, ok = self._run_real_and_ae(
            tmp_path, source_dir, load_ae=_failing_loader)
        assert ok is False
        assert not (root / 'fake_ae').exists()
        # real/ must still have completed fine regardless of the AE failure
        assert len(list((root / REAL_NAME).iterdir())) == 1
