"""tests.test_build_ae_corpus — torch-free tier of
experiments.scripts.build_ae_corpus: discovery, content hashing, manifest
resume, and container-hygiene verification. Pure pathlib + PIL, no torch —
runs on any CPU dev box. The pad/crop + full real+AE pipeline (stem pairing,
pixel alignment, resume-in-practice) needs torch and lives in
tests/test_build_ae_corpus_pipeline.py — split out because a failed
module-level `pytest.importorskip('torch')` skips the WHOLE file at
collection time, which would otherwise take these torch-free tests down too.
"""

import csv
import sys
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.build_ae_corpus import (   # noqa: E402
    MANIFEST_FIELDS,
    _bucket_by_size,
    _content_md5,
    _discover_sources,
    _load_manifest_state,
    _stem_for,
    verify_container_hygiene,
)


def _write_image(path: Path, size=(40, 24), color=(120, 130, 140), mode='RGB', fmt=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', size, color)
    if mode != 'RGB':
        img = img.convert(mode)
    img.save(path, format=fmt)


# ── Torch-free: discovery, hashing, resume, hygiene ──────────────────────────

class TestDiscoverSources:
    def test_finds_valid_exts_recursively(self, tmp_path):
        _write_image(tmp_path / 'a.png')
        _write_image(tmp_path / 'sub' / 'b.jpg', fmt='JPEG')
        (tmp_path / 'readme.txt').write_text('not an image')
        found = _discover_sources(tmp_path)
        assert {p.name for p in found} == {'a.png', 'b.jpg'}

    def test_empty_dir(self, tmp_path):
        assert _discover_sources(tmp_path) == []


class TestContentHash:
    def test_same_bytes_same_hash(self, tmp_path):
        _write_image(tmp_path / 'a.png', color=(1, 2, 3))
        _write_image(tmp_path / 'b.png', color=(1, 2, 3))
        assert _content_md5(tmp_path / 'a.png') == _content_md5(tmp_path / 'b.png')

    def test_different_bytes_different_hash(self, tmp_path):
        _write_image(tmp_path / 'a.png', color=(1, 2, 3))
        _write_image(tmp_path / 'b.png', color=(4, 5, 6))
        assert _content_md5(tmp_path / 'a.png') != _content_md5(tmp_path / 'b.png')

    def test_stem_is_the_hash(self):
        assert _stem_for('deadbeef') == 'deadbeef'


class TestManifestResume:
    def test_missing_manifest_is_empty(self, tmp_path):
        assert _load_manifest_state(tmp_path / 'manifest.csv') == {}

    def test_only_ok_rows_count_as_done(self, tmp_path):
        manifest = tmp_path / 'manifest.csv'
        with manifest.open('w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            w.writeheader()
            w.writerow({'source_md5': 'aaa', 'source_path': 'x', 'ae_name': 'real',
                        'output_path': 'real/aaa.png', 'status': 'ok',
                        'width': 4, 'height': 4, 'mse': '', 'psnr': ''})
            w.writerow({'source_md5': 'bbb', 'source_path': 'y', 'ae_name': 'sd15_vae',
                        'output_path': '', 'status': 'error',
                        'width': '', 'height': '', 'mse': '', 'psnr': ''})
        done = _load_manifest_state(manifest)
        assert ('aaa', 'real') in done
        assert ('bbb', 'sd15_vae') not in done


class TestBucketBySize:
    def test_groups_by_native_h_w(self, tmp_path):
        _write_image(tmp_path / 'wide.png', size=(40, 20))
        _write_image(tmp_path / 'wide2.png', size=(40, 20))
        _write_image(tmp_path / 'tall.png', size=(20, 40))
        buckets = _bucket_by_size(_discover_sources(tmp_path))
        assert buckets[(20, 40)] == [tmp_path / 'wide.png', tmp_path / 'wide2.png']
        assert buckets[(40, 20)] == [tmp_path / 'tall.png']


class TestContainerHygiene:
    def test_clean_corpus_has_no_problems(self, tmp_path):
        _write_image(tmp_path / 'real' / '0000.png', size=(32, 24))
        _write_image(tmp_path / 'fake_ae' / '0000.png', size=(32, 24))
        assert verify_container_hygiene(tmp_path) == []

    def test_format_mismatch_is_flagged(self, tmp_path):
        _write_image(tmp_path / 'real' / '0000.png', size=(32, 24))
        _write_image(tmp_path / 'fake_ae' / '0000.jpg', size=(32, 24), fmt='JPEG')
        problems = verify_container_hygiene(tmp_path)
        assert any('format mismatch' in p for p in problems)

    def test_mode_mismatch_is_flagged(self, tmp_path):
        _write_image(tmp_path / 'real' / '0000.png', size=(32, 24), mode='RGB')
        _write_image(tmp_path / 'fake_ae' / '0000.png', size=(32, 24), mode='L')
        problems = verify_container_hygiene(tmp_path)
        assert any('PIL-mode mismatch' in p for p in problems)

    def test_size_mismatch_against_real_is_flagged(self, tmp_path):
        _write_image(tmp_path / 'real' / 'abc.png', size=(32, 24))
        _write_image(tmp_path / 'fake_ae' / 'abc.png', size=(16, 12))
        problems = verify_container_hygiene(tmp_path)
        assert any('size' in p and 'abc' in p for p in problems)

    def test_partial_run_stem_not_yet_in_real_is_not_a_failure(self, tmp_path):
        _write_image(tmp_path / 'real' / 'only_here.png', size=(32, 24))
        _write_image(tmp_path / 'fake_ae' / 'not_in_real_yet.png', size=(99, 99))
        assert verify_container_hygiene(tmp_path) == []
