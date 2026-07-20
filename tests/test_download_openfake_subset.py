"""tests.test_download_openfake_subset —
experiments.scripts.download_openfake_subset, torch-free surface only.

The download() path itself needs `datasets` + `pillow` (Colab-only deps) and
is out of scope here. What's covered: the train/eval leakage guard
(verify_disjoint / --check_disjoint), which reads only manifest.csv and has
no such dependency — exactly why it can run standalone against two
already-downloaded roots with no network access.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scripts.download_openfake_subset import (  # noqa: E402
    MANIFEST_FIELDS, MANIFEST_NAME, main, verify_disjoint,
)


def _write_manifest(root: Path, rows):
    root.mkdir(parents=True, exist_ok=True)
    with (root / MANIFEST_NAME).open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            full = {k: '' for k in MANIFEST_FIELDS}
            full.update(row)
            writer.writerow(full)


def _row(md5, generator='sdxl', label='fake', file_path=None):
    return {
        'file_path': file_path or f'{generator}/{generator}_{md5[:12]}.png',
        'generator': generator, 'label': label, 'model': generator,
        'md5': md5, 'format': 'PNG', 'width': 64, 'height': 64,
    }


class TestVerifyDisjoint:
    def test_disjoint_roots_return_zero(self, tmp_path):
        root_a = tmp_path / 'eval'
        root_b = tmp_path / 'train'
        _write_manifest(root_a, [_row('a' * 32), _row('b' * 32)])
        _write_manifest(root_b, [_row('c' * 32), _row('d' * 32)])
        assert verify_disjoint(root_a, root_b) == 0

    def test_overlapping_md5_detected(self, tmp_path):
        root_a = tmp_path / 'eval'
        root_b = tmp_path / 'train'
        shared = 'e' * 32
        _write_manifest(root_a, [_row(shared), _row('b' * 32)])
        _write_manifest(root_b, [_row(shared), _row('d' * 32)])
        assert verify_disjoint(root_a, root_b) == 1

    def test_multiple_collisions_counted(self, tmp_path):
        root_a = tmp_path / 'eval'
        root_b = tmp_path / 'train'
        shared = ['a' * 32, 'b' * 32, 'c' * 32]
        _write_manifest(root_a, [_row(m) for m in shared] + [_row('x' * 32)])
        _write_manifest(root_b, [_row(m) for m in shared] + [_row('y' * 32)])
        assert verify_disjoint(root_a, root_b) == 3

    def test_missing_manifest_hard_fails(self, tmp_path):
        root_a = tmp_path / 'eval'
        root_b = tmp_path / 'train'  # never created
        _write_manifest(root_a, [_row('a' * 32)])
        with pytest.raises(SystemExit):
            verify_disjoint(root_a, root_b)

    def test_neither_manifest_present_hard_fails(self, tmp_path):
        with pytest.raises(SystemExit):
            verify_disjoint(tmp_path / 'a', tmp_path / 'b')

    def test_empty_manifests_are_disjoint(self, tmp_path):
        root_a = tmp_path / 'eval'
        root_b = tmp_path / 'train'
        _write_manifest(root_a, [])
        _write_manifest(root_b, [])
        assert verify_disjoint(root_a, root_b) == 0

    def test_rows_without_md5_are_ignored(self, tmp_path):
        # A row with a blank md5 (shouldn't happen in practice, but the
        # loader must not treat two blanks as a collision).
        root_a = tmp_path / 'eval'
        root_b = tmp_path / 'train'
        _write_manifest(root_a, [_row('a' * 32), {'generator': 'sdxl', 'label': 'fake', 'md5': ''}])
        _write_manifest(root_b, [_row('b' * 32), {'generator': 'sdxl', 'label': 'fake', 'md5': ''}])
        assert verify_disjoint(root_a, root_b) == 0

    def test_labels_used_in_output(self, tmp_path, capsys):
        root_a = tmp_path / 'eval'
        root_b = tmp_path / 'train'
        _write_manifest(root_a, [_row('a' * 32)])
        _write_manifest(root_b, [_row('a' * 32)])
        verify_disjoint(root_a, root_b, label_a='EVAL', label_b='TRAIN')
        out = capsys.readouterr().out
        assert 'EVAL' in out and 'TRAIN' in out
        assert 'LEAKAGE' in out


class TestCLIDispatch:
    def test_check_disjoint_exits_zero_when_clean(self, tmp_path, monkeypatch):
        root_a = tmp_path / 'eval'
        root_b = tmp_path / 'train'
        _write_manifest(root_a, [_row('a' * 32)])
        _write_manifest(root_b, [_row('b' * 32)])
        monkeypatch.setattr(sys, 'argv', [
            'download_openfake_subset', '--check_disjoint', str(root_a), str(root_b),
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0

    def test_check_disjoint_exits_nonzero_on_collision(self, tmp_path, monkeypatch):
        root_a = tmp_path / 'eval'
        root_b = tmp_path / 'train'
        shared = 'a' * 32
        _write_manifest(root_a, [_row(shared)])
        _write_manifest(root_b, [_row(shared)])
        monkeypatch.setattr(sys, 'argv', [
            'download_openfake_subset', '--check_disjoint', str(root_a), str(root_b),
        ])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_missing_output_dir_without_check_disjoint_fails(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['download_openfake_subset'])
        with pytest.raises(SystemExit):
            main()

    def test_cli_subprocess_check_disjoint(self, tmp_path):
        # End-to-end through `python -m ...`, no mocking of sys.argv/main.
        root_a = tmp_path / 'eval'
        root_b = tmp_path / 'train'
        _write_manifest(root_a, [_row('a' * 32)])
        _write_manifest(root_b, [_row('b' * 32)])
        result = subprocess.run(
            [sys.executable, '-m', 'experiments.scripts.download_openfake_subset',
             '--check_disjoint', str(root_a), str(root_b)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert 'disjoint' in result.stdout
