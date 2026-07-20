"""analysis.audit_openfake_split_overlap — prove two full_fakes roots are disjoint.

Training on the OpenFake TRAIN split while evaluating on a previously-downloaded
TEST split root is only meaningful if the two share no images. That is checkable
rather than assumable: download_openfake_subset.py keys every image by the md5 of
its RAW BYTES, using it as both the filename stem and a manifest column, so
disjointness is a set intersection.

Reads the md5s from each root's manifest.csv when present (fast path) and falls
back to hashing files on disk — the directory layout, not the manifest, is the
formal index (download_openfake_subset.py's docstring), and a run interrupted
mid-stream can legitimately have files whose manifest rows never flushed.

Exits 1 on ANY overlap so it can gate a launch script.

Usage:
    python -m analysis.audit_openfake_split_overlap \
        --train_root /content/openfake_train_ff \
        --eval_root  /content/openfake_ff
    # more than two roots is fine — every pair is checked
    python -m analysis.audit_openfake_split_overlap --roots A B C
"""

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Set, Tuple

MANIFEST_NAME = 'manifest.csv'
_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'})


def _md5_of_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def collect(root: Path) -> Tuple[Dict[str, Path], int, int]:
    """md5 -> representative path for every image under root.

    Returns (mapping, n_from_manifest, n_hashed). Manifest rows win; files not
    covered by the manifest are hashed so an interrupted download still audits.
    """
    by_md5: Dict[str, Path] = {}
    covered: Set[Path] = set()
    n_manifest = 0

    manifest = root / MANIFEST_NAME
    if manifest.exists():
        with manifest.open(newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                md5 = (row.get('md5') or '').strip()
                rel = (row.get('file_path') or '').strip()
                if not md5:
                    continue
                path = (root / rel) if rel else root
                by_md5.setdefault(md5, path)
                covered.add(path.resolve())
                n_manifest += 1

    n_hashed = 0
    for path in sorted(root.rglob('*')):
        if not path.is_file() or path.suffix.lower() not in _VALID_EXTS:
            continue
        if path.resolve() in covered:
            continue
        by_md5.setdefault(_md5_of_file(path), path)
        n_hashed += 1

    return by_md5, n_manifest, n_hashed


def _pool_of(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).parts[0]
    except (ValueError, IndexError):
        return '?'


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog='audit_openfake_split_overlap',
        description='Fail if two full_fakes roots share any image (md5 of raw bytes).',
    )
    p.add_argument('--train_root', default=None)
    p.add_argument('--eval_root', default=None)
    p.add_argument('--roots', nargs='*', default=None,
                   help='Alternative to --train_root/--eval_root; every pair is checked')
    p.add_argument('--show', type=int, default=10, help='Max colliding files to print per pair')
    args = p.parse_args(argv)

    names = list(args.roots or [r for r in (args.train_root, args.eval_root) if r])
    if len(names) < 2:
        p.error('need at least two roots (--train_root + --eval_root, or --roots A B ...)')

    roots = [Path(n) for n in names]
    missing = [r for r in roots if not r.is_dir()]
    if missing:
        for r in missing:
            print(f'ERROR: root not found: {r}')
        return 2

    tables = {}
    for r in roots:
        by_md5, n_man, n_hash = collect(r)
        tables[r] = by_md5
        print(f'{r}: {len(by_md5)} unique images '
              f'(manifest rows={n_man}, hashed off disk={n_hash})')

    failed = False
    for a, b in combinations(roots, 2):
        shared = tables[a].keys() & tables[b].keys()
        if not shared:
            print(f'\nOK: {a.name} vs {b.name} — DISJOINT (0 shared md5s)')
            continue

        failed = True
        pct_a = 100.0 * len(shared) / max(1, len(tables[a]))
        pct_b = 100.0 * len(shared) / max(1, len(tables[b]))
        print(f'\nFAIL: {a.name} vs {b.name} — {len(shared)} SHARED images '
              f'({pct_a:.2f}% of {a.name}, {pct_b:.2f}% of {b.name})')

        by_pool = defaultdict(int)
        for md5 in shared:
            by_pool[(_pool_of(a, tables[a][md5]), _pool_of(b, tables[b][md5]))] += 1
        print('  collisions by pool:')
        for (pa, pb), n in sorted(by_pool.items(), key=lambda kv: -kv[1]):
            print(f'    {n:>6}  {a.name}/{pa}  <->  {b.name}/{pb}')

        print(f'  first {min(args.show, len(shared))} colliding files:')
        for md5 in sorted(shared)[:args.show]:
            print(f'    {md5}  {tables[a][md5]}  <->  {tables[b][md5]}')

    if failed:
        print('\nOVERLAP DETECTED — the eval set is contaminated. Remove the shared '
              'images from the TRAIN root (never from eval) before training.')
        return 1

    print('\nAll pairs disjoint.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
