"""experiments.scripts.download_openfake_subset — build a full_fakes-layout
OpenFake eval OR training subset (Colab-oriented).

Supersedes the ad-hoc notebook downloader.  Differences that matter:

- **Writes the full_fakes layout directly** (`root/real/` + `root/<generator>/`,
  see lab_utils/data/datasets/full_fakes.py) so the result plugs into every
  eval entry point via `--full_fakes_root <output_dir>` — eval.py, the zoom
  audit (audit_zoom_image_auc.py), robustness — with zero glue.  The manifest
  is bookkeeping; the DIRECTORY LAYOUT is the formal index.
- **Resume-safe**: every image is keyed by the md5 of its raw bytes; the key
  is both the filename stem and a manifest column.  Re-running with a larger
  --n_per_gen tops up instead of duplicating (the HF stream replays from the
  start on every run — a plain per-generator counter would re-save the same
  head of the stream).
- **Keeps original bytes** (with PIL verify() first — the corrupt-EXIF guard
  from the notebook version is preserved).  Decoding + re-encoding to PNG
  inflated JPEG-sourced images ~5-10x for zero information gain; original
  format/compression is itself signal worth keeping for a detector eval.
  --reencode_png restores the old behavior if ever needed.
- **Zips to Drive** at the end (--zip_out), ZIP_STORED (images don't deflate).
- **Reals are stratified**: all reals live in root/real/ (the full_fakes
  layout wants one real pool) but are CAPPED PER SOURCE COLLECTION
  (--n_per_real_source; laion/pexels/docci/imagenet/reddit), so the shared
  AUROC negative set isn't dominated by whichever source the stream serves
  first.  The source survives in the filename (real_<source>_<md5>) and the
  manifest `model` column.
- **Builds train sets too, via the same --split flag.** `--split` still
  defaults to `'test'` (an EVAL pull) — that default is unchanged, see the
  flag help.  Pass `--split train` to pull from OpenFake's ~2.31M-row core
  training split (all in-distribution generators), or `--split validation`
  for the ~59K held-out-IMAGE split (same generators as train, unseen images
  — the in-distribution readout). This maps onto the generators-as-cameras
  noise-head experiment (docs/noise_head_phase0.md): train learns
  fingerprints, validation is the in-distribution check, test (the existing
  default) is the out-of-distribution generator check. No other machinery
  changes — resume/stratify/original-bytes all apply identically to every
  split.
- **Train/eval leakage is provable, not assumed — and checkable standalone.**
  Because every image is keyed by the md5 of its raw bytes (see
  "Resume-safe" above), a train root and an eval root pulled from disjoint HF
  splits should share zero images by construction, unless the splits
  themselves overlap or a run was pointed at the wrong split. `--check_disjoint
  EVAL_ROOT TRAIN_ROOT` verifies this directly against two already-downloaded
  manifest.csv files — no network, no re-download — and hard-fails (nonzero
  exit, offending md5s + file paths printed) on any collision. Run it after
  any train pull that will sit alongside an existing eval root.

Torch-free; needs `datasets` + `pillow` (+ tqdm if available). The leakage
check (--check_disjoint) needs neither — it only reads manifest.csv.

Usage (Colab, eval/test — the default, unchanged):
    python -m experiments.scripts.download_openfake_subset \
        --output_dir /content/openfake_ff \
        --n_per_gen 100 --n_per_real_source 250 \
        --zip_out /content/drive/MyDrive/DINO_SCOPE_DATA/openfake_ff.zip

Usage (Colab, TRAIN set for generators-as-cameras, docs/noise_head_phase0.md):
    python -m experiments.scripts.download_openfake_subset \
        --output_dir /content/openfake_train --split train \
        --n_per_gen 2000 --n_per_real_source 5000 \
        --stop_after_dry 50000 \
        --zip_out /content/drive/MyDrive/DINO_SCOPE_DATA/openfake_train.zip
    # train is ~2.31M rows across 100+ generators; streaming to exhaustion is
    # impractical, so this is bounded by --stop_after_dry rather than left to
    # run to completion. Keep it GENEROUS: per --stop_after_dry's own caution,
    # the stream can be shard/generator-ordered, so a long dry stretch (every
    # known generator already at cap) can still precede an as-yet-unseen
    # generator later in the shard order — 50000 is a reasonable floor for a
    # first pull, not a tuned value. The run prints a per-generator shortfall
    # warning at the end; rerun (resume-safe, tops up) with a larger
    # --stop_after_dry or --max_scan if generators came up short. If you only
    # want a known generator subset, pass --generators explicitly instead —
    # that early-exits as soon as its targets are met and sidesteps the
    # dry-stretch tradeoff entirely.

Usage (Colab, VALIDATION set — in-distribution eval, same generators as train):
    python -m experiments.scripts.download_openfake_subset \
        --output_dir /content/openfake_val --split validation \
        --n_per_gen 200 --n_per_real_source 500

    # after a train pull, verify no train/eval leakage (standalone, no network):
    python -m experiments.scripts.download_openfake_subset \
        --check_disjoint /content/openfake_ff /content/openfake_train

    # later, evaluate:
    python -m experiments.scripts.audit_zoom_image_auc \
        --checkpoint ... --full_fakes_root /content/openfake_ff ...
"""

import argparse
import csv
import hashlib
import io
import zipfile
from collections import defaultdict
from pathlib import Path

MANIFEST_NAME = 'manifest.csv'
# `generator` = the subfolder (full_fakes indexing unit; 'real' for the real
# pool).  `model` = the dataset's raw model field: the generator name for
# fakes, but the SOURCE COLLECTION for reals (laion/pexels/docci/imagenet/
# reddit) — kept so the reals stratum stays recoverable (real-diversity work).
MANIFEST_FIELDS = ['file_path', 'generator', 'label', 'model', 'gen_type',
                   'release_date', 'md5', 'format', 'width', 'height', 'prompt']

_FORMAT_EXT = {'JPEG': '.jpg', 'PNG': '.png', 'WEBP': '.webp', 'TIFF': '.tif'}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='download_openfake_subset',
        description='Stream OpenFake into a full_fakes-layout eval directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--output_dir', default=None,
                   help='Destination root (becomes --full_fakes_root for eval). '
                        'Required unless --check_disjoint is given.')
    p.add_argument('--check_disjoint', nargs=2, default=None,
                   metavar=('EVAL_ROOT', 'TRAIN_ROOT'),
                   help='Skip downloading. Instead verify manifest.csv md5s in '
                        'EVAL_ROOT and TRAIN_ROOT are disjoint (train/eval leakage '
                        'guard) against two already-downloaded roots and hard-fail '
                        '(nonzero exit) on any collision. No network access.')
    p.add_argument('--n_per_gen', type=int, default=100,
                   help='Target images per generator')
    p.add_argument('--n_per_real_source', type=int, default=250,
                   help='Target images per REAL source collection (laion, '
                        'pexels, docci, imagenet, reddit) — reals all land in '
                        'root/real/ (full_fakes layout) but are capped per '
                        'source so the pool stays stratified, not dominated '
                        'by whichever source the stream serves first')
    p.add_argument('--config_name', default='core')
    p.add_argument('--split', default='test',
                   help="Default 'test' ON PURPOSE: an unqualified invocation "
                        "must keep building an EVAL set, so documented eval "
                        "commands don't silently change meaning. 'train' and "
                        "'validation' are supported (opt-in) — see the module "
                        "docstring for train/validation usage and pair any "
                        "'train' pull with --check_disjoint against your eval "
                        "root before training, since drawing eval images from "
                        "train bakes in leakage.")
    p.add_argument('--max_scan', type=int, default=None,
                   help='Stop after scanning this many stream items regardless '
                        'of fill state (bound a slow stream)')
    p.add_argument('--stop_after_dry', type=int, default=None,
                   help='Stop after this many consecutive scanned rows with no '
                        'new save (coupon-collector tail escape). CAUTION: if '
                        'the stream is ordered by shard/generator, a long dry '
                        'stretch can precede an unseen generator — keep this '
                        'generous (e.g. 20000) or leave unset for a full scan.')
    p.add_argument('--generators', nargs='*', default=None,
                   help='Restrict to these generator names (as they appear in '
                        'the dataset "model" field). When given, the stream '
                        'stops as soon as every listed generator + reals are '
                        'full; otherwise it scans to exhaustion/--max_scan '
                        'because new generators may appear late in the stream.')
    p.add_argument('--reencode_png', action='store_true',
                   help='Legacy behavior: decode + re-save everything as PNG '
                        'instead of keeping original bytes')
    p.add_argument('--zip_out', default=None,
                   help='After downloading, zip output_dir to this path '
                        '(e.g. a Drive location). ZIP_STORED — fast, no bloat.')
    return p


def _safe_name(name: str) -> str:
    return ''.join(c if c.isalnum() or c in '._-' else '_' for c in name)


# Real source collections in OpenFake (dataset card, 2026-07). Used ONLY for
# the --generators early-exit check; unknown sources still download fine.
REAL_SOURCES = ('laion', 'pexels', 'docci', 'imagenet', 'reddit')


def _real_key(model: str) -> str:
    return f'real:{_safe_name(model)}'


def _load_manifest_state(manifest_path: Path):
    """(seen md5s, counts) from the manifest — the single resume authority.

    Counts are keyed by generator name for fakes and 'real:<source>' for reals
    (all reals share root/real/ on disk, but caps are per source collection).
    """
    seen = set()
    counts = defaultdict(int)
    if manifest_path.exists():
        with manifest_path.open() as fh:
            for row in csv.DictReader(fh):
                if not row.get('md5'):
                    continue
                seen.add(row['md5'])
                if row.get('label') == 'real':
                    counts[_real_key(row.get('model') or 'unknown')] += 1
                else:
                    counts[row.get('generator') or 'unknown'] += 1
    return seen, counts


def _load_manifest_md5s(manifest_path: Path) -> dict:
    """md5 -> file_path for every row with an md5 (dedup keeps the first)."""
    rows = {}
    with manifest_path.open() as fh:
        for row in csv.DictReader(fh):
            md5 = row.get('md5')
            if md5 and md5 not in rows:
                rows[md5] = row.get('file_path', '')
    return rows


def verify_disjoint(root_a: Path, root_b: Path, *,
                    label_a: str = 'A', label_b: str = 'B') -> int:
    """Hard leakage check between two already-downloaded roots. No network.

    Every image is keyed by the md5 of its raw bytes (see module docstring),
    so this is PROVABLE disjointness, not an assumption: two roots pulled
    from non-overlapping HF splits should share zero md5s by construction.
    A nonzero return means either the splits themselves overlap or a root was
    built from the wrong split — investigate before training.

    Returns the collision count (0 = disjoint). Raises SystemExit if either
    root has no manifest.csv — there is nothing to verify.
    """
    manifest_a = root_a / MANIFEST_NAME
    manifest_b = root_b / MANIFEST_NAME
    for m, label in ((manifest_a, label_a), (manifest_b, label_b)):
        if not m.exists():
            raise SystemExit(f'no {MANIFEST_NAME} at {m} ({label} root) — '
                             f'nothing to verify')

    rows_a = _load_manifest_md5s(manifest_a)
    rows_b = _load_manifest_md5s(manifest_b)
    collisions = sorted(set(rows_a) & set(rows_b))

    print(f'{label_a}: {len(rows_a)} images  ({manifest_a})')
    print(f'{label_b}: {len(rows_b)} images  ({manifest_b})')

    if collisions:
        print(f'\nLEAKAGE: {len(collisions)} md5 collision(s) between '
              f'{label_a} and {label_b}:')
        shown = collisions[:50]
        for md5 in shown:
            print(f'  {md5}  {label_a}:{rows_a[md5]}  {label_b}:{rows_b[md5]}')
        if len(collisions) > len(shown):
            print(f'  ... and {len(collisions) - len(shown)} more')
    else:
        print(f'\nOK: 0 collisions — {label_a} and {label_b} are disjoint')

    return len(collisions)


def download(args) -> Path:
    try:
        from datasets import Features, Value, load_dataset
    except ImportError:
        raise SystemExit('this script needs the `datasets` package '
                         '(pip install datasets) — it is Colab-oriented')
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME

    seen, counts = _load_manifest_state(manifest_path)
    if seen:
        print(f'resuming: {len(seen)} images already in manifest')

    restrict = set(map(_safe_name, args.generators)) if args.generators else None

    def target_for(key: str) -> int:
        return args.n_per_real_source if key.startswith('real:') else args.n_per_gen

    def all_full() -> bool:
        # Only decidable when the generator set is known up front; real
        # sources come from the documented REAL_SOURCES list.
        if restrict is None:
            return False
        want = restrict | {_real_key(s) for s in REAL_SOURCES}
        return all(counts[k] >= target_for(k) for k in want)

    print(f'streaming ComplexDataLab/OpenFake [{args.config_name}/{args.split}] ...')
    ds = load_dataset('ComplexDataLab/OpenFake', name=args.config_name,
                      split=args.split, streaming=True)
    # Raw bytes, no auto-decode: HF's eager Image decode chokes on corrupt EXIF.
    ds = ds.cast_column('image', Features({'bytes': Value('binary'),
                                           'path': Value('string')}))

    new_rows = 0
    write_header = not manifest_path.exists()
    with manifest_path.open('a', newline='', encoding='utf-8') as mfh:
        writer = csv.DictWriter(mfh, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()

        pbar = tqdm(desc='scanning stream') if tqdm else None
        n_scanned = 0
        n_dry = 0
        for item in ds:
            n_scanned += 1
            n_dry += 1
            if pbar is not None:
                pbar.update(1)
            if args.max_scan and n_scanned >= args.max_scan:
                print(f'hit --max_scan={args.max_scan}, stopping')
                break
            if args.stop_after_dry and n_dry >= args.stop_after_dry:
                print(f'{n_dry} consecutive rows with no new save '
                      f'(--stop_after_dry), stopping')
                break
            if all_full():
                print('all targets met, stopping early')
                break

            label = str(item.get('label', '')).lower()
            is_real = label == 'real'
            # Route by the LABEL column, never by the model field: for reals,
            # model holds the source collection (laion/pexels/...), not
            # emptiness — a model-based fallback would file reals under
            # generator-looking folders (the old notebook script did this).
            model = str(item.get('model') or 'unknown')
            gen = 'real' if is_real else _safe_name(model)
            key = _real_key(model) if is_real else gen
            if restrict is not None and not is_real and gen not in restrict:
                continue
            if counts[key] >= target_for(key):
                continue

            image_data = item.get('image')
            if not image_data or not image_data.get('bytes'):
                continue
            raw = image_data['bytes']
            md5 = hashlib.md5(raw).hexdigest()
            if md5 in seen:
                continue

            # Corrupt-EXIF guard: verify() then reopen (verify closes the fp).
            try:
                img = Image.open(io.BytesIO(raw))
                img.verify()
                img = Image.open(io.BytesIO(raw))
                fmt = img.format or 'PNG'
                width, height = img.size
            except Exception as exc:
                msg = f'skipped corrupt image from {gen}: {exc}'
                (pbar.write(msg) if pbar is not None else print(msg))
                continue

            gen_dir = root / gen
            gen_dir.mkdir(parents=True, exist_ok=True)
            # Reals keep their source collection in the filename so the
            # stratum survives even without the manifest.
            stem = f'real_{_safe_name(model)}_{md5[:12]}' if is_real \
                else f'{gen}_{md5[:12]}'
            if args.reencode_png or fmt not in _FORMAT_EXT:
                out = gen_dir / f'{stem}.png'
                img.save(out)
                fmt = 'PNG'
            else:
                out = gen_dir / f'{stem}{_FORMAT_EXT[fmt]}'
                out.write_bytes(raw)

            writer.writerow({
                'file_path': str(out.relative_to(root)),  # portable inside the zip
                'generator': gen, 'label': label, 'model': model,
                'gen_type': str(item.get('type', '')),
                'release_date': str(item.get('release_date', '')),
                'md5': md5, 'format': fmt,
                'width': width, 'height': height,
                'prompt': str(item.get('prompt', ''))[:2000],
            })
            mfh.flush()
            seen.add(md5)
            counts[key] += 1
            new_rows += 1
            n_dry = 0
            if pbar is not None and new_rows % 25 == 0:
                filled = sum(1 for k, c in counts.items() if c >= target_for(k))
                pbar.set_postfix(saved=new_rows, pools=len(counts), full=filled)
        if pbar is not None:
            pbar.close()

    print('\n' + '=' * 56)
    print(f'{"pool":<32} {"have":>6} / target')
    print('-' * 56)
    for key in sorted(counts):
        print(f'{key:<32} {counts[key]:>6} / {target_for(key)}')
    print('=' * 56)

    # Per-generator balance (generators-as-cameras needs this: a class
    # dominated by whichever generator the stream serves first is bad
    # training data). Report every pool under its target PROMINENTLY, not
    # buried in the table above — includes --generators names that never
    # appeared in the stream at all (count 0, not just "under cap").
    shortfalls = {k: counts[k] for k in counts if counts[k] < target_for(k)}
    if restrict is not None:
        for k in restrict:
            shortfalls.setdefault(k, 0)
    if shortfalls:
        print(f'\nWARNING: {len(shortfalls)} pool(s) short of their requested '
              f'cap (underrepresented generator == class imbalance):')
        for k in sorted(shortfalls):
            have, want = shortfalls[k], target_for(k)
            print(f'  {k:<32} {have:>6} / {want}  (short {want - have})')
    else:
        print('\nall pools reached their target cap')

    print(f'\n{new_rows} new images this run; manifest: {manifest_path}')
    print(f'eval with: --full_fakes_root {root.resolve()}')
    if args.split != 'test':
        print(f"\nNOTE: split='{args.split}' — before training on this root "
              f"alongside any existing eval root, verify disjointness:\n"
              f"  python -m experiments.scripts.download_openfake_subset "
              f"--check_disjoint <EVAL_ROOT> {root.resolve()}")
    return root


def zip_dir(root: Path, zip_out: str) -> None:
    zip_path = Path(zip_out)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in root.rglob('*') if p.is_file())
    print(f'zipping {len(files)} files -> {zip_path} (stored, no recompress)')
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_STORED) as zf:
        for p in files:
            zf.write(p, p.relative_to(root))
    size_gb = zip_path.stat().st_size / 1024 ** 3
    print(f'done: {zip_path} ({size_gb:.2f} GiB)')


def main() -> None:
    args = _build_parser().parse_args()

    if args.check_disjoint:
        root_a, root_b = (Path(p) for p in args.check_disjoint)
        n_collisions = verify_disjoint(root_a, root_b, label_a='EVAL', label_b='TRAIN')
        raise SystemExit(1 if n_collisions else 0)

    if not args.output_dir:
        raise SystemExit('--output_dir is required (unless using --check_disjoint)')

    root = download(args)
    if args.zip_out:
        zip_dir(root, args.zip_out)


if __name__ == '__main__':
    main()
