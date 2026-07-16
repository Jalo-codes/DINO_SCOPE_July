"""experiments.scripts.download_openfake_subset — build a full_fakes-layout
OpenFake eval subset (Colab-oriented).

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

Torch-free; needs `datasets` + `pillow` (+ tqdm if available).

Usage (Colab):
    python -m experiments.scripts.download_openfake_subset \
        --output_dir /content/openfake_ff \
        --n_per_gen 100 --n_per_real_source 250 \
        --zip_out /content/drive/MyDrive/DINO_SCOPE_DATA/openfake_ff.zip

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
    p.add_argument('--output_dir', required=True,
                   help='Destination root (becomes --full_fakes_root for eval)')
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
                   help="Default 'test' ON PURPOSE: this builds an EVAL set, "
                        "and training on OpenFake train is on the roadmap — "
                        "drawing eval images from train would bake in leakage.")
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
    print(f'{new_rows} new images this run; manifest: {manifest_path}')
    print(f'eval with: --full_fakes_root {root.resolve()}')
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
    root = download(args)
    if args.zip_out:
        zip_dir(root, args.zip_out)


if __name__ == '__main__':
    main()
