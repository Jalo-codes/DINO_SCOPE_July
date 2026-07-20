"""experiments.scripts.build_ae_corpus — autoencoder-passthrough corpus cacher.

Phase 0.5 (docs/noise_head_phase0.md §10). Pushes a pool of real photographs
through N off-the-shelf autoencoders — ENCODE -> DECODE only, no diffusion,
no sampling, no prompts — and writes the result straight into the full_fakes
layout (`root/real/` + `root/<ae_name>/`, lab_utils/data/datasets/full_fakes.py)
so every downstream eval/train entry point picks it up with zero glue.

WHY this is the strongest control in the program (§10): every "generator"
class contains the EXACT SAME source images, so semantic content cannot leak
into a learned generator/AE fingerprint by construction — the only thing that
differs between root/real/<stem>.png and root/<ae>/<stem>.png is what the AE's
encode-decode round trip did to it.

Design decisions worth writing down (the ways this experiment lies to you if
skipped):

- **Container hygiene (CLAUDE.md; phase0 §3 P1, generalized to AEs).** If
  root/real/ and root/<ae>/ differ in file format, PIL mode, bit depth, or
  resolution handling, a model trained on this corpus learns file format, not
  AE physics, and reports a beautiful, worthless number. Every image in every
  folder — INCLUDING real/ — goes through the identical path: decode with
  PIL, force `.convert('RGB')`, save via the one `_save_png` function with no
  extra kwargs. real/ is RE-SAVED through this path, not copied byte-for-byte,
  specifically so its container statistics match the AE folders exactly.
  `verify_container_hygiene()` tabulates format/mode/size after the fact and
  is a hard failure, not a warning.

- **Stem pairing.** The output stem is the md5 of the SOURCE file's raw bytes
  (mirrors experiments/scripts/download_openfake_subset.py's resume key). The
  same source image therefore produces the identical stem in every folder by
  construction — real/<md5>.png, sd15_vae/<md5>.png, taesd/<md5>.png, ... —
  which is what makes Type A/B triplet construction (phase0 §10) a plain stem
  lookup instead of a fragile filename-matching problem.

- **Resume.** Keyed the same way as the manifest itself: (source_md5, ae_name)
  pairs already recorded with status='ok' are skipped. Re-running with a new
  --aes entry, or a source_dir that grew, tops up only the missing pairs —
  it never redoes or duplicates finished work. Manifest is LONG format (one
  row per source x ae_name, not one wide row per source) so adding/removing
  AE names across runs never requires a schema migration.

- **Arbitrary resolution -> AE -> exact original resolution.** AEs need H, W
  divisible by their spatial downsample factor (8 for the KL-f8 family, 4 for
  the VQ-f4 arm here). Images are reflect-padded (replicate if the pad would
  exceed the image's own extent) up to the next multiple, run through the AE,
  then CROPPED BACK to the original (H, W) before saving. Output is therefore
  pixel-aligned with the source, which triplet construction assumes.

- **All AEs assumed to share the standard SD [-1, 1] pixel normalization**
  (verified for the KL-f8 family, SDXL, Flux, TAESD, and the LDM VQ-f4 model
  used here — TAESD and Flux's AE are explicitly designed as drop-ins for the
  same latent convention). Each AE_SPECS entry carries its own `input_range`
  so a future AE with a different convention is a one-line addition, not a
  code change.

- **fp16 only, never bf16 (Colab T4 target; Turing has no native bf16).**
  `--dtype` therefore only offers {fp16, fp32} — bf16 is not an option that
  exists to be picked wrong. fp16 on CPU is impractical (many CPU kernels
  don't support it), so a CPU device silently forces fp32 with a warning
  rather than either crashing or silently corrupting output.

- **Batched, not image-at-a-time.** Sources are bucketed by their native
  (H, W) so a GPU batch is only ever formed from images that pad to the same
  shape; batches within a bucket are `--batch_size` images at a time.

- Torch/diffusers ARE needed here (unlike download_openfake_subset.py, which
  is deliberately torch-free) — but both are imported lazily, at the point of
  use, so the resume/manifest/discovery/hygiene machinery stays importable
  and unit-testable on a plain CPU box with neither installed.

Usage (Colab T4)::

    python -m experiments.scripts.build_ae_corpus \\
        --source_dir /content/openfake_ff/real \\
        --output_dir /content/ae_corpus \\
        --batch_size 8 --dtype fp16 \\
        --zip_out /content/drive/MyDrive/DINO_SCOPE_DATA/ae_corpus.zip

    # later, evaluate/train against it like any full_fakes root:
    #   --full_fakes_root /content/ae_corpus
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

MANIFEST_NAME = 'manifest.csv'
# Long format: one row per (source image, ae_name) pair, ae_name == 'real' for
# the re-saved source itself. Avoids a schema migration every time an AE is
# added/removed/renamed across runs (a wide per-source row would not).
MANIFEST_FIELDS = ['source_md5', 'source_path', 'ae_name', 'output_path',
                   'status', 'width', 'height', 'mse', 'psnr']

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'})

REAL_NAME = 'real'


# ── AE registry ──────────────────────────────────────────────────────────────
# `kind` selects the diffusers class in _load_ae; `divisor` is the AE's
# spatial downsample factor (padding target); `input_range` is the pixel
# normalization the AE was trained on. Resolved lazily — see module docstring.
AE_SPECS: List[dict] = [
    dict(name='sd15_vae', repo='stabilityai/sd-vae-ft-mse', kind='kl',
         divisor=8, input_range=(-1.0, 1.0),
         note='SD1.5 KL-f8 VAE, mse-finetuned standalone repo (same '
              'architecture as runwayml/stable-diffusion-v1-5:vae)'),
    dict(name='sdxl_vae', repo='madebyollin/sdxl-vae-fp16-fix', kind='kl',
         divisor=8, input_range=(-1.0, 1.0),
         note='fp16-fix fork — the stock SDXL VAE overflows to NaN in fp16, '
              'the only dtype this corpus is built in on the T4 target'),
    dict(name='flux_vae', repo='black-forest-labs/FLUX.1-schnell', subfolder='vae',
         kind='kl', divisor=8, input_range=(-1.0, 1.0),
         note='Apache-2.0, ungated — 16-channel latent space, architecturally '
              'distinct from the 4-channel SD1.5/SDXL KL-f8 family'),
    dict(name='taesd', repo='madebyollin/taesd', kind='tiny',
         divisor=8, input_range=(-1.0, 1.0),
         note='deliberately different/lower-quality architecture tier '
              '(distilled tiny AE) for reconstruction-artifact diversity'),
    dict(name='vqgan_f4', repo='CompVis/ldm-celebahq-256', subfolder='vqvae',
         kind='vq', divisor=4, input_range=(-1.0, 1.0),
         note='VQ (discrete codebook) architecture — the non-KL diversity arm'),
]


# ── Torch-free helpers (discovery, hashing, resume, hygiene) ─────────────────

def _discover_sources(source_dir: Path) -> List[Path]:
    return sorted(p for p in Path(source_dir).rglob('*')
                  if p.is_file() and p.suffix.lower() in _VALID_EXTS)


def _content_md5(path: Path) -> str:
    import hashlib
    return hashlib.md5(path.read_bytes()).hexdigest()


def _stem_for(md5: str) -> str:
    return md5


def _load_manifest_state(manifest_path: Path) -> Dict[Tuple[str, str], dict]:
    """(source_md5, ae_name) -> row, for every row already recorded status='ok'."""
    done: Dict[Tuple[str, str], dict] = {}
    if manifest_path.exists():
        with manifest_path.open() as fh:
            for row in csv.DictReader(fh):
                if row.get('status') == 'ok':
                    done[(row['source_md5'], row['ae_name'])] = row
    return done


def _bucket_by_size(paths: List[Path]) -> Dict[Tuple[int, int], List[Path]]:
    """Group paths by native (H, W) so a GPU batch pads to one common shape."""
    from collections import defaultdict
    from PIL import Image
    buckets: Dict[Tuple[int, int], List[Path]] = defaultdict(list)
    for p in paths:
        with Image.open(p) as img:
            w, h = img.size
        buckets[(h, w)].append(p)
    return buckets


def _save_png(img, path: Path) -> None:
    """The ONE save path used for real/ and every AE folder — container hygiene."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format='PNG')


def verify_container_hygiene(root: Path) -> List[str]:
    """Tabulate format/PIL-mode/size across root/real/ + root/<ae>/; return problems (empty == clean).

    Two independent checks, both hard-fail material:
      1. Every non-empty folder must share the SAME set of formats and PIL
         modes as every other folder (rule P1 — a format/mode split means the
         model would learn container, not AE physics).
      2. Every AE-folder file must be pixel-aligned (identical (W, H)) with
         root/real/'s file of the same stem — the pixel-alignment contract
         triplet construction assumes. A stem present in an AE folder but not
         (yet) in real/ is a partial-run artifact, not a hygiene failure.
    """
    root = Path(root)
    problems: List[str] = []
    from PIL import Image

    by_folder: Dict[str, dict] = {}
    all_formats: set = set()
    all_modes: set = set()
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in folder.iterdir()
                        if f.is_file() and f.suffix.lower() in _VALID_EXTS)
        if not files:
            continue
        f_formats, f_modes = set(), set()
        for f in files:
            with Image.open(f) as img:
                f_formats.add(img.format)
                f_modes.add(img.mode)
        by_folder[folder.name] = {'n': len(files), 'formats': f_formats, 'modes': f_modes}
        all_formats |= f_formats
        all_modes |= f_modes

    if len(all_formats) > 1:
        detail = {k: v['formats'] for k, v in by_folder.items()}
        problems.append(f'format mismatch across folders (want one shared format): {detail}')
    if len(all_modes) > 1:
        detail = {k: v['modes'] for k, v in by_folder.items()}
        problems.append(f'PIL-mode mismatch across folders (want one shared mode): {detail}')

    real_dir = root / REAL_NAME
    if real_dir.is_dir():
        real_sizes: Dict[str, tuple] = {}
        for f in real_dir.iterdir():
            if f.is_file() and f.suffix.lower() in _VALID_EXTS:
                with Image.open(f) as img:
                    real_sizes[f.stem] = img.size
        for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name != REAL_NAME):
            for f in folder.iterdir():
                if not (f.is_file() and f.suffix.lower() in _VALID_EXTS):
                    continue
                expected = real_sizes.get(f.stem)
                if expected is None:
                    continue  # stem not (yet) in real/ — partial run, not a hygiene bug
                with Image.open(f) as img:
                    if img.size != expected:
                        problems.append(
                            f'{folder.name}/{f.name} size {img.size} != '
                            f'{REAL_NAME}/{f.stem}{f.suffix} size {expected}'
                        )
    return problems


# ── Torch-dependent helpers (imported lazily) ────────────────────────────────

def _load_rgb_tensor(path: Path):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as img:
        img.load()
        img = img.convert('RGB')
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C, H, W) in [0, 1]


def _tensor_to_pil(t):
    import numpy as np
    from PIL import Image
    t = t.clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).float().cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode='RGB')


def _pad_to_multiple(x, multiple: int):
    """Pad (..., H, W) up to the next multiple of `multiple`; return (padded, (h, w))."""
    import torch.nn.functional as F
    h, w = x.shape[-2], x.shape[-1]
    pad_h = (-h) % multiple
    pad_w = (-w) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (h, w)
    # reflect needs the pad amount strictly less than the dimension it pads;
    # only relevant for images smaller than one `multiple` block.
    mode = 'reflect' if pad_h < h and pad_w < w else 'replicate'
    return F.pad(x, (0, pad_w, 0, pad_h), mode=mode), (h, w)


def _crop_to(x, hw: Tuple[int, int]):
    h, w = hw
    return x[..., :h, :w]


def _load_ae(spec: dict, dtype, device: str):
    from diffusers import AutoencoderKL, AutoencoderTiny, VQModel
    loader = {'kl': AutoencoderKL, 'tiny': AutoencoderTiny, 'vq': VQModel}[spec['kind']]
    kwargs = {}
    if spec.get('subfolder'):
        kwargs['subfolder'] = spec['subfolder']
    model = loader.from_pretrained(spec['repo'], torch_dtype=dtype, **kwargs)
    model = model.to(device)
    model.eval()
    return model


def _apply_ae(model, batch):
    """ENCODE -> DECODE only. diffusers' AutoencoderKL/AutoencoderTiny/VQModel
    all implement forward() as exactly this round trip (deterministic — KL
    models use the posterior MODE, not a sampled draw — no diffusion, no
    sampling, no prompts) and all return `.sample`."""
    import torch
    with torch.no_grad():
        return model(batch, return_dict=True).sample


def _run_ae_batch(model, tensors: List, divisor: int, dtype, device,
                   input_range: Tuple[float, float] = (-1.0, 1.0)):
    import torch
    lo, hi = input_range
    batch = torch.stack(tensors).to(device=device, dtype=dtype)
    batch = lo + batch * (hi - lo)                       # [0, 1] -> input_range
    padded, hw = _pad_to_multiple(batch, divisor)
    out = _apply_ae(model, padded)
    out = _crop_to(out, hw)
    out = (out.float() - lo) / (hi - lo)                 # input_range -> [0, 1]
    return out.clamp(0.0, 1.0).cpu()


# ── Orchestration ─────────────────────────────────────────────────────────────

def _materialize_real(sources: List[Path], md5_by_path: Dict[Path, str], root: Path,
                       writer: 'csv.DictWriter', done: Dict[Tuple[str, str], dict],
                       mfh) -> None:
    from PIL import Image
    out_dir = root / REAL_NAME
    for p in sources:
        md5 = md5_by_path[p]
        if (md5, REAL_NAME) in done:
            continue
        with Image.open(p) as img:
            img.load()
            img = img.convert('RGB')
            out_path = out_dir / f'{_stem_for(md5)}.png'
            _save_png(img, out_path)
            w, h = img.size
        row = {'source_md5': md5, 'source_path': str(p), 'ae_name': REAL_NAME,
               'output_path': str(out_path.relative_to(root)), 'status': 'ok',
               'width': w, 'height': h, 'mse': '', 'psnr': ''}
        writer.writerow(row)
        mfh.flush()
        done[(md5, REAL_NAME)] = row


def _materialize_ae(spec: dict, sources: List[Path], md5_by_path: Dict[Path, str],
                     root: Path, batch_size: int, device: str, dtype,
                     writer: 'csv.DictWriter', done: Dict[Tuple[str, str], dict], mfh,
                     load_ae: Callable = _load_ae) -> bool:
    """Returns False if the AE failed to load (caller logs it as skipped)."""
    import torch
    name = spec['name']
    pending = [p for p in sources if (md5_by_path[p], name) not in done]
    if not pending:
        return True

    try:
        model = load_ae(spec, dtype, device)
    except Exception as exc:  # noqa: BLE001 — deliberately broad: "skip, don't fail the run"
        print(f'[skip] {name}: failed to load ({exc})')
        return False

    out_dir = root / name
    for hw, paths in _bucket_by_size(pending).items():
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            tensors = [_load_rgb_tensor(p) for p in chunk]
            recon = _run_ae_batch(model, tensors, spec['divisor'], dtype, device,
                                   input_range=spec.get('input_range', (-1.0, 1.0)))
            for p, orig_t, rec_t in zip(chunk, tensors, recon):
                md5 = md5_by_path[p]
                pil = _tensor_to_pil(rec_t)
                out_path = out_dir / f'{_stem_for(md5)}.png'
                _save_png(pil, out_path)
                mse = float(torch.mean((orig_t - rec_t) ** 2))
                psnr = 99.0 if mse <= 1e-10 else float(-10.0 * torch.log10(torch.tensor(mse)))
                w, h = pil.size
                row = {'source_md5': md5, 'source_path': str(p), 'ae_name': name,
                       'output_path': str(out_path.relative_to(root)), 'status': 'ok',
                       'width': w, 'height': h, 'mse': f'{mse:.6f}', 'psnr': f'{psnr:.3f}'}
                writer.writerow(row)
                mfh.flush()
                done[(md5, name)] = row

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return True


def run(args) -> Path:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / MANIFEST_NAME
    done = _load_manifest_state(manifest_path)
    if done:
        print(f'resuming: {len(done)} (source, ae) pairs already in manifest')

    sources = _discover_sources(Path(args.source_dir))
    if not sources:
        raise SystemExit(f'no images found under {args.source_dir}')
    print(f'{len(sources)} source images found under {args.source_dir}')
    md5_by_path = {p: _content_md5(p) for p in sources}

    specs = AE_SPECS
    if args.aes is not None:
        wanted = set(args.aes)
        specs = [s for s in AE_SPECS if s['name'] in wanted]
        missing = wanted - {s['name'] for s in specs}
        if missing:
            print(f'WARNING: unknown --aes names ignored: {sorted(missing)}')

    import torch
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('WARNING: --device cuda requested but unavailable, falling back to cpu')
        device = 'cpu'
    dtype = {'fp16': torch.float16, 'fp32': torch.float32}[args.dtype]
    if device == 'cpu' and dtype == torch.float16:
        print('WARNING: fp16 on cpu is impractical (unsupported kernels for many '
              'ops) — forcing fp32 for this run')
        dtype = torch.float32

    write_header = not manifest_path.exists()
    with manifest_path.open('a', newline='', encoding='utf-8') as mfh:
        writer = csv.DictWriter(mfh, fieldnames=MANIFEST_FIELDS)
        if write_header:
            writer.writeheader()

        _materialize_real(sources, md5_by_path, root, writer, done, mfh)

        skipped = []
        for spec in specs:
            ok = _materialize_ae(spec, sources, md5_by_path, root, args.batch_size,
                                  device, dtype, writer, done, mfh)
            if not ok:
                skipped.append(spec['name'])
        if skipped:
            print(f'skipped AEs (failed to load — see messages above): {skipped}')

    problems = verify_container_hygiene(root)
    if problems:
        for msg in problems:
            print(f'[HYGIENE FAIL] {msg}')
        raise SystemExit(f'{len(problems)} container-hygiene problem(s) — '
                          f'fix before using this corpus for anything')
    print('container hygiene check: OK (uniform format/mode, pixel-aligned per stem)')
    print(f'done: {root}')
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='build_ae_corpus',
        description='Push a real-image pool through N off-the-shelf autoencoders '
                    '(encode-decode only) into a full_fakes-layout content-controlled '
                    'corpus (docs/noise_head_phase0.md §10).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--source_dir', required=True,
                   help='Directory of real photographs (searched recursively). '
                        'In practice root/real/ of a download_openfake_subset.py run.')
    p.add_argument('--output_dir', required=True,
                   help='Destination root (becomes --full_fakes_root for eval/train)')
    p.add_argument('--aes', nargs='*', default=None,
                   help='Restrict to these AE names (see AE_SPECS: '
                        f'{[s["name"] for s in AE_SPECS]}). Default: all of them.')
    p.add_argument('--batch_size', type=int, default=8,
                   help='Images per GPU batch WITHIN one (H, W) resolution bucket')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['fp16', 'fp32'], default='fp16',
                   help='fp16 for the T4 target. NEVER bf16 — not offered as a '
                        'choice on purpose (Turing has no native bf16 support). '
                        'fp32 is the cpu/debug fallback only.')
    p.add_argument('--zip_out', default=None,
                   help='After building, zip output_dir to this path (e.g. a '
                        'Drive location). ZIP_STORED — images do not deflate.')
    return p


def main() -> None:
    args = _build_parser().parse_args()
    root = run(args)
    if args.zip_out:
        zip_dir(root, args.zip_out)


if __name__ == '__main__':
    main()
