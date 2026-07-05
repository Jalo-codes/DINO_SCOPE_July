"""experiments.scripts.export_pico_masks — offline pseudo-mask export for PicoBanana.

Runs the raw-DINOv3 feature-diff prototype (experiments/labs/dino_diff_lab.py)
over real/modified PicoBanana pairs and materializes a triplet dataset on disk
for the ``pico_pseudo`` builder::

    out_root/modified/<case_id>.png      edited image, crop_frac-CROPPED, lossless PNG
    out_root/original/<case_id>.png      source image, crop_frac-CROPPED, lossless PNG
    out_root/mask/<case_id>_mask.png     pseudo-mask at the SAME cropped geometry
    out_root/export_format.json          {'version': 2, 'crop_baked_in': True, ...}

THE CROP IS BAKED INTO THE DATA. The border trim that stabilizes the diff
(crop_frac — Gemini's re-encode fingerprints the frame edge) is applied to the
exported images themselves, and the mask is rendered at exactly that cropped
size. Image size == mask size on disk, by construction; no loader, trainer, or
eval script performs any geometry adjustment downstream (the old runtime
edge_crop_frac threading is gone).

Cropping forces a re-encode, so images are saved as LOSSLESS PNG: the decoded
pixels are identical to the cropped region of the decoded source — only the
container changes, no recompression perturbs the statistics the detector
trains on.

v1 exports (full-frame verbatim copies + zero-border masks) are INCOMPATIBLE
and are refused at startup: a populated out_root without the v2 format marker
aborts with instructions to use a fresh directory.

NOTE: any mask files already present under the PicoBanana root are known
noise and are never read — this script only consumes originals/ + modified/
via the pico_banana indexer (which likewise ignores them).

Mask cleanup (hot_mask, in order): threshold → drop straggler components
smaller than hot_min_patches (the patches leave the TP set; the image stays)
→ plug fully-enclosed background holes (a splice's interior is part of the
splice; no false-negative holes in the pseudo-mask). Pairs whose real/modified
native sizes differ are dropped (pair_size_mismatch) so all three exported
files always share one geometry. Preview the whole operating point with
experiments/scripts/viz_pico_masks.py before a full run.

Decisiveness filter — a pair is exported only if its diff map splits cleanly:
  * otsu_eta   between-class/total variance ratio at the Otsu split (bimodality,
                in [0,1]) must be >= min_otsu_eta. Calibration note: a pure
                unimodal Gaussian already scores ~0.65 on this criterion, a
                cleanly bimodal map ~0.99 — hence the 0.75 default. Every
                pair's eta lands in the manifest, so re-tune from real data;
  * hot_frac   fraction of hot patches must lie in [min_hot_frac, max_hot_frac]
                (empty and near-full-frame masks are both rejected);
  * hot_mask's own min_patches component filter applies before both checks.
Every processed pair (kept or dropped, with stats + reason) is recorded in
``out_root/export_manifest.json`` so rejects can be eyeballed.

Throughput on an L4 (or any single GPU): pairs are processed in GPU batches
(``batch_size`` pairs = 2*batch_size images in ONE backbone forward call, via
torch.inference_mode, no autograd graph), with PIL image loading/cropping and
final disk writes (copy + mask save) done on a background thread pool so I/O
overlaps the next batch's GPU compute instead of stalling it. The neighborhood
-diff math is fully vectorized over the batch dimension (verified numerically
equivalent to the single-pair version in dino_diff_lab.py — identical max abs
diff of 0.0 across a randomized check). If a batch overflows GPU memory, it is
bisected and retried automatically, and batch_size is permanently halved for
the remainder of the run so later batches don't repeat the stall.

Processing continues until ``n_pairs`` pairs are KEPT (or the pool runs out),
sampled round-robin across edit categories so no category dominates. A run may
keep up to batch_size-1 pairs beyond n_pairs (the batch that crosses the
target is finished, not truncated mid-batch). Already-exported cases are
skipped on re-run (Colab-disconnect friendly).

Usage (Colab)::

    from experiments.scripts.export_pico_masks import run_export
    run_export(root='/content/pico_banana_native_s3',
               out_root='/content/pico_gemini_triplets',
               n_pairs=4000,
               zip_out='/content/drive/MyDrive/DINO_SCOPE_DATA/pico_gemini_triplets.zip')

CLI::

    python -m experiments.scripts.export_pico_masks \\
        --root /content/pico_banana_native_s3 \\
        --out_root /content/pico_gemini_triplets \\
        --n_pairs 4000 \\
        --zip_out /content/drive/MyDrive/DINO_SCOPE_DATA/pico_gemini_triplets.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiments.labs.dino_diff_lab import (
    CROP_FRAC,
    HOT_MIN_PATCHES,
    HOT_PERCENTILE,
    HOT_THRESH_MULT,
    MAX_HOT_FRAC,
    MIN_HOT_FRAC,
    MIN_OTSU_ETA,
    PLUG_HOLES,
    _DEFAULT_MODEL_NAME,
    _crop_edges,
    _group_case_pairs,
    _load_raw_backbone,
    _round_robin_pairs,
    _safe_name,
    hot_mask,
    otsu_eta,
    render_cropped_mask,
)
from lab_utils.data.datasets.registry import build as build_source
from lab_utils.data.resolution import Resolution
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.logging.text import log_line

# The operating point (CROP_FRAC, HOT_*, PLUG_HOLES, MIN_*/MAX_*), the model
# default (_DEFAULT_MODEL_NAME) and the
# shared helpers (otsu_eta, render_cropped_mask, ...) live in
# experiments.labs.dino_diff_lab — one source of truth for this script and
# viz_pico_masks.py (scripts must not import each other).


# ── batched GPU inference ───────────────────────────────────────────────────────
# Batched analogs of dino_diff_lab.encode_patches / neighborhood_max_diff — one
# backbone call and one vectorized diff per batch instead of per pair. The
# neighborhood-max logic is unchanged except for a leading batch dim; verified
# numerically equivalent to the single-pair version (max abs diff 0.0 over a
# randomized check against dino_diff_lab.neighborhood_max_diff's algorithm).

def _encode_patches_batch(backbone: torch.nn.Module, x: torch.Tensor, res: Resolution) -> torch.Tensor:
    """(B, 3, S, S) -> (B, num_patches, feat_dim), L2-normalized, fp32."""
    with torch.inference_mode():
        out = backbone(pixel_values=x).last_hidden_state
    feats = out[:, -res.num_patches:, :].float()
    return F.normalize(feats, dim=-1)


def _pool_grid_batch(feats: torch.Tensor, grid_hw: Tuple[int, int], ksize: int) -> torch.Tensor:
    if ksize <= 1:
        return feats
    b, rows, cols = feats.shape[0], *grid_hw
    grid = feats.reshape(b, rows, cols, -1).permute(0, 3, 1, 2)
    pad = ksize // 2
    pooled = F.avg_pool2d(grid, kernel_size=ksize, stride=1, padding=pad, count_include_pad=False)
    pooled = pooled.permute(0, 2, 3, 1).reshape(b, rows * cols, -1)
    return F.normalize(pooled, dim=-1)


def _neighborhood_max_diff_batch(
    feats_real: torch.Tensor,
    feats_mod: torch.Tensor,
    grid_hw: Tuple[int, int],
    *,
    radius: int = 1,
    pool_ksize: int = 1,
) -> np.ndarray:
    """Batched change-score maps: (B, num_patches, D) x2 -> (B, rows, cols) float32."""
    b, rows, cols = feats_real.shape[0], *grid_hw
    a = _pool_grid_batch(feats_real, grid_hw, pool_ksize).reshape(b, rows, cols, -1)
    m = _pool_grid_batch(feats_mod, grid_hw, pool_ksize).reshape(b, rows, cols, -1)

    best_sim = torch.full((b, rows, cols), -1.0, dtype=a.dtype, device=a.device)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r0, r1 = max(0, -dr), rows - max(0, dr)
            c0, c1 = max(0, -dc), cols - max(0, dc)
            if r0 >= r1 or c0 >= c1:
                continue
            m_win = m[:, r0:r1, c0:c1]
            a_win = a[:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            sim = (m_win * a_win).sum(dim=-1)
            best_sim[:, r0:r1, c0:c1] = torch.maximum(best_sim[:, r0:r1, c0:c1], sim)

    diff = (1.0 - best_sim).clamp(min=0.0)
    return diff.cpu().numpy().astype(np.float32)


def _load_pair(real_path, mod_path, crop_frac: float, res: Resolution):
    """CPU-only (PIL decode + crop + normalize); safe to run in a thread —
    holds the GIL only briefly per call (PIL/numpy release it for the bulk of
    the work), so many of these overlap real GPU compute on the main thread."""
    from PIL import Image as PILImage

    real_img = PILImage.open(real_path).convert('RGB')
    mod_img = PILImage.open(mod_path).convert('RGB')
    native_sizes = (mod_img.size, real_img.size)
    real_t = load_image_tensor(_crop_edges(real_img, crop_frac), res, device=None, add_batch_dim=False)
    mod_t = load_image_tensor(_crop_edges(mod_img, crop_frac), res, device=None, add_batch_dim=False)
    return real_t, mod_t, native_sizes


def _write_outputs(mod_src, real_src, mod_out, orig_out, mask_img, mask_file, crop_frac):
    """Crop both images by crop_frac and save LOSSLESS PNG (see module doc);
    runs on the I/O thread pool, overlapped with the next batch's GPU work."""
    from PIL import Image as PILImage

    for src, out in ((mod_src, mod_out), (real_src, orig_out)):
        img = PILImage.open(src).convert('RGB')
        _crop_edges(img, crop_frac).save(out)  # .png suffix → lossless
    mask_img.save(mask_file)


# ── export ─────────────────────────────────────────────────────────────────────

def run_export(
    root: str,
    out_root: str,
    *,
    source: str = 'pico_banana',
    n_pairs: int = 4000,
    image_size: int = 688,
    model_name: str = _DEFAULT_MODEL_NAME,
    radius: int = 1,
    pool_ksize: int = 1,
    crop_frac: float = CROP_FRAC,
    hot_percentile=HOT_PERCENTILE,
    hot_thresh_mult: float = HOT_THRESH_MULT,
    hot_min_patches: int = HOT_MIN_PATCHES,
    plug_holes: bool = PLUG_HOLES,
    min_otsu_eta: float = MIN_OTSU_ETA,
    min_hot_frac: float = MIN_HOT_FRAC,
    max_hot_frac: float = MAX_HOT_FRAC,
    device: str = 'cuda',
    dtype: str = 'fp16',
    batch_size: int = 16,
    io_workers: int = 8,
    seed: int = 42,
    zip_out: Optional[str] = None,
) -> Dict:
    """Export up to n_pairs KEPT pseudo-mask triplets; returns summary dict.

    hot_* defaults are the visually validated operating point (otsu @
    thresh_mult=0.5 — tuned for TIGHT masks, precision over recall).
    dtype: backbone inference dtype ('fp16' is the fast path on an L4/T4;
    masks are thresholded so the tiny numeric drift vs fp32 is immaterial).
    batch_size: pairs per GPU forward call (2*batch_size images/call — real
    and modified batches are concatenated into one backbone invocation).
    Tune upward while VRAM allows; auto-halves permanently on OOM.
    io_workers: threads for PIL decode/crop and final disk writes, overlapped
    with GPU compute of the surrounding batches.
    """
    dev = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    if dev.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    res = Resolution(image_size=image_size, patch_size=16)
    grid_hw = (res.num_patches_per_side, res.num_patches_per_side)

    out_path = Path(out_root)
    mod_dir, orig_dir, mask_dir = out_path / 'modified', out_path / 'original', out_path / 'mask'

    # Refuse to resume into a v1 export (full-frame verbatim copies + zero-border
    # masks): mixing geometries in one triplet dir would be a silent train-time
    # misalignment. The v2 marker also pins crop_frac so a resume can't change it.
    format_path = out_path / 'export_format.json'
    if format_path.exists():
        with open(format_path) as f:
            fmt = json.load(f)
        if fmt.get('version') != 2 or fmt.get('crop_frac') != crop_frac:
            raise RuntimeError(
                f'export_pico_masks: {out_root} was exported with format={fmt}, '
                f'this run wants version=2 crop_frac={crop_frac}. Use a fresh '
                f'out_root — geometries must not be mixed in one triplet dir.'
            )
    elif mask_dir.is_dir() and any(mask_dir.iterdir()):
        raise RuntimeError(
            f'export_pico_masks: {out_root} contains masks but no '
            f'export_format.json — this is a v1 (full-frame) export, which is '
            f'incompatible with the baked-in-crop v2 layout. Use a fresh '
            f'out_root; discard the v1 triplets.'
        )

    for d in (mod_dir, orig_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)
    with open(format_path, 'w') as f:
        json.dump({'version': 2, 'crop_baked_in': True, 'crop_frac': crop_frac}, f, indent=2)

    _, val_ds = build_source(source, Path(root).expanduser(), res=res)
    pairs = _group_case_pairs(val_ds.items)
    if not pairs:
        raise RuntimeError(f'export_pico_masks: no complete pairs for source={source!r} root={root!r}')
    ordered = _round_robin_pairs(pairs, seed)
    log_line(f'[dd] export: {len(ordered)} candidate pairs, target n_pairs={n_pairs}, '
             f'batch_size={batch_size}, io_workers={io_workers}')

    backbone = _load_raw_backbone(model_name, dev)
    torch_dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[dtype]
    if torch_dtype is not torch.float32:
        backbone = backbone.to(torch_dtype)

    records: List[dict] = []
    n_kept = n_dropped = n_resumed = 0
    state = {'batch_size': max(1, int(batch_size))}

    io_pool = ThreadPoolExecutor(max_workers=io_workers)
    write_futures: List = []

    def _drain(exhaustive: bool = False) -> None:
        nonlocal write_futures
        if exhaustive:
            for f in write_futures:
                f.result()  # surface exceptions from the write thread
            write_futures = []
        else:
            still_pending = []
            for f in write_futures:
                if f.done():
                    f.result()  # surface exceptions from the write thread now, while cheap
                else:
                    still_pending.append(f)
            write_futures = still_pending

    def _run_forward(batch_pairs: List[Tuple[str, Dict, str]]):
        """Load + GPU-forward one batch. Bisects and retries on CUDA OOM,
        permanently shrinking state['batch_size'] so later batches don't stall."""
        try:
            load_results = list(io_pool.map(
                lambda item: _load_pair(item[1]['real'].image, item[1]['modified'].image, crop_frac, res),
                batch_pairs,
            ))
            real_list = [r[0] for r in load_results]
            mod_list = [r[1] for r in load_results]
            native_sizes = [r[2] for r in load_results]
            stacked = torch.stack(real_list + mod_list).to(dev).to(torch_dtype)

            feats = _encode_patches_batch(backbone, stacked, res)
            b = len(batch_pairs)
            feats_real, feats_mod = feats[:b], feats[b:]
            diff_maps = _neighborhood_max_diff_batch(
                feats_real, feats_mod, grid_hw, radius=radius, pool_ksize=pool_ksize)
            return list(zip(batch_pairs, diff_maps, native_sizes))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch_pairs) == 1:
                raise
            state['batch_size'] = max(1, len(batch_pairs) // 2)
            log_line(f'[dd] export WARN: CUDA OOM at batch={len(batch_pairs)}; '
                     f'bisecting, batch_size now {state["batch_size"]}')
            mid = len(batch_pairs) // 2
            return _run_forward(batch_pairs[:mid]) + _run_forward(batch_pairs[mid:])

    pending_iter = iter(ordered)
    while n_kept < n_pairs:
        batch_pairs: List[Tuple[str, Dict, str]] = []
        for case_id, d in pending_iter:
            name = _safe_name(case_id)
            mask_file = mask_dir / f'{name}_mask.png'
            if mask_file.exists():  # resume: already exported on a prior run
                n_kept += 1
                n_resumed += 1
                if n_kept >= n_pairs:
                    break
                continue
            batch_pairs.append((case_id, d, name))
            if len(batch_pairs) >= state['batch_size']:
                break
        if not batch_pairs:
            break  # pool exhausted (or hit target while skipping resumed items)

        for (case_id, d, name), diff_map, (mod_native_size, real_native_size) in _run_forward(batch_pairs):
            real_it, mod_it = d['real'], d['modified']
            category = real_it.meta.get('category', '')
            mask_file = mask_dir / f'{name}_mask.png'

            hot = hot_mask(diff_map, grid_hw, percentile=hot_percentile,
                           thresh_mult=hot_thresh_mult, min_patches=hot_min_patches,
                           plug_holes=plug_holes)
            eta = otsu_eta(diff_map)
            hot_frac = float(hot.mean())

            reason = None
            if real_native_size != mod_native_size:
                # Both images get the same fractional crop, so equal native
                # sizes are what guarantee all three exported files share one
                # geometry. Unequal source pairs are dropped, never exported.
                reason = f'pair_size_mismatch real={real_native_size} mod={mod_native_size}'
            elif eta < min_otsu_eta:
                reason = f'otsu_eta {eta:.3f} < {min_otsu_eta}'
            elif hot_frac < min_hot_frac:
                reason = f'hot_frac {hot_frac:.4f} < {min_hot_frac}'
            elif hot_frac > max_hot_frac:
                reason = f'hot_frac {hot_frac:.4f} > {max_hot_frac}'

            rec = {'case_id': case_id, 'category': category,
                   'otsu_eta': round(eta, 4), 'hot_frac': round(hot_frac, 4),
                   'diff_mean': round(float(diff_map.mean()), 4),
                   'diff_max': round(float(diff_map.max()), 4)}

            if reason is not None:
                n_dropped += 1
                rec.update(kept=False, reason=reason)
                records.append(rec)
                continue

            # Cropped, lossless-PNG exports — crop baked in, mask at identical
            # geometry. Writes run on the I/O pool, overlapped with GPU work.
            mod_out = mod_dir / f'{name}.png'
            orig_out = orig_dir / f'{name}.png'
            mask_img = render_cropped_mask(hot, mod_native_size, crop_frac)
            write_futures.append(io_pool.submit(
                _write_outputs, mod_it.image, real_it.image, mod_out, orig_out,
                mask_img, mask_file, crop_frac))

            n_kept += 1
            rec.update(kept=True, reason=None)
            records.append(rec)

        _drain()
        log_line(f'[dd] export: kept={n_kept}/{n_pairs} dropped={n_dropped} '
                 f'(resumed={n_resumed}) batch_size={state["batch_size"]}')

    _drain(exhaustive=True)
    io_pool.shutdown(wait=True)

    summary = {
        'config': {
            'root': str(root), 'source': source, 'n_pairs': n_pairs,
            'image_size': image_size, 'model_name': model_name,
            'radius': radius, 'pool_ksize': pool_ksize, 'crop_frac': crop_frac,
            'hot_percentile': str(hot_percentile), 'hot_thresh_mult': hot_thresh_mult,
            'hot_min_patches': hot_min_patches, 'plug_holes': plug_holes,
            'min_otsu_eta': min_otsu_eta,
            'min_hot_frac': min_hot_frac, 'max_hot_frac': max_hot_frac,
            'dtype': dtype, 'batch_size': batch_size, 'io_workers': io_workers, 'seed': seed,
        },
        'n_kept': n_kept, 'n_dropped': n_dropped, 'n_resumed': n_resumed,
        'final_batch_size': state['batch_size'],
        'records': records,
    }
    with open(out_path / 'export_manifest.json', 'w') as f:
        json.dump(summary, f, indent=1)

    log_line(f'[dd] export done: kept={n_kept} (resumed={n_resumed}) dropped={n_dropped} '
             f'→ {out_path} (manifest: export_manifest.json)')

    if zip_out is not None:
        zip_path = Path(zip_out)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        log_line(f'[dd] zipping {out_path} → {zip_path} ...')
        made = shutil.make_archive(str(zip_path.with_suffix('')), 'zip',
                                   root_dir=out_path.parent, base_dir=out_path.name)
        log_line(f'[dd] zip written: {made}')
        summary['zip'] = made

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog='export_pico_masks',
        description='Export PicoBanana pseudo-mask triplets (raw-DINO diff masks).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--root', required=True, help='PicoBanana dataset root')
    p.add_argument('--out_root', required=True, help='Output triplet dataset root')
    p.add_argument('--n_pairs', type=int, default=4000, help='Target KEPT pairs')
    p.add_argument('--image_size', type=int, default=688)
    p.add_argument('--model_name', default=_DEFAULT_MODEL_NAME)
    p.add_argument('--radius', type=int, default=1)
    p.add_argument('--pool_ksize', type=int, default=1)
    p.add_argument('--crop_frac', type=float, default=CROP_FRAC)
    p.add_argument('--hot_thresh_mult', type=float, default=HOT_THRESH_MULT)
    p.add_argument('--hot_min_patches', type=int, default=HOT_MIN_PATCHES)
    p.add_argument('--min_otsu_eta', type=float, default=MIN_OTSU_ETA)
    p.add_argument('--min_hot_frac', type=float, default=MIN_HOT_FRAC)
    p.add_argument('--max_hot_frac', type=float, default=MAX_HOT_FRAC)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    p.add_argument('--dtype', default='fp16', choices=['fp32', 'fp16', 'bf16'])
    p.add_argument('--batch_size', type=int, default=16, help='Pairs per GPU forward call')
    p.add_argument('--io_workers', type=int, default=8, help='Threads for image I/O')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--zip_out', default=None, help='Optional .zip destination (e.g. on Drive)')
    a = p.parse_args()

    run_export(
        root=a.root, out_root=a.out_root, n_pairs=a.n_pairs,
        image_size=a.image_size, model_name=a.model_name,
        radius=a.radius, pool_ksize=a.pool_ksize, crop_frac=a.crop_frac,
        hot_thresh_mult=a.hot_thresh_mult, hot_min_patches=a.hot_min_patches,
        min_otsu_eta=a.min_otsu_eta, min_hot_frac=a.min_hot_frac,
        max_hot_frac=a.max_hot_frac, device=a.device, dtype=a.dtype,
        batch_size=a.batch_size, io_workers=a.io_workers,
        seed=a.seed, zip_out=a.zip_out,
    )


if __name__ == '__main__':
    main()
