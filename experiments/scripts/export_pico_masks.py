"""experiments.scripts.export_pico_masks — offline pseudo-mask export for PicoBanana.

Runs the raw-DINOv3 feature-diff prototype (experiments/labs/dino_diff_lab.py)
over real/modified PicoBanana pairs and materializes an **inpaint-triplet
dataset** on disk that the existing ``inpaint`` builder ingests unchanged::

    out_root/modified/<case_id>.<ext>    byte-for-byte copy of the edited image
    out_root/original/<case_id>.<ext>    byte-for-byte copy of the source image
    out_root/mask/<case_id>_mask.png     pseudo-mask (raw-backbone diff, adaptive
                                         Otsu threshold, component filtering)

Images are copied VERBATIM (no re-encode — recompression would perturb the
exact statistics the detector trains on). The edge crop used to stabilize the
diff (crop_frac) is applied only in feature space; the exported mask is placed
back into full-image coordinates with a zero border, so masks always align
with the untouched copied images.

NOTE: any mask files already present under the PicoBanana root are known
noise and are never read — this script only consumes originals/ + modified/
via the pico_banana indexer (which likewise ignores them).

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

Processing continues until ``n_pairs`` pairs are KEPT (or the pool runs out),
sampled round-robin across edit categories so no category dominates.
Already-exported cases are skipped on re-run (Colab-disconnect friendly).

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
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line

_DEFAULT_MODEL_NAME = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'


# ── decisiveness statistics ────────────────────────────────────────────────────

def otsu_eta(values: np.ndarray) -> float:
    """Otsu's separability criterion: between-class variance at the best split
    divided by total variance. 0 = unimodal mush, 1 = two perfectly separated
    clusters. The direct "was there a decisive split" measure for a diff map.
    """
    flat = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    n = len(flat)
    var_total = flat.var()
    if n < 3 or var_total <= 0:
        return 0.0
    csum = np.cumsum(flat)
    total = csum[-1]
    i = np.arange(1, n)
    m0 = csum[:-1] / i
    m1 = (total - csum[:-1]) / (n - i)
    var_between = (i * (n - i) * (m0 - m1) ** 2) / (n * n)
    return float(var_between.max() / var_total)


# ── mask rendering ─────────────────────────────────────────────────────────────

def render_full_mask(
    hot: np.ndarray,
    mod_size: Tuple[int, int],
    crop_frac: float,
):
    """Map a patch-grid hot mask (computed on the crop_frac-cropped image) back
    into FULL modified-image pixel coordinates.

    The diff ran on the interior region [dx:W-dx, dy:H-dy]; the grid is
    nearest-upsampled to that interior and pasted into a zero canvas, so the
    border (never scored) is labeled negative and the exported images can be
    byte-for-byte copies of the untouched files.
    """
    from PIL import Image

    w, h = mod_size
    dx, dy = int(round(w * crop_frac)), int(round(h * crop_frac))
    iw, ih = w - 2 * dx, h - 2 * dy

    grid = Image.fromarray((hot.astype(np.uint8)) * 255, mode='L')
    interior = grid.resize((iw, ih), Image.NEAREST)
    canvas = Image.new('L', (w, h), 0)
    canvas.paste(interior, (dx, dy))
    return canvas


# ── export ─────────────────────────────────────────────────────────────────────

def _round_robin_pairs(pairs: Dict[str, Dict], seed: int) -> List[Tuple[str, Dict]]:
    """Order case pairs round-robin across categories (shuffled within each),
    so a prefix of any length stays category-balanced."""
    rng = random.Random(seed)
    by_cat: Dict[str, List[Tuple[str, Dict]]] = {}
    for cid, d in sorted(pairs.items()):
        cat = d['real'].meta.get('category', '')
        by_cat.setdefault(cat, []).append((cid, d))
    for lst in by_cat.values():
        rng.shuffle(lst)
    ordered: List[Tuple[str, Dict]] = []
    max_len = max(len(v) for v in by_cat.values())
    for i in range(max_len):
        for cat in sorted(by_cat):
            if i < len(by_cat[cat]):
                ordered.append(by_cat[cat][i])
    return ordered


def _safe_name(case_id: str) -> str:
    return case_id.replace('/', '_').replace(' ', '_')


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
    crop_frac: float = 0.05,
    hot_percentile='otsu',
    hot_thresh_mult: float = 0.5,
    hot_min_patches: int = 3,
    min_otsu_eta: float = 0.75,
    min_hot_frac: float = 0.002,
    max_hot_frac: float = 0.25,
    device: str = 'cuda',
    dtype: str = 'fp16',
    seed: int = 42,
    zip_out: Optional[str] = None,
) -> Dict:
    """Export up to n_pairs KEPT pseudo-mask triplets; returns summary dict.

    hot_* defaults are the visually validated operating point (otsu @
    thresh_mult=0.5 — tuned for TIGHT masks, precision over recall).
    dtype: backbone inference dtype ('fp16' halves T4 wall-clock; masks are
    thresholded so the tiny numeric drift vs fp32 is immaterial).
    """
    import torch

    from experiments.labs.dino_diff_lab import (
        _crop_edges, _group_case_pairs, _load_raw_backbone,
        encode_patches, hot_mask, neighborhood_max_diff,
    )
    from lab_utils.data.datasets.registry import build as build_source
    from lab_utils.eval.preprocess import load_image_tensor

    dev = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    res = Resolution(image_size=image_size, patch_size=16)
    grid_hw = (res.num_patches_per_side, res.num_patches_per_side)

    out_path = Path(out_root)
    mod_dir, orig_dir, mask_dir = out_path / 'modified', out_path / 'original', out_path / 'mask'
    for d in (mod_dir, orig_dir, mask_dir):
        d.mkdir(parents=True, exist_ok=True)

    _, val_ds = build_source(source, Path(root).expanduser(), res=res)
    pairs = _group_case_pairs(val_ds.items)
    if not pairs:
        raise RuntimeError(f'export_pico_masks: no complete pairs for source={source!r} root={root!r}')
    ordered = _round_robin_pairs(pairs, seed)
    log_line(f'[dd] export: {len(ordered)} candidate pairs, target n_pairs={n_pairs}')

    backbone = _load_raw_backbone(model_name, dev)
    torch_dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[dtype]
    if torch_dtype is not torch.float32:
        backbone = backbone.to(torch_dtype)

    from PIL import Image as PILImage

    records: List[dict] = []
    n_kept = n_dropped = n_resumed = 0

    for case_id, d in ordered:
        if n_kept >= n_pairs:
            break
        real_it, mod_it = d['real'], d['modified']
        category = real_it.meta.get('category', '')
        name = _safe_name(case_id)
        mask_file = mask_dir / f'{name}_mask.png'

        if mask_file.exists():  # resume: already exported on a prior run
            n_kept += 1
            n_resumed += 1
            continue

        real_img = PILImage.open(real_it.image).convert('RGB')
        mod_img = PILImage.open(mod_it.image).convert('RGB')
        mod_native_size = mod_img.size

        real_x = load_image_tensor(_crop_edges(real_img, crop_frac), res, device=dev)
        mod_x = load_image_tensor(_crop_edges(mod_img, crop_frac), res, device=dev)
        feats_real = encode_patches(backbone, real_x.to(torch_dtype), res).float()
        feats_mod = encode_patches(backbone, mod_x.to(torch_dtype), res).float()
        diff_map = neighborhood_max_diff(feats_real, feats_mod, grid_hw,
                                         radius=radius, pool_ksize=pool_ksize)

        hot = hot_mask(diff_map, grid_hw, percentile=hot_percentile,
                       thresh_mult=hot_thresh_mult, min_patches=hot_min_patches)
        eta = otsu_eta(diff_map)
        hot_frac = float(hot.mean())

        reason = None
        if eta < min_otsu_eta:
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

        # Byte-for-byte copies — no re-encode, original statistics preserved.
        mod_out = mod_dir / f'{name}{Path(mod_it.image).suffix.lower()}'
        orig_out = orig_dir / f'{name}{Path(real_it.image).suffix.lower()}'
        shutil.copy2(mod_it.image, mod_out)
        shutil.copy2(real_it.image, orig_out)
        render_full_mask(hot, mod_native_size, crop_frac).save(mask_file)

        n_kept += 1
        rec.update(kept=True, reason=None)
        records.append(rec)
        if n_kept % 100 == 0:
            log_line(f'[dd] export: kept={n_kept}/{n_pairs} dropped={n_dropped}')

    summary = {
        'config': {
            'root': str(root), 'source': source, 'n_pairs': n_pairs,
            'image_size': image_size, 'model_name': model_name,
            'radius': radius, 'pool_ksize': pool_ksize, 'crop_frac': crop_frac,
            'hot_percentile': str(hot_percentile), 'hot_thresh_mult': hot_thresh_mult,
            'hot_min_patches': hot_min_patches, 'min_otsu_eta': min_otsu_eta,
            'min_hot_frac': min_hot_frac, 'max_hot_frac': max_hot_frac,
            'dtype': dtype, 'seed': seed,
        },
        'n_kept': n_kept, 'n_dropped': n_dropped, 'n_resumed': n_resumed,
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
    p.add_argument('--crop_frac', type=float, default=0.05)
    p.add_argument('--hot_thresh_mult', type=float, default=0.5)
    p.add_argument('--hot_min_patches', type=int, default=3)
    p.add_argument('--min_otsu_eta', type=float, default=0.75)
    p.add_argument('--min_hot_frac', type=float, default=0.002)
    p.add_argument('--max_hot_frac', type=float, default=0.25)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    p.add_argument('--dtype', default='fp16', choices=['fp32', 'fp16', 'bf16'])
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
        seed=a.seed, zip_out=a.zip_out,
    )


if __name__ == '__main__':
    main()
