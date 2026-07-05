"""experiments.scripts.viz_pico_masks — eyeball the pico pseudo-mask operating point.

Runs N candidate pairs through EXACTLY the export pipeline — every default is
imported from export_pico_masks, no competing copies — and renders one figure
per pair, inline in the Colab cell output (display_image_inline; also works in
iTerm2/kitty, and saves PNGs when out_dir is given):

    original (cropped) | modified (cropped) | diff heatmap |
    raw hot overlay    | FINAL mask overlay

'raw hot' is the bare adaptive threshold; 'FINAL' is what the export would
actually write: stragglers (< HOT_MIN_PATCHES) removed from the TP set,
fully-enclosed background holes plugged, rendered at the crop-baked geometry
via render_cropped_mask. The suptitle carries the export's keep/drop verdict
with the same decisiveness stats (otsu_eta, hot_frac) — so the set you eyeball
here is the set an export run would keep.

Run this BEFORE a full export to sanity-check the configuration.

Usage (Colab)::

    from experiments.scripts.viz_pico_masks import run_viz
    run_viz(root='/content/pico_banana_native_s3', n=12)

CLI::

    python -m experiments.scripts.viz_pico_masks \\
        --root /content/pico_banana_native_s3 --n 12 --out_dir /content/mask_viz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch

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
    _group_case_pairs,
    _load_raw_backbone,
    _round_robin_pairs,
    _safe_name,
    diff_one,
    hot_mask,
    otsu_eta,
    render_cropped_mask,
)
from lab_utils.data.datasets.registry import build as build_source
from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line


def _native_mask(hot: np.ndarray, native_size, crop_frac: float) -> np.ndarray:
    """Patch-grid mask → bool array at the cropped image's pixel size."""
    return np.asarray(render_cropped_mask(hot, native_size, crop_frac)) > 127


def _verdict(eta: float, hot_frac: float) -> Optional[str]:
    """The export's decisiveness filter, verbatim. None = pair would be KEPT."""
    if eta < MIN_OTSU_ETA:
        return f'otsu_eta {eta:.3f} < {MIN_OTSU_ETA}'
    if hot_frac < MIN_HOT_FRAC:
        return f'hot_frac {hot_frac:.4f} < {MIN_HOT_FRAC}'
    if hot_frac > MAX_HOT_FRAC:
        return f'hot_frac {hot_frac:.4f} > {MAX_HOT_FRAC}'
    return None


def run_viz(
    root: str,
    *,
    source: str = 'pico_banana',
    n: int = 12,
    image_size: int = 688,
    model_name: str = _DEFAULT_MODEL_NAME,
    radius: int = 1,
    pool_ksize: int = 1,
    device: str = 'cuda',
    seed: int = 42,
    out_dir: Optional[str] = None,
    include_drops: bool = True,
) -> dict:
    """Visualize n pairs under the export operating point; returns summary.

    include_drops: when True (default) the n pairs are rendered as sampled,
    drops included with their reason in the title — that IS the point of the
    preview. When False, keeps rendering until n would-be-KEPT pairs shown.
    Runs fp32 (diff_one's path); masks are thresholded, so the fp16-vs-fp32
    drift of a real export run is immaterial to what you see.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    from experiments.labs.dino_diff_lab import _crop_edges
    from experiments.labs.viz import display_image_inline, mask_overlay

    dev = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    res = Resolution(image_size=image_size, patch_size=16)
    grid_hw = (res.num_patches_per_side, res.num_patches_per_side)

    _, val_ds = build_source(source, Path(root).expanduser(), res=res)
    pairs = _group_case_pairs(val_ds.items)
    if not pairs:
        raise RuntimeError(f'viz_pico_masks: no complete pairs for source={source!r} root={root!r}')
    ordered = _round_robin_pairs(pairs, seed)

    backbone = _load_raw_backbone(model_name, dev)
    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    log_line(f'[dd] viz: {len(ordered)} candidate pairs; rendering {n} '
             f'(operating point: {HOT_PERCENTILE}@{HOT_THRESH_MULT}, '
             f'min_patches={HOT_MIN_PATCHES}, plug_holes={PLUG_HOLES}, '
             f'crop_frac={CROP_FRAC})')

    n_shown = n_kept = n_dropped = 0
    records = []
    for case_id, d in ordered:
        if n_shown >= n:
            break
        real_it, mod_it = d['real'], d['modified']

        with Image.open(real_it.image) as ri, Image.open(mod_it.image) as mi:
            real_native, mod_native = ri.size, mi.size
        if real_native != mod_native:
            log_line(f'[dd] viz: {case_id}: pair_size_mismatch real={real_native} '
                     f'mod={mod_native} — export would drop; skipping')
            n_dropped += 1
            continue

        r = diff_one(backbone, res, real_it.image, mod_it.image, device=dev,
                     radius=radius, pool_ksize=pool_ksize, crop_frac=CROP_FRAC)
        diff_map = r['diff_map']

        raw = hot_mask(diff_map, grid_hw, percentile=HOT_PERCENTILE,
                       thresh_mult=HOT_THRESH_MULT, min_patches=1, plug_holes=False)
        final = hot_mask(diff_map, grid_hw, percentile=HOT_PERCENTILE,
                         thresh_mult=HOT_THRESH_MULT, min_patches=HOT_MIN_PATCHES,
                         plug_holes=PLUG_HOLES)

        eta = otsu_eta(diff_map)
        hot_frac = float(final.mean())
        reason = _verdict(eta, hot_frac)
        if reason is None:
            n_kept += 1
        else:
            n_dropped += 1
            if not include_drops:
                continue

        # Cropped-native frames — the exact geometry the export writes.
        real_arr = np.array(_crop_edges(Image.open(real_it.image).convert('RGB'), CROP_FRAC))
        mod_arr  = np.array(_crop_edges(Image.open(mod_it.image).convert('RGB'), CROP_FRAC))
        raw_px   = _native_mask(raw, mod_native, CROP_FRAC)
        final_px = _native_mask(final, mod_native, CROP_FRAC)

        fig, axes = plt.subplots(1, 5, figsize=(3.2 * 5, 3.8))
        verdict = 'KEEP' if reason is None else f'DROP: {reason}'
        fig.suptitle(f'{case_id}  [{verdict}]  otsu_eta={eta:.3f}  '
                     f'hot_frac={hot_frac:.4f}', fontsize=11)
        panels = [
            (real_arr, 'original (cropped)', None),
            (mod_arr,  'modified (cropped)', None),
            (diff_map, 'diff heatmap',       'hot'),
            (mask_overlay(mod_arr, raw_px,   color=(230, 160, 20), alpha=0.45),
             f'raw hot ({HOT_PERCENTILE}@{HOT_THRESH_MULT})', None),
            (mask_overlay(mod_arr, final_px, color=(220, 30, 30), alpha=0.45),
             f'FINAL (≥{HOT_MIN_PATCHES} patches, holes plugged)', None),
        ]
        for ax, (arr, title, cmap) in zip(axes, panels):
            ax.imshow(arr, cmap=cmap, interpolation='nearest')
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        plt.tight_layout()

        if out_path:
            fig.savefig(out_path / f'{_safe_name(case_id)}.png', dpi=110, bbox_inches='tight')
        display_image_inline(fig)
        plt.close(fig)

        n_shown += 1
        records.append({'case_id': case_id, 'kept': reason is None, 'reason': reason,
                        'otsu_eta': round(eta, 4), 'hot_frac': round(hot_frac, 4)})

    log_line(f'[dd] viz done: shown={n_shown} (would keep {n_kept}, '
             f'drop {n_dropped} of those sampled)'
             + (f' -> {out_path}' if out_path else ''))
    return {'shown': n_shown, 'kept': n_kept, 'dropped': n_dropped, 'records': records}


def main() -> None:
    p = argparse.ArgumentParser(
        prog='viz_pico_masks',
        description='Preview pico pseudo-masks under the export operating point.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--root', required=True, help='PicoBanana dataset root')
    p.add_argument('--n', type=int, default=12, help='Pairs to render')
    p.add_argument('--image_size', type=int, default=688)
    p.add_argument('--model_name', default=_DEFAULT_MODEL_NAME)
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', default=None, help='Also save PNGs here')
    p.add_argument('--kept_only', action='store_true',
                   help='Render only pairs the export would keep')
    a = p.parse_args()
    run_viz(root=a.root, n=a.n, image_size=a.image_size, model_name=a.model_name,
            device=a.device, seed=a.seed, out_dir=a.out_dir,
            include_drops=not a.kept_only)


if __name__ == '__main__':
    main()
