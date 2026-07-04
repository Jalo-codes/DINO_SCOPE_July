"""experiments.scripts.predict — GT-free qualitative inference on raw images.

For a quick "does the model completely fall over on this image" check with no
dataset root, no masks, no labels — just a checkpoint and some image files.
Reuses the same fetch → decode → viz pipeline as eval.py (I2: model_info is
the only forward-pass call site), just skips the metric() step since there is
no GT here.

Two ways to use it:

  1. CLI (arbitrary image list or glob)::

        python -m experiments.scripts.predict \\
            --checkpoint /runs/exp01/best.pt \\
            --images ~/Downloads/Gemini_Generated_Image_*.png \\
            --out_dir predict_out --decoder kmeans

  2. Notebook (paired real/modified sample via the registered indexer,
     inline display, no shell-out — this is the recommended way to run
     the model over pico_banana; it uses the real indexer/case_id pairing
     instead of ad hoc filename globbing)::

        from experiments.scripts.predict import run_predict
        results = run_predict(
            checkpoint='/content/drive/MyDrive/DINO_SCOPE_RUNS/optimal/optimal_h16plus_688_r16/epoch_0004.pt',
            source='pico_banana', root='/content/pico_banana_native_s3',
            k=8, decoder='kmeans', show=True, crop_frac=0.10,
        )

  3. Notebook (real/modified pairs via the registered dataset indexer, no
     checkpoint needed — sanity-checks the indexer's pairing directly)::

        from experiments.scripts.predict import show_pairs
        pairs = show_pairs(
            root='/content/pico_banana_native_s3', source='pico_banana', k=8,
        )
"""

import argparse
import glob
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from lab_utils.eval.decode.hdbscan import decode_hdbscan
from lab_utils.eval.decode.kmeans import decode_kmeans
from lab_utils.eval.decode.threshold import decode_threshold
from lab_utils.eval.fetch import ModelInfo, model_info
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.logging.text import log_line
from lab_utils.train.distributed import unwrap_model

from experiments.configs.zoom import DEFAULT_ZOOM

_DECODERS = {
    'kmeans': decode_kmeans,
    'threshold': decode_threshold,
    'hdbscan': decode_hdbscan,
}


def _resolve_images(patterns: List[str]) -> List[Path]:
    paths: List[Path] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(str(Path(pattern).expanduser())))
        if expanded:
            paths.extend(Path(p) for p in expanded)
        else:
            paths.append(Path(pattern).expanduser())
    return paths


def _image_score(image_logit: Optional[float]) -> float:
    if image_logit is None or not math.isfinite(image_logit):
        return float('nan')
    return float(1.0 / (1.0 + math.exp(-image_logit)))


def _crop_edges(img, frac: float):
    """Crop `frac` off each of the four edges (e.g. 0.05 removes a 5%-wide border)."""
    if not frac:
        return img
    w, h = img.size
    dx, dy = int(round(w * frac)), int(round(h * frac))
    return img.crop((dx, dy, w - dx, h - dy))


def sample_real_modified(
    real_dir: str,
    modified_dir: str,
    k: int,
    *,
    seed: int = 42,
) -> List[Tuple[str, Path]]:
    """Sample up to k (real, path) and k (modified, path) items, paired by leading id.

    Matches files by the numeric/id prefix before the first underscore (the
    `00049_original.jpg` / `00049_modified.jpg` naming convention). Returns a
    flat list of (label, path) tuples — 'real' items first, then 'modified'.
    """
    real_dir_p, mod_dir_p = Path(real_dir).expanduser(), Path(modified_dir).expanduser()
    real_files = {p.name.split('_')[0]: p for p in sorted(real_dir_p.iterdir()) if p.is_file()}
    mod_files = {p.name.split('_')[0]: p for p in sorted(mod_dir_p.iterdir()) if p.is_file()}
    common_ids = sorted(set(real_files) & set(mod_files))
    if not common_ids:
        raise RuntimeError(
            f'sample_real_modified: no matching id prefixes between {real_dir} and {modified_dir}'
        )
    rng = random.Random(seed)
    chosen = rng.sample(common_ids, k=min(k, len(common_ids)))
    items: List[Tuple[str, Path]] = [('real', real_files[i]) for i in chosen]
    items += [('modified', mod_files[i]) for i in chosen]
    return items


def _group_case_pairs(items: List) -> Dict[str, Dict[str, object]]:
    """Group Items by meta['case_id'] into {'real': Item, 'modified': Item}.

    Every builder in lab_utils.data.datasets.registry shares a case_id between
    the real and fake item of a pair. Only case_ids with BOTH sides are kept.
    """
    by_case: Dict[str, Dict[str, object]] = {}
    for it in items:
        slot = by_case.setdefault(it.meta.get('case_id', it.item_id), {})
        slot['real' if it.is_real else 'modified'] = it
    return {cid: d for cid, d in by_case.items() if 'real' in d and 'modified' in d}


def _sample_indexed_pairs(
    source: str,
    root: str,
    res,
    k: int,
    *,
    seed: int = 42,
) -> List[Tuple[str, Path, dict]]:
    """Sample k (real, modified) pairs from a registered dataset indexer.

    Returns a flat list of (label, path, meta) tuples, INTERLEAVED as
    real, modified, real, modified, ... (each consecutive pair shares a
    case_id) so predictions render back-to-back for easy comparison.
    """
    from lab_utils.data.datasets.registry import build as build_source

    _, val_ds = build_source(source, Path(root).expanduser(), res=res)
    items = val_ds.items
    if not items:
        raise RuntimeError(f'_sample_indexed_pairs: indexer found no items for source={source!r} root={root!r}')

    pairs = list(_group_case_pairs(items).items())
    if not pairs:
        raise RuntimeError(f'_sample_indexed_pairs: no case_id in {source!r} had both a real and a modified item')

    rng = random.Random(seed)
    chosen = rng.sample(pairs, k=min(k, len(pairs)))

    out: List[Tuple[str, Path, dict]] = []
    for _, d in chosen:
        out.append(('real', d['real'].image, d['real'].meta))
        out.append(('modified', d['modified'].image, d['modified'].meta))
    return out


def show_pairs(
    root: str,
    source: str = 'pico_banana',
    k: int = 8,
    image_size: int = 448,
    crop_frac: float = 0.05,
    show: bool = True,
    out_dir: Optional[str] = None,
    seed: int = 42,
) -> List[Dict]:
    """Sample k (real, modified) pairs straight from a registered dataset
    indexer and display them side by side — no checkpoint/model needed.

    Uses lab_utils.data.datasets.registry (the same indexer eval.py/training
    use). This is a sanity check on the INDEXER's pairing, independent of
    any model.
    """
    from PIL import Image as PILImage

    from lab_utils.data.datasets.registry import build as build_source
    from lab_utils.data.resolution import Resolution
    from experiments.labs.viz import display_image_inline

    res = Resolution(image_size=image_size, patch_size=16)
    _, val_ds = build_source(source, Path(root).expanduser(), res=res)
    items = val_ds.items
    if not items:
        raise RuntimeError(f'show_pairs: indexer found no items for source={source!r} root={root!r}')

    pairs = list(_group_case_pairs(items).items())
    if not pairs:
        raise RuntimeError(f'show_pairs: no case_id in {source!r} had both a real and a modified item')

    rng = random.Random(seed)
    chosen = rng.sample(pairs, k=min(k, len(pairs)))

    out_path = Path(out_dir) if out_dir else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    results: List[Dict] = []
    for case_id, d in chosen:
        real_it, mod_it = d['real'], d['modified']
        real_pil = PILImage.open(real_it.image).convert('RGB')
        mod_pil = PILImage.open(mod_it.image).convert('RGB')
        if crop_frac:
            real_pil = _crop_edges(real_pil, crop_frac)
            mod_pil = _crop_edges(mod_pil, crop_frac)

        category = real_it.meta.get('category', '')
        log_line(
            f'[predict] pair case_id={case_id} category={category!r}: '
            f'real={real_it.image.name}  modified={mod_it.image.name}'
        )

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        fig.suptitle(f'{case_id}  |  {category}', fontsize=10)
        axes[0].imshow(np.array(real_pil)); axes[0].set_title('real'); axes[0].axis('off')
        axes[1].imshow(np.array(mod_pil)); axes[1].set_title('modified'); axes[1].axis('off')
        fig.tight_layout()

        if out_path is not None:
            fig.savefig(out_path / f'{case_id}.png', dpi=130, bbox_inches='tight')
        if show:
            display_image_inline(fig)
        plt.close(fig)

        results.append({
            'case_id': case_id, 'category': category,
            'real': real_it, 'modified': mod_it,
        })

    log_line(f'[predict] showed {len(results)} real/modified pair(s) from {source!r}')
    return results


def _load_model(checkpoint: str, device):
    import torch

    log_line(f'[predict] loading checkpoint: {checkpoint}')
    model, cfg, res = load_eval_model(checkpoint, device=device, strict=False)
    bare_model = unwrap_model(model)
    has_localization = (
        getattr(bare_model, 'contrastive_proj', None) is not None
        or getattr(bare_model, 'patch_head', None) is not None
    )
    return bare_model, res, has_localization


def predict_one(
    bare_model,
    res,
    path,
    *,
    decoder: str,
    device,
    use_amp: bool,
    amp_dtype: str,
    crop_frac: float = 0.05,
    zoom: bool = False,
    attn_percentile=DEFAULT_ZOOM.attn_percentile,
    attn_thresh_mult: float = DEFAULT_ZOOM.attn_thresh_mult,
    attn_pad_frac: float = DEFAULT_ZOOM.attn_pad_frac,
    min_crop_frac: float = DEFAULT_ZOOM.min_crop_frac,
    min_box_size: int = DEFAULT_ZOOM.min_box_size,
    attn_min_pad_frac: float = DEFAULT_ZOOM.attn_min_pad_frac,
) -> Dict:
    """Run one image through the model. Returns a dict of image, mask, info, scores.

    crop_frac: fraction to crop off each of the four edges before resizing
    (e.g. 0.05 discards a 5%-wide border) — use when a source has an
    encode/decode or upload artifact right at the frame edge.

    zoom: attention-guided two-pass decode — same geometry as eval.py's
    --zoom / experiments.labs.attention_zoom.attention_zoom_single, minus
    the eval_metric() call (this stays GT-free — no Item, no mask needed).
    Pass 1 gets a full-frame attention map; if the attention bbox is not
    ~the whole frame, pass 2 crops to it, re-decodes, and places the crop
    mask back at pixel resolution (keeps the finer crop-grid detail instead
    of collapsing to the coarse full-frame patch grid). Falls back to the
    pass-1 decode when attention is unavailable or the bbox is trivial.
    """
    src = path
    if crop_frac:
        from PIL import Image as PILImage
        src = _crop_edges(PILImage.open(path).convert('RGB'), crop_frac)
    img_t, img_pil = load_image_tensor(src, res, device=device, return_pil=True)
    import torch
    with torch.no_grad():
        info: ModelInfo = model_info(bare_model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)

    score = _image_score(info.image_logit)
    zoomed = False
    bbox = None
    mask_zoom = None
    crop_pil = None
    attn_crop = None
    crop_grid_hw = None
    if decoder != 'none':
        mask_full = _DECODERS[decoder](info)
        patch_mask = mask_full
        if zoom and info.attention is not None:
            from lab_utils.eval.zoom import (
                attention_to_bbox, bbox_is_trivial, crop_to_bbox, place_mask_in_frame_pixels,
            )
            bbox = attention_to_bbox(
                info.attention, info.grid_hw,
                percentile=attn_percentile, thresh_mult=attn_thresh_mult,
                pad_frac=attn_pad_frac, min_box_size=min_box_size,
                min_pad_frac=attn_min_pad_frac,
            )
            if not bbox_is_trivial(bbox, min_crop_frac=min_crop_frac):
                crop_pil = crop_to_bbox(img_pil, bbox)
                crop_t = load_image_tensor(crop_pil, res, device=device)
                with torch.no_grad():
                    info_crop: ModelInfo = model_info(
                        bare_model, crop_t, device=device, amp=use_amp, amp_dtype=amp_dtype
                    )
                crop_mask = _DECODERS[decoder](info_crop)
                crop2d = np.asarray(crop_mask, dtype=bool)
                if crop2d.ndim == 1:
                    crop2d = crop2d.reshape(info_crop.grid_hw)
                full_px = (int(res.image_size), int(res.image_size))
                mask_zoom = place_mask_in_frame_pixels(crop2d, bbox, full_px)
                patch_mask = mask_zoom
                attn_crop = info_crop.attention
                crop_grid_hw = info_crop.grid_hw
                zoomed = True
        splice_frac = float(np.asarray(patch_mask).mean())
    else:
        mask_full = np.zeros(info.grid_hw, dtype=bool)
        patch_mask = mask_full
        splice_frac = float('nan')

    return {
        'path': Path(path),
        'img_pil': img_pil,
        'info': info,
        'patch_mask': patch_mask,
        'image_score': score,
        'splice_area_frac': splice_frac,
        'zoomed': zoomed,
        'zoom_bbox': bbox,
        'mask_full': mask_full,
        'mask_zoom': mask_zoom,
        'zoom_crop_pil': crop_pil,
        'zoom_attn_crop': attn_crop,
        'zoom_crop_grid_hw': crop_grid_hw,
    }


def run_predict(
    checkpoint: str,
    images: Optional[List[str]] = None,
    real_dir: Optional[str] = None,
    modified_dir: Optional[str] = None,
    source: Optional[str] = None,
    root: Optional[str] = None,
    k: int = 8,
    decoder: str = 'kmeans',
    out_dir: Optional[str] = None,
    show: bool = True,
    device: str = 'cuda',
    no_amp: bool = False,
    amp_dtype: str = 'float16',
    seed: int = 42,
    crop_frac: float = 0.05,
    zoom: bool = False,
    attn_percentile=DEFAULT_ZOOM.attn_percentile,
    attn_thresh_mult: float = DEFAULT_ZOOM.attn_thresh_mult,
    attn_pad_frac: float = DEFAULT_ZOOM.attn_pad_frac,
    min_crop_frac: float = DEFAULT_ZOOM.min_crop_frac,
    min_box_size: int = DEFAULT_ZOOM.min_box_size,
    attn_min_pad_frac: float = DEFAULT_ZOOM.attn_min_pad_frac,
) -> List[Dict]:
    """Notebook-friendly entry point — no shell-out, no CLI parsing.

    Pass exactly one of:
      - `images` (paths/globs),
      - `real_dir`/`modified_dir` (ad hoc id-prefix matching), or
      - `source`/`root` (a registered dataset indexer, e.g. source='pico_banana' —
        the same registry eval.py/training use; samples k matched real/
        modified pairs via meta['case_id'], the correct pairing since it
        goes through the actual indexer rather than filename globbing).

    Displays each prediction inline when `show=True` (auto-detects the
    notebook kernel) and optionally saves PNGs to `out_dir`. Returns the
    per-image result dicts for further inspection.

    crop_frac: fraction to crop off each of the four edges before resizing
    (e.g. 0.05 discards a 5%-wide border on all sides) — use when the
    images have an edge artifact (upload/encode border, watermark strip).

    zoom: attention-guided two-pass decode per image (see predict_one) —
    the same mechanism as eval.py's --zoom, GT-free here. Useful when an
    edit region (e.g. pico_banana's localized object edits) is small
    relative to the frame and gets diluted in a single full-frame decode.
    """
    import torch

    dev = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not no_amp) and (dev.type == 'cuda')

    bare_model, res, has_localization = _load_model(checkpoint, dev)
    decoder_name = decoder
    if not has_localization and decoder_name != 'none':
        log_line('[predict] no localization heads in checkpoint — defaulting decoder to none')
        decoder_name = 'none'

    meta_by_path: Dict[Path, dict] = {}
    labeled_paths: List[Tuple[str, Path]]
    if source is not None or root is not None:
        if source is None or root is None:
            raise ValueError('run_predict: pass both source and root together')
        triples = _sample_indexed_pairs(source, root, res, k, seed=seed)
        labeled_paths = [(label, path) for label, path, _ in triples]
        meta_by_path = {path: meta for _, path, meta in triples}
    elif real_dir is not None or modified_dir is not None:
        if real_dir is None or modified_dir is None:
            raise ValueError('run_predict: pass both real_dir and modified_dir together')
        labeled_paths = sample_real_modified(real_dir, modified_dir, k, seed=seed)
    elif images is not None:
        labeled_paths = [('image', p) for p in _resolve_images(images)]
    else:
        raise ValueError('run_predict: pass images=, real_dir=+modified_dir=, or source=+root=')

    out_path = Path(out_dir) if out_dir else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    from experiments.labs.viz import display_image_inline, plot_hdbscan_result, plot_prediction

    results: List[Dict] = []
    for label, path in labeled_paths:
        if not Path(path).exists():
            log_line(f'[predict] WARN: missing file, skipped: {path}')
            continue

        r = predict_one(
            bare_model, res, path,
            decoder=decoder_name, device=dev, use_amp=use_amp, amp_dtype=amp_dtype,
            crop_frac=crop_frac, zoom=zoom, attn_percentile=attn_percentile,
            attn_thresh_mult=attn_thresh_mult, attn_pad_frac=attn_pad_frac,
            min_crop_frac=min_crop_frac, min_box_size=min_box_size,
            attn_min_pad_frac=attn_min_pad_frac,
        )
        r['label'] = label
        category = meta_by_path.get(path, {}).get('category', '')
        r['category'] = category
        results.append(r)

        zoom_tag = ' zoomed' if r['zoomed'] else (' zoom-fallback' if zoom else '')
        log_line(
            f'[predict] [{label}] {path.name}'
            + (f' ({category})' if category else '')
            + f': image_score={r["image_score"]:.4f}  '
            f'splice_area_frac={r["splice_area_frac"]:.4f}  decoder={decoder_name}{zoom_tag}'
        )

        title_prefix = f'[{label}] {category} — {path.name}' if category else f'[{label}] {path.name}'
        title = f'{title_prefix} | score={r["image_score"]:.3f} | area={r["splice_area_frac"]:.3f}{zoom_tag}'
        if zoom:
            # Richer panel set — draws the attention-zoom window on the input,
            # shows the actual crop the model saw on pass 2, and separates the
            # flat (full-frame) mask from the pixel-res zoom mask so a zoom that
            # "fired" is visibly different, not just a differently-log-tagged
            # copy of the same overlay.
            fig = plot_hdbscan_result(
                r['img_pil'], patch_mask=r['mask_full'], info=r['info'],
                zoom_mask=r['mask_zoom'], crop_box=r['zoom_bbox'],
                crop_pil=r['zoom_crop_pil'], attn_crop=r['zoom_attn_crop'],
                crop_grid_hw=r['zoom_crop_grid_hw'], title=title, decoder_name=decoder_name,
            )
        else:
            fig = plot_prediction(
                r['img_pil'], patch_mask=r['patch_mask'], info=r['info'], title=title,
            )

        if out_path is not None:
            fig_name = f'{label}_{path.stem}.png'
            if hasattr(fig, 'savefig'):
                fig.savefig(out_path / fig_name, dpi=130, bbox_inches='tight')
            else:
                fig.save(out_path / fig_name)

        if show:
            display_image_inline(fig)

        if hasattr(fig, 'savefig'):
            import matplotlib.pyplot as plt
            plt.close(fig)

    log_line(f'[predict] done — {len(results)} image(s) processed'
              + (f', visualizations saved to {out_path}' if out_path else ''))
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='predict',
        description='GT-free qualitative inference on raw image files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint file')
    p.add_argument('--images', nargs='+', default=None,
                   help='Image paths or glob patterns')
    p.add_argument('--real_dir', default=None, help='Directory of real/original images')
    p.add_argument('--modified_dir', default=None, help='Directory of modified/edited images')
    p.add_argument('--source', default=None,
                   help='Registered dataset source name (e.g. pico_banana) — use with --root')
    p.add_argument('--root', default=None, help='Dataset root for --source')
    p.add_argument('--k', type=int, default=8,
                   help='Number of matched (real, modified) pairs to sample when using '
                        '--real_dir/--modified_dir or --source/--root')
    p.add_argument('--out_dir', default=None,
                   help='Directory to write per-image visualization PNGs (optional)')
    p.add_argument('--decoder', default='kmeans', choices=['kmeans', 'threshold', 'hdbscan', 'none'])
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'])
    p.add_argument('--no_show', action='store_true', help='Skip inline display (CLI/headless use)')
    p.add_argument('--crop_frac', type=float, default=0.05,
                   help='Crop this fraction off each of the four edges before resizing '
                        '(e.g. 0.05 discards a 5%% border on all sides)')
    p.add_argument('--zoom', action='store_true',
                   help='Attention-guided two-pass decode (crop to the attention bbox, '
                        're-decode, place back) — same geometry as eval.py --zoom, GT-free')
    p.add_argument('--attn_percentile', default=DEFAULT_ZOOM.attn_percentile,
                   help="Attention threshold method for --zoom: 'peak', 'otsu', 'gap', or a numeric percentile "
                        "(default: the shared operating point in experiments/configs/zoom.py)")
    p.add_argument('--attn_thresh_mult', type=float, default=DEFAULT_ZOOM.attn_thresh_mult)
    p.add_argument('--attn_pad_frac', type=float, default=DEFAULT_ZOOM.attn_pad_frac)
    p.add_argument('--min_crop_frac', type=float, default=DEFAULT_ZOOM.min_crop_frac,
                   help='Attention bbox above this fraction of the frame is treated as '
                        'trivial (falls back to the unzoomed decode)')
    p.add_argument('--min_box_size', type=int, default=DEFAULT_ZOOM.min_box_size)
    p.add_argument('--attn_min_pad_frac', type=float, default=DEFAULT_ZOOM.attn_min_pad_frac)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_predict(
        checkpoint=args.checkpoint,
        images=args.images,
        real_dir=args.real_dir,
        modified_dir=args.modified_dir,
        source=args.source,
        root=args.root,
        k=args.k,
        decoder=args.decoder,
        out_dir=args.out_dir,
        show=not args.no_show,
        device=args.device,
        no_amp=args.no_amp,
        amp_dtype=args.amp_dtype,
        crop_frac=args.crop_frac,
        zoom=args.zoom,
        attn_percentile=args.attn_percentile,
        attn_thresh_mult=args.attn_thresh_mult,
        attn_pad_frac=args.attn_pad_frac,
        min_crop_frac=args.min_crop_frac,
        min_box_size=args.min_box_size,
        attn_min_pad_frac=args.attn_min_pad_frac,
    )


if __name__ == '__main__':
    main()
