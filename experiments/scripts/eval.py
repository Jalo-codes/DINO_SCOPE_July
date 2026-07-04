"""experiments.scripts.eval — standalone evaluation over one or more datasets.

Runs the canonical fetch → decode → metric → aggregate pipeline with no
oracle assistance.  All GT is read ONLY inside metric() (I3).  No attention-
zoom or multi-pass logic here — see experiments/labs/attention_zoom.py.

Usage:
    python -m experiments.scripts.eval \\
        --checkpoint /runs/exp01/best.pt \\
        --imd2020_root /data/imd2020 \\
        --decoder kmeans

    # Multiple decoders (decoder bench):
    python -m experiments.scripts.eval \\
        --checkpoint /runs/exp01/best.pt \\
        --imd2020_root /data/imd2020 \\
        --decoder kmeans threshold \\
        --bench

    # Cache model outputs for repeated eval:
    python -m experiments.scripts.eval \\
        --checkpoint /runs/exp01/best.pt \\
        --imd2020_root /data/imd2020 \\
        --cache_dir /tmp/eval_cache
"""

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from lab_utils.compat import trapz
from lab_utils.eval.aggregate import decoder_bench, save_summary_json, summarize
from lab_utils.eval.cache import build_cache, iter_cache
from lab_utils.eval.decode.hdbscan import decode_hdbscan
from lab_utils.eval.decode.kmeans import decode_kmeans
from lab_utils.eval.decode.threshold import decode_threshold
from lab_utils.eval.fetch import model_info
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.record import EvalRecord
from lab_utils.eval.val_sources import add_source_root_args, collect_val_items_by_source
from lab_utils.logging.text import log_line
from lab_utils.train.distributed import unwrap_model


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_percentile(val: str):
    try:
        return float(val)
    except ValueError:
        return val


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='eval',
        description='Evaluate a DINO_SCOPE_final checkpoint.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', required=True,
                   help='Path to .pt checkpoint file')
    p.add_argument('--summary_out', default=None,
                   help='Path to write a flat JSON file of summary metrics')
    p.add_argument('--decoder', nargs='+', default=['kmeans'],
                   choices=['kmeans', 'threshold', 'hdbscan', 'none'],
                   help='Decoder(s) to evaluate. "none" skips localization '
                        '(image-level AUC only).')
    p.add_argument('--bench', action='store_true',
                   help='Print decoder comparison table when >1 decoder given')
    p.add_argument('--cache_dir', default=None,
                   help='Directory to cache ModelInfo npz files. '
                        'If already populated, skip the forward pass.')
    p.add_argument('--overwrite_cache', action='store_true',
                   help='Re-run forward pass even if cache exists')

    g = p.add_argument_group('dataset roots (at least one required)')
    add_source_root_args(g)

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'],
                   help='Data type for mixed precision (float16 or bfloat16)')
    g.add_argument('--compile', action='store_true',
                   help='Compile model with torch.compile for faster execution')

    g = p.add_argument_group('eval control')
    g.add_argument('--max_items', type=int, default=None,
                   help='Limit items evaluated per source (smoke test mode)')
    g.add_argument('--sources', nargs='*', default=None,
                   help='Restrict to these source names (default: all configured)')
    g.add_argument('--subgroup', type=str, default=None,
                   help='Restrict evaluation to items in this comma-separated subgroup/cell (reals are preserved)')
    g.add_argument('--edge_crop_frac', type=float, default=0.0,
                   help='Border-crop this fraction off each of the four edges before '
                        'the forward pass (both flat and --zoom paths), matching '
                        'train.py/export_pico_masks.py. The GT mask is cropped by the '
                        'identical fraction (lab_utils.eval.metric.metric) so scores '
                        'stay geometry-aligned with what the model actually saw. '
                        'MUST match the edge_crop_frac the checkpoint was trained with '
                        'for a fair number — 0.0 = old behaviour (no crop).')
    g.add_argument('--zoom', action='store_true',
                   help='Run attention-guided zoom (two-pass evaluation)')
    g.add_argument('--attn_percentile', default='peak',
                   help="Attention threshold method: 'peak' (>= thresh_mult*max, "
                        "recall-first), 'otsu', 'gap', or a numeric percentile.")
    g.add_argument('--attn_thresh_mult', type=float, default=0.08,
                   help="Threshold scale for 'peak'/'otsu'/'gap'. With 'peak', "
                        'fraction of the max attention; lower = broader single box.')
    g.add_argument('--attn_pad_frac', type=float, default=0.10,
                   help='Padding fraction around the attention crop box')
    g.add_argument('--min_box_size', type=int, default=8,
                   help='Minimum crop size in patches')
    g.add_argument('--attn_min_pad_frac', type=float, default=0.06,
                   help='Floor on per-side crop padding fraction so the margin does '
                        'not collapse to ~0 on medium/large boxes. 0 = legacy.')
    g.add_argument('--zoom_pad_frac', type=float, default=None,
                   help='AREA-BASED crop padding: pad each side by this fraction of '
                        'the frame (resolution-invariant). Set it to switch off the '
                        'patch-based pad/min_box_size math. Default None = legacy.')
    g.add_argument('--zoom_min_area', type=float, default=0.0,
                   help='With --zoom_pad_frac, floor the padded crop to this fraction '
                        'of the frame area (grown symmetrically about center).')

    g = p.add_argument_group('visualizations')
    g.add_argument('--out_dir', default=None,
                   help='Directory to save visualization figures. If set, enables visualization.')
    g.add_argument('--viz_n', type=int, default=0,
                   help='Number of positive/manipulated images to visualize (requires --out_dir and/or --show)')
    g.add_argument('--show', action='store_true',
                   help='Display figures inline in an IPython/Colab kernel or a '
                        'graphics terminal (iTerm2/kitty/chafa). Works with or '
                        'without --out_dir.')
    return p


# ── Decode dispatch ────────────────────────────────────────────────────────────

_DECODERS = {
    'kmeans':    decode_kmeans,
    'threshold': decode_threshold,
    'hdbscan':   decode_hdbscan,
    'none':      lambda info: np.zeros(info.grid_hw, dtype=bool),
}


def _decode(decoder_name: str, info) -> 'np.ndarray':
    fn = _DECODERS.get(decoder_name)
    if fn is None:
        raise ValueError(f'eval.py: unknown decoder {decoder_name!r}')
    return fn(info)


def _log_image_auc(records: List[EvalRecord], *, log_tag: str = '[eval]') -> None:
    """Compute and log image-level AUC from records.  Self-contained."""
    if not records:
        return
    scores = np.array([r.image_score for r in records], dtype=np.float64)
    labels = np.array([0 if r.is_real else 1 for r in records], dtype=np.int32)
    if np.any(np.isnan(scores)):
        log_line(f'{log_tag} image_auc: NaN (some image_scores are NaN)')
        return
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        log_line(f'{log_tag} image_auc: N/A (need both reals and splices)')
        return
    order = np.argsort(-scores)
    sl    = labels[order]
    tpr   = np.cumsum(sl) / n_pos
    fpr   = np.cumsum(1 - sl) / n_neg
    auc   = float(trapz(tpr, fpr))
    auc   = 1.0 + auc if auc < 0 else auc
    log_line(f'{log_tag} image_auc: {auc:.4f}')


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not args.no_amp) and (device.type == 'cuda')

    # Crop-window padding floor for --zoom.  The default widens the crop so the
    # second pass keeps lightly-attended splice margins; 0.0 = legacy padding.
    zoom_crop_kwargs = {'attn_min_pad_frac': args.attn_min_pad_frac,
                        'attn_thresh_mult': args.attn_thresh_mult,
                        'pad_side_frac':    args.zoom_pad_frac,
                        'min_area_frac':    args.zoom_min_area}

    # ── Load checkpoint + build model (shared loader) ──────────────────────────
    log_line(f'[eval] loading checkpoint: {args.checkpoint}')
    model, cfg, res = load_eval_model(args.checkpoint, device=device, strict=False)
    if args.compile:
        log_line('[eval] compiling model with torch.compile...')
        model = torch.compile(model)
    bare_model = unwrap_model(model)

    # Auto-default decoder to 'none' when model has no localization heads.
    has_localization = (
        getattr(bare_model, 'contrastive_proj', None) is not None
        or getattr(bare_model, 'patch_head', None) is not None
    )
    if not has_localization and args.decoder == ['kmeans']:
        log_line('[eval] no localization heads in checkpoint — defaulting --decoder to none')
        args.decoder = ['none']

    # ── Datasets ──────────────────────────────────────────────────────────────
    val_items_by_source = collect_val_items_by_source(args, res)
    if not val_items_by_source:
        raise RuntimeError(
            'eval.py: no dataset roots configured or found. '
            'Pass at least one of --imd2020_root, --casia_root, etc.'
        )

    all_items = [item for items in val_items_by_source.values() for item in items]
    if args.subgroup:
        subgroups = [s.strip() for s in args.subgroup.split(',')]
        all_items = [
            item for item in all_items
            if item.is_real or (item.meta.get('generator') or item.meta.get('tgif_subcat')) in subgroups
            or item.meta.get('tgif_model') in subgroups
        ]
        log_line(f'[eval] filtered to subgroup(s)={subgroups}, remaining items: {len(all_items)}')

    # ── Cache forward pass ────────────────────────────────────────────────────
    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
        log_line(f'[eval] building/loading cache: {cache_dir}')
        build_cache(
            bare_model, all_items,
            device=device, amp=use_amp, amp_dtype=args.amp_dtype,
            cache_dir=cache_dir,
            overwrite=args.overwrite_cache,
        )

    # ── Eval per decoder ──────────────────────────────────────────────────────
    records_by_decoder: Dict[str, List[EvalRecord]] = {}
    summaries_dict: Dict[str, Dict] = {}

    from lab_utils.eval.preprocess import load_image_tensor

    for decoder_name in args.decoder:
        log_line(f'[eval] decoder={decoder_name}')
        records: List[EvalRecord] = []

        if args.zoom and args.cache_dir:
            log_line('[eval] WARN: --cache_dir is ignored because zoom uses dynamic two-pass crops')
        if args.edge_crop_frac and args.cache_dir:
            log_line(f'[eval] WARN: --edge_crop_frac={args.edge_crop_frac} has no effect on '
                     f'cached ModelInfo (--cache_dir) — the cache was built from uncropped '
                     f'forward passes. Rebuild with --overwrite_cache after cropping, or drop '
                     f'--cache_dir, for a correct number.')

        n_viz = 0
        from tqdm import tqdm
        import sys
        disable_tqdm = not sys.stdout.isatty()
        if disable_tqdm:
            log_line(f'[eval] processing {len(all_items)} items for {decoder_name}...')
        for item in tqdm(all_items, desc=f'[eval] {decoder_name}', unit='item', disable=disable_tqdm):
            try:
                need_viz = (args.out_dir or args.show) and (args.viz_n > 0) and (n_viz < args.viz_n) and (not item.is_real)
                
                img_pil = None
                if need_viz:
                    from PIL import Image
                    img_pil = Image.open(item.image).convert('RGB')
                    if args.edge_crop_frac:
                        from lab_utils.data.dataset import _crop_edges
                        img_pil = _crop_edges(img_pil, args.edge_crop_frac)

                if args.zoom:
                    from experiments.labs.attention_zoom import attention_zoom_single
                    rec_out = attention_zoom_single(
                        bare_model, item, res,
                        device=device, use_amp=use_amp, amp_dtype=args.amp_dtype,
                        decoder=decoder_name,
                        attn_percentile=_parse_percentile(args.attn_percentile),
                        attn_pad_frac=args.attn_pad_frac,
                        min_box_size=args.min_box_size,
                        return_debug=need_viz,
                        edge_crop_frac=args.edge_crop_frac,
                        **zoom_crop_kwargs,
                    )
                    rec, debug = rec_out if need_viz else (rec_out, None)
                else:
                    if args.cache_dir:
                        from lab_utils.eval.cache import load_cache
                        cache   = load_cache(args.cache_dir, item_ids=[item.item_id])
                        info    = cache.get(item.item_id)
                        if info is None:
                            log_line(f'[eval] WARN: no cache entry for {item.item_id}')
                            continue
                    else:
                        img_src = item
                        if args.edge_crop_frac:
                            from PIL import Image as PILImage

                            from lab_utils.data.dataset import _crop_edges
                            img_src = _crop_edges(PILImage.open(item.image).convert('RGB'), args.edge_crop_frac)
                        img_t = load_image_tensor(img_src, res, device=device)
                        with torch.no_grad():
                            info = model_info(bare_model, img_t, device=device, amp=use_amp, amp_dtype=args.amp_dtype)

                    patch_mask = _decode(decoder_name, info)
                    rec        = eval_metric(patch_mask, info, item, decoder=decoder_name,
                                              edge_crop_frac=args.edge_crop_frac)

                import dataclasses
                subgroup = item.meta.get('generator') or item.meta.get('tgif_subcat')
                rec = dataclasses.replace(rec, subgroup=subgroup)
                records.append(rec)

                if need_viz:
                    out_path = None
                    if args.out_dir:
                        out_path = Path(args.out_dir) / f"{decoder_name}_viz"
                        out_path.mkdir(parents=True, exist_ok=True)
                    fig_name = f"{item.source}_{item.item_id}.png"

                    if args.zoom:
                        from experiments.labs.viz import plot_hdbscan_result
                        from lab_utils.eval.zoom import mask_to_bbox

                        gt_box = None
                        if rec.gt_mask is not None and rec.gt_mask.any():
                            gt_box = mask_to_bbox(rec.gt_mask.astype(bool))

                        class MockInfo:
                            def __init__(self, attn, hw):
                                self.attention = attn
                                self.grid_hw = hw

                        mock_info = MockInfo(debug.get('attn1'), debug.get('grid_hw'))

                        fig = plot_hdbscan_result(
                            img_pil,
                            patch_mask=debug.get('mask_full').reshape(debug.get('grid_hw')) if debug.get('mask_full').ndim == 1 else debug.get('mask_full'),
                            info=mock_info,
                            gt_mask=rec.gt_mask.astype(bool) if rec.gt_mask is not None else None,
                            zoom_mask=debug.get('mask_zoom'),
                            gt_box=gt_box,
                            crop_box=debug.get('bbox'),
                            crop_pil=debug.get('crop_pil'),
                            attn_crop=debug.get('attn_crop'),
                            crop_grid_hw=debug.get('crop_grid_hw'),
                            title=f"ID: {item.item_id} | F1 (flat): {rec.f1:.4f}",
                            decoder_name=decoder_name,
                        )
                    else:
                        from experiments.labs.viz import plot_prediction
                        fig = plot_prediction(
                            img_pil,
                            patch_mask=patch_mask,
                            info=info,
                            gt_mask=rec.gt_mask.astype(bool) if rec.gt_mask is not None else None,
                            title=f"ID: {item.item_id} | F1: {rec.f1:.4f}",
                        )

                    if out_path is not None:
                        if hasattr(fig, 'savefig'):
                            fig.savefig(out_path / fig_name, dpi=130, bbox_inches='tight')
                        else:
                            fig.save(out_path / fig_name)
                    if args.show:
                        from experiments.labs.viz import display_image_inline
                        display_image_inline(fig)
                    if hasattr(fig, 'savefig'):
                        try:
                            import matplotlib.pyplot as plt
                            plt.close(fig)
                        except ImportError:
                            pass
                    n_viz += 1
            except Exception as exc:
                log_line(f'[eval] WARN: skipped item={item.item_id}: {exc}')

        records_by_decoder[decoder_name] = records
        summary_res = summarize(records, log_tag='[eval]', tag=decoder_name)
        summaries_dict[decoder_name] = summary_res

        has_subgroups = any(r.subgroup is not None for r in records)
        if has_subgroups:
            from lab_utils.eval.aggregate import summarize_by_subgroup
            summarize_by_subgroup(records, log_tag='[eval]', tag=decoder_name)

        # When using the 'none' decoder (image-level only), log image AUC.
        if decoder_name == 'none':
            _log_image_auc(records, log_tag='[eval]')

    # ── Decoder bench ─────────────────────────────────────────────────────────
    if args.bench and len(args.decoder) > 1:
        log_line('[eval] decoder bench:')
        decoder_bench(records_by_decoder, log_tag='[eval]')

    # ── Save flat JSON summary if requested ───────────────────────────────────
    if args.summary_out:
        log_line(f'[eval] saving flat summary to {args.summary_out}')
        save_summary_json(args.summary_out, summaries_dict)


if __name__ == '__main__':
    main()
