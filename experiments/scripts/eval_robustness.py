"""experiments.scripts.eval_robustness — evaluate localization robustness under image corruptions.

Usage:
    python -m experiments.scripts.eval_robustness \
        --checkpoint /runs/exp01/best.pt \
        --imd2020_root /data/imd2020 \
        --decoder kmeans \
        --max_items 500
"""

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass
import argparse
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageFilter
from tqdm import tqdm

from lab_utils.eval.aggregate import save_summary_json, write_records_csv
from lab_utils.eval.decode.hdbscan import decode_hdbscan
from lab_utils.eval.decode.kmeans import decode_kmeans
from lab_utils.eval.decode.threshold import decode_threshold
from lab_utils.eval.fetch import model_info
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.record import EvalRecord
from lab_utils.eval.val_sources import add_source_root_args, collect_val_items_by_source
from lab_utils.errors import DataError
from lab_utils.logging.text import install_log, log_line
from lab_utils.train.distributed import unwrap_model
from lab_utils.eval.preprocess import _resolve_pil, load_image_tensor
from lab_utils.eval.robustness import robustness_sweep

from experiments.configs.zoom import DEFAULT_ZOOM


# ── Corruption Helpers ─────────────────────────────────────────────────────────

def apply_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def apply_noise(img: Image.Image, std: float) -> Image.Image:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.clip(arr + np.random.normal(0.0, float(std), arr.shape), 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def apply_blur(img: Image.Image, radius: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=float(radius)))


CONDITIONS = {
    'clean': lambda img: img,
    # JPEG compression ladder (quality 90→20, step 10) — 8 levels.
    'jpeg_90': lambda img: apply_jpeg(img, 90),
    'jpeg_80': lambda img: apply_jpeg(img, 80),
    'jpeg_70': lambda img: apply_jpeg(img, 70),
    'jpeg_60': lambda img: apply_jpeg(img, 60),
    'jpeg_50': lambda img: apply_jpeg(img, 50),
    'jpeg_40': lambda img: apply_jpeg(img, 40),
    'jpeg_30': lambda img: apply_jpeg(img, 30),
    'jpeg_20': lambda img: apply_jpeg(img, 20),
    # Gaussian-noise ladder (std 0.01→0.20) — 8 levels.
    'noise_0.01': lambda img: apply_noise(img, 0.01),
    'noise_0.02': lambda img: apply_noise(img, 0.02),
    'noise_0.04': lambda img: apply_noise(img, 0.04),
    'noise_0.06': lambda img: apply_noise(img, 0.06),
    'noise_0.08': lambda img: apply_noise(img, 0.08),
    'noise_0.10': lambda img: apply_noise(img, 0.10),
    'noise_0.15': lambda img: apply_noise(img, 0.15),
    'noise_0.20': lambda img: apply_noise(img, 0.20),
    # Blur kept for the "etc." column (Gaussian radius).
    'blur_1.0': lambda img: apply_blur(img, 1.0),
    'blur_2.0': lambda img: apply_blur(img, 2.0),
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_percentile(val: str):
    try:
        return float(val)
    except ValueError:
        return val


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='eval_robustness',
        description='Evaluate a checkpoint under image corruptions.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', required=True,
                   help='Path to .pt checkpoint file')
    p.add_argument('--summary_out', default=None,
                   help='Path to write a flat JSON file of summary metrics')
    p.add_argument('--decoder', nargs='+', default=['kmeans'],
                   choices=['kmeans', 'threshold', 'hdbscan', 'none'],
                   help='Decoder(s) to evaluate.')

    g = p.add_argument_group('dataset roots (at least one required)')
    add_source_root_args(g)

    # Zoom defaults from experiments.configs.zoom.DEFAULT_ZOOM (shared with
    # eval.py / predict.py / val_zoom — one operating point).
    g = p.add_argument_group('zoom (two-pass evaluation)')
    g.add_argument('--zoom', action='store_true',
                   help='Run attention-guided zoom (two-pass evaluation)')
    g.add_argument('--attn_percentile', default=DEFAULT_ZOOM.attn_percentile,
                   help="Attention threshold method: 'peak' (>= thresh_mult*max, "
                        "recall-first), 'otsu', 'gap', or a numeric percentile.")
    g.add_argument('--attn_thresh_mult', type=float, default=DEFAULT_ZOOM.attn_thresh_mult,
                   help="Threshold scale for 'peak'/'otsu'/'gap'. With 'peak', "
                        'fraction of the max attention; lower = broader single box.')
    g.add_argument('--attn_pad_frac', type=float, default=DEFAULT_ZOOM.attn_pad_frac,
                   help='Padding fraction around the attention crop box')
    g.add_argument('--min_box_size', type=int, default=DEFAULT_ZOOM.min_box_size,
                   help='Minimum crop size in patches')
    g.add_argument('--attn_min_pad_frac', type=float, default=DEFAULT_ZOOM.attn_min_pad_frac,
                   help='Floor on per-side crop padding fraction so the margin does '
                        'not collapse to ~0 on medium/large boxes. 0 = legacy.')

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'],
                   help='Data type for mixed precision')
    g.add_argument('--compile', action='store_true',
                   help='Compile model with torch.compile')

    g = p.add_argument_group('eval control')
    g.add_argument('--max_items', type=int, default=None,
                   help='Limit items evaluated per source (flat slice; for per-cell '
                        'caps on TGIF2 use --tgif_eval_per_cell instead)')
    g.add_argument('--sources', nargs='*', default=None,
                   help='Restrict to these source names')
    g.add_argument('--subgroup', type=str, default=None,
                   help='Restrict to items in this comma-separated subgroup/cell '
                        '(e.g. "flux1dev|sp|semantic"); matches item tgif_subcat/generator. '
                        'Reals are dropped, so the run is exactly that cell.')
    g.add_argument('--conditions', nargs='+', default=None,
                   help='Subset of corruption conditions to run (default: all). Names from '
                        'CONDITIONS, e.g. "clean jpeg_50 noise_0.10". Splitting a large run '
                        'into per-condition-group cells keeps each process from accumulating '
                        'all conditions in RAM (the cause of host-OOM SIGKILLs on big TGIF runs).')
    g.add_argument('--corrupt_at', default='native', choices=['native', 'model_input'],
                   help="Where the corruption is applied. 'native': on the native-resolution "
                        'image (probe items: the cropped window) BEFORE the model resize — '
                        'the laundering threat model; artifact scale in model space then '
                        'varies with each crop\'s upsample factor. \'model_input\': resize to '
                        'the model\'s square input first, corrupt after — every item gets '
                        'IDENTICAL model-space frequency destruction (the signal-isolation '
                        'instrument: use this to ask WHAT frequency band a model relies on).')
    g.add_argument('--out_dir', default=None,
                   help='Directory for a durable eval.log and per-item records CSVs '
                        '(one CSV per decoder x condition, write_records_csv format). '
                        'Required for any analysis beyond the printed summary tables.')

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
        raise ValueError(f'eval_robustness.py: unknown decoder {decoder_name!r}')
    return fn(info)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    # Resolve the active corruption set (default: all). Splitting into subsets
    # lets each orchestrate cell run a fresh process that frees RAM between runs.
    if args.conditions:
        unknown = [c for c in args.conditions if c not in CONDITIONS]
        if unknown:
            parser.error(f'unknown --conditions {unknown}; valid: {sorted(CONDITIONS)}')
        active_conditions = {c: CONDITIONS[c] for c in args.conditions}
    else:
        active_conditions = dict(CONDITIONS)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not args.no_amp) and (device.type == 'cuda')

    # Durable log — installed before dataset builders so their [data]/[probe]
    # lines land in the file, and any crash leaves evidence on disk.
    if args.out_dir:
        install_log(str(Path(args.out_dir) / 'eval.log'))

    # ── Load checkpoint + build model ─────────────────────────────────────────
    log_line(f'[robust] loading checkpoint: {args.checkpoint}')
    model, cfg, res = load_eval_model(args.checkpoint, device=device, strict=False)
    if args.compile:
        log_line('[robust] compiling model with torch.compile...')
        model = torch.compile(model)
    bare_model = unwrap_model(model)

    # Auto-default decoder to 'none' when model has no localization heads
    has_localization = (
        getattr(bare_model, 'contrastive_proj', None) is not None
        or getattr(bare_model, 'patch_head', None) is not None
    )
    if not has_localization and args.decoder == ['kmeans']:
        log_line('[robust] no localization heads in checkpoint — defaulting --decoder to none')
        args.decoder = ['none']

    # ── Datasets ──────────────────────────────────────────────────────────────
    val_items_by_source = collect_val_items_by_source(args, res, log_tag='[robust]')
    if not val_items_by_source:
        raise RuntimeError('eval_robustness.py: no dataset roots configured or found.')

    all_items = [item for items in val_items_by_source.values() for item in items]

    # Restrict to one cell (e.g. "flux1dev|sp|semantic") so the robustness sweep
    # is per-subgroup. Matches tgif_subcat/generator; reals (no cell) are dropped.
    if args.subgroup:
        subgroups = {s.strip() for s in args.subgroup.split(',') if s.strip()}
        before = len(all_items)
        all_items = [
            it for it in all_items
            if (it.meta.get('generator') or it.meta.get('tgif_subcat')) in subgroups
        ]
        log_line(f'[robust] filtered to subgroup(s)={sorted(subgroups)}: '
                 f'{len(all_items)}/{before} items')
        if not all_items:
            raise RuntimeError(
                f'eval_robustness.py: no items match subgroup(s) {sorted(subgroups)}. '
                f'Check the cell label against the index (model|type|family).')

    dataset_names = sorted(list(set(item.source for item in all_items)))

    # Initialize records maps: dataset -> decoder -> condition -> list of EvalRecords
    records_by_dataset_decoder_condition = {
        ds: {dec: {cond: [] for cond in active_conditions} for dec in args.decoder}
        for ds in dataset_names
    }

    zoom_crop_kwargs = {
        'attn_min_pad_frac': args.attn_min_pad_frac,
        'attn_thresh_mult': args.attn_thresh_mult,
    }

    # ── Robustness sweep loop ─────────────────────────────────────────────────
    import sys
    disable_tqdm = not sys.stdout.isatty()
    for cond_name, apply_fn in active_conditions.items():
        log_line(f'[robust] running condition: {cond_name}')
        if disable_tqdm:
            log_line(f'[robust] processing {len(all_items)} items for {cond_name}...')
        
        for item in tqdm(all_items, desc=f'[robust] {cond_name}', unit='item', disable=disable_tqdm):
            try:
                # 1. Load image and apply corruption in PIL. _resolve_pil (NOT a
                # bare Image.open) applies region-probe crop windows — probe
                # items must be corrupted as the crop the model actually sees,
                # never as the full frame.
                img_pil = _resolve_pil(item)
                if args.corrupt_at == 'model_input':
                    S = res.image_size
                    if img_pil.size != (S, S):
                        img_pil = img_pil.resize((S, S), Image.BILINEAR)
                img_corrupted_pil = apply_fn(img_pil)

                if args.zoom:
                    from experiments.labs.attention_zoom import attention_zoom_single
                    # For zoom, evaluate each decoder individually using the corrupted PIL image
                    for decoder_name in args.decoder:
                        rec = attention_zoom_single(
                            bare_model, item, res,
                            device=device, use_amp=use_amp, amp_dtype=args.amp_dtype,
                            decoder=decoder_name,
                            attn_percentile=_parse_percentile(args.attn_percentile),
                            attn_pad_frac=args.attn_pad_frac,
                            min_box_size=args.min_box_size,
                            override_image_pil=img_corrupted_pil,
                            **zoom_crop_kwargs,
                        )
                        import dataclasses
                        subgroup = item.meta.get('generator') or item.meta.get('tgif_subcat')
                        # Strip pixel-res arrays before accumulating: scores are
                        # already computed, and holding gt/pred masks across
                        # items x conditions is the documented host-OOM cause.
                        rec = dataclasses.replace(rec, subgroup=subgroup,
                                                  gt_mask=None, pred_mask=None,
                                                  attention=None)
                        records_by_dataset_decoder_condition[item.source][decoder_name][cond_name].append(rec)
                else:
                    # 2. Preprocess to normalized tensor and forward
                    img_t = load_image_tensor(img_corrupted_pil, res, device=device)
                    with torch.no_grad():
                        info = model_info(bare_model, img_t, device=device, amp=use_amp, amp_dtype=args.amp_dtype)

                    # 3. Decode + Metric for each decoder
                    for decoder_name in args.decoder:
                        patch_mask = _decode(decoder_name, info)
                        rec = eval_metric(patch_mask, info, item, decoder=decoder_name)

                        import dataclasses
                        subgroup = item.meta.get('generator') or item.meta.get('tgif_subcat')
                        # Strip pixel-res arrays before accumulating (see zoom
                        # branch comment — the documented host-OOM cause).
                        rec = dataclasses.replace(rec, subgroup=subgroup,
                                                  gt_mask=None, pred_mask=None,
                                                  attention=None)
                        records_by_dataset_decoder_condition[item.source][decoder_name][cond_name].append(rec)
            except DataError:
                raise  # alignment/pairing bug — abort the sweep, never a skip
            except Exception as exc:
                log_line(f'[robust] WARN: skipped item={item.item_id} under {cond_name}: {exc}')

    # Run robustness sweeps and collect summaries
    summaries_dict = {}
    for ds in dataset_names:
        for decoder_name in args.decoder:
            log_line(f'[robust] ─── Robustness Sweep Table: dataset={ds}, decoder={decoder_name} ───')
            recs_by_cond = records_by_dataset_decoder_condition[ds][decoder_name]
            
            has_records = any(len(recs) > 0 for recs in recs_by_cond.values())
            if not has_records:
                continue
                
            sweep_res = robustness_sweep(
                recs_by_cond,
                metric='f1',
                baseline_name='clean',
                log_tag='[robust]',
                tag=f"{ds}_{decoder_name}",
            )
            
            # Save flat dictionary of metrics for saving
            decoder_summary = {}
            for cond_name, stats in sweep_res.items():
                for k, v in stats.items():
                    if not isinstance(v, dict) and not np.isnan(v):
                        decoder_summary[f"{cond_name}_{k}"] = v
            summaries_dict[f"{ds}_{decoder_name}"] = decoder_summary

    # ── Per-item records CSVs (the analysis-side artifact) ────────────────────
    # One CSV per decoder x condition, all datasets flattened (source column
    # disambiguates) — same format as eval.py's records CSVs so downstream
    # stratified-AUROC analysis joins on item_id against the probe manifest.
    if args.out_dir:
        for decoder_name in args.decoder:
            for cond_name in active_conditions:
                recs = [
                    r for ds in dataset_names
                    for r in records_by_dataset_decoder_condition[ds][decoder_name][cond_name]
                ]
                if not recs:
                    continue
                csv_path = Path(args.out_dir) / f'{decoder_name}_{cond_name}_records.csv'
                write_records_csv(recs, str(csv_path))
                log_line(f'[robust] wrote {len(recs)} records -> {csv_path}')

    # ── Save flat JSON summary if requested ───────────────────────────────────
    if args.summary_out:
        log_line(f'[robust] saving flat robustness summary to {args.summary_out}')
        
        flat = {}
        for key, summary in summaries_dict.items():
            for k, v in summary.items():
                flat[f"{key}_{k}"] = float(v)
                
        import os
        out_dir = os.path.dirname(args.summary_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.summary_out, 'w') as f:
            json.dump(flat, f, indent=2)


if __name__ == '__main__':
    main()
