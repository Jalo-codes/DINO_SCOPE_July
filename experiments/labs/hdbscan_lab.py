"""experiments.labs.hdbscan_lab — HDBSCAN + HDBSCAN-zoom: viz, boxes, numbers.

Runs two decode modes on a checkpoint and compares them:
  * hdbscan       — full-frame HDBSCAN decode on patch embeddings
  * hdbscan_zoom  — attention-bbox crop → HDBSCAN decode in the crop → place back

For each visualised item it renders a 4-panel figure with bounding boxes:
  predicted-region boxes (red), GT box (green), attention-zoom window (yellow).

Numbers come from the standard pipeline (metric → summarize / decoder_bench);
nothing here re-implements scoring.  Image loading, zoom geometry, and box
drawing all come from the shared utilities (eval.preprocess / eval.zoom / labs.viz).

Example (Colab):
    python -m experiments.labs.hdbscan_lab \\
        --checkpoint /content/drive/MyDrive/DINO_SCOPE_RUNS/.../epoch_004.pt \\
        --imd2020_root /content/IMD2020 \\
        --out_dir /content/hdbscan_out \\
        --viz_n 24 --max_items 200 \\
        --min_cluster_size 8 --theta_x 0.5 --polarity size
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution
from lab_utils.data.verify import SKIP_VERIFY, verify_all as _verify_all
from lab_utils.eval.decode.hdbscan import hdbscan_available, hdbscan_decode
from lab_utils.eval.decode.kmeans import decode_kmeans
from lab_utils.eval.fetch import model_info
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.record import EvalRecord
from lab_utils.eval.zoom import mask_to_bbox
from lab_utils.logging.text import log_line

_SOURCE_ROOT_MAP = {
    'imd2020': 'imd2020_root', 'casia': 'casia_root', 'indoor': 'indoor_root',
    'coco_inpaint': 'coco_inpaint_root', 'sagid': 'sagid_root',
    'bfree': 'bfree_root', 'anyedit': 'anyedit_root', 'tgif2': 'tgif2_root',
    'cocoglide': 'cocoglide_root', 'opensdi': 'opensdi_root',
}


# ── dataset collection ──────────────────────────────────────────────────────────

def _collect_items(args, res: Resolution) -> List[Item]:
    """Collect eval items.  This is a pure eval harness, so every item the
    builder produces (train + val split combined) is an eval item — IMD2020 is
    held out of training entirely, so its 90/10 internal split is irrelevant
    here and we'd otherwise hide 90% of it behind the 'train' bucket.

    Verify runs ONLY on the capped slice (max_items), not the full index.
    The inference loop catches corrupt/degenerate files at runtime anyway.
    """
    from lab_utils.data.datasets.registry import REGISTRY
    items: List[Item] = []
    for source, attr in _SOURCE_ROOT_MAP.items():
        root_str = getattr(args, attr, None)
        if not root_str:
            continue
        root = Path(root_str)
        if not root.exists():
            log_line(f'[buckets] WARN: root not found for {source}: {root}')
            continue
        train_ds, val_ds = REGISTRY[source](root, res=res, verify_policy=SKIP_VERIFY)
        got = list(train_ds.items) + list(val_ds.items)
        if args.max_items:
            got = got[:args.max_items]
        got, _ = _verify_all(got, log_tag=f'[buckets] {source}')
        log_line(f'[buckets] {source}: {len(got)} eval items (all splits)')
        items.extend(got)
    return items


# ── per-item viz ────────────────────────────────────────────────────────────────

def _visualise_item(
    item: Item, img_pil, patch_mask: np.ndarray, hinfo: dict, info,
    rec: EvalRecord, crop_box, out_path: Path,
    zoom_mask=None, rec_zoom=None,
    crop_pil=None, attn_crop=None, crop_grid_hw=None,
    decoder_name: str = 'hdbscan',
) -> None:
    from experiments.labs.viz import plot_hdbscan_result

    gt_box = None
    if rec.gt_mask is not None and rec.gt_mask.any():
        gt_box = mask_to_bbox(rec.gt_mask.astype(bool))

    zoom_iou = f' zoom_iou={rec_zoom.iou:.3f}' if rec_zoom is not None else ''
    n_clusters_str = f'  clusters={hinfo.get("n_clusters")}' if decoder_name == 'hdbscan' else ''
    title = (f'{item.item_id}  src={item.source}  '
             f'flat_f1={rec.f1:.3f} flat_iou={rec.iou:.3f}{zoom_iou}  '
             f'bucket={rec.bucket}{n_clusters_str}')
    # Pass ModelInfo straight through (carries MIL attention + grid geometry).
    # Also forward the pass-2 crop image and crop attention so the viz can show
    # what the model actually saw at the zoom and how it attended there.
    fig = plot_hdbscan_result(
        img_pil, patch_mask, info,
        gt_mask=rec.gt_mask.astype(bool) if rec.gt_mask is not None else None,
        zoom_mask=zoom_mask, gt_box=gt_box, crop_box=crop_box, title=title,
        crop_pil=crop_pil, attn_crop=attn_crop, crop_grid_hw=crop_grid_hw,
        decoder_name=decoder_name,
    )
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


# ── main lab ────────────────────────────────────────────────────────────────────

def run_hdbscan_lab(args) -> Dict[str, List[EvalRecord]]:
    import torch
    from lab_utils.eval.aggregate import decoder_bench, summarize
    from experiments.labs.attention_zoom import attention_zoom_single

    if args.decoder == 'hdbscan' and not hdbscan_available():
        raise RuntimeError(
            'hdbscan_lab: no HDBSCAN backend (need sklearn>=1.3 or the hdbscan package)'
        )

    device  = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    log_line(f'[buckets] loading checkpoint: {args.checkpoint}')
    model, cfg, res = load_eval_model(
        args.checkpoint, device=device, strict=not args.non_strict,
        model_name=args.model_name, image_size=args.image_size,
        patch_size=args.patch_size,
    )
    if args.compile:
        log_line('[buckets] compiling model with torch.compile...')
        model = torch.compile(model)

    if cfg is not None and cfg.contrastive_dim <= 0:
        raise RuntimeError('hdbscan_lab: checkpoint has no contrastive head; embeddings required')

    items = _collect_items(args, res)
    if not items:
        raise RuntimeError('hdbscan_lab: no dataset roots configured/found')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hk = dict(
        min_cluster_size=args.min_cluster_size,
        theta_x=args.theta_x,
        polarity=args.polarity,
        spatial_weight=args.spatial_weight,
    )

    flat_records: List[EvalRecord] = []
    zoom_records: List[EvalRecord] = []
    n_viz = 0

    from tqdm import tqdm
    from experiments.labs.attention_zoom import multi_zoom_single
    for i, item in enumerate(tqdm(items, desc=f'[buckets] {args.decoder}', unit='item')):
        try:
            img_t, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
            with torch.no_grad():
                info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=args.amp_dtype)
            if info.embeddings is None:
                log_line(f'[buckets] WARN: no embeddings for {item.item_id}, skip')
                continue

            if args.decoder == 'hdbscan':
                mask_flat, hinfo = hdbscan_decode(
                    info.embeddings, attention=info.attention, grid_hw=info.grid_hw, **hk
                )
                patch_mask = np.asarray(mask_flat, dtype=bool).reshape(info.grid_hw)
            else:  # kmeans
                patch_mask = decode_kmeans(info)
                hinfo = {"n_clusters": 2}

            rec_flat   = eval_metric(patch_mask, info, item, decoder=args.decoder)
            flat_records.append(rec_flat)

            crop_box     = None
            zoom_mask    = None
            rec_zoom     = None
            crop_pil     = None
            attn_crop    = None
            crop_grid_hw = None
            if not args.no_zoom:
                if args.zoom_mode == 'mask':
                    # Seed the crop from where HDBSCAN already predicted the splice,
                    # not from MIL pool attention (which is a proxy, not a locator).
                    rec_zoom, debug = multi_zoom_single(
                        model, item, res, device=device, use_amp=use_amp,
                        amp_dtype=args.amp_dtype, box_source='decode',
                        decoder=args.decoder, return_debug=True,
                    )
                    first_box = debug.get('per_box', [{}])
                    first_box = first_box[0] if first_box else {}
                    crop_box     = first_box.get('bbox')
                    crop_pil     = first_box.get('crop_pil')
                    attn_crop    = first_box.get('attn_crop')
                    crop_grid_hw = first_box.get('crop_grid_hw')
                else:
                    rec_zoom, debug = attention_zoom_single(
                        model, item, res, device=device, use_amp=use_amp,
                        amp_dtype=args.amp_dtype,
                        decoder=args.decoder, attn_percentile=args.attn_percentile,
                        min_box_size=args.min_box_size,
                        return_debug=True,
                    )
                    crop_box     = debug.get('bbox')
                    crop_pil     = debug.get('crop_pil')
                    attn_crop    = debug.get('attn_crop')
                    crop_grid_hw = debug.get('crop_grid_hw')
                zoom_records.append(rec_zoom)
                zoom_mask = debug.get('mask_zoom')    # pixel-res placed-back mask

            if n_viz < args.viz_n and not item.is_real:
                _visualise_item(
                    item, img_pil, patch_mask, hinfo, info, rec_flat, crop_box,
                    out_dir / f'{i:04d}_{item.source}_{item.item_id}_{args.decoder}.png',
                    zoom_mask=zoom_mask, rec_zoom=rec_zoom,
                    crop_pil=crop_pil, attn_crop=attn_crop, crop_grid_hw=crop_grid_hw,
                    decoder_name=args.decoder,
                )
                n_viz += 1
        except Exception as exc:
            log_line(f'[buckets] WARN: item={item.item_id} failed: {exc}')

    # ── numerical results ──────────────────────────────────────────────────────
    flat_tag = args.decoder
    zoom_tag = f'{args.decoder}_zoom'

    log_line(f'[buckets] ── {flat_tag.upper()} (full-frame) ──')
    summarize(flat_records, log_tag='[buckets]', tag=flat_tag)

    results = {flat_tag: flat_records}
    if not args.no_zoom and zoom_records:
        log_line(f'[buckets] ── {flat_tag.upper()} + attention-zoom ──')
        summarize(zoom_records, log_tag='[buckets]', tag=zoom_tag)
        results[zoom_tag] = zoom_records
        decoder_bench(results, log_tag='[buckets]')

    _dump_records_csv(results, out_dir / 'per_item_results.csv')
    log_line(f'[buckets] wrote {n_viz} figures + per_item_results.csv → {out_dir}')
    return results


def _dump_records_csv(results: Dict[str, List[EvalRecord]], path: Path) -> None:
    from lab_utils.logging.csv_logger import CsvLogger
    logger = CsvLogger(path)
    for mode, recs in results.items():
        for r in recs:
            logger.write(
                mode=mode, item_id=r.item_id, source=r.source, is_real=r.is_real,
                bucket=r.bucket, mask_area=round(r.mask_area, 5),
                f1=round(r.f1, 5), iou=round(r.iou, 5),
                precision=round(r.precision, 5), recall=round(r.recall, 5),
                image_score=round(r.image_score, 5),
            )


# ── CLI ─────────────────────────────────────────────────────────────────────────

def _parse_percentile(val: str):
    try:
        return float(val)
    except ValueError:
        return val


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='hdbscan_lab',
                                description='HDBSCAN / KMeans + zoom viz/results on a checkpoint.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out_dir',    required=True)
    p.add_argument('--decoder',    default='hdbscan', choices=['hdbscan', 'kmeans'],
                   help='Decoder to run and compare (flat vs zoom)')

    g = p.add_argument_group('dataset roots')
    for source, attr in _SOURCE_ROOT_MAP.items():
        g.add_argument(f'--{attr}', default=None)

    g = p.add_argument_group('hdbscan params (ignored for kmeans)')
    g.add_argument('--min_cluster_size', type=int,   default=8)
    g.add_argument('--theta_x',          type=float, default=0.5)
    g.add_argument('--polarity',         default='attention', choices=['size', 'attention'])
    g.add_argument('--spatial_weight',   type=float, default=0.0)

    g = p.add_argument_group('zoom + viz')
    g.add_argument('--no_zoom',         action='store_true')
    g.add_argument('--zoom_mode',       default='attention', choices=['attention', 'mask'],
                   help='"attention": crop guided by MIL pool attention (default). '
                        '"mask": crop guided by flat HDBSCAN prediction components — '
                        'more reliable when the model has no patch-level supervision.')
    g.add_argument('--attn_percentile', type=_parse_percentile, default='otsu',
                   help='Only used with --zoom_mode attention')
    g.add_argument('--min_box_size',    type=int,   default=8,
                   help='Minimum zoom window size in patches on each side')
    g.add_argument('--viz_n',           type=int,   default=24)
    g.add_argument('--max_items',       type=int,   default=None)

    g = p.add_argument_group('backbone override (for cfg-less legacy checkpoints)')
    g.add_argument('--model_name',  default=None,
                   help='HF backbone id; default = project DINOv3 ViT-H/16+')
    g.add_argument('--image_size',  type=int, default=None, help='default 448')
    g.add_argument('--patch_size',  type=int, default=None, help='default 16')
    g.add_argument('--non_strict',  action='store_true',
                   help='load_state_dict(strict=False) — tolerate key mismatches')

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'],
                   help='Data type for mixed precision (float16 or bfloat16)')
    g.add_argument('--compile', action='store_true',
                   help='Compile model with torch.compile for faster execution')
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_hdbscan_lab(args)


if __name__ == '__main__':
    main()
