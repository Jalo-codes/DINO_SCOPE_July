"""experiments.scripts.gen_size_bucket_visuals — viz panels for IMD2020 + TGIF2
partitioned by splice-area bucket (tiny / small / medium / large).

Pre-buckets items via a cheap PIL mask read so inference only runs on items
that will actually be saved.  Output layout::

    {out_dir}/{source}/{bucket}/  e.g.  results/visuals/size_buckets/imd2020/tiny/

Bucket thresholds (lab_utils.eval.buckets):
    tiny   ≤ 5 %   of pixels  (includes most crops / small objects)
    small   5 – 15 %
    medium 15 – 30 %
    large  > 30 %   (near-full-frame manipulations)

Usage (on the 2080 box):
    export PY=/home/fri-team-4/dino_venv/bin/python
    GPU=1 bash run_scripts/gen_visuals_by_size.sh
    # or directly:
    CUDA_VISIBLE_DEVICES=1 $PY -m experiments.scripts.gen_size_bucket_visuals \\
        --checkpoint /media/ssd/runs/ablation/lora_rank_sweep/r032/best.pt \\
        --out_dir results/visuals/size_buckets \\
        --imd2020_root /media/ssd/DINO_SCOPE_DATA/IMD2020 \\
        --tgif2_root /media/ssd/DINO_SCOPE_DATA/content/flux_originals \\
        --imd_val_split 1.0 --tgif_eval_per_cell 60 \\
        --max_per_bucket 75 --zoom
"""

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lab_utils.eval.buckets import area_to_bucket
from lab_utils.eval.fetch import model_info
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.val_sources import add_source_root_args, collect_val_items_by_source
from lab_utils.logging.text import log_line
from lab_utils.train.distributed import unwrap_model


def _get_decoder(name: str):
    if name == 'kmeans':
        from lab_utils.eval.decode.kmeans import decode_kmeans
        return decode_kmeans
    if name == 'threshold':
        from lab_utils.eval.decode.threshold import decode_threshold
        return decode_threshold
    if name == 'hdbscan':
        from lab_utils.eval.decode.hdbscan import decode_hdbscan
        return decode_hdbscan
    raise ValueError(f'unknown decoder: {name!r}')


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description='Viz panels for IMD2020 + TGIF2 partitioned by splice-area bucket.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    p.add_argument('--out_dir', required=True, help='Output root; sub-dirs created automatically')
    p.add_argument('--decoder', default='kmeans', choices=['kmeans', 'threshold', 'hdbscan'])
    p.add_argument('--zoom', action='store_true', help='Attention-guided two-pass zoom')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    p.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'])
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--sources', nargs='*', default=['imd2020', 'tgif2'],
                   help='Sources to visualize')
    p.add_argument('--max_per_bucket', type=int, default=None,
                   help='Max panels saved per (source × bucket) cell. '
                        'Pre-buckets cheaply so inference only runs on items that will be saved.')
    # max_items is consumed by collect_val_items_by_source as a per-source cap
    p.add_argument('--max_items', type=int, default=None,
                   help='Hard cap on items collected per source (before pre-bucketing)')
    g = p.add_argument_group('dataset roots')
    add_source_root_args(g)
    return p


def _prescan_buckets(items, res) -> dict:
    """Return {item_id: bucket} via cheap mask-area reads (no inference)."""
    out = {}
    for item in items:
        if item.is_real:
            continue
        try:
            area = item.mask_area(res)
            out[item.item_id] = area_to_bucket(area)
        except Exception:
            out[item.item_id] = 'tiny'
    return out


def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(
        args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    )
    use_amp = not args.no_amp and device.type == 'cuda'
    decode_fn = _get_decoder(args.decoder)

    log_line(f'[size_viz] loading checkpoint: {args.checkpoint}')
    model, cfg, res = load_eval_model(args.checkpoint, device=device, strict=False)
    bare_model = unwrap_model(model)

    val_by_source = collect_val_items_by_source(args, res, log_tag='[size_viz]')

    out_root = Path(args.out_dir)
    bucket_counts: dict = {}   # (source, bucket) → saved count

    disable_tqdm = not sys.stdout.isatty()
    try:
        from tqdm import tqdm as _tqdm
        def tqdm(it, **kw): return _tqdm(it, disable=disable_tqdm, **kw)
    except ImportError:
        def tqdm(it, **kw): return it

    for source, items in val_by_source.items():
        splices = [it for it in items if not it.is_real]
        if not splices:
            continue

        # ── Pre-bucket (PIL only, no GPU) ───────────────────────────────────────
        log_line(f'[size_viz] pre-scanning {len(splices)} splice items from {source} ...')
        id_to_bucket = _prescan_buckets(splices, res)

        # Filter: only keep items that still have room in their bucket
        to_infer = []
        seen_counts: dict = {}   # bucket → how many we've already accepted
        for item in splices:
            bkt = id_to_bucket.get(item.item_id, 'tiny')
            n = seen_counts.get(bkt, 0)
            if args.max_per_bucket is None or n < args.max_per_bucket:
                to_infer.append((item, bkt))
                seen_counts[bkt] = n + 1

        log_line(
            f'[size_viz] {source}: {len(to_infer)} items selected for inference '
            f'(buckets: ' +
            ', '.join(f'{b}={seen_counts.get(b, 0)}' for b in ('tiny', 'small', 'medium', 'large')) +
            ')'
        )

        # ── Inference + viz ─────────────────────────────────────────────────────
        for item, bucket in tqdm(to_infer, desc=f'[size_viz] {source}', unit='img'):
            key = (source, bucket)
            if bucket_counts.get(key, 0) >= (args.max_per_bucket or 10**9):
                continue   # safety: already filled (shouldn't happen given pre-filter)

            try:
                if args.zoom:
                    from experiments.labs.attention_zoom import attention_zoom_single
                    rec, debug = attention_zoom_single(
                        bare_model, item, res,
                        device=device, use_amp=use_amp, amp_dtype=args.amp_dtype,
                        decoder=args.decoder,
                        return_debug=True,
                    )
                    patch_mask_for_viz = None   # debug has mask_full
                    info_for_viz = None         # debug has attn1 / grid_hw
                else:
                    img_t = load_image_tensor(item, res, device=device)
                    with torch.no_grad():
                        info_for_viz = model_info(
                            bare_model, img_t,
                            device=device, amp=use_amp, amp_dtype=args.amp_dtype,
                        )
                    patch_mask_for_viz = decode_fn(info_for_viz)
                    rec = eval_metric(
                        patch_mask_for_viz, info_for_viz, item, decoder=args.decoder
                    )
                    debug = None
            except Exception as exc:
                log_line(f'[size_viz] WARN inference skipped {item.item_id}: {exc}')
                continue

            # ── Build panel ─────────────────────────────────────────────────────
            try:
                img_pil = Image.open(item.image).convert('RGB')
                title = f"{item.item_id} | F1={rec.f1:.3f} | {bucket}"

                if args.zoom:
                    from experiments.labs.viz import plot_hdbscan_result
                    from lab_utils.eval.zoom import mask_to_bbox

                    class _MockInfo:
                        def __init__(self, attn, hw):
                            self.attention = attn
                            self.grid_hw   = hw

                    grid_hw   = debug.get('grid_hw')
                    mask_full = np.asarray(debug.get('mask_full'), dtype=bool)
                    if mask_full.ndim == 1:
                        mask_full = mask_full.reshape(grid_hw)

                    gt_box = None
                    if rec.gt_mask is not None and rec.gt_mask.any():
                        gt_box = mask_to_bbox(rec.gt_mask.astype(bool))

                    fig = plot_hdbscan_result(
                        img_pil,
                        patch_mask=mask_full,
                        info=_MockInfo(debug.get('attn1'), grid_hw),
                        gt_mask=rec.gt_mask.astype(bool) if rec.gt_mask is not None else None,
                        zoom_mask=debug.get('mask_zoom'),
                        gt_box=gt_box,
                        crop_box=debug.get('bbox'),
                        crop_pil=debug.get('crop_pil'),
                        attn_crop=debug.get('attn_crop'),
                        crop_grid_hw=debug.get('crop_grid_hw'),
                        title=title,
                        decoder_name=args.decoder,
                    )
                else:
                    from experiments.labs.viz import plot_prediction
                    fig = plot_prediction(
                        img_pil,
                        patch_mask=patch_mask_for_viz,
                        info=info_for_viz,
                        gt_mask=rec.gt_mask.astype(bool) if rec.gt_mask is not None else None,
                        title=title,
                    )

                out_path = out_root / source / bucket
                out_path.mkdir(parents=True, exist_ok=True)
                fname = item.item_id.replace('/', '_').replace('\\', '_') + '.png'
                if hasattr(fig, 'savefig'):
                    fig.savefig(out_path / fname, dpi=130, bbox_inches='tight')
                    try:
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                    except ImportError:
                        pass
                else:
                    fig.save(out_path / fname)

                bucket_counts[key] = bucket_counts.get(key, 0) + 1

            except Exception as exc:
                log_line(f'[size_viz] WARN viz failed {item.item_id}: {exc}')

    log_line(f'[size_viz] done → {out_root}')
    for (src, bkt), n in sorted(bucket_counts.items()):
        log_line(f'  {src:12s}  {bkt:8s}  {n} panels')


if __name__ == '__main__':
    main()
