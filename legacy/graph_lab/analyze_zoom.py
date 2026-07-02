"""graph_lab.analyze_zoom — analyze ATTENTION-zoom performance on a checkpoint.

Uses the eval utility itself (``lab_utils/eval/localization.py``): every splice
goes through ``attention_zoom_item`` (the exact per-item core of
``collect_attention_zoom_samples``), and the collected samples are reported with
``report_zoom_eval`` — so the printed FULL/ZOOM IoU numbers and the pictures come
from one code path and cannot drift.

Per splice it shows, on one row:

    Original+bbox | Attention | GT | Full decode (iou) | Zoom localize (iou) | Zoom attention

- bbox LOCATION   : the attention-nominated box drawn on the original.
- bbox LOCALIZATION: the zoom crop's decode placed back in the full frame, tinted
  on the full-frame original.
- the ATTENTION map: full-frame, and the zoom crop's own attention.

The headline ZOOM iou is the zoom crop's decode-vs-crop-GT, deferred to the zoom
as-is — NO and/or with the full-frame mask. The box comes straight from
attention; ground truth is only ever used to score.

Model-bearing (re-embeds crops); loads a checkpoint like viz_decode.

Usage:
    python -m graph_lab.analyze_zoom \\
        --ckpt /content/drive/MyDrive/DINO_SCOPE_RUNS/<run>/best.pt \\
        --imd2020_root /content/IMD2020 --imd_val_only \\
        --eval_decode kmeans --zoom_thresh otsu --pad_frac 0.25 \\
        --n_items 24 --out /content/analyze_zoom [--show]
"""

import argparse
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
from PIL import Image, ImageDraw

from lab_utils.data.sampling import deterministic_subsample
from lab_utils.train.checkpoint import load as ckpt_load
from lab_utils.train.amp import resolve_amp
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.eval.partition import DecodeSpec
from lab_utils.eval.localization import (
    attention_zoom_item, report_zoom_eval, _load_gt_pixel_mask,
)
from lab_utils.viz import heatmap_rgb, overlay_blend, mask_tint, save_composite

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec

_SPLICE_KINDS = ('imd_splice', 'casia_splice')
_SHOW_WARNED = False
_COLS = 6


def _build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--imd2020_root', default=None)
    p.add_argument('--casia_root', default=None)
    p.add_argument('--indoor_root', default=None)
    p.add_argument('--casia_train', action='store_true', default=False)
    p.add_argument('--imd_val_only', action='store_true', default=False)
    p.add_argument('--n_items', type=int, default=24, help='Splices per split.')
    p.add_argument('--out', required=True)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--panel_size', type=int, default=240)
    p.add_argument('--show', action='store_true', default=False,
                   help='Render each composite inline (run in-cell: analyze_zoom.main([...])).')
    # Decode (kmeans/graph go through DecodeSpec; hdbscan via the decode_fn hook).
    p.add_argument('--eval_decode', choices=('kmeans', 'graph', 'hdbscan'), default='kmeans')
    p.add_argument('--tau_pos', type=float, default=None)
    p.add_argument('--tau_neg', type=float, default=None)
    p.add_argument('--graph_s_edge', type=float, default=None)
    p.add_argument('--graph_knn', type=int, default=10)
    # HDBSCAN knobs (only used when --eval_decode hdbscan).
    p.add_argument('--hdb_min_cluster_size', type=int, default=8)
    p.add_argument('--hdb_min_samples', type=int, default=2,
                   help='Min samples for HDBSCAN (lower = less noise, default 2).')
    p.add_argument('--hdb_theta_x', type=float, default=0.5)
    p.add_argument('--hdb_polarity', choices=('size', 'attention'), default='attention',
                   help='Background polarity rule for HDBSCAN (default attention).')
    p.add_argument('--hdb_spatial_weight', type=float, default=0.15,
                   help='Spatial coordinate scaling weight to enforce adjacent grouping (default 0.15).')
    # Attention bbox knobs.
    p.add_argument('--zoom_thresh', choices=('gap', 'otsu', 'hyst'), default='otsu')
    p.add_argument('--zoom_thresh_mult', type=float, default=0.85,
                   help='Multiplier applied to attention thresholds (default 0.85).')
    p.add_argument('--pad_frac', type=float, default=0.25)
    p.add_argument('--base_padding', type=int, default=2,
                   help='Base padding in patches around the zoom region (default 2).')
    p.add_argument('--min_crop_patches', type=int, default=8)
    return p


def _in_notebook_kernel() -> bool:
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and getattr(ip, 'kernel', None) is not None
    except Exception:
        return False


def _display_inline(path, show):
    global _SHOW_WARNED
    if not show:
        return
    if not _in_notebook_kernel():
        if not _SHOW_WARNED:
            print('[analyze_zoom] --show only renders inside the notebook kernel; run in-cell '
                  'via analyze_zoom.main([...]). PNGs are still saved to --out.')
            _SHOW_WARNED = True
        return
    from IPython.display import Image as _IPyImage, display as _display
    _display(_IPyImage(filename=path))


def _draw_box(disp, bbox_native, src_wh, color=(255, 255, 0), width=3):
    """Draw the native-pixel bbox on the (S,S) display frame."""
    W, H = src_wh
    S = disp.shape[0]
    x0, y0, x1, y1 = bbox_native
    sx, sy = S / float(W), S / float(H)
    im = Image.fromarray(disp.copy())
    ImageDraw.Draw(im).rectangle(
        [x0 * sx, y0 * sy, x1 * sx, y1 * sy], outline=color, width=width)
    return np.asarray(im, dtype=np.uint8)


def main(argv=None):
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        if _in_notebook_kernel():
            print('[analyze_zoom] bad/missing args (usage above). In a cell pass a list, e.g. '
                  'analyze_zoom.main(["--ckpt", "...", "--imd2020_root", "...", "--out", "..."]).')
            return
        raise
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    resolve_amp(device, want_amp=True)
    cfg = Config()
    res = cfg.resolution
    S = res.image_size
    P = res.num_patches_per_side
    mean, std = tuple(cfg.IMAGENET_MEAN), tuple(cfg.IMAGENET_STD)

    spec_method = args.eval_decode if args.eval_decode in ('kmeans', 'graph') else 'kmeans'
    decode_spec = DecodeSpec(
        method=spec_method,
        tau_pos=float(args.tau_pos) if args.tau_pos is not None else float(cfg.TAU_POS),
        tau_neg=float(args.tau_neg) if args.tau_neg is not None else float(cfg.TAU_NEG),
        n_init=2, s_edge=args.graph_s_edge, mutual_knn_k=int(args.graph_knn))

    # HDBSCAN drives the SAME attention-zoom path via the decode_fn hook, so its
    # dependency stays in graph_lab and never enters lab_utils.
    decode_fn = None
    if args.eval_decode == 'hdbscan':
        from graph_lab.hdbscan_decode import hdbscan_decode, hdbscan_available
        if not hdbscan_available():
            print('[analyze_zoom] ERROR: HDBSCAN unavailable (need scikit-learn>=1.3 '
                  'or the hdbscan package).')
            return

        def decode_fn(z, attention, _P=P):
            m, _ = hdbscan_decode(
                z, attention=attention, grid_hw=(_P, _P),
                min_cluster_size=int(args.hdb_min_cluster_size),
                min_samples=args.hdb_min_samples,
                spatial_weight=float(args.hdb_spatial_weight),
                theta_x=float(args.hdb_theta_x), polarity=args.hdb_polarity)
            return m.astype(np.int64)   # (N,) {0,1}; 1 = accepted foreground

    # ── items / model (mirror viz_decode) ──────────────────────────────────────
    espec = IMD2020BCESpec(
        imd2020_root=args.imd2020_root, casia_root=args.casia_root,
        indoor_root=args.indoor_root,
        imd_train=not args.imd_val_only, casia_train=args.casia_train)
    _, val_items = espec.build_items(cfg)
    by_split = {
        'imd_val':   [it for it in val_items if it.get('source') == 'imd2020'
                      and it.get('kind') in _SPLICE_KINDS and it.get('mask')],
        'casia_val': [it for it in val_items if it.get('source') == 'casia'
                      and it.get('kind') in _SPLICE_KINDS and it.get('mask')],
    }
    for k in by_split:
        by_split[k] = deterministic_subsample(by_split[k], args.n_items, seed='analyzezoom')
    print(f'[analyze_zoom] items: imd={len(by_split["imd_val"])} casia={len(by_split["casia_val"])} '
          f'decode={args.eval_decode} thresh={args.zoom_thresh} pad_frac={args.pad_frac}')

    ckpt = ckpt_load(args.ckpt, map_location=str(device))
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    c_dim = int(sd['contrastive_proj.weight'].shape[0]) if 'contrastive_proj.weight' in sd else 0
    p_hidden = int(sd['pool.V.weight'].shape[0]) if 'pool.V.weight' in sd else 0
    has_patch = 'patch_head.weight' in sd
    if c_dim <= 0:
        print('[analyze_zoom] ERROR: checkpoint has no contrastive head.')
        return
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME, resolution=res,
        lora_rank=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT, lora_targets=cfg.LORA_TARGETS,
        contrastive_dim=c_dim, pool_hidden=p_hidden, patch_bce=has_patch, device=device)
    model.load_state_dict(sd); model.eval()
    print(f'[analyze_zoom] loaded epoch={ckpt.get("epoch","?")} c_dim={c_dim} pool_hidden={p_hidden}')

    GT, FULL, ZOOM = (0, 255, 0), (0, 140, 255), (255, 0, 255)
    by_tag = {}                      # tag -> list[_ZoomSample] for report_zoom_eval
    n_saved = 0
    for split, items in by_split.items():
        tag_samples = []
        for idx, it in enumerate(items):
            img_path = str(it.get('img', '')); mask_path = str(it.get('mask', ''))
            stem = os.path.splitext(os.path.basename(img_path))[0]
            try:
                src = Image.open(img_path).convert('RGB')
                mask = Image.open(mask_path).convert('L')
            except Exception:
                continue
            gt_full = _load_gt_pixel_mask({'mask_path': mask_path}, res)
            if gt_full is None:
                continue

            viz = attention_zoom_item(
                model, src, mask, gt_full, res=res, device=device,
                normalize_mean=mean, normalize_std=std, decode_spec=decode_spec,
                base_padding=int(args.base_padding), pad_frac=float(args.pad_frac),
                min_crop_patches=int(args.min_crop_patches),
                zoom_thresh_mode=str(args.zoom_thresh),
                zoom_thresh_mult=float(args.zoom_thresh_mult), decode_fn=decode_fn,
                kind=it.get('kind', ''))
            tag_samples.append(viz.sample)
            s = viz.sample

            disp = viz.disp
            # bbox LOCATION on the original.
            orig = _draw_box(disp, viz.bbox, src.size) if viz.bbox is not None else disp
            # full-frame attention.
            attn_panel = (overlay_blend(disp, heatmap_rgb(viz.attn_grid, (S, S)))
                          if viz.attn_grid is not None else disp)
            panels = [
                (f'orig+bbox  area={s.area_full:.3f}', orig),
                ('attention', attn_panel),
                ('GT', mask_tint(disp, gt_full, (S, S), GT)),
                (f'full decode\niou={s.iou_full:.2f}', mask_tint(disp, viz.full_pred, (S, S), FULL)),
            ]
            # bbox LOCALIZATION (zoom crop decode placed back in full frame) + zoom-crop attention.
            if viz.bbox is not None and viz.crop_disp is not None:
                loc = mask_tint(disp, viz.zoom_pred_full, (S, S), ZOOM)
                zattn = (overlay_blend(viz.crop_disp, heatmap_rgb(viz.crop_attn, (S, S)))
                         if viz.crop_attn is not None else viz.crop_disp)
                panels.append((f'zoom localize\niou={s.iou_zoom:.2f}', loc))
                panels.append(('zoom attention', zattn))
            else:
                blank = np.zeros((S, S, 3), dtype=np.uint8)
                panels.append((f'zoom localize\n(no zoom) iou={s.iou_zoom:.2f}', blank))
                panels.append(('zoom attention\n(no zoom)', blank.copy()))

            path = os.path.join(args.out, f'{split}_{idx:03d}_{stem}.png')
            save_composite(panels, path, panel_size=int(args.panel_size), cols=_COLS)
            _display_inline(path, args.show)
            n_saved += 1
        by_tag[split] = tag_samples

    # ── authoritative numbers, straight from the eval util ──────────────────────
    print(f'\n[analyze_zoom] {n_saved} composites → {args.out}/')
    for tag, samples in by_tag.items():
        if samples:
            report_zoom_eval(samples, condensed=False, log_tag='[zoom]', tag=tag)


if __name__ == '__main__':
    main()
