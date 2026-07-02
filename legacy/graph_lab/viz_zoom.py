"""graph_lab.viz_zoom — VISUALIZE the repo's coarse→fine attention zoom,
decoder by decoder (K-means / graph / HDBSCAN), in a MULTI-ROW composite.

Reuses the eval suite's zoom path (the same functions as
``collect_coarse_to_fine_samples`` in ``lab_utils/eval/localization.py`` and the
bbox utilities in ``lab_utils/eval/zoom.py``) — it does NOT reimplement the
geometry. The only new thing is the picture: it runs the whole coarse→fine
independently for each decoder and shows what the zoom actually *sees*.

Each splice is ONE PNG of several rows:

    Original | Attention(full) | GT          | ·       | ·
    K-means  | coarse  | refined+bbox | crop  | crop-heat | crop-decode
    Graph    | coarse  | refined+bbox | crop  | crop-heat | crop-decode
    HDBSCAN  | coarse  | refined+bbox | crop  | crop-heat | crop-decode

So per decoder you see, on the SAME image: the full-frame coarse mask, the
refined (post-zoom) mask with its zoom bbox(es) drawn, the cropped region the
model re-embedded, that crop's ZOOMED attention heatmap, and the crop's decode —
labelled with the coarse→refined IoU.

Flow per decoder (identical to the deployed coarse→fine):
  1. Pass 1: embed full frame, decode.
  2. Bbox: ``_minority_bbox`` (single) or attention via ``multi_zoom_bboxes``
     (multi, with that decoder's foreground as hot_mask).
  3. Pass 2: crop → re-embed → decode → ``_place_fine_in_pixel_frame`` paste back.
     Single mode applies the refine AS-IS (matches collect_coarse_to_fine_samples);
     multi mode keeps each crop only when ``p_zoom >= p_full`` (the per-crop ratchet).

Model-bearing (re-embeds crops), so it loads a checkpoint like viz_decode.

Usage:
    python -m graph_lab.viz_zoom \\
        --ckpt /content/drive/MyDrive/DINO_SCOPE_RUNS/<run>/best.pt \\
        --imd2020_root /content/IMD2020 --imd_val_only \\
        --tau_pos 0.60 --tau_neg 0.15 --s_edge 0.97 \\
        --hdb_min_cluster_size 8 --hdb_theta_x 0.5 \\
        --zoom_mode single --pad_frac 0.25 --n_items 24 \\
        --out /content/viz_zoom_margin1560 [--show]
"""

import argparse
import math
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch
from PIL import Image, ImageDraw

from lab_utils.data.sampling import deterministic_subsample
from lab_utils.data.resolution import resize_only
from lab_utils.train.checkpoint import load as ckpt_load
from lab_utils.train.amp import resolve_amp
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.eval.partition import DecodeSpec, decode_oracle_labels, graph_components_decode
from lab_utils.eval.localization import (
    _embed_logit_attn_pil, _minority_bbox, _place_fine_in_pixel_frame,
    _load_gt_pixel_mask, _patches_to_pixels, _mask_metrics,
)
from lab_utils.eval.zoom import multi_zoom_bboxes
from lab_utils.viz import heatmap_rgb, overlay_blend, mask_tint, save_composite

from graph_lab.hdbscan_decode import hdbscan_decode, hdbscan_available

from contrastive_inpainting_v1.configs.base import Config
from contrastive_inpainting_v1.experiments.imd2020_bce import IMD2020BCESpec

_SPLICE_KINDS = ('imd_splice', 'casia_splice')
_SHOW_WARNED = False
_COLS = 6  # panels per row (context row is padded to this width)


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
                   help='Render each composite inline (run in-cell: viz_zoom.main([...])).')
    # Graph decode knobs.
    p.add_argument('--tau_pos', type=float, default=0.60)
    p.add_argument('--tau_neg', type=float, default=0.15)
    p.add_argument('--s_edge', type=float, default=None)
    p.add_argument('--knn', type=int, default=10)
    p.add_argument('--m_min', type=int, default=4)
    p.add_argument('--theta_w', type=float, default=None)
    p.add_argument('--theta_x', type=float, default=None)
    # HDBSCAN knobs.
    p.add_argument('--hdb_min_cluster_size', type=int, default=8)
    p.add_argument('--hdb_min_samples', type=int, default=None)
    p.add_argument('--hdb_theta_x', type=float, default=0.5)
    p.add_argument('--hdb_polarity', choices=('size', 'attention'), default='size')
    # Zoom knobs (same as collect_coarse_to_fine_samples).
    p.add_argument('--zoom_mode', choices=('single', 'multi'), default='single')
    p.add_argument('--pad_frac', type=float, default=0.25)
    p.add_argument('--refine_max_frac', type=float, default=0.40)
    p.add_argument('--zoom_thresh', choices=('gap', 'otsu', 'hyst'), default='otsu')
    p.add_argument('--max_regions', type=int, default=3)
    p.add_argument('--methods', default='kmeans,graph,hdbscan',
                   help='Comma list subset of kmeans,graph,hdbscan.')
    return p


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def _in_notebook_kernel() -> bool:
    """True only when running inside a live notebook kernel (not a subprocess)."""
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
            print('[viz_zoom] --show only renders inside the notebook kernel; run in-cell '
                  'via viz_zoom.main([...]). PNGs are still saved to --out.')
            _SHOW_WARNED = True
        return
    from IPython.display import Image as _IPyImage, display as _display
    _display(_IPyImage(filename=path))


def _draw_boxes(base_rgb, rects, color=(255, 255, 0), width=3):
    im = Image.fromarray(base_rgb.copy())
    dr = ImageDraw.Draw(im)
    for (x0, y0, x1, y1) in rects:
        dr.rectangle([int(x0), int(y0), int(x1), int(y1)], outline=color, width=width)
    return np.asarray(im, dtype=np.uint8)


def _pick(mask_a, mask_b, gt):
    """Oracle polarity: better-by-F1 of a labeling and its complement."""
    fa = _mask_metrics(mask_a.reshape(-1), gt.reshape(-1))[0]
    fb = _mask_metrics(mask_b.reshape(-1), gt.reshape(-1))[0]
    return mask_a if fa >= fb else mask_b


def _blank(S, label_color=(70, 70, 70)):
    """A dark filler panel (keeps each decoder on its own grid row)."""
    a = np.zeros((S, S, 3), dtype=np.uint8)
    a[:] = label_color
    return a


def main(argv=None):
    args = _build_parser().parse_args(argv)
    from contrastive_inpainting_v1.pipeline.cli import apply_path_defaults
    apply_path_defaults(args)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    resolve_amp(device, want_amp=True)
    cfg = Config()
    res = cfg.resolution
    P = res.num_patches_per_side
    ps = res.patch_size
    S = res.image_size
    mean, std = tuple(cfg.IMAGENET_MEAN), tuple(cfg.IMAGENET_STD)

    km_spec = DecodeSpec()  # default = k-means
    graph_spec = DecodeSpec(
        method='graph', tau_pos=float(args.tau_pos), tau_neg=float(args.tau_neg),
        s_edge=args.s_edge, mutual_knn_k=int(args.knn), m_min=int(args.m_min),
        theta_w=args.theta_w, theta_x=args.theta_x)

    # Methods to compare (drop hdbscan if the dep is missing).
    palette = {'kmeans': ('K-means', (0, 140, 255)),
               'graph': ('Graph', (255, 165, 0)),
               'hdbscan': ('HDBSCAN', (255, 0, 255))}
    methods = [m.strip() for m in args.methods.split(',') if m.strip() in palette]
    if 'hdbscan' in methods and not hdbscan_available():
        print('[viz_zoom] WARN: HDBSCAN unavailable (need scikit-learn>=1.3 or the hdbscan '
              'package) — dropping it from the comparison.')
        methods = [m for m in methods if m != 'hdbscan']

    def _labels(method, z, att):
        """(N,) {0,1} labeling for a decoder — graph/kmeans via the shared
        dispatcher, hdbscan via its accepted-foreground mask."""
        if method == 'kmeans':
            return decode_oracle_labels(z, km_spec)
        if method == 'graph':
            return decode_oracle_labels(z, graph_spec)
        m, _ = hdbscan_decode(
            z, attention=att, grid_hw=(P, P),
            min_cluster_size=int(args.hdb_min_cluster_size), min_samples=args.hdb_min_samples,
            theta_x=float(args.hdb_theta_x), polarity=args.hdb_polarity)
        return m.astype(np.int64)

    def _crop_panels(crop_pil, a2, decode_grid, color):
        """Three zoom-crop panels: the (S,S)-resized crop the model re-embeds,
        its ZOOMED attention heatmap, and the crop's decode tinted on it."""
        cdisp = np.asarray(resize_only(crop_pil, res), dtype=np.uint8)
        heat = (overlay_blend(cdisp, heatmap_rgb(a2.reshape(P, P), (S, S)))
                if a2 is not None else cdisp)
        dec = mask_tint(cdisp, decode_grid, (S, S), color)
        return cdisp, heat, dec

    def _coarse_to_fine(method, z1, attn_full, p_full, src, gt_px, color):
        """Full coarse→fine for one decoder. Returns
        (coarse(S,S), refined(S,S), rects[disp px], c_iou, r_iou, fired, zoom)
        where zoom = (crop_rgb, crop_heat, crop_decode) of the PRIMARY crop or
        None if the zoom didn't fire."""
        W, H = src.size
        raw1 = _labels(method, z1, attn_full)
        grid1 = raw1.reshape(P, P)
        n1 = int((grid1 == 1).sum())
        minority = 1 if n1 <= (grid1.size - n1) else 0
        coarse = _pick(_patches_to_pixels(raw1 == 1, P, ps),
                       _patches_to_pixels(raw1 == 0, P, ps), gt_px)
        c_iou = _mask_metrics(coarse.reshape(-1), gt_px.reshape(-1))[1]
        refined = coarse.copy(); rects = []; fired = False; zoom = None

        if args.zoom_mode == 'single':
            bbox, frac = _minority_bbox(raw1, P, args.pad_frac)
            if bbox is not None and frac <= args.refine_max_frac:
                r0, r1, c0, c1 = bbox
                x0 = int(round(c0 / P * W)); x1 = max(x0 + 1, int(round(c1 / P * W)))
                y0 = int(round(r0 / P * H)); y1 = max(y0 + 1, int(round(r1 / P * H)))
                rects.append((c0 * ps, r0 * ps, c1 * ps, r1 * ps))
                crop_pil = src.crop((x0, y0, x1, y1))
                # NOTE: single-mode canonical (collect_coarse_to_fine_samples) embeds
                # with _embed_pil and applies the refine AS-IS — no p_zoom ratchet
                # (that gate lives only in the multi path). We use the logit/attn variant
                # only to get a2 for the zoom-heatmap panel; the decode path is unchanged.
                z2, lz, a2 = _embed_logit_attn_pil(model, crop_pil, res, device, mean, std)
                if z2 is not None:
                    raw2 = _labels(method, z2, a2)
                    m1 = _place_fine_in_pixel_frame(raw2 == 1, bbox, res)
                    m0 = _place_fine_in_pixel_frame(raw2 == 0, bbox, res)
                    f1 = _mask_metrics(m1.reshape(-1), gt_px.reshape(-1))[0]
                    f0 = _mask_metrics(m0.reshape(-1), gt_px.reshape(-1))[0]
                    chosen = 1 if f1 >= f0 else 0
                    refined = m1 if chosen == 1 else m0
                    decode_grid = (raw2 == chosen).reshape(P, P)
                    zoom = _crop_panels(crop_pil, a2, decode_grid, color)
                    fired = True
        else:  # multi — attention bboxes, this decoder's foreground as hot_mask
            attn_grid = attn_full.reshape(P, P) if attn_full is not None else np.zeros((P, P))
            bboxes = multi_zoom_bboxes(
                attn_grid, H, W, max_regions=int(args.max_regions), theta_fill=0.45,
                base_padding=2, pad_frac=args.pad_frac, min_crop_patches=8,
                thresh_mode=args.zoom_thresh, hot_mask=(grid1 == minority))
            refined = np.zeros((S, S), dtype=bool)
            for (x0, y0, x1, y1) in bboxes:
                c0 = max(0, min(P - 1, int(round(x0 * P / W)))); c1 = max(c0 + 1, min(P, int(round(x1 * P / W))))
                r0 = max(0, min(P - 1, int(round(y0 * P / H)))); r1 = max(r0 + 1, min(P, int(round(y1 * P / H))))
                crop_pil = src.crop((x0, y0, x1, y1))
                z2, lz, a2 = _embed_logit_attn_pil(model, crop_pil, res, device, mean, std)
                pz = _sigmoid(lz) if lz is not None else 1.0
                if z2 is None or pz < p_full:
                    continue
                raw2 = _labels(method, z2, a2)
                g2 = raw2.reshape(P, P); m2 = int((g2 == 1).sum())
                mlab = 1 if m2 <= (g2.size - m2) else 0   # splice ≈ minority region
                refined |= _place_fine_in_pixel_frame(raw2 == mlab, (r0, r1, c0, c1), res)
                rects.append((c0 * ps, r0 * ps, c1 * ps, r1 * ps))
                if zoom is None:  # primary crop = first (highest hot-mass) to fire
                    zoom = _crop_panels(crop_pil, a2, (raw2 == mlab).reshape(P, P), color)
            fired = bool(rects)
            if not fired:
                refined = coarse.copy()

        r_iou = _mask_metrics(refined.reshape(-1), gt_px.reshape(-1))[1]
        return coarse, refined, rects, c_iou, r_iou, fired, zoom

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
        by_split[k] = deterministic_subsample(by_split[k], args.n_items, seed='vizzoom')
    print(f'[viz_zoom] items: imd={len(by_split["imd_val"])} casia={len(by_split["casia_val"])} '
          f'mode={args.zoom_mode} methods={methods}')

    ckpt = ckpt_load(args.ckpt, map_location=str(device))
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    c_dim = int(sd['contrastive_proj.weight'].shape[0]) if 'contrastive_proj.weight' in sd else 0
    p_hidden = int(sd['pool.V.weight'].shape[0]) if 'pool.V.weight' in sd else 0
    has_patch = 'patch_head.weight' in sd
    if c_dim <= 0:
        print('[viz_zoom] ERROR: checkpoint has no contrastive head.')
        return
    model = build_multi_head_detector(
        model_name=cfg.MODEL_NAME, resolution=res,
        lora_rank=cfg.LORA_RANK, lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT, lora_targets=cfg.LORA_TARGETS,
        contrastive_dim=c_dim, pool_hidden=p_hidden, patch_bce=has_patch, device=device)
    model.load_state_dict(sd); model.eval()
    print(f'[viz_zoom] loaded epoch={ckpt.get("epoch","?")} c_dim={c_dim} pool_hidden={p_hidden}')

    agg = {m: {'c': [], 'r': [], 'fired': 0, 'imp': 0} for m in methods}
    n_saved = 0
    for split, items in by_split.items():
        for idx, it in enumerate(items):
            img_path = str(it.get('img', ''))
            stem = os.path.splitext(os.path.basename(img_path))[0]
            try:
                src = Image.open(img_path).convert('RGB')
            except Exception:
                continue
            gt_px = _load_gt_pixel_mask({'mask_path': str(it.get('mask'))}, res)
            if gt_px is None:
                continue
            disp = np.asarray(resize_only(src, res), dtype=np.uint8)

            z1, logit_full, attn_full = _embed_logit_attn_pil(model, src, res, device, mean, std)
            if z1 is None:
                continue
            p_full = _sigmoid(logit_full) if logit_full is not None else 0.0

            # Row 0: shared full-frame context, padded to _COLS.
            attn_panel = (overlay_blend(disp, heatmap_rgb(attn_full.reshape(P, P), (S, S)))
                          if attn_full is not None else _blank(S))
            panels = [('Original', disp),
                      (f'Attention  p={p_full:.2f}', attn_panel),
                      ('GT', mask_tint(disp, gt_px, (S, S), (0, 255, 0)))]
            while len(panels) % _COLS != 0:
                panels.append(('', _blank(S)))

            # One row per decoder: coarse | refined+bbox | crop | crop-heat | crop-decode | (pad)
            for m in methods:
                name, color = palette[m]
                coarse, refined, rects, c_iou, r_iou, fired, zoom = _coarse_to_fine(
                    m, z1, attn_full, p_full, src, gt_px, color)
                agg[m]['c'].append(c_iou); agg[m]['r'].append(r_iou)
                agg[m]['fired'] += int(fired); agg[m]['imp'] += int(r_iou > c_iou)
                refined_base = _draw_boxes(disp, rects) if rects else disp
                tag = 'zoom' if fired else 'no-zoom'
                row = [
                    (f'{name}\ncoarse c{c_iou:.2f}', mask_tint(disp, coarse, (S, S), color)),
                    (f'refined r{r_iou:.2f}\n{tag}', mask_tint(refined_base, refined, (S, S), color)),
                ]
                if zoom is not None:
                    crop_rgb, crop_heat, crop_dec = zoom
                    row += [('zoom crop', crop_rgb),
                            ('zoom heatmap', crop_heat),
                            ('zoom decode', crop_dec)]
                else:
                    row += [('zoom crop\n(none)', _blank(S)),
                            ('zoom heatmap\n(none)', _blank(S)),
                            ('zoom decode\n(none)', _blank(S))]
                while len(row) < _COLS:
                    row.append(('', _blank(S)))
                panels.extend(row)

            path = os.path.join(args.out, f'{split}_{idx:03d}_{stem}.png')
            save_composite(panels, path, panel_size=int(args.panel_size), cols=_COLS)
            _display_inline(path, args.show)
            n_saved += 1

    print(f'\n[viz_zoom] {n_saved} splices — coarse→refined median IoU by decoder:')
    print(f'  {"method":>9} | coarse | refined | zoom fired | improved')
    print('  ' + '-' * 56)
    for m in methods:
        c = np.array(agg[m]['c']); r = np.array(agg[m]['r'])
        if c.size == 0:
            continue
        print(f'  {palette[m][0]:>9} |  {np.median(c):.3f} |  {np.median(r):.3f}  |'
              f'   {agg[m]["fired"]}/{n_saved}   |  {agg[m]["imp"]}/{n_saved}')
    print(f'[viz_zoom] saved {n_saved} composites → {args.out}/')


if __name__ == '__main__':
    main()
