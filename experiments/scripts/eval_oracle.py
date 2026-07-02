"""experiments.scripts.eval_oracle — ⚠️ CHEATING ORACLE eval (GT-leaking, ISOLATED).

==============================================================================
THIS IS NOT PART OF THE CONTAMINATION-FREE EVAL PIPELINE.
==============================================================================
The normal pipeline has a single GT touch (lab_utils.eval.metric.metric) and the
decoders pick the splice cluster *blind*, using attention only. This script
deliberately violates that contract: it forms the SAME clusters as the real
decoders, then picks the splice cluster by looking at the ground-truth mask
(highest F1 cluster). That number is an UPPER BOUND ("if the polarity oracle were
perfect"), not a reportable result.

Isolation guarantees:
  * Lives in its own module; nothing in lab_utils/ or the normal eval scripts
    imports it.
  * Re-implements its own tiny GT-scoring helpers so it does NOT touch / extend
    lab_utils.eval.metric (which remains the *sole* GT touch of the real path).
  * Only the GT-free building blocks are reused: model load, the single forward
    (fetch.model_info), the clustering kernels, and OpenSDI item discovery.

What it reports, partitioned by generator:
  * baseline_<decoder>  — the real, contamination-free number (attention polarity)
  * ORACLE_<decoder>    — the cheat (best-cluster-by-GT polarity)
Both use the identical clustering / identical forward pass, so the gap isolates
exactly one factor: the cluster-selection (polarity) decision.

Usage (Colab A100, OpenSDI on Drive):
    python -m experiments.scripts.eval_oracle \
        --checkpoint /content/drive/MyDrive/DINO_SCOPE_RUNS/optimal/optimal_h16plus_688_r16/best.pt \
        --opensdi_root /content/drive/MyDrive/DINO_SCOPE_DATA/OpenSDI_eval/OpenSDI \
        --decoders kmeans \
        --out_json /content/drive/MyDrive/DINO_SCOPE_RUNS/optimal/oracle_opensdi.json
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# ── GT-FREE building blocks (safe to reuse — none of these read the mask) ────────
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.fetch import model_info
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.val_sources import add_source_root_args, collect_val_items_by_source
from lab_utils.eval.decode.kmeans import spherical_kmeans2, polarity_attn
from lab_utils.eval.decode.hdbscan import hdbscan_decode
from lab_utils.eval.zoom import (
    attention_to_bbox, bbox_is_trivial, crop_to_bbox, place_mask_in_frame_pixels,
)
from lab_utils.train.distributed import unwrap_model
from lab_utils.logging.text import log_line


# ── Local GT scoring (a deliberate, ISOLATED copy of metric.py's helpers) ───────
# Re-implemented here on purpose: the oracle is the cheating path, so it must not
# extend lab_utils.eval.metric (the real pipeline's sole GT touch). Kept byte-for-
# byte equivalent to metric.py so baseline numbers here match the real pipeline.

def _load_gt_pixels(mask_path, threshold: float = 0.5) -> Optional[np.ndarray]:
    if mask_path is None:
        return None
    pil = Image.open(mask_path).convert('L')
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return arr >= threshold


def _upsample_pred_to(patch_mask: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    H, W = hw
    pil = Image.fromarray(patch_mask.astype(np.uint8) * 255, mode='L')
    if pil.size != (W, H):
        pil = pil.resize((W, H), Image.NEAREST)
    return np.asarray(pil) > 127


def _binary_scores(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = int((pred & gt).sum())
    p_n = int(pred.sum())
    g_n = int(gt.sum())
    union = int((pred | gt).sum())
    return {
        'f1':        (2.0 * inter / (p_n + g_n)) if (p_n + g_n) > 0 else 0.0,
        'iou':       (inter / union)             if union > 0       else 0.0,
        'precision': (inter / p_n)               if p_n > 0         else 0.0,
        'recall':    (inter / g_n)               if g_n > 0         else 0.0,
        'accuracy':  float((pred == gt).mean()),
    }


def _score_patch_mask(patch_mask_2d: np.ndarray, gt_pix: Optional[np.ndarray],
                      img_size: int) -> Dict[str, float]:
    """Score a (n_side, n_side) bool patch mask per-pixel, exactly like metric.py.

    gt_pix is the native-resolution GT (None for reals → all-zero @ input frame).
    """
    if gt_pix is not None:
        pred = _upsample_pred_to(patch_mask_2d, gt_pix.shape)
        gt = gt_pix
    else:
        gt = np.zeros((img_size, img_size), dtype=bool)
        pred = _upsample_pred_to(patch_mask_2d, (img_size, img_size))
    return _binary_scores(pred, gt)


# ── Cluster enumeration (GT-free) + oracle selection (GT-cheating) ──────────────

def _clusters_and_baseline(decoder: str, z: np.ndarray, attn: Optional[np.ndarray],
                           grid_hw: Tuple[int, int]) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return (list of per-cluster bool masks (N,), baseline splice mask (N,)).

    The clustering is identical to the real decoders; only the *selection* differs
    downstream. baseline = the real attention-polarity choice.
    """
    n = z.shape[0]
    if decoder == 'kmeans':
        raw, _ = spherical_kmeans2(z)
        clusters = [(raw == 0), (raw == 1)]
        baseline = polarity_attn(raw, attn)                       # attention polarity
        return clusters, np.asarray(baseline, dtype=bool).reshape(-1)
    if decoder == 'hdbscan':
        acc, hinfo = hdbscan_decode(z, attention=attn, grid_hw=grid_hw,
                                    polarity='attention')
        labels = np.asarray(hinfo.get('labels', np.full(n, -1)), dtype=np.int64)
        uniq = [int(c) for c in np.unique(labels) if c != -1]
        clusters = [(labels == c) for c in uniq]
        return clusters, np.asarray(acc, dtype=bool).reshape(-1)
    raise ValueError(f'eval_oracle: unknown decoder {decoder!r}')


def _oracle_pick(clusters: List[np.ndarray], gt_pix: np.ndarray,
                 grid_hw: Tuple[int, int], img_size: int) -> Tuple[np.ndarray, float]:
    """⚠️ CHEAT: pick the single cluster with the highest F1 vs the GT mask.

    Returns (best cluster bool mask (N,), its f1). gt_pix must be non-None
    (oracle is only meaningful for spliced items).
    """
    n_side = grid_hw[0]
    best_mask = np.zeros(clusters[0].shape, dtype=bool) if clusters else None
    best_f1 = -1.0
    for c in clusters:
        f1 = _score_patch_mask(c.reshape(n_side, n_side), gt_pix, img_size)['f1']
        if f1 > best_f1:
            best_f1 = f1
            best_mask = c
    return best_mask, best_f1


# ── Visualization ───────────────────────────────────────────────────────────────

def _attn_heatmap(attn_flat: np.ndarray, n_side: int,
                  orig_arr: np.ndarray, display: int) -> 'Image.Image':
    """Hot-colormap attention overlay on orig_arr (H=display, W=display, float32 [0,255])."""
    a = np.asarray(attn_flat, dtype=np.float32).reshape(n_side, n_side)
    lo, hi = a.min(), a.max()
    a = (a - lo) / (hi - lo + 1e-8)
    pil = Image.fromarray((a * 255).astype(np.uint8), mode='L')
    a_up = np.array(pil.resize((display, display), Image.NEAREST), dtype=np.float32) / 255.0
    # black→red→yellow→white hot ramp
    r = np.clip(a_up * 3,       0, 1)
    g = np.clip(a_up * 3 - 1,   0, 1)
    b = np.clip(a_up * 3 - 2,   0, 1)
    heat = np.stack([r, g, b], axis=-1)
    blended = orig_arr / 255.0 * 0.35 + heat * 0.65
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


def _make_viz_strip(
    orig_path,
    gt_pix,
    baseline_2d,
    oracle_2d,
    *,
    attn_full: Optional[np.ndarray] = None,
    n_side_full: int = 0,
    zoom_pil: Optional['Image.Image'] = None,
    attn_zoom: Optional[np.ndarray] = None,
    n_side_zoom: int = 0,
    display: int = 256,
) -> 'Image.Image':
    """7-panel strip: Original | Full-attn | Zoom-crop | Zoom-attn | GT | Baseline | Oracle.

    Panels 3-4 (zoom) fall back to the original / full-attn when no zoom bbox was found.
    All panels are display×display. Predictions are patch-grid masks nearest-upsampled.
    """
    from PIL import ImageDraw, ImageFont

    orig = Image.open(orig_path).convert('RGB').resize((display, display), Image.LANCZOS)
    orig_arr = np.array(orig, dtype=np.float32)

    def _mask_to_display(m: np.ndarray) -> np.ndarray:
        pil = Image.fromarray(m.astype(np.uint8) * 255, mode='L')
        return np.array(pil.resize((display, display), Image.NEAREST)) > 127

    def _tint(base_arr: np.ndarray, mask_d: np.ndarray, rgb: tuple) -> 'Image.Image':
        arr = base_arr.copy()
        r, g, b = rgb
        arr[mask_d, 0] = arr[mask_d, 0] * 0.35 + r * 0.65
        arr[mask_d, 1] = arr[mask_d, 1] * 0.35 + g * 0.65
        arr[mask_d, 2] = arr[mask_d, 2] * 0.35 + b * 0.65
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    # ── panels ────────────────────────────────────────────────────────────────────
    # 1. Original
    p_orig = orig

    # 2. Full attention heatmap
    if attn_full is not None and n_side_full:
        p_attn_full = _attn_heatmap(attn_full, n_side_full, orig_arr, display)
    else:
        p_attn_full = orig

    # 3. Zoom crop (fall back to original if no zoom)
    if zoom_pil is not None:
        p_zoom = zoom_pil.resize((display, display), Image.LANCZOS)
        zoom_arr = np.array(p_zoom, dtype=np.float32)
    else:
        p_zoom = orig
        zoom_arr = orig_arr

    # 4. Zoom attention heatmap
    if attn_zoom is not None and n_side_zoom:
        p_attn_zoom = _attn_heatmap(attn_zoom, n_side_zoom, zoom_arr, display)
    else:
        p_attn_zoom = p_zoom

    # 5. GT mask (green tint on original)
    gt_d = (_mask_to_display(gt_pix) if gt_pix is not None
            else np.zeros((display, display), bool))
    p_gt = _tint(orig_arr, gt_d, (20, 210, 80))

    # 6. Baseline prediction (red tint on original)
    base_d = (_mask_to_display(baseline_2d) if baseline_2d is not None
              else np.zeros((display, display), bool))
    p_base = _tint(orig_arr, base_d, (220, 50, 50))

    # 7. Oracle prediction (blue tint on original)
    ora_d = (_mask_to_display(oracle_2d) if oracle_2d is not None
             else np.zeros((display, display), bool))
    p_ora = _tint(orig_arr, ora_d, (50, 120, 220))

    panels = [p_orig, p_attn_full, p_zoom, p_attn_zoom, p_gt, p_base, p_ora]
    labels = ['Original', 'Attn (full)', 'Zoom crop', 'Attn (zoom)',
              'GT', 'Baseline', 'Oracle']

    gap = 3
    header_h = 20
    W = display * len(panels) + gap * (len(panels) - 1)
    strip = Image.new('RGB', (W, display + header_h), (24, 24, 24))
    draw = ImageDraw.Draw(strip)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 12)
    except Exception:
        font = ImageFont.load_default()
    for i, (panel, lbl) in enumerate(zip(panels, labels)):
        x0 = i * (display + gap)
        strip.paste(panel, (x0, header_h))
        draw.text((x0 + display // 2, 3), lbl, fill=(210, 210, 210), font=font, anchor='mt')

    return strip


# ── Aggregation ─────────────────────────────────────────────────────────────────

def _stats(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return dict(n=0, mean=float('nan'), median=float('nan'), std=float('nan'))
    a = np.asarray(vals, dtype=np.float64)
    return dict(n=int(a.size), mean=float(a.mean()), median=float(np.median(a)),
                std=float(a.std()))


def _summarize_split(records: List[dict], key: str) -> Dict[str, Dict[str, float]]:
    """records: per-item dicts. key in {'baseline','oracle'}. Splice metrics only."""
    splices = [r for r in records if not r['is_real'] and r.get(key) is not None]
    return {m: _stats([r[key][m] for r in splices]) for m in ('f1', 'iou', 'precision', 'recall')}


# ── Main ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='eval_oracle',
        description='⚠️ ISOLATED CHEATING ORACLE eval (best-cluster-by-GT). Not a reportable number.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint.')
    p.add_argument('--decoders', nargs='+', default=['kmeans'],
                   choices=['kmeans', 'hdbscan'],
                   help='Clustering kernel(s); oracle picks best cluster per kernel.')
    p.add_argument('--out_json', '--summary_out', dest='out_json', default=None,
                   help='Where to write the per-generator JSON summary.')
    p.add_argument('--max_items', type=int, default=None,
                   help='Optional per-source cap (default: ALL available images).')
    p.add_argument('--skip_reals', action='store_true',
                   help='Evaluate only spliced (fake) items. The oracle is a '
                        'localization upper bound (meaningless for reals), so this '
                        'is the natural mode and also avoids forwards on reals.')
    p.add_argument('--save_viz', default=None, metavar='DIR',
                   help='Save 4-panel visualization strips (original|GT|baseline|oracle) '
                        'here. Organised as DIR/{generator}/{case_id}.jpg. Point to a '
                        'LOCAL path (e.g. /content/viz) and copy to Drive afterwards — '
                        'writing 5k small files direct to Drive FUSE is very slow.')
    p.add_argument('--viz_per_gen', type=int, default=1000,
                   help='Max visualizations saved per generator (default 1000).')
    p.add_argument('--viz_display', type=int, default=336,
                   help='Pixel size of each panel in the strip (default 336).')
    p.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'])

    g = p.add_argument_group('dataset roots (use --opensdi_root for this task)')
    add_source_root_args(g)
    return p


def main() -> None:
    args = _build_parser().parse_args()

    log_line('[oracle] ' + '=' * 69)
    log_line('[oracle] ⚠️  CHEATING ORACLE EVAL — GT is used to pick the splice cluster.')
    log_line('[oracle] ⚠️  These ORACLE_* numbers are an UPPER BOUND, not a real result.')
    log_line('[oracle] ⚠️  Fully isolated from the contamination-free pipeline.')
    log_line('[oracle] ' + '=' * 69)

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not args.no_amp) and (device.type == 'cuda')

    log_line(f'[oracle] loading checkpoint: {args.checkpoint}')
    model, _cfg, res = load_eval_model(args.checkpoint, device=device, strict=False)
    bare = unwrap_model(model)
    img_size = int(res.image_size)

    items_by_source = collect_val_items_by_source(args, res, log_tag='[oracle]')
    all_items = [it for items in items_by_source.values() for it in items]
    if not all_items:
        raise RuntimeError('eval_oracle: no items found — set --opensdi_root (or another root).')
    if args.skip_reals:
        before = len(all_items)
        all_items = [it for it in all_items if not it.is_real]
        log_line(f'[oracle] skip_reals: kept {len(all_items)}/{before} spliced items')
    log_line(f'[oracle] evaluating {len(all_items)} items across decoders={args.decoders}')

    # per decoder → per generator → list of per-item record dicts
    records: Dict[str, Dict[str, List[dict]]] = {
        d: defaultdict(list) for d in args.decoders
    }

    viz_counts: Dict[str, int] = defaultdict(int)   # generator → strips saved so far
    if args.save_viz:
        log_line(f'[oracle] viz will be saved to {args.save_viz} '
                 f'(≤{args.viz_per_gen}/gen, {args.viz_display}px panels, JPEG)')

    full_px = (img_size, img_size)

    n = len(all_items)
    every = max(1, n // 20)
    for i, item in enumerate(all_items):
        gen = item.meta.get('generator', item.source)
        try:
            img_t, img_pil = load_image_tensor(item.image, res, device=device, return_pil=True)
            info = model_info(bare, img_t, device=device, amp=use_amp, amp_dtype=args.amp_dtype)
        except Exception as exc:
            log_line(f'[oracle] WARN fetch failed {item.item_id}: {exc}')
            continue
        if info.embeddings is None:
            raise ValueError('eval_oracle: model has no contrastive embeddings — '
                             'kmeans/hdbscan clustering is unavailable.')

        # ── Zoom pass (shared across decoders; mirrors the real eval pipeline) ──
        zoom_pil  = None
        info_zoom = None
        if info.attention is not None:
            bbox = attention_to_bbox(info.attention, info.grid_hw,
                                     percentile='otsu', thresh_mult=0.08,
                                     pad_frac=0.10, min_box_size=8, min_pad_frac=0.06)
            if not bbox_is_trivial(bbox, min_crop_frac=0.25):
                try:
                    zoom_pil  = crop_to_bbox(img_pil, bbox)
                    zoom_t    = load_image_tensor(zoom_pil, res, device=device)
                    info_zoom = model_info(bare, zoom_t, device=device,
                                          amp=use_amp, amp_dtype=args.amp_dtype)
                except Exception as exc:
                    log_line(f'[oracle] WARN zoom crop failed {item.item_id}: {exc}')
                    zoom_pil  = None
                    info_zoom = None

        # Use zoom embeddings when available (matches numbers.py zoom=True behaviour);
        # fall back to full-image embeddings when bbox was trivial or crop failed.
        if info_zoom is not None and info_zoom.embeddings is not None:
            z      = np.ascontiguousarray(info_zoom.embeddings, dtype=np.float32)
            info_c = info_zoom   # grid_hw for clustering = zoom grid
            use_zoom = True
        else:
            z      = np.ascontiguousarray(info.embeddings, dtype=np.float32)
            info_c = info
            use_zoom = False

        gt_pix = _load_gt_pixels(item.mask)     # the GT touch (cheat lives below)
        n_side = info_c.grid_hw[0]

        baseline_2d = None
        oracle_2d   = None
        for d in args.decoders:
            try:
                clusters, baseline_flat = _clusters_and_baseline(
                    d, z, info_c.attention, info_c.grid_hw)
            except Exception as exc:
                log_line(f'[oracle] WARN {d} cluster failed {item.item_id}: {exc}')
                continue

            # Place mask back into the full frame if we used zoom embeddings.
            if use_zoom:
                crop2d = baseline_flat.reshape(n_side, n_side)
                baseline_2d = place_mask_in_frame_pixels(crop2d, bbox, full_px)
            else:
                baseline_2d = baseline_flat.reshape(n_side, n_side)

            base_scores = _score_patch_mask(baseline_2d, gt_pix, img_size)
            rec = {'is_real': item.is_real, 'baseline': base_scores, 'oracle': None}

            # ⚠️ oracle: pick best cluster, place in full frame, score vs GT
            if gt_pix is not None and clusters:
                if use_zoom:
                    # Score each cluster after placing it in the full frame
                    best_mask_full = None
                    best_f1 = -1.0
                    for c in clusters:
                        c2d   = c.reshape(n_side, n_side)
                        c_full = place_mask_in_frame_pixels(c2d, bbox, full_px)
                        f1 = _score_patch_mask(c_full, gt_pix, img_size)['f1']
                        if f1 > best_f1:
                            best_f1 = f1
                            best_mask_full = c_full
                    oracle_2d = best_mask_full
                else:
                    oracle_mask, _ = _oracle_pick(clusters, gt_pix, info_c.grid_hw, img_size)
                    oracle_2d = oracle_mask.reshape(n_side, n_side)
                if oracle_2d is not None:
                    rec['oracle'] = _score_patch_mask(oracle_2d, gt_pix, img_size)
            records[d][gen].append(rec)

        # ── Save viz strip (first decoder only, up to viz_per_gen per generator) ──
        if args.save_viz and baseline_2d is not None and viz_counts[gen] < args.viz_per_gen:
            try:
                case_id = item.meta.get('case_id', item.item_id).replace('/', '_')
                out_dir = os.path.join(args.save_viz, gen)
                os.makedirs(out_dir, exist_ok=True)
                strip = _make_viz_strip(
                    item.image, gt_pix, baseline_2d, oracle_2d,
                    attn_full=info.attention,
                    n_side_full=info.grid_hw[0],
                    zoom_pil=zoom_pil,
                    attn_zoom=(info_zoom.attention if info_zoom is not None else None),
                    n_side_zoom=(info_zoom.grid_hw[0] if info_zoom is not None else 0),
                    display=args.viz_display,
                )
                strip.save(os.path.join(out_dir, f'{case_id}.jpg'), quality=88)
                viz_counts[gen] += 1
            except Exception as exc:
                log_line(f'[oracle] WARN viz save failed {item.item_id}: {exc}')

        if (i + 1) % every == 0 or (i + 1) == n:
            log_line(f'[oracle] {i + 1}/{n}')

    if args.save_viz and viz_counts:
        total_viz = sum(viz_counts.values())
        log_line(f'[oracle] saved {total_viz} viz strips to {args.save_viz} '
                 f'({dict(viz_counts)})')
        log_line(f'[oracle] to copy to Drive: '
                 f'cp -r {args.save_viz} /content/drive/MyDrive/DINO_SCOPE_RUNS/optimal/')

    # ── Build + print summary, partitioned by generator ─────────────────────────
    summary: Dict[str, dict] = {'_meta': {
        'checkpoint': args.checkpoint,
        'oracle_definition': 'best-cluster-by-GT (F1) polarity; CHEAT / upper bound',
        'baseline_definition': 'attention polarity, zoom pass (otsu bbox → crop → 2nd forward; falls back to flat when bbox trivial)',
        'decoders': args.decoders,
        'image_size': img_size,
    }}

    for d in args.decoders:
        log_line('[oracle]')
        log_line(f'[oracle] ┌─ decoder={d} ' + '─' * 74)
        log_line(f'[oracle] │ {"generator":<14} {"n_spl":>6} '
                 f'{"base_F1":>9} {"base_IOU":>9} '
                 f'{"ORA_F1":>9} {"ORA_IOU":>9} '
                 f'{"ΔF1":>7} {"ΔIOU":>7}')
        dec_out: Dict[str, dict] = {}
        gens = sorted(records[d].keys())
        for gen in gens + ['_overall']:
            recs = ([r for g in gens for r in records[d][g]] if gen == '_overall'
                    else records[d][gen])
            if not recs:
                continue
            n_spl = sum(1 for r in recs if not r['is_real'])
            n_real = sum(1 for r in recs if r['is_real'])
            base = _summarize_split(recs, 'baseline')
            orac = _summarize_split(recs, 'oracle')
            reals_acc = _stats([r['baseline']['accuracy'] for r in recs if r['is_real']])
            dec_out[gen] = {
                'n_splice': n_spl, 'n_real': n_real,
                'baseline': base, 'oracle': orac,
                'reals_accuracy': reals_acc,
            }
            bf1  = base['f1']['mean'];  biou  = base['iou']['mean']
            of1  = orac['f1']['mean'];  oiou  = orac['iou']['mean']
            df1  = (of1 - bf1)  if (np.isfinite(bf1)  and np.isfinite(of1))  else float('nan')
            diou = (oiou - biou) if (np.isfinite(biou) and np.isfinite(oiou)) else float('nan')
            label = '─OVERALL' if gen == '_overall' else gen
            log_line(f'[oracle] │ {label:<14} {n_spl:>6} '
                     f'{bf1:>9.4f} {biou:>9.4f} '
                     f'{of1:>9.4f} {oiou:>9.4f} '
                     f'{df1:>+7.4f} {diou:>+7.4f}')
        log_line('[oracle] └' + '─' * 87)
        summary[d] = dec_out

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or '.', exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump(summary, f, indent=2)
        log_line(f'[oracle] wrote summary → {args.out_json}')


if __name__ == '__main__':
    main()
