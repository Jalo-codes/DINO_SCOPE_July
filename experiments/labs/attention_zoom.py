"""experiments.labs.attention_zoom — attention-guided two-pass eval, GT-free.

Two passes, no GT ever read:
  1. Full-image pass → ModelInfo (attention map + coarse decode for fallback)
  2. Attention bbox (top-percentile attention patches) → crop → second pass →
     decode the crop → place the crop mask back into the full-frame patch grid

The crop window comes from the *attention map*; the *decoder* is pluggable, so
"attention-zoom + kmeans", "+ threshold", "+ hdbscan" all run through the same
seam.  Geometry lives in lab_utils.eval.zoom; loading in lab_utils.eval.preprocess.

Zoom→metric seam (GAMEPLAN Phase 6): metric() receives the placed-back full-frame
patch mask plus the *pass-1* ModelInfo (full-frame grid_hw / res).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import torch

from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution
from lab_utils.eval.decode.kmeans import decode_kmeans
from lab_utils.eval.decode.threshold import decode_threshold
from lab_utils.eval.fetch import ModelInfo, model_info, repool_hidden
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.record import EvalRecord
from lab_utils.eval.hide import HIDE_THRESH_MULT, build_hide_mask
from lab_utils.eval.multibox import cover_bboxes, gate_boxes_by_logit
from lab_utils.eval.zoom import (
    BBox,
    attention_hot_mask,
    attention_to_bbox,
    bbox_is_trivial,
    crop_to_bbox,
    place_mask_in_frame_pixels,
)
from lab_utils.logging.text import log_line
from lab_utils.train.distributed import unwrap_model

DecodeFn = Callable[[ModelInfo], np.ndarray]

_NAMED_DECODERS: dict = {
    'kmeans':    decode_kmeans,
    'threshold': decode_threshold,
}


def _resolve_decoder(decoder) -> tuple:
    """Accept a callable or a known name → (decode_fn, name)."""
    if callable(decoder):
        return decoder, getattr(decoder, '__name__', 'custom').replace('decode_', '')
    fn = _NAMED_DECODERS.get(decoder)
    if fn is None:
        # Allow hdbscan / graph without importing them at module load.
        if decoder == 'hdbscan':
            from lab_utils.eval.decode.hdbscan import decode_hdbscan
            fn = decode_hdbscan
        elif decoder == 'graph':
            from lab_utils.eval.decode.graph import decode_graph
            fn = decode_graph
        else:
            raise ValueError(f'attention_zoom: unknown decoder {decoder!r}')
    return fn, decoder


# ── single item ────────────────────────────────────────────────────────────────

@torch.no_grad()
def attention_zoom_single(
    model: torch.nn.Module,
    item: Item,
    res: Resolution,
    *,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
    decoder='kmeans',
    attn_percentile: float | str = 'otsu',
    attn_thresh_mult: float = 1.0,
    attn_pad_frac: float = 0.10,
    min_crop_frac: float = 0.25,
    min_box_size: int = 8,
    attn_min_pad_frac: float = 0.06,
    pad_side_frac: float | None = None,
    min_area_frac: float = 0.0,
    return_debug: bool = False,
    override_image_pil = None,
):
    """Attention-zoom inference for one item with a pluggable decoder.

    Falls back to the single-pass full-frame decode when attention is missing
    or the attention bbox is ~the whole frame (nothing to zoom into).

    By default the bbox uses an Otsu hot/cold split of the attention map
    (`attn_percentile='otsu'`, `attn_thresh_mult=1.0`) — the threshold every
    prior config (DINO_SCOPE eval_zoom_tgif, legacy val-zoom early-stop) used,
    giving a tight, genuinely-magnified crop.  Alternatives: `'gap'` (largest-gap
    split) or `'peak'` (keep patches ≥ `attn_thresh_mult` × the peak — recall-first
    / very broad; with a small mult the box can balloon past `min_crop_frac` and
    trip the whole-frame fallback, no-opping the zoom).

    `attn_min_pad_frac` floors the per-side crop padding so the breathing room
    does not collapse to ~0 on medium/large boxes (the inverse-area scaling
    otherwise drives it there).  0.0 = legacy padding.

    `pad_side_frac` switches the crop box to RESOLUTION-INVARIANT area-based
    padding: each side is grown by that fraction of the frame (so 0.05 means 5%
    at any grid size), and `min_area_frac` floors the padded box to that fraction
    of the frame area.  When set, the patch-unit pad/min_box_size math is skipped
    entirely.  Leave it None to keep the legacy patch-based padding.

    Returns an EvalRecord, or (EvalRecord, debug_dict) when return_debug=True.
    The debug dict carries the pass-1 mask, pass-2 placed mask, and the bbox —
    everything the visualiser needs to draw boxes.
    """
    decode_fn, decoder_name = _resolve_decoder(decoder)
    zoom_label = f'{decoder_name}_zoom'

    # Pass 1 — full image
    if override_image_pil is not None:
        img_pil = override_image_pil.convert('RGB')
        img_tensor = load_image_tensor(img_pil, res, device=device)
    else:
        img_tensor, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    info1 = model_info(model, img_tensor, device=device, amp=use_amp, amp_dtype=amp_dtype)
    mask1 = decode_fn(info1)

    debug = {'bbox': None, 'mask_full': mask1, 'mask_zoom': None,
             'grid_hw': info1.grid_hw, 'zoomed': False,
             'crop_pil': None, 'attn_crop': None, 'crop_grid_hw': None,
             'attn1': info1.attention, 'img_pil': img_pil}

    if info1.attention is None:
        rec = eval_metric(mask1, info1, item, decoder=zoom_label)
        return (rec, debug) if return_debug else rec

    bbox = attention_to_bbox(
        info1.attention, info1.grid_hw,
        percentile=attn_percentile, thresh_mult=attn_thresh_mult,
        pad_frac=attn_pad_frac,
        min_box_size=min_box_size, min_pad_frac=attn_min_pad_frac,
        pad_side_frac=pad_side_frac, min_area_frac=min_area_frac,
    )
    debug['bbox'] = bbox

    if bbox_is_trivial(bbox, min_crop_frac=min_crop_frac):
        rec = eval_metric(mask1, info1, item, decoder=zoom_label)
        return (rec, debug) if return_debug else rec

    # Pass 2 — cropped region: decode in the crop, then place the crop mask back
    # at PIXEL resolution so the finer crop-grid detail survives (the coarse
    # full-frame patch grid would throw most of it away).
    crop_pil    = crop_to_bbox(img_pil, bbox)
    crop_tensor = load_image_tensor(crop_pil, res, device=device)
    info2       = model_info(model, crop_tensor, device=device, amp=use_amp, amp_dtype=amp_dtype)
    mask2_crop  = decode_fn(info2)

    crop2d = np.asarray(mask2_crop, dtype=bool)
    if crop2d.ndim == 1:
        crop2d = crop2d.reshape(info2.grid_hw)
    full_px   = (int(res.image_size), int(res.image_size))
    full_mask = place_mask_in_frame_pixels(crop2d, bbox, full_px)

    debug.update({
        'mask_zoom': full_mask, 'mask_crop': crop2d, 'zoomed': True,
        'crop_pil': crop_pil, 'attn_crop': info2.attention,
        'crop_grid_hw': info2.grid_hw,
    })
    rec = eval_metric(full_mask, info1, item, decoder=zoom_label)
    return (rec, debug) if return_debug else rec


# ── batch ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def attention_zoom_eval(
    model: torch.nn.Module,
    items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    use_amp: bool = False,
    decoder='kmeans',
    log_tag: str = '[zoom]',
    summarize_results: bool = True,
    **kwargs,
) -> List[EvalRecord]:
    """Run attention_zoom_single over items; return EvalRecord list."""
    from lab_utils.eval.aggregate import summarize

    bare = unwrap_model(model)
    bare.eval()

    records: List[EvalRecord] = []
    for item in items:
        try:
            rec = attention_zoom_single(
                bare, item, res,
                device=device, use_amp=use_amp, decoder=decoder, **kwargs,
            )
            records.append(rec)
        except Exception as exc:
            log_line(f'{log_tag} WARN: skipped item={item.item_id}: {exc}')

    if summarize_results and records:
        summarize(records, log_tag=log_tag)
    elif not records:
        log_line(f'{log_tag} no records (n_items={len(items)})')
    return records


# ── multi-bbox runner ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_bbox_zoom(
    model: torch.nn.Module,
    img_pil,
    bboxes: List[BBox],
    res: Resolution,
    *,
    device: torch.device,
    decode_fn: DecodeFn,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
    min_crop_frac: float = 0.25,
):
    """Crop → re-decode → place-back over a list of bboxes; union the masks.

    Generalises the single-box pass-2 of `attention_zoom_single` to K boxes.
    Trivial / whole-frame boxes are skipped.  Each surviving box is cropped,
    re-run through the model, decoded, and placed back at PIXEL resolution; the
    placed masks are OR'd into one full-frame mask.

    Returns (union_mask_px, per_box) where union_mask_px is a bool (S, S) pixel
    mask (or None if no box survived) and per_box is a list of debug dicts.
    """
    full_px = (int(res.image_size), int(res.image_size))
    placed: List[np.ndarray] = []
    per_box: List[dict] = []

    for bbox in bboxes:
        if bbox is None or bbox_is_trivial(bbox, min_crop_frac=min_crop_frac):
            continue
        crop_pil    = crop_to_bbox(img_pil, bbox)
        crop_tensor = load_image_tensor(crop_pil, res, device=device)
        info_c      = model_info(model, crop_tensor, device=device,
                                 amp=use_amp, amp_dtype=amp_dtype)
        mask_c = decode_fn(info_c)
        crop2d = np.asarray(mask_c, dtype=bool)
        if crop2d.ndim == 1:
            crop2d = crop2d.reshape(info_c.grid_hw)
        mask_px = place_mask_in_frame_pixels(crop2d, bbox, full_px)
        placed.append(mask_px)
        per_box.append({
            'bbox': bbox, 'mask_crop': crop2d, 'mask_px': mask_px,
            'crop_pil': crop_pil, 'crop_grid_hw': info_c.grid_hw,
            'attn_crop': info_c.attention, 'image_logit': info_c.image_logit,
        })

    if not placed:
        return None, []
    return np.logical_or.reduce(placed), per_box


def _gate_box_union(per_box: List[dict], full_logit, *, margin: float = 0.0):
    """Relative-to-full-image logit gate over run_bbox_zoom's per_box list.

    Keeps the boxes whose crop image_logit clears `full_logit - margin` and ORs
    their pixel masks.  Returns (union_px, keep_indices), or (None, []) when no
    box clears the bar — the caller then defers to the original unzoomed decode.
    When `full_logit` is None (image head disabled) gating is skipped and every
    box is kept (the union is unchanged).
    """
    if full_logit is None:
        keep = list(range(len(per_box)))
    else:
        keep = gate_boxes_by_logit([pb.get('image_logit') for pb in per_box],
                                   full_logit, margin=margin)
    if not keep:
        return None, []
    union = np.logical_or.reduce([per_box[i]['mask_px'] for i in keep])
    return union, keep


# ── multi-window zoom (efficient box cover over a patch mask, single pass) ───────

@torch.no_grad()
def multi_zoom_single(
    model: torch.nn.Module,
    item: Item,
    res: Resolution,
    *,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
    decoder='kmeans',
    box_source: str = 'attention',
    attn_percentile: float | str = 'otsu',
    box_area_weight: float = 0.04,
    min_patches: int = 2,
    max_regions: int = 4,
    box_pad_frac: float = 0.08,
    square_cap: float = 1.4,
    min_crop_frac: float = 0.25,
    gate_logit: bool = True,
    gate_margin: float = 0.0,
    return_debug: bool = False,
):
    """Single-pass multi-window zoom via an efficient box cover.

    One forward → a binary ON/OFF patch mask → `multibox.cover_bboxes` finds the
    cheapest set of windows covering the ON patches (small wasted area vs few
    boxes) → zoom each → MIL-gate → union.  No second pool pass, no decode-fragment
    boxes.

    box_source:
      'attention' (default) — ON = the thresholded MIL attention hot set.
      'decode'              — ON = the full-frame decode mask (seed from where the
                              decoder already fired, e.g. for the hdbscan lab).

    Falls back to the unzoomed pass-1 decode when nothing survives the cover, or
    when the gate rejects every window.
    """
    decode_fn, decoder_name = _resolve_decoder(decoder)
    label = f'{decoder_name}_multi'

    img_tensor, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    info1 = model_info(model, img_tensor, device=device, amp=use_amp, amp_dtype=amp_dtype)
    mask1 = decode_fn(info1)

    debug = {'bboxes': [], 'mask_full': mask1, 'mask_zoom': None,
             'grid_hw': info1.grid_hw, 'zoomed': False, 'per_box': [],
             'attn1': info1.attention, 'img_pil': img_pil,
             'full_logit': info1.image_logit, 'gated_boxes': None}

    # Binary ON/OFF patch mask that the cover operates on.
    if box_source == 'attention':
        if info1.attention is None:
            rec = eval_metric(mask1, info1, item, decoder=label)
            return (rec, debug) if return_debug else rec
        hot = attention_hot_mask(info1.attention, info1.grid_hw, percentile=attn_percentile)
    elif box_source == 'decode':
        hot = np.asarray(mask1, dtype=bool)
        if hot.ndim == 1:
            hot = hot.reshape(info1.grid_hw)
    else:
        raise ValueError(f"multi_zoom_single: unknown box_source {box_source!r} "
                         "(attention|decode)")

    bboxes = cover_bboxes(
        hot, box_area_weight=box_area_weight, min_patches=min_patches,
        max_regions=max_regions, pad_frac=box_pad_frac, square_cap=square_cap,
    )
    bboxes = [b for b in bboxes if not bbox_is_trivial(b, min_crop_frac=min_crop_frac)]
    debug['bboxes'] = bboxes

    if not bboxes:
        rec = eval_metric(mask1, info1, item, decoder=label)
        return (rec, debug) if return_debug else rec

    union, per_box = run_bbox_zoom(
        model, img_pil, bboxes, res, device=device, decode_fn=decode_fn,
        use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
    )
    debug['per_box'] = per_box      # store before gating so the viz can show why
    if union is None:
        rec = eval_metric(mask1, info1, item, decoder=label)
        return (rec, debug) if return_debug else rec

    # Gate the windows against the full-image MIL logit; if none clear the bar,
    # defer to the unzoomed pass-1 decode (see _gate_box_union).
    if gate_logit:
        gated, keep = _gate_box_union(per_box, info1.image_logit, margin=gate_margin)
        debug['gated_boxes'] = keep
        if gated is None:
            rec = eval_metric(mask1, info1, item, decoder=label)
            return (rec, debug) if return_debug else rec
        union = gated

    debug.update({'mask_zoom': union, 'zoomed': True})
    rec = eval_metric(union, info1, item, decoder=label)
    return (rec, debug) if return_debug else rec


@torch.no_grad()
def multi_zoom_eval(
    model: torch.nn.Module,
    items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    use_amp: bool = False,
    decoder='kmeans',
    log_tag: str = '[zoom]',
    summarize_results: bool = True,
    **kwargs,
) -> List[EvalRecord]:
    """Run multi_zoom_single over items; return EvalRecord list."""
    from lab_utils.eval.aggregate import summarize

    bare = unwrap_model(model)
    bare.eval()

    records: List[EvalRecord] = []
    for item in items:
        try:
            rec = multi_zoom_single(
                bare, item, res,
                device=device, use_amp=use_amp, decoder=decoder, **kwargs,
            )
            records.append(rec)
        except Exception as exc:
            log_line(f'{log_tag} WARN: skipped item={item.item_id}: {exc}')

    if summarize_results and records:
        summarize(records, log_tag=log_tag)
    elif not records:
        log_line(f'{log_tag} no records (n_items={len(items)})')
    return records


# ── second-best finder (MIL hide → re-pool → 2nd bbox) ──────────────────────────

@torch.no_grad()
def second_best_zoom_single(
    model: torch.nn.Module,
    item: Item,
    res: Resolution,
    *,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
    decoder='kmeans',
    attn_percentile: float | str = 'otsu',
    attn_pad_frac: float = 0.10,
    min_crop_frac: float = 0.25,
    min_box_size: int = 8,
    hide_thresh_mult: float = HIDE_THRESH_MULT,
    hide_dilate: int = 1,
    gate_logit: bool = True,
    gate_margin: float = 0.0,
    return_debug: bool = False,
):
    """Attention-zoom guided by MIL hiding on the FULL image (before zooming).

    [PAUSED while multi-window is rebuilt — kept working, not under active dev.]

    Pass 1 (with return_feats) → top bbox.  Hide region 1's connected hot
    component from the MIL pool, re-pool the cached features (no second backbone
    forward), and read a second bbox off the renormalised attention — a different
    spot in the full image.  Zoom the top box AND the second box, then union.

    Caveat (by design): MIL re-pool re-ranks fixed gating scores, so the second
    box is the next-strongest *fixed* region with region 1 removed — useful and
    cheap, not a feature-level independence test (that needs the deferred
    backbone hide).

    Box gating (gate_logit=True): each zoom crop is scored by the MIL head and a
    box is kept only if its crop image_logit clears the full-image logit minus
    gate_margin (a real splice crop concentrates the manipulation, so it should
    out-score the diluted full frame).  If NO box clears the bar the zoom is just
    noise → defer to the unzoomed pass-1 decode.

    Falls back to the pass-1 full-frame decode when attention is missing or no
    non-trivial box survives.
    """
    decode_fn, decoder_name = _resolve_decoder(decoder)
    label = f'{decoder_name}_second_best'

    img_tensor, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    info1 = model_info(model, img_tensor, device=device, amp=use_amp,
                       amp_dtype=amp_dtype, return_feats=True)
    mask1 = decode_fn(info1)

    debug = {'bbox1': None, 'bbox2': None, 'mask_full': mask1, 'mask_zoom': None,
             'grid_hw': info1.grid_hw, 'zoomed': False, 'per_box': [],
             'attn1': info1.attention, 'attn2': None, 'img_pil': img_pil,
             'full_logit': info1.image_logit, 'gated_boxes': None}

    if info1.attention is None:
        rec = eval_metric(mask1, info1, item, decoder=label)
        return (rec, debug) if return_debug else rec

    bbox1 = attention_to_bbox(
        info1.attention, info1.grid_hw,
        percentile=attn_percentile, pad_frac=attn_pad_frac, min_box_size=min_box_size,
    )
    debug['bbox1'] = bbox1

    # Hide region 1's connected hot component on the full image → re-pool →
    # second attention.  The hide set (positive patches + an 8-neighbour margin,
    # built a touch broader than the bbox) is owned by lab_utils.eval.hide.
    bbox2 = None
    if info1.patch_feats is not None:
        hide = build_hide_mask(
            info1.attention, info1.grid_hw, mode='component',
            percentile=attn_percentile, thresh_mult=hide_thresh_mult,
            dilate=hide_dilate,
        )
        if hide.any() and hide.sum() < hide.size:   # don't hide everything
            info2 = repool_hidden(model, info1, hide)
            debug['attn2'] = info2.attention
            cand = attention_to_bbox(
                info2.attention, info2.grid_hw,
                percentile=attn_percentile, pad_frac=attn_pad_frac, min_box_size=min_box_size,
            )
            if not bbox_is_trivial(cand, min_crop_frac=min_crop_frac):
                bbox2 = cand
    debug['bbox2'] = bbox2

    bboxes = [b for b in (bbox1, bbox2)
              if b is not None and not bbox_is_trivial(b, min_crop_frac=min_crop_frac)]
    if not bboxes:
        rec = eval_metric(mask1, info1, item, decoder=label)
        return (rec, debug) if return_debug else rec

    union, per_box = run_bbox_zoom(
        model, img_pil, bboxes, res, device=device, decode_fn=decode_fn,
        use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
    )
    debug['per_box'] = per_box      # store before gating so the viz can show why
    if union is None:
        rec = eval_metric(mask1, info1, item, decoder=label)
        return (rec, debug) if return_debug else rec

    # Gate the zoom boxes against the full-image MIL logit.  If every crop scores
    # below the full image, the zoom only adds noise → defer to the unzoomed
    # pass-1 decode (the trusted floor).
    if gate_logit:
        gated, keep = _gate_box_union(per_box, info1.image_logit, margin=gate_margin)
        debug['gated_boxes'] = keep
        if gated is None:
            rec = eval_metric(mask1, info1, item, decoder=label)
            return (rec, debug) if return_debug else rec
        union = gated

    debug.update({'mask_zoom': union, 'zoomed': True})
    rec = eval_metric(union, info1, item, decoder=label)
    return (rec, debug) if return_debug else rec


@torch.no_grad()
def second_best_zoom_eval(
    model: torch.nn.Module,
    items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    use_amp: bool = False,
    decoder='kmeans',
    log_tag: str = '[zoom]',
    summarize_results: bool = True,
    **kwargs,
) -> List[EvalRecord]:
    """Run second_best_zoom_single over items; return EvalRecord list."""
    from lab_utils.eval.aggregate import summarize

    bare = unwrap_model(model)
    bare.eval()

    records: List[EvalRecord] = []
    for item in items:
        try:
            rec = second_best_zoom_single(
                bare, item, res,
                device=device, use_amp=use_amp, decoder=decoder, **kwargs,
            )
            records.append(rec)
        except Exception as exc:
            log_line(f'{log_tag} WARN: skipped item={item.item_id}: {exc}')

    if summarize_results and records:
        summarize(records, log_tag=log_tag)
    elif not records:
        log_line(f'{log_tag} no records (n_items={len(items)})')
    return records


# ── public mode registry ─────────────────────────────────────────────────────────
# Name → finder, so any caller (multi_zoom_bench, experiments/scripts/eval.py, a
# notebook) can dispatch a zoom mode by string.  Active set:
#   single       — one attention bbox.
#   multi        — efficient box cover over the attention hot set (gated).
#   second_best  — MIL pool-peel (PAUSED; kept working, not under active dev).

ZOOM_SINGLE_FNS = {
    'single':      attention_zoom_single,
    'multi':       multi_zoom_single,
    'second_best': second_best_zoom_single,
}

ZOOM_EVAL_FNS = {
    'single':      attention_zoom_eval,
    'multi':       multi_zoom_eval,
    'second_best': second_best_zoom_eval,
}

ZOOM_MODES = tuple(ZOOM_SINGLE_FNS.keys())
