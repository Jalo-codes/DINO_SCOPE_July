"""experiments.labs.box_heatmap_lab — supervised single-box heatmap orchestration.

The MVP path: frozen detector → per-patch input → :class:`BoxHeatmap` → per-patch
box-membership logit, supervised by weighted BCE toward a binary box mask.

    full-frame forward (frozen)  →  [z | attn | patch_logit]
        →  BoxHeatmap  →  per-patch logit (heatmap)
        →  BCE+Dice vs the GT splice mask          (train; GT touched via metric, I3)
        →  threshold + read-off bbox  →  run_bbox_zoom (frozen)  →  F1   (eval only)

Target (per patch, over the grid):
    1  the raw GT splice patches (a patch is 1 when ≥ ``patch_frac`` of its pixels
       are GT),
    0  everything else.
The label is the clean splice mask — NO geometry is baked in.  All read-off
geometry (proximity grouping, padding, don't-zoom-large, squaring) is an EVAL
concern; baking it into the label muddies it and hurts convergence.

Reuses the RL path's pure helpers (`build_policy_input`, `policy_input_dim`,
`run_bbox_zoom`) but trains nothing in them.
"""

from __future__ import annotations

import dataclasses
import random
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lab_utils.data.item import Item
from lab_utils.data.datasets.registry import REGISTRY
from lab_utils.data.resolution import Resolution
from lab_utils.eval.fetch import ModelInfo, model_info
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.multibox import proximity_bboxes, suppress_contained_boxes
from lab_utils.eval.zoom import gt_grid_mask
from lab_utils.logging.text import log_line
from lab_utils.model.box_heatmap import BoxHeatmap

from experiments.labs.attention_zoom import (
    run_bbox_zoom, _gate_box_union, attention_zoom_single, multi_zoom_single,
)
from experiments.labs.box_policy_zoom import build_policy_input  # pure helper


# Eval sources and the arg-attr that carries each dataset root.  Shared by the
# train and eval scripts (both pass an argparse namespace with these attrs).
_EVAL_SOURCES = ('sagid', 'casia', 'imd2020')
_SOURCE_ROOT = {'sagid': 'sagid_root', 'casia': 'casia_root', 'imd2020': 'imd2020_root'}

DecodeFn = Callable[[ModelInfo], np.ndarray]


# ── training ──────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class HeatStats:
    loss:    float
    pos:     float    # number of target-1 patches
    n:       int      # total patches
    kind:    str      # box | large | no_gt


def _gt_pixels_via_metric(info: ModelInfo, item: Item, decoder_name: str) -> Optional[np.ndarray]:
    """Extract the native-resolution GT mask through the sanctioned metric path (I3)."""
    n_side = info.grid_hw[0]
    rec = eval_metric(np.zeros((n_side, n_side), dtype=bool), info, item, decoder=decoder_name)
    return rec.gt_mask


def box_heatmap_features_target(
    model: torch.nn.Module,
    item: Item,
    res: Resolution,
    *,
    device: torch.device,
    decoder_name: str,
    use_attn: bool = True,
    use_patch_logit: bool = True,
    patch_frac: float = 0.25,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    """Frozen forward → (per-patch features, BCE target, kind), or None.

    This is the ONLY part of a training step that touches the backbone, so it can
    be run ONCE per item and cached — the head then trains on the cached tensors
    with no further backbone forwards (the whole point of "cheap cached vectors").

    The target is the raw GT splice mask over the patch grid (a patch is 1 when
    ≥ ``patch_frac`` of its pixels are GT) — the clean, convergent label.  All
    read-off geometry is an eval concern, never baked into the target.
    """
    img_t = load_image_tensor(item, res, device=device)
    with torch.no_grad():
        info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    if info.embeddings is None:
        return None

    gt_pixels = _gt_pixels_via_metric(info, item, decoder_name)
    gm = gt_grid_mask(gt_pixels, info.grid_hw, patch_frac=patch_frac)
    tgt_np = gm.reshape(-1).astype(np.float32)
    kind = 'box' if gm.any() else 'no_gt'
    feats_np = build_policy_input(info, use_attn=use_attn, use_patch_logit=use_patch_logit)
    return feats_np.astype(np.float32), tgt_np.astype(np.float32), kind


def heatmap_loss(head: BoxHeatmap, feats_t: torch.Tensor, target_t: torch.Tensor,
                 *, loss_mode: str = 'bce_dice', pos_weight: float = 8.0,
                 dice_smooth: float = 1.0) -> torch.Tensor:
    """Per-image heatmap loss against the splice target.

    ``loss_mode``:
      'bce'      — weighted BCE.  Loss mass scales with splice size, so large
                   splices dominate each accumulated gradient step (small splices
                   under-trained).
      'dice'     — soft Dice.  SIZE-INVARIANT: every image contributes a [0,1]
                   score regardless of patch count, so each pulls the gradient
                   equally — fixes the size bias.  No pos_weight needed.
      'bce_dice' — BCE + Dice: BCE's stable per-pixel gradient plus Dice's
                   size-invariance (the default).
    """
    logit = head(feats_t)
    if loss_mode in ('bce', 'bce_dice'):
        bce = F.binary_cross_entropy_with_logits(
            logit, target_t, pos_weight=torch.tensor(float(pos_weight), device=feats_t.device),
        )
        if loss_mode == 'bce':
            return bce
    prob = torch.sigmoid(logit)
    inter = (prob * target_t).sum()
    denom = prob.sum() + target_t.sum()
    dice = 1.0 - (2.0 * inter + dice_smooth) / (denom + dice_smooth)
    return dice if loss_mode == 'dice' else bce + dice


def box_heatmap_train_item(
    model: torch.nn.Module,
    head: BoxHeatmap,
    item: Item,
    res: Resolution,
    *,
    device: torch.device,
    decoder_name: str,
    use_attn: bool = True,
    use_patch_logit: bool = True,
    pos_weight: float = 8.0,
    loss_mode: str = 'bce_dice',
    patch_frac: float = 0.25,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
) -> Optional[Tuple[torch.Tensor, HeatStats]]:
    """One supervised step body: per-image ``loss_mode`` loss vs the splice target.

    Returns (loss, stats), or None if the item is unusable (no embeddings).  The
    detector forward is no-grad; only the head sees gradient.

    ``loss_mode`` defaults to 'bce_dice' (Dice makes each image contribute equally
    regardless of splice size — fixes the size-driven gradient bias).  The target
    is the raw GT splice mask (geometry is a read-off concern, not a label one).
    """
    out = box_heatmap_features_target(
        model, item, res, device=device, decoder_name=decoder_name,
        use_attn=use_attn, use_patch_logit=use_patch_logit,
        patch_frac=patch_frac, use_amp=use_amp, amp_dtype=amp_dtype,
    )
    if out is None:
        return None
    feats_np, tgt_np, kind = out
    feats_t = torch.from_numpy(feats_np).float().to(device)
    target_t = torch.from_numpy(tgt_np).float().to(device)
    loss = heatmap_loss(head, feats_t, target_t, loss_mode=loss_mode, pos_weight=pos_weight)
    stats = HeatStats(loss=float(loss.detach()), pos=float(target_t.sum()),
                      n=int(target_t.numel()), kind=kind)
    return loss, stats


# ── deterministic inference (eval) ───────────────────────────────────────────────

@torch.no_grad()
def box_heatmap_single(
    model: torch.nn.Module,
    head: BoxHeatmap,
    item: Item,
    res: Resolution,
    *,
    device: torch.device,
    decode_fn: DecodeFn,
    decoder_name: str = 'kmeans',
    use_attn: bool = True,
    use_patch_logit: bool = True,
    thresh: float = 0.5,
    min_patches: int = 2,
    dilate: int = 1,
    max_regions: int = 3,
    readoff_pad_frac: float = 0.05,
    readoff_min_pad_frac: float = 0.0,
    readoff_min_box_size: int = 6,
    square_cap: float = 1.4,
    overlap_kill_frac: float = 0.30,
    large_area_frac: float = 0.6,
    gate_logit: bool = True,
    gate_margin: float = 0.0,
    min_crop_frac: float = 0.25,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
    return_debug: bool = False,
):
    """Predict the heatmap, group it into ≤ ``max_regions`` boxes, zoom, OR-union,
    score.  Flat decode when no box.

    The head predicts the raw splice patches; ALL geometry happens here:

      * grouping is `proximity_bboxes`: ON patches are joined in PATCH space
        (dilate → connected components → one box per component).  This avoids
        hull-space containment merges (a small region inside a larger one's bbox
        getting swallowed).
      * padding is light and shrinks with box size: small splices get one patch
        of margin, large boxes ~none (``readoff_pad_frac`` proportional term +
        the per-size base pad in ``_pad_bbox``).  ``readoff_min_box_size`` is the
        real floor that stops a tiny splice from being over-magnified into a
        sliver.  ``square_cap`` partially squares (1.4; 1.0 over-expands thins).
      * ``overlap_kill_frac``: drop a smaller box when > this fraction of it lies
        inside a larger box (kills redundant nested/overlapping windows).
      * ``large_area_frac``: if the RAW ON set already covers this fraction of the
        frame it's a large splice → defer to flat (decided pre-pad, not on the
        padded box).
      * ``gate_logit``: keep only zoom crops whose MIL logit out-scores the full
        frame; if none do, the zoom is noise → defer to the trusted flat decode
        (the safety net `multi_zoom_single` already has, which the read-off lacked).

    When ``return_debug`` the debug dict carries gate diagnostics so eval can
    quantify the gate's value: ``gate_status`` ∈ {no_boxes, kept, fired} and
    ``f1_ungated`` (the F1 the zoom WOULD have scored without the gate).
    """
    label = f'{decoder_name}_boxheatmap'
    head.eval()

    img_t, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    mask1 = decode_fn(info)

    debug = {'mask_full': mask1, 'mask_zoom': None, 'grid_hw': info.grid_hw,
             'zoomed': False, 'boxes': [], 'heat': None,
             'attn1': info.attention, 'img_pil': img_pil,
             'gate_status': 'no_boxes', 'f1_ungated': None}

    if info.embeddings is None:
        rec = eval_metric(mask1, info, item, decoder=label)
        return (rec, debug) if return_debug else rec

    feats = torch.from_numpy(
        build_policy_input(info, use_attn=use_attn, use_patch_logit=use_patch_logit)
    ).float().to(device)
    prob = torch.sigmoid(head(feats)).detach().cpu().numpy()
    debug['heat'] = prob
    on = prob.reshape(info.grid_hw) >= float(thresh)

    # Hot-area fallback: decide "large splice → don't zoom" on the RAW ON set,
    # before any padding.  A near-whole-frame hot set should defer to the flat
    # decode, not zoom a barely-magnified crop that just resamples the full image.
    if on.mean() >= float(large_area_frac):
        debug['boxes'] = []
        rec = eval_metric(mask1, info, item, decoder=label)
        return (rec, debug) if return_debug else rec

    boxes = proximity_bboxes(
        on, dilate=dilate, min_patches=min_patches, max_regions=max_regions,
        pad_frac=readoff_pad_frac, min_box_size=readoff_min_box_size,
        min_pad_frac=readoff_min_pad_frac, square_cap=square_cap,
    )
    # Kill a smaller box when > overlap_kill_frac of it sits inside a larger one
    # (redundant inner/overlapping windows from padding + squaring).
    boxes = suppress_contained_boxes(boxes, frac=overlap_kill_frac)
    debug['boxes'] = boxes

    if not boxes:
        rec = eval_metric(mask1, info, item, decoder=label)   # no zoom (large / cold)
        return (rec, debug) if return_debug else rec

    union, per_box = run_bbox_zoom(
        model, img_pil, boxes, res, device=device, decode_fn=decode_fn,
        use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
    )
    debug['per_box'] = per_box
    ungated_union = union          # the zoom result BEFORE the gate (diagnostics)

    # MIL logit gate: keep only crops that out-score the full frame; if none do,
    # the zoom is just noise → defer to the trusted flat decode (the safety net
    # multi_zoom_single already has).  No-op when the image head is off.
    gate_status = 'kept' if union is not None else 'no_boxes'
    if union is not None and gate_logit:
        gated, keep = _gate_box_union(per_box, info.image_logit, margin=gate_margin)
        debug['gated_boxes'] = keep
        if gated is None:
            gate_status = 'fired'          # gate rejected every box → flat fallback
            union = None
        else:
            union = gated

    pred = union if union is not None else mask1
    debug.update({'mask_zoom': union, 'zoomed': union is not None,
                  'gate_status': gate_status})
    rec = eval_metric(pred, info, item, decoder=label)

    # Gate diagnostics: F1 the ungated zoom WOULD have scored, so eval can measure
    # whether deferring to flat actually helped (only the cheap metric, no forward).
    if return_debug:
        if ungated_union is not None:
            debug['f1_ungated'] = float(eval_metric(ungated_union, info, item, decoder=label).f1)
        else:
            debug['f1_ungated'] = float(rec.f1)
    return (rec, debug) if return_debug else rec


# ── seeding ──────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── data ───────────────────────────────────────────────────────────────────────

def collect_splices(args, sources, res, *, split: str) -> Dict[str, List[Item]]:
    """Per-source splice (non-real) items for ``split``, keyed by source.

    ``args`` is an argparse namespace carrying a ``<source>_root`` attr per source
    (see ``_SOURCE_ROOT``); missing/absent roots are skipped with a warning.
    """
    out: Dict[str, List[Item]] = {}
    for source in sources:
        root_str = getattr(args, _SOURCE_ROOT[source], None)
        if not root_str:
            continue
        root = Path(root_str)
        if not root.exists():
            log_line(f'[sb] WARN: root not found for {source}: {root}')
            continue
        train_ds, val_ds = REGISTRY[source](root, res=res)
        ds = train_ds if split == 'train' else val_ds
        splices = [it for it in ds.items if not it.is_real]
        out[source] = splices
        log_line(f'[sb] {source} ({split}): {len(splices)} splices')
    return out


# ── viz ──────────────────────────────────────────────────────────────────────────

def save_viz(path, debug, rec, *, source, item_id, flat_f1, attn_f1, attn_box,
             patch_frac, max_regions, readoff_pad_frac):
    """Full-frame panels + one row per zoom window.

    Row 0: input+boxes | pred heatmap | full attention | GT mask | union zoom decode.
      Boxes: green = the GT splice run through the SAME read-off (group + pad) as the
      head, red = predicted, cyan = attn-zoom.  So green = "ideal boxes".
    Rows 1..K: for each window — the crop the model saw | its (new) attention map |
      what it decoded in that crop.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    grid_hw = debug['grid_hw']
    n_rows, n_cols = grid_hw
    img = np.asarray(debug['img_pil'])
    H, W = img.shape[:2]

    gm = gt_grid_mask(rec.gt_mask, grid_hw, patch_frac=patch_frac)
    kind = 'box' if gm.any() else 'no_gt'
    # Green = GT run through the SAME read-off the head is graded against
    # (patch-space proximity grouping + padding floor + partial squaring).
    tgt_fboxes = proximity_bboxes(gm, dilate=1, min_patches=1, max_regions=max_regions,
                                  pad_frac=readoff_pad_frac, min_pad_frac=0.0, square_cap=1.4)

    per_box = debug.get('per_box') or []
    ncols = 5
    nrows = 1 + len(per_box)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows), squeeze=False)
    for a in axes.ravel():
        a.axis('off')

    # ── Row 0: full-frame ──
    top = axes[0]
    top[0].imshow(img); top[0].set_title('input + boxes', fontsize=9)
    drew = {}

    def _draw(fbox, color, lab):
        if fbox is None:
            return
        y0, x0, y1, x1 = fbox
        top[0].add_patch(Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H,
                                   fill=False, edgecolor=color, lw=2,
                                   label=(lab if lab not in drew else '_nolegend_')))
        drew[lab] = True

    for fb in tgt_fboxes:
        _draw(fb, 'lime', 'target')
    _draw(attn_box, 'cyan', 'attn-zoom')
    for fb in debug['boxes']:
        _draw(fb, 'red', 'pred')
    if drew:
        top[0].legend(loc='lower right', fontsize=7)

    heat = debug.get('heat')
    if heat is not None:
        top[1].imshow(np.asarray(heat).reshape(n_rows, n_cols), cmap='magma', vmin=0, vmax=1)
    top[1].set_title(f'pred heatmap (kind={kind})', fontsize=9)

    attn1 = debug.get('attn1')
    if attn1 is not None:
        top[2].imshow(np.asarray(attn1).reshape(n_rows, n_cols), cmap='magma')
    top[2].set_title('full attention', fontsize=9)

    if rec.gt_mask is not None:
        top[3].imshow(rec.gt_mask, cmap='gray')
    top[3].set_title('GT mask', fontsize=9)

    mz = debug.get('mask_zoom')
    if mz is not None:
        top[4].imshow(np.asarray(mz), cmap='gray'); top[4].set_title('union zoom decode', fontsize=9)
    else:
        top[4].set_title('no zoom (flat)', fontsize=9)

    # ── Window rows: crop the model saw | its new attention | its decode ──
    for k, pb in enumerate(per_box):
        row = axes[k + 1]
        crop = pb.get('crop_pil')
        if crop is not None:
            row[0].imshow(np.asarray(crop))
        row[0].set_title(f'window {k}: crop', fontsize=9)

        ac, cg = pb.get('attn_crop'), pb.get('crop_grid_hw')
        if ac is not None and cg is not None:
            row[1].imshow(np.asarray(ac).reshape(cg), cmap='magma')
        row[1].set_title(f'window {k}: attention', fontsize=9)

        mc = pb.get('mask_crop')
        if mc is not None:
            row[2].imshow(np.asarray(mc), cmap='gray')
        row[2].set_title(f'window {k}: decode', fontsize=9)

    fig.suptitle(f'{source}  {item_id}   f1={rec.f1:.3f}  flat={flat_f1:.3f}  '
                 f'attn={attn_f1:.3f}  zoomed={debug["zoomed"]}', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


# ── per-epoch eval ─────────────────────────────────────────────────────────────

def _stats(x):
    """(median, mean, p25, p75) of a list — median-led with mean + quartiles."""
    if not x:
        nan = float('nan')
        return nan, nan, nan, nan
    a = np.asarray(x, dtype=float)
    return (float(np.median(a)), float(a.mean()),
            float(np.percentile(a, 25)), float(np.percentile(a, 75)))


@torch.no_grad()
def _flat_f1(model, item, res, *, device, decode_fn, decoder_name, use_amp, amp_dtype) -> float:
    img_t = load_image_tensor(item, res, device=device)
    info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    rec = eval_metric(decode_fn(info), info, item, decoder=decoder_name)
    return float(rec.f1)


@torch.no_grad()
def _hdbscan_zoom_f1(model, item, res, *, device, use_amp, amp_dtype) -> float:
    """F1 of zooming the region(s) the HDBSCAN partition predicts (box_source=decode)."""
    rec = multi_zoom_single(
        model, item, res, device=device, use_amp=use_amp, amp_dtype=amp_dtype,
        decoder='hdbscan', box_source='decode',
    )
    return float(rec.f1)


@torch.no_grad()
def evaluate(model, head, eval_by_source, res, *, device, decode_fn, decoder_name,
            use_amp, amp_dtype, flat_cache, attn_cache, hdb_cache, with_hdbscan,
            viz_per_source, viz_dir, epoch,
            single_kwargs, patch_frac, max_regions,
            readoff_pad_frac) -> float:
    """Score the head per source vs flat / attn / hdbscan references; return overall
    median policy F1.  ``flat_cache``/``attn_cache``/``hdb_cache`` carry the static
    references (they depend only on the frozen detector) so callers can share them
    across epochs OR across a read-off sweep — they're computed once per item.
    """
    head.eval()
    overall: List[float] = []
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    for source, items in eval_by_source.items():
        pol, flat, attn, hdb, n_zoom, n_viz = [], [], [], [], 0, 0
        pol_nogate = []                                  # policy F1 with the gate OFF
        gate_kept, gate_fired = 0, 0                     # gate decision counts
        gf_ungated, gf_flat = [], []                     # on FIRED items: zoom vs flat
        for item in items:
            rec, debug = box_heatmap_single(
                model, head, item, res, device=device, decode_fn=decode_fn,
                decoder_name=decoder_name, use_amp=use_amp, amp_dtype=amp_dtype,
                return_debug=True, **single_kwargs,
            )
            pol.append(float(rec.f1))
            n_zoom += int(debug['zoomed'])

            # Gate diagnostics: f1_ungated = what the zoom would have scored with
            # the gate off; gate_status ∈ {no_boxes, kept, fired}.
            ug = debug.get('f1_ungated')
            pol_nogate.append(float(ug) if ug is not None else float(rec.f1))
            gs = debug.get('gate_status')
            if gs == 'kept':
                gate_kept += 1
            elif gs == 'fired':
                gate_fired += 1

            f = flat_cache.get(item.item_id)
            if f is None:
                f = _flat_f1(model, item, res, device=device, decode_fn=decode_fn,
                             decoder_name=decoder_name, use_amp=use_amp, amp_dtype=amp_dtype)
                flat_cache[item.item_id] = f
            flat.append(f)
            if gs == 'fired':                # gate sent this one back to flat
                gf_ungated.append(float(ug) if ug is not None else f)
                gf_flat.append(f)

            # Attention-zoom reference (static on a frozen detector ⇒ cached across
            # epochs).  Fetch the bbox too for the items we visualize.
            want_viz = (viz_dir is not None and (viz_per_source <= 0 or n_viz < viz_per_source))
            a = attn_cache.get(item.item_id)
            attn_box = None
            if want_viz or a is None:
                arec, adbg = attention_zoom_single(
                    model, item, res, device=device, decoder=decoder_name,
                    use_amp=use_amp, amp_dtype=amp_dtype, return_debug=True)
                a = float(arec.f1)
                attn_cache[item.item_id] = a
                attn_box = adbg.get('bbox')
            attn.append(a)

            if with_hdbscan:
                h = hdb_cache.get(item.item_id)
                if h is None:
                    h = _hdbscan_zoom_f1(model, item, res, device=device,
                                         use_amp=use_amp, amp_dtype=amp_dtype)
                    hdb_cache[item.item_id] = h
                hdb.append(h)

            if want_viz:
                save_viz(viz_dir / f'{n_viz:02d}_{source}_{item.item_id}.png', debug, rec,
                         source=source, item_id=item.item_id, flat_f1=f, attn_f1=a,
                         attn_box=attn_box, patch_frac=patch_frac,
                         max_regions=max_regions, readoff_pad_frac=readoff_pad_frac)
                n_viz += 1

        def _fmt(x):
            m, mean, q1, q3 = _stats(x)
            return f'med={m:.4f} mean={mean:.4f} p25={q1:.4f} p75={q3:.4f}'
        log_line(f'[sb-eval] {source:>8} (n={len(pol)} zoomed={n_zoom}/{len(pol)})')
        log_line(f'[sb-eval]   policy  {_fmt(pol)}')
        log_line(f'[sb-eval]   nogate  {_fmt(pol_nogate)}')
        log_line(f'[sb-eval]   flat    {_fmt(flat)}')
        log_line(f'[sb-eval]   attn    {_fmt(attn)}')
        if with_hdbscan:
            log_line(f'[sb-eval]   hdbscan {_fmt(hdb)}')
        # Gate diagnostics: how often it fired, and whether the flat fallback it
        # forced actually beat the ungated zoom on those items (Δ>0 ⇒ gate helped).
        pol_mean = _stats(pol)[1]
        nog_mean = _stats(pol_nogate)[1]
        gate_line = (f'[sb-eval]   gate    kept={gate_kept} fired={gate_fired} '
                     f'| mean policy={pol_mean:.4f} vs nogate={nog_mean:.4f} '
                     f'(Δ={pol_mean - nog_mean:+.4f})')
        if gf_flat:
            ung_m = float(np.mean(gf_ungated))
            flt_m = float(np.mean(gf_flat))
            gate_line += (f' | on fired (n={len(gf_flat)}): '
                          f'ungated_zoom={ung_m:.4f} → flat={flt_m:.4f} (Δ={flt_m - ung_m:+.4f})')
        log_line(gate_line)
        overall.extend(pol)

    om, omean, o25, o75 = _stats(overall)
    log_line(f'[sb-eval] epoch={epoch} OVERALL policy: '
             f'med={om:.4f} mean={omean:.4f} p25={o25:.4f} p75={o75:.4f} (n={len(overall)})')
    return om
