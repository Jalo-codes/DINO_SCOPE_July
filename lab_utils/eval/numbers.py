"""lab_utils.eval.numbers — general numerical-eval engine (image + localization).

Shared logic for the numerical eval, kept in lab_utils so multiple thin script
entry points can reuse it without importing each other (the C-script invariant).

Configure any subset of the registry sources via --<source>_root (imd2020,
casia, coco_inpaint, sagid, bfree, anyedit, indoor, tgif2, cocoglide, opensdi).
TGIF additionally accepts split kwargs (--tgif_types/--tgif_eval_per_cell/
--tgif_split_seed) so its hidden held-out split reproduces a training run
exactly; per-(model|type|family) subgroup cells are emitted automatically for
any source whose items carry a subgroup label.

Reports BOTH image-level detection (AUC + acc/precision/recall/F1 @ 0.5) and
localization (F1 / IoU / precision / recall, median-led with area buckets and —
for TGIF — per-subcategory cells), and dumps everything to one JSON.

Efficiency contract:
  * ONE DINO forward per item feeds BOTH decoders (kmeans + hdbscan) — the same
    ModelInfo is decoded twice, the backbone is NOT re-run per decoder.
  * zoom adds exactly ONE more shared pair of forwards (full pass-1 + one crop
    pass-2); the bbox comes from attention (decoder-independent) so both decoders
    reuse the same two infos.  K decoders cost 2 forwards, not 2K.

Multi-checkpoint: pass >1 checkpoint and the eval item sets are built ONCE and
scored by each checkpoint over the identical images.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from lab_utils.compat import trapz
from lab_utils.data.datasets.registry import REGISTRY
from lab_utils.data.resolution import Resolution
from lab_utils.eval.aggregate import (
    summarize, summarize_by_subgroup, summarize_image_only,
)
from lab_utils.eval.decode.hdbscan import decode_hdbscan
from lab_utils.eval.decode.kmeans import decode_kmeans
from lab_utils.eval.decode.threshold import decode_threshold
from lab_utils.eval.fetch import model_info
from lab_utils.eval.load_model import _infer_heads, load_eval_model
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.record import EvalRecord
from lab_utils.eval.val_sources import SOURCE_ROOT_ARGS, add_source_root_args
from lab_utils.eval.zoom import (
    attention_to_bbox, bbox_is_trivial, crop_to_bbox, place_mask_in_frame_pixels,
)
from lab_utils.logging.text import log_line
from lab_utils.train.checkpoint import load as load_ckpt
from lab_utils.train.distributed import unwrap_model

_DECODE_FNS = {
    'kmeans': decode_kmeans, 'hdbscan': decode_hdbscan, 'threshold': decode_threshold,
    # Image-level only: emit an empty mask and skip localization entirely. For a
    # checkpoint with no localization head, every real decoder raises per item,
    # which the handler logs and skips — leaving NO records, hence no image-level
    # AUC either, even though that metric is decoder-independent. Matches the
    # 'none' option eval.py and eval_robustness already expose.
    'none': lambda info: np.zeros(info.grid_hw, dtype=bool),
}

# Canonical backbone for this project — pinned as the default override so a
# checkpoint carrying a stale cfg.model_name (e.g. 'dinov2-base') still rebuilds
# the right arch. Heads are still inferred from the state_dict by load_eval_model.
_DINOV3 = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='eval_numbers',
        description='General numerical eval (image + localization) over any source(s).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', nargs='+', required=True,
                   help='One or more .pt checkpoints. First is the reference; the '
                        'rest are compared against it (Δ in the printed table).')
    p.add_argument('--label', nargs='+', default=None,
                   help='Optional label per checkpoint (default: file stem).')

    g = p.add_argument_group('datasets (configure any subset via --<source>_root)')
    add_source_root_args(g)   # --imd2020_root, --casia_root, …, --tgif2_root, --cocoglide_root, …
    g.add_argument('--sources', nargs='*', default=None,
                   help='Restrict to these source names (default: every configured root).')
    g.add_argument('--max_items', type=int, default=None,
                   help='Cap items per source (smoke test).')

    g = p.add_argument_group('TGIF split (only used when --tgif2_root is set)')
    g.add_argument('--tgif_types', nargs='+', default=None, choices=['sp', 'fr'],
                   help='Restrict TGIF manipulation types (e.g. fr). Default: all. '
                        'MUST match the training run to reproduce its hidden split.')
    # NOTE: --tgif_eval_per_cell is registered by add_source_root_args (default None);
    # re-adding it here raises an argparse conflict. run() falls back to 500 when None
    # so the held-out split still matches training (leakage-free).
    g.add_argument('--tgif_split_seed', default='tgif_fr_half',
                   help='MUST match the training run for an identical hidden split.')

    g = p.add_argument_group('decode / readouts')
    g.add_argument('--decoders', nargs='+', default=['kmeans', 'hdbscan'],
                   # 'none' = image-level only; required for image-head-only
                   # checkpoints, where any real decoder has nothing to decode.
                   choices=['kmeans', 'hdbscan', 'threshold', 'none'])
    g.add_argument('--zoom', action=argparse.BooleanOptionalAction, default=True,
                   help='Also run attention-zoom readouts (shared forwards).')
    g.add_argument('--attn_percentile', default='otsu',
                   help="Zoom threshold method: 'otsu' (default), 'gap', 'peak', or "
                        'a numeric percentile.')
    g.add_argument('--attn_thresh_mult', type=float, default=1.0,
                   help='Threshold scale; 1.0 for otsu/gap, ~0.08 for peak.')
    g.add_argument('--launder_mode', default='none',
                   choices=['none', 'bicubic_x2', 'bicubic_x4', 'real_esrgan_x2', 'real_esrgan_x4'],
                   help='Upsampling-downsampling laundering attack simulation.')
    g.add_argument('--prelaundered_root', default=None,
                   help='Path to the pre-laundered directory for offline Real-ESRGAN/SR.')

    g = p.add_argument_group('arch overrides (pin the backbone)')
    g.add_argument('--model_name', default=_DINOV3)
    g.add_argument('--image_size', type=int, default=448)
    g.add_argument('--patch_size', type=int, default=16)

    g = p.add_argument_group('output / hardware')
    g.add_argument('--out_json', default=None, help='Where to write the results JSON.')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default='bfloat16', choices=['float16', 'bfloat16'])
    return p


# ── Item building (once, reused across checkpoints) ──────────────────────────

def _assert_tgif_heldout(train_ds, val_ds, args) -> None:
    """Hard guard that the evaluated TGIF items are the HELD-OUT scenes only.

    The split is deterministic in (split_seed, eval_per_cell, types) and assigns
    each whole scene (coco_id) to exactly one side, so — as long as those params
    match the training run — the val side is precisely the scenes training never
    saw.  We rebuild both sides here and assert the partition is disjoint, then
    report the held-out count so the eval set is auditable.  A coco_id appearing
    on both sides would mean the split is broken; we refuse to report numbers.
    """
    train_ids = {it.meta.get('tgif_coco_id') for it in train_ds.items}
    val_ids   = {it.meta.get('tgif_coco_id') for it in val_ds.items}
    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(
            f'tgif2 LEAKAGE: {len(overlap)} coco_id(s) appear in BOTH the trained-on '
            f'and held-out splits (e.g. {sorted(overlap)[:5]}). The split is broken — '
            f'do not trust these eval numbers.')
    log_line(
        f'[eval-num] tgif2 leakage check OK: evaluating {len(val_ids)} HELD-OUT coco_ids, '
        f'0 overlap with {len(train_ids)} trained-on; '
        f'split seed={args.tgif_split_seed} eval_per_cell={args.tgif_eval_per_cell} '
        f'types={sorted(args.tgif_types) if args.tgif_types else "all"} '
        f'(these MUST match the training run for val to be the same held-out set)')


def _build_item_sets(args, res: Resolution) -> Dict[str, List]:
    """{source_name: [Item]} for every configured --<source>_root.  GT-free; model-free.

    Any registry source works.  TGIF additionally takes the split kwargs so its
    hidden held-out split matches a training run exactly.  Per-(model|type|family)
    subgroup cells are emitted automatically for any source whose items carry a
    subgroup label (currently TGIF).
    """
    sets: Dict[str, List] = {}
    restrict = getattr(args, 'sources', None)

    for source, attr in SOURCE_ROOT_ARGS.items():
        if restrict and source not in restrict:
            continue
        root_str = getattr(args, attr, None)
        if not root_str:
            continue
        root = Path(root_str)
        if not root.exists():
            log_line(f'[eval-num] WARNING: --{attr} {root} not found — skipping {source}')
            continue

        kw: dict = {}
        if source == 'tgif2':
            kw = dict(eval_per_cell=args.tgif_eval_per_cell,
                      split_seed=args.tgif_split_seed,
                      types=set(args.tgif_types) if args.tgif_types else None)
        train_ds, val_ds = REGISTRY[source](root, res=res, **kw)
        if source == 'tgif2':
            _assert_tgif_heldout(train_ds, val_ds, args)   # prove eval = held-out only
        items = val_ds.items
        if args.max_items:
            if source == 'tgif2':
                from collections import defaultdict
                by_subcat = defaultdict(list)
                for it in items:
                    sub = it.meta.get('tgif_subcat')
                    by_subcat[sub].append(it)
                
                # Take max_items from each subgroup cell
                capped_items = []
                for sub in sorted(by_subcat):
                    capped_items.extend(by_subcat[sub][:args.max_items])
                items = capped_items
            else:
                items = items[:args.max_items]
        sets[source] = items

        extra = ''
        if source == 'tgif2':
            extra = (f' (eval_per_cell={args.tgif_eval_per_cell} seed={args.tgif_split_seed} '
                     f'types={sorted(args.tgif_types) if args.tgif_types else "all"})')
        log_line(f'[eval-num] {source}: {len(items)} held-out items{extra}')

    if not sets:
        raise RuntimeError('eval_numbers: no dataset roots configured/found. Pass at least one '
                           '--<source>_root (e.g. --imd2020_root, --tgif2_root, --casia_root).')
    return sets


# ── Per-item readouts (shared forward) ───────────────────────────────────────

class PrelaunderedMissing(Exception):
    """A real_esrgan launder mode could not find a pre-laundered image for an item.

    Raised instead of silently substituting the original image — the caller skips
    and counts the item so partial pre-laundered coverage is visible, never faked.
    """


def _load_and_launder(item, launder_mode, prelaundered_root, args=None):
    """Load the image from disk and apply laundering.
    
    Returns (img_down, img_up) where:
      - img_down is the laundered image of the original resolution.
      - img_up is the upscaled image (if upscaling occurred) or the original image.
    """
    from PIL import Image as PILImage
    img_pil = PILImage.open(item.image).convert('RGB')
    w, h = img_pil.size
    
    if launder_mode == 'none':
        return img_pil, img_pil
        
    if launder_mode.startswith('bicubic_'):
        factor = float(launder_mode.split('_x')[-1])
        img_up = img_pil.resize((int(w * factor), int(h * factor)), PILImage.BICUBIC)
        img_down = img_up.resize((w, h), PILImage.BICUBIC)
        return img_down, img_up
        
    if launder_mode.startswith('real_esrgan_'):
        # Load upscaled image
        img_up = _load_prelaundered_image(item, launder_mode, prelaundered_root, img_pil, args=args)
        img_down = img_up.resize((w, h), PILImage.BICUBIC)
        return img_down, img_up
        
    return img_pil, img_pil


def _load_prelaundered_image(item, launder_mode, prelaundered_root, default_img, args=None):
    from PIL import Image as PILImage
    suffix = launder_mode.replace('real_', '') # e.g. esrgan_x2 or esrgan_x4
    
    # 1. Try to find the root path for this source
    root_path = None
    if args is not None:
        source_root_attr = SOURCE_ROOT_ARGS.get(item.source)
        if source_root_attr:
            root_path_str = getattr(args, source_root_attr, None)
            if root_path_str:
                root_path = Path(root_path_str)
                
    # 2. If we know the root path, resolve relative path and construct new path
    if root_path:
        try:
            relative_path = Path(item.image).relative_to(root_path)
            # Try prelaundered_root if provided
            if prelaundered_root:
                new_path = Path(prelaundered_root) / relative_path
                if new_path.exists():
                    return PILImage.open(new_path).convert('RGB')
            # Try folder swap by appending suffix to root folder name
            new_root = root_path.parent / f"{root_path.name}_{suffix}"
            new_path = new_root / relative_path
            if new_path.exists():
                return PILImage.open(new_path).convert('RGB')
        except Exception as e:
            # Present-but-unreadable (corrupt) or unresolvable: log with an allowed
            # tag (NOT '[laundering]', which log_line rejects) and fall through so
            # the item is counted as missing + skipped, never silently substituted.
            log_line(f"[eval-num] laundering WARN: prelaundered load failed for {item.image}: {e}")

    # 3. Fallback to the existing heuristics
    orig_path_str = str(item.image)
    new_dir = f'tgif2_flux_{suffix}'
    
    # Try simple folder swap
    if 'tgif2_flux' in orig_path_str:
        new_path_str = orig_path_str.replace('tgif2_flux', new_dir)
        if os.path.exists(new_path_str):
            return PILImage.open(new_path_str).convert('RGB')
            
    # Try prelaundered_root with 'val' heuristic
    if prelaundered_root:
        p = Path(item.image)
        parts = p.parts
        if 'val' in parts:
            val_idx = parts.index('val')
            rel_path = Path(*parts[val_idx:])
            new_path = Path(prelaundered_root) / rel_path
            if new_path.exists():
                return PILImage.open(new_path).convert('RGB')

    # Not found by any strategy.  Do NOT substitute the original — that would
    # silently contaminate real_esrgan numbers with un-laundered images.  Record
    # the miss and raise so the caller skips + counts this item.
    tr = getattr(args, '_prelaundered', None) if args is not None else None
    if tr is not None:
        tr['missing'].add(str(item.image))
    raise PrelaunderedMissing(str(item.image))


@torch.no_grad()
def _flat_records(model, items, res, *, decoders, device, use_amp, amp_dtype,
                  log_tag, launder_mode='none', prelaundered_root=None, args=None) -> Dict[str, List[EvalRecord]]:
    """ONE forward per item → decode each decoder → {decoder: [records]}."""
    out = {f'{d}_flat': [] for d in decoders}
    n = len(items)
    every = max(1, n // 10)
    for i, item in enumerate(items):
        sub = item.meta.get('tgif_subcat') or item.meta.get('generator')
        try:
            img_down, _ = _load_and_launder(item, launder_mode, prelaundered_root, args=args)
            img_t = load_image_tensor(img_down, res, device=device)
            info  = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
        except PrelaunderedMissing:
            continue   # recorded for the coverage summary; skip, never substitute
        except Exception as exc:
            log_line(f'{log_tag} WARN flat fetch failed {item.item_id}: {exc}')
            continue
        for d in decoders:
            try:
                m = _DECODE_FNS[d](info)
                out[f'{d}_flat'].append(eval_metric(m, info, item, decoder=f'{d}_flat', subgroup=sub))
            except Exception as exc:
                log_line(f'{log_tag} WARN {d}_flat decode failed {item.item_id}: {exc}')
        if (i + 1) % every == 0 or (i + 1) == n:
            log_line(f'{log_tag} flat {i + 1}/{n}')
    return out


@torch.no_grad()
def _zoom_records(model, items, res, *, decoders, device, use_amp, amp_dtype,
                  attn_percentile, attn_thresh_mult, log_tag, launder_mode='none', prelaundered_root=None, args=None) -> Dict[str, List[EvalRecord]]:
    """Attention-zoom with forwards SHARED across decoders.

    Per item: one pass-1 (full) forward + one bbox (decoder-independent, from
    attention) + one pass-2 (crop) forward — then every decoder reuses those two
    ModelInfos.  So K decoders cost 2 forwards, not 2K.
    """
    out = {f'{d}_zoom': [] for d in decoders}
    full_px = (int(res.image_size), int(res.image_size))
    n = len(items)
    every = max(1, n // 10)
    for i, item in enumerate(items):
        sub = item.meta.get('tgif_subcat') or item.meta.get('generator')
        try:
            img_down, img_up = _load_and_launder(item, launder_mode, prelaundered_root, args=args)
            img_t = load_image_tensor(img_down, res, device=device)
            info1 = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
        except PrelaunderedMissing:
            continue   # recorded for the coverage summary; skip, never substitute
        except Exception as exc:
            log_line(f'{log_tag} WARN zoom fetch failed {item.item_id}: {exc}')
            continue

        bbox = None
        info2 = None
        if info1.attention is not None:
            bbox = attention_to_bbox(
                info1.attention, info1.grid_hw,
                percentile=attn_percentile, thresh_mult=attn_thresh_mult,
                pad_frac=0.10, min_box_size=8, min_pad_frac=0.06,
            )
            if bbox_is_trivial(bbox, min_crop_frac=0.25):
                bbox = None
            else:
                try:
                    crop_pil = crop_to_bbox(img_up, bbox)
                    crop_t = load_image_tensor(crop_pil, res, device=device)
                    info2  = model_info(model, crop_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
                except Exception as exc:
                    log_line(f'{log_tag} WARN zoom crop failed {item.item_id}: {exc}')
                    bbox = None

        for d in decoders:
            try:
                if bbox is None or info2 is None:        # fallback → flat decode, zoom label
                    mask = _DECODE_FNS[d](info1)
                else:
                    crop2d = np.asarray(_DECODE_FNS[d](info2), dtype=bool)
                    if crop2d.ndim == 1:
                        crop2d = crop2d.reshape(info2.grid_hw)
                    mask = place_mask_in_frame_pixels(crop2d, bbox, full_px)
                out[f'{d}_zoom'].append(eval_metric(mask, info1, item, decoder=f'{d}_zoom', subgroup=sub))
            except Exception as exc:
                log_line(f'{log_tag} WARN {d}_zoom decode failed {item.item_id}: {exc}')
        if (i + 1) % every == 0 or (i + 1) == n:
            log_line(f'{log_tag} zoom {i + 1}/{n}')
    return out


# ── Image-level detection metrics (decoder-independent) ──────────────────────

def _image_level_metrics(records: List[EvalRecord]) -> dict:
    """AUC + accuracy/precision/recall/F1 @ image_score>=0.5 over splices+reals."""
    scores = np.array([r.image_score for r in records], dtype=np.float64)
    labels = np.array([0 if r.is_real else 1 for r in records], dtype=np.int64)
    keep = ~np.isnan(scores)
    scores, labels = scores[keep], labels[keep]
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    res: dict = {'n_pos': n_pos, 'n_neg': n_neg, 'auc': float('nan'),
                 'acc': float('nan'), 'precision': float('nan'),
                 'recall': float('nan'), 'f1': float('nan')}
    if n_pos and n_neg:
        order = np.argsort(-scores)
        sl = labels[order]
        tpr = np.cumsum(sl) / n_pos
        fpr = np.cumsum(1 - sl) / n_neg
        auc = float(trapz(tpr, fpr))
        res['auc'] = 1.0 + auc if auc < 0 else auc
    if len(scores):
        pred = (scores >= 0.5).astype(np.int64)
        tp = int(((pred == 1) & (labels == 1)).sum())
        fp = int(((pred == 1) & (labels == 0)).sum())
        fn = int(((pred == 0) & (labels == 1)).sum())
        res['acc'] = float((pred == labels).mean())
        res['precision'] = tp / (tp + fp) if (tp + fp) else float('nan')
        res['recall'] = tp / (tp + fn) if (tp + fn) else float('nan')
        if res['precision'] == res['precision'] and res['recall'] == res['recall'] \
                and (res['precision'] + res['recall']) > 0:
            res['f1'] = 2 * res['precision'] * res['recall'] / (res['precision'] + res['recall'])
    return res


# ── One checkpoint over all sources ──────────────────────────────────────────

def _eval_checkpoint(ckpt_path: str, label: str, item_sets: Dict[str, List],
                     res: Resolution, *, args, device, use_amp) -> dict:
    log_line(f'[eval-num] ═══ checkpoint={label} ({ckpt_path}) ═══')
    # Head dims (contrastive_dim / pool_hidden / patch_bce) are inferred from the
    # saved weights and passed as overrides — these tgif-finetune checkpoints
    # carry a stale cfg slot (e.g. contrastive_dim=128, model_name=dinov2-base)
    # that load_eval_model would otherwise trust, causing a size-mismatch on a
    # present key (which strict=False does NOT suppress).
    state = load_ckpt(ckpt_path)
    c_dim, p_hidden, p_bce = _infer_heads(state.get('model', state))
    log_line(f'[eval-num] inferred heads: contrastive_dim={c_dim} pool_hidden={p_hidden} patch_bce={p_bce}')
    model, _cfg, _res = load_eval_model(
        ckpt_path, device=device, strict=False,
        model_name=args.model_name, image_size=args.image_size, patch_size=args.patch_size,
        contrastive_dim=c_dim, pool_hidden=p_hidden, patch_bce=p_bce,
    )
    bare = unwrap_model(model)
    bare.eval()

    decoders = list(args.decoders)
    launder_mode = getattr(args, 'launder_mode', 'none')
    prelaundered_root = getattr(args, 'prelaundered_root', None)
    common = dict(decoders=decoders, device=device, use_amp=use_amp, amp_dtype=args.amp_dtype,
                  launder_mode=launder_mode, prelaundered_root=prelaundered_root, args=args)

    is_prelaundered = launder_mode.startswith('real_esrgan')

    out: dict = {'checkpoint': str(ckpt_path), 'sources': {}}
    for src, items in item_sets.items():
        tag = f'[eval-num] {label}/{src}'
        log_line(f'{tag} n={len(items)} decoders={decoders} zoom={args.zoom}')

        # Reset per-source pre-laundered coverage tracking (read inside _load_and_launder).
        if is_prelaundered:
            args._prelaundered = {'missing': set()}

        readouts: Dict[str, List[EvalRecord]] = {}
        readouts.update(_flat_records(bare, items, res, log_tag=tag, **common))
        if args.zoom:
            readouts.update(_zoom_records(
                bare, items, res, log_tag=tag,
                attn_percentile=_parse_pct(args.attn_percentile),
                attn_thresh_mult=args.attn_thresh_mult, **common))

        if is_prelaundered:
            n_missing = len(args._prelaundered['missing'])
            n_attempt = len(items)
            n_scored = n_attempt - n_missing
            cov = 100.0 * n_scored / max(1, n_attempt)
            if n_missing:
                bar = '!' * 64
                log_line(f'{tag} {bar}')
                log_line(f'{tag} !! PRE-LAUNDERED COVERAGE ({launder_mode}): '
                         f'{n_scored}/{n_attempt} scored ({cov:.1f}%), '
                         f'{n_missing} SKIPPED — laundered file missing or unreadable.')
                log_line(f'{tag} !! Skipped items were NOT substituted with originals. '
                         f'These numbers reflect ONLY the laundered subset.')
                log_line(f'{tag} {bar}')
            else:
                log_line(f'{tag} pre-laundered coverage ({launder_mode}): '
                         f'full ({n_attempt}/{n_attempt}).')

        # image-level is decoder/mode-independent (image_score from the full pass)
        any_recs = next((r for r in readouts.values() if r), [])
        src_out: dict = {'n_items': len(items), 'image_level': _image_level_metrics(any_recs),
                         'localization': {}}
        for rlabel, recs in readouts.items():
            # decoder 'none' emits empty masks, so every localization block —
            # overall, per bucket, per subgroup — reads a meaningless 0.0000.
            # Report image-level separability per subgroup instead.
            if rlabel.startswith('none_'):
                entry = {'overall': summarize_image_only(
                    recs, log_tag=f'{tag}', tag=rlabel)}
                subs = {}
            else:
                entry = {'overall': summarize(recs, log_tag=f'{tag}', tag=rlabel)}
                subs = summarize_by_subgroup(recs, log_tag=f'{tag}', tag=rlabel)
            if subs:   # only sources whose items carry a subgroup label (TGIF)
                entry['subgroups'] = subs
            src_out['localization'][rlabel] = entry
        out['sources'][src] = src_out
    return out


def _parse_pct(val: str):
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


# ── Comparison print ─────────────────────────────────────────────────────────

def _med_f1(src_out: dict, rlabel: str) -> float:
    loc = src_out.get('localization', {}).get(rlabel, {})
    return loc.get('overall', {}).get('splices', {}).get('f1', {}).get('median', float('nan'))


def _print_comparison(results: Dict[str, dict], labels: Sequence[str]) -> None:
    log_line('[eval-num] ════════════ COMPARISON ════════════')
    ref = labels[0]
    sources = sorted(results[ref]['sources'])
    for src in sources:
        log_line(f'[eval-num] ── source={src} ──')
        line = '   image_auc : ' + '  '.join(
            f'{lab}={results[lab]["sources"][src]["image_level"]["auc"]:.4f}' for lab in labels)
        log_line('[eval-num]' + line)
        rlabels = sorted(results[ref]['sources'][src]['localization'])
        for rl in rlabels:
            vals = {lab: _med_f1(results[lab]['sources'][src], rl) for lab in labels}
            base = vals[ref]
            parts = []
            for lab in labels:
                v = vals[lab]
                d = (v - base)
                delta = '' if lab == ref or d != d else f' (Δ{d:+.4f})'
                parts.append(f'{lab}={v:.4f}{delta}')
            log_line(f'[eval-num]   {rl:>15} medF1 : ' + '  '.join(parts))


# ── Entry ─────────────────────────────────────────────────────────────────────

def _to_py(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run(args) -> dict:
    """Execute the eval for an already-parsed args namespace; return the payload.

    Both the general script and any preset entry point build args via
    ``build_parser()`` and call this — no eval logic is duplicated in scripts.
    """
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not args.no_amp) and (device.type == 'cuda')

    # --tgif_eval_per_cell comes from add_source_root_args with default None; this
    # path requires a concrete cap so the held-out coco_id split matches training.
    if getattr(args, 'tgif_eval_per_cell', None) is None:
        args.tgif_eval_per_cell = 500

    labels = args.label or [Path(c).stem for c in args.checkpoint]
    if len(labels) != len(args.checkpoint):
        raise SystemExit('eval_numbers: --label count must match --checkpoint count')

    res = Resolution(image_size=args.image_size, patch_size=args.patch_size)
    item_sets = _build_item_sets(args, res)   # built ONCE, reused per checkpoint

    results: Dict[str, dict] = {}
    for ckpt, lab in zip(args.checkpoint, labels):
        results[lab] = _eval_checkpoint(ckpt, lab, item_sets, res,
                                        args=args, device=device, use_amp=use_amp)

    _print_comparison(results, labels)

    out_json = args.out_json
    if out_json is None:
        out_json = str(Path(args.checkpoint[0]).resolve().parent / 'eval_numbers.json')
    payload = {'labels': list(labels), 'decoders': list(args.decoders),
               'zoom': bool(args.zoom), 'results': results}
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(payload, f, indent=2, default=_to_py)
    log_line(f'[eval-num] wrote results → {out_json}')
    return payload
