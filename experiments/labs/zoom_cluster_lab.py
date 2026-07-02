"""experiments.labs.zoom_cluster_lab — learned zoom head: clustering, advantage
reward, training, inference, and robust eval. Implements docs/zoom_head_spec.md.

Per item, frozen detector ⇒ features are cheap and the reward is a deterministic
queryable function, so this is an offline contextual bandit, not RL:

    frozen forward → z (contrastive), feats=[z|attn|patch_logit]
      ProjectionHead:  z → z'          → HDBSCAN(z') + connected-components → REGIONS
      ZoomValueHead:   feats → per-patch scalar → mean over a region = predicted advantage
    reward(region) = F1(zoom→region) − F1(no-zoom baseline)        [run_bbox_zoom, frozen]
    train: value head regresses region-mean toward realized advantage (search→distil);
           projection head trained by a GT-instance metric loss (clean clusters).
    decode: gate region iff predicted-advantage > δ → zoom union → score; else flat.

v1 scope (flagged): per-region advantage is scored STANDALONE (regions are spatially
disjoint after CC, so marginal ≈ standalone); full leave-one-out union credit and
soft-IoU reward are v2 knobs. δ is a tuned operating point, not learned.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution
from lab_utils.eval.fetch import ModelInfo, model_info
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.zoom import gt_grid_mask, _label_components, _dilate8, _pad_bbox, BBox, bbox_is_trivial
from lab_utils.eval.multibox import _square, _inter_area
from lab_utils.eval.decode.hdbscan import _load_hdbscan
from lab_utils.logging.text import log_line

from experiments.labs.attention_zoom import run_bbox_zoom, attention_zoom_single
from experiments.labs.box_policy_zoom import build_policy_input

_TRAIN_SOURCES = ('casia', 'sagid')
_EVAL_SOURCES = ('imd2020', 'sagid')


# ── clustering: HDBSCAN(z') → spatial regions ────────────────────────────────────

def cluster_regions(
    zp: np.ndarray,
    grid_hw: Tuple[int, int],
    *,
    min_cluster_size: int = 8,
    min_samples: Optional[int] = None,
    dilate: int = 1,
    min_patches: int = 4,
    max_regions: int = 6,
    pad_frac: float = 0.06,
    min_box_size: int = 6,
    min_pad_frac: float = 0.04,
    square_cap: float = 1.4,
    overlap_kill_frac: float = 0.30,
) -> List[dict]:
    """HDBSCAN on the projected embedding `zp` (N, d'), then split each semantic
    cluster into SPATIAL blobs (connected-components in patch space) so a cluster
    spanning two separate splices does not become one huge low-magnification box.

    Returns a list of regions, each: {'bbox': BBox, 'idx': (k,) int patch indices,
    'size': int}. Largest `max_regions` by patch count are kept.
    """
    n_rows, n_cols = grid_hw
    n = int(zp.shape[0])
    if n < int(min_cluster_size):
        return []
    backend, HDB = _load_hdbscan()
    if backend is None:
        raise ImportError('zoom_cluster_lab: HDBSCAN backend required '
                          '(scikit-learn >= 1.3 or the standalone hdbscan package).')
    kw = dict(min_cluster_size=int(min_cluster_size))
    if min_samples is not None:
        kw['min_samples'] = int(min_samples)
    labels = np.asarray(HDB(**kw).fit_predict(np.ascontiguousarray(zp, dtype=np.float64)))

    regions: List[dict] = []
    for c in sorted(set(labels.tolist())):
        if c < 0:                                            # HDBSCAN noise → ignore
            continue
        cmask = (labels == c).reshape(n_rows, n_cols)
        if int(cmask.sum()) < int(min_patches):
            continue
        grouped = _dilate8(cmask, dilate) if dilate > 0 else cmask
        for cells in _label_components(grouped):
            on = [(r, cc) for (r, cc) in cells if cmask[r, cc]]   # original ON patches
            if len(on) < int(min_patches):
                continue
            rs = [r for r, _ in on]
            cs = [cc for _, cc in on]
            box = _pad_bbox(min(rs), min(cs), max(rs) + 1, max(cs) + 1,
                            n_rows, n_cols, pad_frac, min_box_size=min_box_size,
                            min_pad_frac=min_pad_frac, small_base_pad=1)
            box = _square(box, square_cap)
            idx = np.asarray([r * n_cols + cc for (r, cc) in on], dtype=np.int64)
            regions.append({'bbox': box, 'idx': idx, 'size': len(on)})

    regions.sort(key=lambda r: r['size'], reverse=True)
    regions = regions[:int(max_regions)]

    # Suppress overlapping boxes: after padding + squaring, adjacent clusters
    # can produce heavily overlapping boxes.  Drop a smaller box when > frac
    # of it sits inside a larger one (greedy, descending area — same logic as
    # suppress_contained_boxes but operating on the full region dicts).
    if overlap_kill_frac > 0 and len(regions) > 1:
        suppressed: List[dict] = []
        for rg in sorted(regions, key=lambda r: (r['bbox'][2] - r['bbox'][0])
                         * (r['bbox'][3] - r['bbox'][1]), reverse=True):
            b = rg['bbox']
            ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            if ab <= 0:
                continue
            if any(_inter_area(b, k['bbox']) / ab > overlap_kill_frac
                   for k in suppressed):
                continue
            suppressed.append(rg)
        regions = suppressed

    return regions


# ── reward: realized per-region advantage over the no-zoom baseline ──────────────

def _reward_val(rec, reward: str) -> float:
    return float(rec.iou if reward == 'iou' else rec.f1)


@torch.no_grad()
def region_advantage(
    model, img_pil, info: ModelInfo, item: Item, res: Resolution, bbox: BBox,
    *, decode_fn, decoder_name: str, flat_val: float, reward: str = 'f1',
    use_amp: bool = False, amp_dtype: str = 'float16', min_crop_frac: float = 0.25,
    device=None,
) -> float:
    """Realized advantage of zooming ONE region = F1(zoom→region) − flat baseline.
    Frozen model ⇒ no grad; this is the queryable reward function."""
    union, _ = run_bbox_zoom(
        model, img_pil, [bbox], res, device=device, decode_fn=decode_fn,
        use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
    )
    if union is None:                       # trivial / whole-frame box → no zoom
        return 0.0
    rec = eval_metric(union, info, item, decoder=decoder_name)
    return _reward_val(rec, reward) - float(flat_val)


# ── projection metric loss: clean, zoom-coherent clusters from GT instances ──────

def instance_grid_labels(info: ModelInfo, item: Item, *, decoder_name: str,
                         patch_frac: float = 0.25) -> np.ndarray:
    """(N,) per-patch instance id: 0 = background/clean, 1..K = GT splice instances
    (connected components of the GT patch-grid mask). GT touched via metric (I3)."""
    n_side = info.grid_hw[0]
    rec = eval_metric(np.zeros((n_side, n_side), dtype=bool), info, item, decoder=decoder_name)
    gm = gt_grid_mask(rec.gt_mask, info.grid_hw, patch_frac=patch_frac)
    labels = np.zeros(gm.shape, dtype=np.int64)
    for i, cells in enumerate(_label_components(gm), start=1):
        for (r, c) in cells:
            labels[r, c] = i
    return labels.reshape(-1)


def projection_instance_loss(zp_t: torch.Tensor, inst_labels: torch.Tensor,
                             *, margin: float = 0.2) -> torch.Tensor:
    """Supervised metric loss on L2-normed z': pull WITHIN a GT instance together,
    push ACROSS instances (and instance-vs-background) apart. Smooth, DINO-native
    (no sharp per-patch targets). Returns 0 (graph-connected) when no instances."""
    inst = inst_labels.long()
    has_inst = (inst > 0)
    if int(has_inst.sum()) == 0:
        return zp_t.sum() * 0.0
    sim = zp_t @ zp_t.t()                                   # (N, N) cosine (z' is L2-normed)
    eq = inst.unsqueeze(0) == inst.unsqueeze(1)
    any_inst = has_inst.unsqueeze(0) | has_inst.unsqueeze(1)
    eye = torch.eye(inst.shape[0], dtype=torch.bool, device=zp_t.device)
    pos = eq & has_inst.unsqueeze(0) & has_inst.unsqueeze(1) & ~eye   # same real instance
    neg = (~eq) & any_inst & ~eye                                     # differ; ≥1 real instance
    loss = zp_t.sum() * 0.0
    if int(pos.sum()) > 0:
        loss = loss + (1.0 - sim[pos]).mean()
    if int(neg.sum()) > 0:
        loss = loss + torch.clamp(sim[neg] - float(margin), min=0.0).mean()
    return loss


# ── training step ────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ZoomStats:
    loss: float
    value_loss: float
    proj_loss: float
    n_regions: int
    mean_adv: float
    pos_frac: float        # fraction of regions with positive realized advantage


def zoom_head_train_item(
    model, zoomhead, item: Item, res: Resolution, *, device,
    decode_fn, decoder_name: str, use_attn: bool = True, use_patch_logit: bool = True,
    patch_frac: float = 0.25, cluster_kwargs: Optional[dict] = None,
    min_crop_frac: float = 0.25, lambda_value: float = 1.0, lambda_proj: float = 1.0,
    proj_margin: float = 0.2, reward: str = 'f1', huber_beta: float = 0.1,
    delta: float = 0.0,
    use_amp: bool = False, amp_dtype: str = 'float16',
) -> Optional[Tuple[torch.Tensor, ZoomStats]]:
    """One bandit step: cluster z', gate by predicted advantage > δ, compute
    leave-one-out marginal credit for kept regions (union-aligned reward) and
    standalone advantage for rejected regions, + the projection instance loss.
    Backbone + advantage scoring are no-grad; only the two light heads see gradient.
    """
    cluster_kwargs = cluster_kwargs or {}
    img_t, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    with torch.no_grad():
        info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    if info.embeddings is None:
        return None

    emb_t = torch.from_numpy(np.asarray(info.embeddings, dtype=np.float32)).to(device)
    feats_t = torch.from_numpy(
        build_policy_input(info, use_attn=use_attn, use_patch_logit=use_patch_logit).astype(np.float32)
    ).to(device)

    zp_t = zoomhead.project(emb_t)                          # (N, d')   grad
    value_t = zoomhead.value_logit(feats_t)                 # (N,)      grad

    regions = cluster_regions(zp_t.detach().cpu().numpy(), info.grid_hw, **cluster_kwargs)

    flat_rec = eval_metric(decode_fn(info), info, item, decoder=decoder_name)
    flat_val = _reward_val(flat_rec, reward)

    # ── credit assignment: standalone advantage for all proposed regions ─────────────
    advs: List[float] = [0.0] * len(regions)

    if regions:
        for i in range(len(regions)):
            advs[i] = region_advantage(
                model, img_pil, info, item, res, regions[i]['bbox'],
                device=device, decode_fn=decode_fn, decoder_name=decoder_name,
                flat_val=flat_val, reward=reward, use_amp=use_amp,
                amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
            )

    # value loss: regress per-region mean(value) → credit target
    if regions:
        preds = torch.stack([
            value_t[torch.as_tensor(rg['idx'], device=device)].mean() for rg in regions
        ])
        targets = torch.as_tensor(advs, dtype=preds.dtype, device=device)
        value_loss = F.smooth_l1_loss(preds, targets, beta=huber_beta)
    else:
        value_loss = value_t.sum() * 0.0

    # projection loss: GT-instance metric on z'
    inst = instance_grid_labels(info, item, decoder_name=decoder_name, patch_frac=patch_frac)
    proj_loss = projection_instance_loss(zp_t, torch.from_numpy(inst).to(device), margin=proj_margin)

    loss = lambda_value * value_loss + lambda_proj * proj_loss
    stats = ZoomStats(
        loss=float(loss.detach()), value_loss=float(value_loss.detach()),
        proj_loss=float(proj_loss.detach()), n_regions=len(regions),
        mean_adv=float(np.mean(advs)) if advs else 0.0,
        pos_frac=float(np.mean([a > 0 for a in advs])) if advs else 0.0,
    )
    return loss, stats


# ── deterministic inference (gate by predicted advantage > δ) ────────────────────

@torch.no_grad()
def zoom_head_single(
    model, zoomhead, item: Item, res: Resolution, *, device,
    decode_fn, decoder_name: str = 'kmeans', delta: float = 0.0,
    use_attn: bool = True, use_patch_logit: bool = True,
    cluster_kwargs: Optional[dict] = None, min_crop_frac: float = 0.25,
    use_amp: bool = False, amp_dtype: str = 'float16', return_debug: bool = False,
):
    """Cluster → predict per-region advantage → gate (`> δ`) → zoom union → score.
    Flat-decode fallback (safety floor) when no region clears δ."""
    cluster_kwargs = cluster_kwargs or {}
    label = f'{decoder_name}_zoomhead'
    img_t, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    mask1 = decode_fn(info)
    debug = {'n_regions': 0, 'n_kept': 0, 'zoomed': False, 'preds': []}

    if info.embeddings is None:
        rec = eval_metric(mask1, info, item, decoder=label)
        return (rec, debug) if return_debug else rec

    emb_t = torch.from_numpy(np.asarray(info.embeddings, dtype=np.float32)).to(device)
    feats_t = torch.from_numpy(
        build_policy_input(info, use_attn=use_attn, use_patch_logit=use_patch_logit).astype(np.float32)
    ).to(device)
    zp_np = zoomhead.project(emb_t).detach().cpu().numpy()
    value_np = zoomhead.value_logit(feats_t).detach().cpu().numpy()

    regions = cluster_regions(zp_np, info.grid_hw, **cluster_kwargs)
    preds = [float(value_np[rg['idx']].mean()) for rg in regions]
    kept = [rg for rg, p in zip(regions, preds) if p > float(delta)]
    debug.update({'n_regions': len(regions), 'n_kept': len(kept), 'preds': preds})

    if not kept:
        rec = eval_metric(mask1, info, item, decoder=label)
        return (rec, debug) if return_debug else rec

    union, _ = run_bbox_zoom(
        model, img_pil, [rg['bbox'] for rg in kept], res, device=device,
        decode_fn=decode_fn, use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
    )
    pred_mask = union if union is not None else mask1
    debug['zoomed'] = union is not None
    rec = eval_metric(pred_mask, info, item, decoder=label)
    return (rec, debug) if return_debug else rec


# ── robust per-epoch eval ──────────────────────────────────────────────────────────

def _stats(x):
    if not x:
        nan = float('nan')
        return nan, nan, nan, nan
    a = np.asarray(x, dtype=float)
    return (float(np.median(a)), float(a.mean()),
            float(np.percentile(a, 25)), float(np.percentile(a, 75)))


@torch.no_grad()
def _flat_f1(model, item, res, *, device, decode_fn, decoder_name, use_amp, amp_dtype):
    img_t = load_image_tensor(item, res, device=device)
    info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    return float(eval_metric(decode_fn(info), info, item, decoder=decoder_name).f1)


@torch.no_grad()
def evaluate_zoom_head(
    model, zoomhead, eval_by_source, res, *, device, decode_fn, decoder_name,
    delta: float, cluster_kwargs: dict, flat_cache: dict, attn_cache: dict,
    use_attn: bool = True, use_patch_logit: bool = True, min_crop_frac: float = 0.25,
    reward: str = 'f1', use_amp: bool = False, amp_dtype: str = 'float16',
    delta_grid=(-0.05, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20), epoch: int = 0,
) -> float:
    """Per-source policy F1 (gated at the operating δ) vs flat / attention-zoom
    references, plus rich diagnostics: gate counts, predicted-vs-realized advantage
    calibration, and a δ-sweep (mean captured realized advantage per δ) to pick the
    operating point WITHOUT retraining. Returns overall median policy F1."""
    zoomhead.eval()
    overall: List[float] = []
    all_pred, all_real = [], []                         # calibration scatter (all regions)
    sweep_capture = {d: [] for d in delta_grid}         # per-item captured realized adv

    for source, items in eval_by_source.items():
        pol, flat, attn, n_zoom, n_reg = [], [], [], 0, 0
        for item in items:
            # policy F1 at the operating δ (gated union zoom)
            rec, debug = zoom_head_single(
                model, zoomhead, item, res, device=device, decode_fn=decode_fn,
                decoder_name=decoder_name, delta=delta, use_attn=use_attn,
                use_patch_logit=use_patch_logit, cluster_kwargs=cluster_kwargs,
                min_crop_frac=min_crop_frac, use_amp=use_amp, amp_dtype=amp_dtype,
                return_debug=True,
            )
            pol.append(float(rec.f1)); n_zoom += int(debug['zoomed']); n_reg += debug['n_regions']

            # flat baseline (cached, static on a frozen detector)
            f = flat_cache.get(item.item_id)
            if f is None:
                f = _flat_f1(model, item, res, device=device, decode_fn=decode_fn,
                             decoder_name=decoder_name, use_amp=use_amp, amp_dtype=amp_dtype)
                flat_cache[item.item_id] = f
            flat.append(f)

            # attention-zoom reference (cached)
            a = attn_cache.get(item.item_id)
            if a is None:
                a = float(attention_zoom_single(
                    model, item, res, device=device, decoder=decoder_name,
                    use_amp=use_amp, amp_dtype=amp_dtype).f1)
                attn_cache[item.item_id] = a
            attn.append(a)

            # calibration + δ-sweep: predicted vs realized advantage per region
            img_t, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
            info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
            if info.embeddings is None:
                continue
            emb_t = torch.from_numpy(np.asarray(info.embeddings, dtype=np.float32)).to(device)
            feats_t = torch.from_numpy(
                build_policy_input(info, use_attn=use_attn, use_patch_logit=use_patch_logit).astype(np.float32)
            ).to(device)
            zp_np = zoomhead.project(emb_t).detach().cpu().numpy()
            value_np = zoomhead.value_logit(feats_t).detach().cpu().numpy()
            regions = cluster_regions(zp_np, info.grid_hw, **cluster_kwargs)
            for rg in regions:
                p = float(value_np[rg['idx']].mean())
                r = region_advantage(model, img_pil, info, item, res, rg['bbox'], device=device,
                                     decode_fn=decode_fn, decoder_name=decoder_name, flat_val=f,
                                     reward=reward, use_amp=use_amp, amp_dtype=amp_dtype,
                                     min_crop_frac=min_crop_frac)
                all_pred.append(p); all_real.append(r)
                for d in delta_grid:
                    if p > d:
                        sweep_capture[d].append(r)

        def _fmt(x):
            m, mean, q1, q3 = _stats(x)
            return f'med={m:.4f} mean={mean:.4f} p25={q1:.4f} p75={q3:.4f}'
        log_line(f'[zh-eval] {source:>8} (n={len(pol)} zoomed={n_zoom}/{len(pol)} '
                 f'regions/img={n_reg / max(1, len(pol)):.2f})')
        log_line(f'[zh-eval]   policy  {_fmt(pol)}')
        log_line(f'[zh-eval]   flat    {_fmt(flat)}')
        log_line(f'[zh-eval]   attn    {_fmt(attn)}')
        log_line(f'[zh-eval]   Δ vs flat={_stats(pol)[1] - _stats(flat)[1]:+.4f}  '
                 f'vs attn={_stats(pol)[1] - _stats(attn)[1]:+.4f}')
        overall.extend(pol)

    # calibration (does predicted advantage track realized?)
    if len(all_pred) >= 3 and np.std(all_pred) > 1e-9 and np.std(all_real) > 1e-9:
        corr = float(np.corrcoef(all_pred, all_real)[0, 1])
    else:
        corr = float('nan')
    log_line(f'[zh-eval] calibration: pred↔realized advantage corr={corr:.3f} '
             f'(n_regions={len(all_pred)})')

    # δ-sweep: mean captured realized advantage (disjoint-region additive proxy)
    sweep = '  '.join(
        f'δ={d:+.2f}:{(np.sum(v) / max(1, len(overall))):+.4f}(n={len(v)})'
        for d, v in sweep_capture.items()
    )
    log_line(f'[zh-eval] δ-sweep mean captured-advantage/img:  {sweep}')

    om, omean, o25, o75 = _stats(overall)
    log_line(f'[zh-eval] epoch={epoch} OVERALL policy: med={om:.4f} mean={omean:.4f} '
             f'p25={o25:.4f} p75={o75:.4f} (n={len(overall)})  operating δ={delta:+.2f}')
    return om
