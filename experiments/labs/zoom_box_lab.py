"""experiments.labs.zoom_box_lab — dense per-patch box head trained as a contextual
bandit (AWR / search-and-distill).  Implements docs/zoom_box_spec.md.

Per item, the FROZEN detector's features are cheap and the reward is a deterministic,
queryable function — so this is an offline contextual bandit, not RL:

    frozen forward → feats=[z|attn|patch_logit]
      ZoomBoxHead:  feats → per-patch (box distances, confidence)
        box  → fractional bbox anchored at the patch centre (FCOS)
        conf → predicted zoom-ADVANTAGE of that box
    reward(box) = m(zoom→box) − baseline                     [run_bbox_zoom, frozen]
    train (bandit): per proposing patch, jitter the box → score each candidate's
        advantage → advantage-weighted-regress the box toward the winners (AWR);
        regress confidence toward the greedy box's realized advantage.
    decode: gate region iff conf > δ → NMS by conf (least overlap) → zoom union;
        else fall back to attention-zoom (a FIXED, GT-free default — NOT an oracle).

Baseline = the incumbent **attention-zoom** (deployable): the gate means "zoom only where
the learned box beats attn", and the system is a cascade over attn.  Early bandit epochs
use baseline=flat (curriculum) for positive cold-start signal.  max(flat,attn) is reported
ONLY as an oracle ceiling (per-image GT-selected) — never deployed.  A short SUPERVISED
warm-start (GT-component boxes) seeds the box head before the bandit.
"""

from __future__ import annotations

import dataclasses
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution
from lab_utils.eval.fetch import ModelInfo, model_info
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.zoom import (
    BBox, gt_grid_mask, all_component_bboxes, grid_bbox_to_frac, pad_grid_bbox,
    bbox_is_trivial,
)
from lab_utils.eval.multibox import _area, _inter_area
from lab_utils.logging.text import log_line

from experiments.labs.attention_zoom import run_bbox_zoom, attention_zoom_single
from experiments.labs.box_policy_zoom import build_policy_input
from lab_utils.model.zoom_box_head import (
    patch_centers, boxes_from_distances,
)

_TRAIN_SOURCES = ('casia', 'sagid')
_EVAL_SOURCES = ('imd2020', 'sagid')


# ── geometry helpers (fractional boxes) ──────────────────────────────────────────

def _iou(a: BBox, b: BBox) -> float:
    inter = _inter_area(a, b)
    if inter <= 0.0:
        return 0.0
    return inter / (_area(a) + _area(b) - inter + 1e-9)


def _nms(boxes: List[BBox], scores: List[float], *, iou_thresh: float,
         max_keep: int, min_crop_frac: float) -> List[int]:
    """Greedy NMS by descending score; suppress boxes overlapping a kept one and drop
    trivial (~whole-frame) boxes.  Returns kept indices into ``boxes``."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    kept: List[int] = []
    for i in order:
        b = boxes[i]
        if bbox_is_trivial(b, min_crop_frac=min_crop_frac):
            continue
        if any(_iou(b, boxes[j]) > iou_thresh for j in kept):
            continue
        kept.append(i)
        if len(kept) >= max_keep:
            break
    return kept


def _reward_val(rec, reward: str) -> float:
    return float(rec.iou if reward == 'iou' else rec.f1)


def _box_tuple(box_t: torch.Tensor) -> BBox:
    return tuple(float(x) for x in box_t.detach().cpu().tolist())          # type: ignore


# ── baseline (max of flat decode and attention-zoom) ─────────────────────────────

@torch.no_grad()
def baseline_scores(
    model, item: Item, res: Resolution, *, device, decode_fn, decoder_name: str,
    reward: str, use_amp: bool, amp_dtype: str,
    flat_cache: dict, attn_cache: dict, ref: str = 'attn',
) -> Tuple[float, float, float]:
    """(flat, attn, baseline) localization scores for one item.  ``ref`` picks the
    baseline the advantage is measured against:
      'attn' (default, DEPLOYABLE) — the incumbent attention-zoom (fixed, GT-free).
      'flat'                        — no-zoom (the early-bandit curriculum: gives POSITIVE
                                      advantage on zoom-favorable images to climb toward).
      'max'  (ORACLE — do NOT deploy) — per-image max(flat,attn); needs GT to select, so
                                      it is a ceiling reference only, never a fallback.
    Both references are static on a frozen detector ⇒ cached across epochs."""
    iid = item.item_id
    f = flat_cache.get(iid)
    if f is None:
        img_t = load_image_tensor(item, res, device=device)
        info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
        f = _reward_val(eval_metric(decode_fn(info), info, item, decoder=decoder_name), reward)
        flat_cache[iid] = f
    a = attn_cache.get(iid)
    if a is None:
        a = _reward_val(attention_zoom_single(
            model, item, res, device=device, decoder=decoder_name,
            use_amp=use_amp, amp_dtype=amp_dtype), reward)
        attn_cache[iid] = a
    base = {'attn': a, 'flat': f, 'max': max(f, a)}[ref]
    return f, a, base


# ── reward: realized advantage of zooming ONE box over the baseline ──────────────

@torch.no_grad()
def box_advantage(
    model, img_pil, info: ModelInfo, item: Item, res: Resolution, bbox: BBox, *, device,
    decode_fn, decoder_name: str, baseline: float, flat_val: float, reward: str = 'f1',
    use_amp: bool = False, amp_dtype: str = 'float16', min_crop_frac: float = 0.25,
) -> float:
    """A(box) = m(zoom→box) − baseline.  A trivial / failed zoom collapses to the flat
    decode, so its advantage is flat − baseline (≤ 0 when attn-zoom is the baseline)."""
    if bbox is None or bbox_is_trivial(bbox, min_crop_frac=min_crop_frac):
        return float(flat_val) - float(baseline)
    union, _ = run_bbox_zoom(
        model, img_pil, [bbox], res, device=device, decode_fn=decode_fn,
        use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
    )
    if union is None:
        return float(flat_val) - float(baseline)
    rec = eval_metric(union, info, item, decoder=decoder_name)
    return _reward_val(rec, reward) - float(baseline)


@torch.no_grad()
def _score_boxes(
    model, img_pil, info, item, res, boxes: List[BBox], *, device, decode_fn,
    decoder_name, baseline, flat_val, reward, use_amp, amp_dtype, min_crop_frac,
) -> List[float]:
    """Advantage of each box, deduplicated (consensus ⇒ many near-identical boxes) by
    rounding to a 0.02-frac grid so each distinct crop is scored once."""
    uniq: Dict[tuple, float] = {}
    keys: List[tuple] = []
    for b in boxes:
        k = (round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2))
        keys.append(k)
        if k not in uniq:
            uniq[k] = 0.0
    for k in list(uniq.keys()):
        uniq[k] = box_advantage(
            model, img_pil, info, item, res, tuple(k), device=device, decode_fn=decode_fn,
            decoder_name=decoder_name, baseline=baseline, flat_val=flat_val, reward=reward,
            use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
        )
    return [uniq[k] for k in keys]


# ── training step ────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class ZoomBoxStats:
    loss: float
    box_loss: float
    conf_loss: float
    n_propose: int
    mean_adv: float          # mean greedy-box advantage over proposing patches
    best_adv: float          # mean over patches of the best candidate advantage
    pos_frac: float          # fraction of proposing patches whose greedy adv > 0
    mean_box_area: float     # mean fractional area of the greedy boxes
    phase: str


def _feats_tensor(info: ModelInfo, *, use_attn: bool, use_patch_logit: bool, device) -> torch.Tensor:
    return torch.from_numpy(
        build_policy_input(info, use_attn=use_attn, use_patch_logit=use_patch_logit).astype(np.float32)
    ).to(device)


def _warmstart_item(
    model, head, item: Item, res: Resolution, *, device, decode_fn, decoder_name: str,
    use_attn: bool, use_patch_logit: bool, patch_frac: float, pad_min_patches: int,
    huber_beta: float, conf_warm: float, use_amp: bool, amp_dtype: str,
) -> Optional[Tuple[torch.Tensor, ZoomBoxStats]]:
    """Phase 0 (supervised): for patches inside a GT component, regress the box toward
    that component's (lightly padded) frac box and the confidence toward 1; elsewhere
    confidence toward 0.  Seeds the box head into a sensible basin for the bandit."""
    img_t = load_image_tensor(item, res, device=device)
    with torch.no_grad():
        info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    if info.embeddings is None:
        return None
    feats_t = _feats_tensor(info, use_attn=use_attn, use_patch_logit=use_patch_logit, device=device)
    grid_hw = info.grid_hw
    n_rows, n_cols = grid_hw
    centers = patch_centers(grid_hw, device=device)
    dist, conf = head(feats_t)                                  # (N,4),(N,)
    N = feats_t.shape[0]

    rec0 = eval_metric(np.zeros(grid_hw, dtype=bool), info, item, decoder=decoder_name)
    gt_grid = gt_grid_mask(rec0.gt_mask, grid_hw, patch_frac=patch_frac)
    comps = all_component_bboxes(gt_grid)

    inside = np.zeros(N, dtype=bool)
    tboxes = np.zeros((N, 4), dtype=np.float32)
    for gb, _size in comps:
        fb = grid_bbox_to_frac(pad_grid_bbox(gb, grid_hw, 0.0, pad_min_patches=pad_min_patches), grid_hw)
        r0, c0, r1, c1 = gb
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if gt_grid[r, c]:
                    p = r * n_cols + c
                    inside[p] = True
                    tboxes[p] = fb

    inside_t = torch.from_numpy(inside).to(device)
    if bool(inside_t.any()):
        tb = torch.from_numpy(tboxes).to(device)
        cy, cx = centers[:, 0], centers[:, 1]
        tdist = torch.stack([
            (cy - tb[:, 0]).clamp(min=0.0), (cx - tb[:, 1]).clamp(min=0.0),
            (tb[:, 2] - cy).clamp(min=0.0), (tb[:, 3] - cx).clamp(min=0.0),
        ], dim=-1)
        box_loss = F.smooth_l1_loss(dist[inside_t], tdist[inside_t].detach(), beta=huber_beta)
    else:
        box_loss = dist.sum() * 0.0
    # conf warm-start: regress toward a SMALL advantage-scale target (±conf_warm), NOT
    # BCE — BCE drives logits to ±5 and the bandit hand-off (target → realized advantage
    # ~±0.3) would then spike the conf loss and (via grad) corrupt the box trunk.
    conf_target = torch.where(inside_t, torch.full_like(conf, conf_warm),
                              torch.full_like(conf, -conf_warm))
    conf_loss = F.smooth_l1_loss(conf, conf_target, beta=huber_beta)
    loss = box_loss + conf_loss

    areas = []
    if comps:
        for gb, _ in comps:
            fb = grid_bbox_to_frac(gb, grid_hw)
            areas.append(_area(fb))
    stats = ZoomBoxStats(
        loss=float(loss.detach()), box_loss=float(box_loss.detach()),
        conf_loss=float(conf_loss.detach()), n_propose=int(inside.sum()),
        mean_adv=0.0, best_adv=0.0, pos_frac=float(len(comps) > 0),
        mean_box_area=float(np.mean(areas)) if areas else 0.0, phase='warmstart',
    )
    return loss, stats


def _bandit_item(
    model, head, item: Item, res: Resolution, *, device, decode_fn, decoder_name: str,
    baseline: float, flat_val: float, use_attn: bool, use_patch_logit: bool,
    n_propose: int, n_candidates: int, n_background: int, sigma: float, awr_temp: float,
    reward: str, min_crop_frac: float, huber_beta: float, lambda_box: float,
    lambda_conf: float, use_amp: bool, amp_dtype: str, rng: random.Random,
) -> Optional[Tuple[torch.Tensor, ZoomBoxStats]]:
    """One bandit step: propose boxes from high-prior patches, jitter each into K
    candidates (exploration), score every candidate's frozen advantage, advantage-weight
    regress the box toward the winners (AWR), and regress confidence toward the greedy
    box's realized advantage.  Backbone + scoring are no-grad; only the head sees grad."""
    img_t, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    with torch.no_grad():
        info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    if info.embeddings is None:
        return None
    feats_t = _feats_tensor(info, use_attn=use_attn, use_patch_logit=use_patch_logit, device=device)
    grid_hw = info.grid_hw
    centers = patch_centers(grid_hw, device=device)
    N = feats_t.shape[0]

    raw, conf = head.box_logits(feats_t)                        # (N,4) pre-softplus, (N,)
    mu = head.distances(raw)                                    # (N,4) frac distances (floored)

    # proposing set: top patches by manipulation prior (or confidence) + a few background
    if info.patch_logits is not None:
        prior = torch.sigmoid(torch.from_numpy(np.asarray(info.patch_logits, dtype=np.float32)).to(device))
    else:
        prior = conf.detach()
    k_prop = int(min(n_propose, N))
    prop_idx = torch.topk(prior, k_prop).indices.tolist()
    bg_idx: List[int] = []
    if n_background > 0:
        pool = list(set(range(N)) - set(prop_idx))
        rng.shuffle(pool)
        bg_idx = pool[:n_background]

    # candidate distances (greedy + jittered in pre-softplus space) and their frac boxes
    cand_dist: Dict[int, List[torch.Tensor]] = {}
    cand_box: Dict[int, List[BBox]] = {}
    for p in prop_idx:
        ds = [mu[p]]
        for _ in range(n_candidates):
            eps = torch.randn(4, device=device)
            ds.append(head.distances(raw[p] + sigma * eps))
        cand_dist[p] = ds
        cand_box[p] = [_box_tuple(boxes_from_distances(d.unsqueeze(0), centers[p:p + 1])[0]) for d in ds]
    bg_box: Dict[int, BBox] = {
        p: _box_tuple(boxes_from_distances(mu[p].unsqueeze(0), centers[p:p + 1])[0]) for p in bg_idx
    }

    # score every candidate (deduped)
    flat_boxes: List[BBox] = []
    index: List[tuple] = []
    for p in prop_idx:
        for j, b in enumerate(cand_box[p]):
            flat_boxes.append(b)
            index.append((p, j))
    for p in bg_idx:
        flat_boxes.append(bg_box[p])
        index.append((p, 'bg'))
    advs = _score_boxes(
        model, img_pil, info, item, res, flat_boxes, device=device, decode_fn=decode_fn,
        decoder_name=decoder_name, baseline=baseline, flat_val=flat_val, reward=reward,
        use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
    )
    adv_map = {key: a for key, a in zip(index, advs)}

    # AWR box loss + confidence regression
    box_terms: List[torch.Tensor] = []
    conf_terms: List[torch.Tensor] = []
    greedy_advs: List[float] = []
    best_advs: List[float] = []
    box_areas: List[float] = []
    for p in prop_idx:
        cand_a = [adv_map[(p, j)] for j in range(len(cand_dist[p]))]
        a_t = torch.tensor(cand_a, device=device, dtype=torch.float32)
        w = torch.softmax(a_t / max(awr_temp, 1e-6), dim=0).detach()
        bl = mu[p].sum() * 0.0
        for j, d in enumerate(cand_dist[p]):
            bl = bl + w[j] * F.smooth_l1_loss(mu[p], d.detach(), beta=huber_beta)
        box_terms.append(bl)
        ga = float(adv_map[(p, 0)])
        greedy_advs.append(ga)
        best_advs.append(float(max(cand_a)))
        box_areas.append(_area(cand_box[p][0]))
        conf_terms.append(F.smooth_l1_loss(
            conf[p], torch.tensor(ga, device=device), beta=huber_beta))
    for p in bg_idx:
        conf_terms.append(F.smooth_l1_loss(
            conf[p], torch.tensor(float(adv_map[(p, 'bg')]), device=device), beta=huber_beta))

    box_loss = torch.stack(box_terms).mean() if box_terms else mu.sum() * 0.0
    conf_loss = torch.stack(conf_terms).mean() if conf_terms else conf.sum() * 0.0
    loss = lambda_box * box_loss + lambda_conf * conf_loss

    stats = ZoomBoxStats(
        loss=float(loss.detach()), box_loss=float(box_loss.detach()),
        conf_loss=float(conf_loss.detach()), n_propose=len(prop_idx),
        mean_adv=float(np.mean(greedy_advs)) if greedy_advs else 0.0,
        best_adv=float(np.mean(best_advs)) if best_advs else 0.0,
        pos_frac=float(np.mean([a > 0 for a in greedy_advs])) if greedy_advs else 0.0,
        mean_box_area=float(np.mean(box_areas)) if box_areas else 0.0, phase='bandit',
    )
    return loss, stats


def zoom_box_train_item(
    model, head, item: Item, res: Resolution, *, phase: str, device, decode_fn,
    decoder_name: str, flat_cache: dict, attn_cache: dict, use_attn: bool = True,
    use_patch_logit: bool = True, patch_frac: float = 0.25, pad_min_patches: int = 1,
    n_propose: int = 6, n_candidates: int = 5, n_background: int = 4, sigma: float = 0.5,
    awr_temp: float = 0.05, reward: str = 'f1', min_crop_frac: float = 0.25,
    huber_beta: float = 0.1, lambda_box: float = 1.0, lambda_conf: float = 1.0,
    conf_warm: float = 0.2, ref: str = 'attn',
    use_amp: bool = False, amp_dtype: str = 'float16', rng: Optional[random.Random] = None,
) -> Optional[Tuple[torch.Tensor, ZoomBoxStats]]:
    """Dispatch a single training item to the warm-start (supervised) or bandit step."""
    if phase == 'warmstart':
        return _warmstart_item(
            model, head, item, res, device=device, decode_fn=decode_fn,
            decoder_name=decoder_name, use_attn=use_attn, use_patch_logit=use_patch_logit,
            patch_frac=patch_frac, pad_min_patches=pad_min_patches, huber_beta=huber_beta,
            conf_warm=conf_warm, use_amp=use_amp, amp_dtype=amp_dtype,
        )
    _f, _a, baseline = baseline_scores(
        model, item, res, device=device, decode_fn=decode_fn, decoder_name=decoder_name,
        reward=reward, use_amp=use_amp, amp_dtype=amp_dtype,
        flat_cache=flat_cache, attn_cache=attn_cache, ref=ref,
    )
    return _bandit_item(
        model, head, item, res, device=device, decode_fn=decode_fn, decoder_name=decoder_name,
        baseline=baseline, flat_val=_f, use_attn=use_attn, use_patch_logit=use_patch_logit,
        n_propose=n_propose, n_candidates=n_candidates, n_background=n_background, sigma=sigma,
        awr_temp=awr_temp, reward=reward, min_crop_frac=min_crop_frac, huber_beta=huber_beta,
        lambda_box=lambda_box, lambda_conf=lambda_conf, use_amp=use_amp, amp_dtype=amp_dtype,
        rng=rng or random.Random(0),
    )


# ── deterministic inference (gate by conf > δ → NMS → zoom union) ────────────────

@torch.no_grad()
def _head_boxes(head, info: ModelInfo, *, use_attn, use_patch_logit, device):
    feats_t = _feats_tensor(info, use_attn=use_attn, use_patch_logit=use_patch_logit, device=device)
    dist, conf = head(feats_t)
    centers = patch_centers(info.grid_hw, device=device)
    boxes = boxes_from_distances(dist, centers).detach().cpu().numpy()
    return boxes, conf.detach().cpu().numpy()


@torch.no_grad()
def zoom_box_single(
    model, head, item: Item, res: Resolution, *, device, decode_fn, decoder_name: str = 'kmeans',
    delta: float = 0.0, iou_thresh: float = 0.5, max_boxes: int = 4, use_attn: bool = True,
    use_patch_logit: bool = True, min_crop_frac: float = 0.25, use_amp: bool = False,
    amp_dtype: str = 'float16', return_debug: bool = False,
):
    """Gate (conf > δ) → NMS by confidence (least overlap) → zoom union → score.
    Flat-decode fallback when no box clears the gate (the caller substitutes the
    max(flat, attn) baseline for the reported policy score)."""
    label = f'{decoder_name}_zoombox'
    img_t, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    flat_mask = decode_fn(info)
    debug = {'zoomed': False, 'n_gated': 0, 'n_kept': 0, 'kept_boxes': []}

    if info.embeddings is None:
        rec = eval_metric(flat_mask, info, item, decoder=label)
        return (rec, debug) if return_debug else rec

    boxes, conf = _head_boxes(head, info, use_attn=use_attn, use_patch_logit=use_patch_logit, device=device)
    gated = [i for i in range(boxes.shape[0]) if float(conf[i]) > float(delta)]
    box_list = [tuple(float(x) for x in boxes[i]) for i in gated]
    score_list = [float(conf[i]) for i in gated]
    keep = _nms(box_list, score_list, iou_thresh=iou_thresh, max_keep=max_boxes, min_crop_frac=min_crop_frac)
    kept_boxes = [box_list[k] for k in keep]
    debug.update({'n_gated': len(gated), 'n_kept': len(kept_boxes), 'kept_boxes': kept_boxes})

    if not kept_boxes:
        rec = eval_metric(flat_mask, info, item, decoder=label)
        return (rec, debug) if return_debug else rec

    union, _ = run_bbox_zoom(
        model, img_pil, kept_boxes, res, device=device, decode_fn=decode_fn,
        use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
    )
    pred = union if union is not None else flat_mask
    debug['zoomed'] = union is not None
    rec = eval_metric(pred, info, item, decoder=label)
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
def _eval_item(
    model, head, item: Item, res: Resolution, *, device, decode_fn, decoder_name,
    delta: float, delta_grid, iou_thresh, max_boxes, baseline: float, flat_val: float,
    use_attn, use_patch_logit, min_crop_frac, reward, use_amp, amp_dtype,
):
    """One eval item: policy F1 at the operating δ, plus (conf, realized-advantage) for
    the NMS survivors at the MOST PERMISSIVE δ (for calibration + the δ-sweep)."""
    label = f'{decoder_name}_zoombox'
    img_t, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    flat_mask = decode_fn(info)
    if info.embeddings is None:
        return baseline, False, 0, []

    boxes, conf = _head_boxes(head, info, use_attn=use_attn, use_patch_logit=use_patch_logit, device=device)
    n = boxes.shape[0]
    delta_min = min(delta_grid)

    # diagnostic survivors at the permissive δ — scored once, reused for the sweep
    perm_gated = [i for i in range(n) if float(conf[i]) > delta_min]
    perm_boxes = [tuple(float(x) for x in boxes[i]) for i in perm_gated]
    perm_scores = [float(conf[i]) for i in perm_gated]
    perm_keep = _nms(perm_boxes, perm_scores, iou_thresh=iou_thresh, max_keep=max_boxes, min_crop_frac=min_crop_frac)
    survivors = []
    for k in perm_keep:
        a = box_advantage(
            model, img_pil, info, item, res, perm_boxes[k], device=device, decode_fn=decode_fn,
            decoder_name=decoder_name, baseline=baseline, flat_val=flat_val, reward=reward,
            use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
        )
        survivors.append((perm_scores[k], a))

    # policy at the operating δ: keep survivors (already NMS'd at the permissive δ) whose
    # confidence clears the operating δ.  survivors is aligned with perm_keep order.
    kept = [perm_boxes[k] for k, (c, _a) in zip(perm_keep, survivors) if c > float(delta)]
    if not kept:
        return baseline, False, 0, survivors
    union, _ = run_bbox_zoom(
        model, img_pil, kept, res, device=device, decode_fn=decode_fn,
        use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
    )
    if union is None:
        return baseline, False, 0, survivors
    pol = _reward_val(eval_metric(union, info, item, decoder=label), reward)
    return pol, True, len(kept), survivors


@torch.no_grad()
def evaluate_zoom_box(
    model, head, eval_by_source, res, *, device, decode_fn, decoder_name,
    delta: float, iou_thresh: float, max_boxes: int, flat_cache: dict, attn_cache: dict,
    use_attn: bool = True, use_patch_logit: bool = True, min_crop_frac: float = 0.25,
    reward: str = 'f1', use_amp: bool = False, amp_dtype: str = 'float16',
    delta_grid=(-0.05, 0.0, 0.02, 0.05, 0.10, 0.15, 0.20), epoch: int = 0,
) -> float:
    """Per-source DEPLOYABLE policy (learned zoom where gated, else attention-zoom) vs
    flat / attn references, plus a clearly-labeled ORACLE ceiling = per-image max(flat,attn)
    (needs GT to select — a reference only, never deployed).  Confidence↔realized-advantage
    calibration and a δ-sweep.  Returns mean captured advantage over ATTN (policy − attn),
    the deployable quantity — NOT over the oracle ceiling."""
    head.eval()
    captured: List[float] = []
    all_conf: List[float] = []
    all_adv: List[float] = []
    sweep = {d: [] for d in delta_grid}

    for source, items in eval_by_source.items():
        pol, flat, attn, oracle = [], [], [], []
        n_zoom, n_kept = 0, 0
        for item in items:
            # DEPLOYABLE baseline = attn (fixed, GT-free); the fallback is attn, not an
            # oracle max(flat,attn).  Advantages are measured vs attn for honest calibration.
            f, a, _ = baseline_scores(
                model, item, res, device=device, decode_fn=decode_fn, decoder_name=decoder_name,
                reward=reward, use_amp=use_amp, amp_dtype=amp_dtype,
                flat_cache=flat_cache, attn_cache=attn_cache, ref='attn',
            )
            pf, zoomed, nk, survivors = _eval_item(
                model, head, item, res, device=device, decode_fn=decode_fn, decoder_name=decoder_name,
                delta=delta, delta_grid=delta_grid, iou_thresh=iou_thresh, max_boxes=max_boxes,
                baseline=a, flat_val=f, use_attn=use_attn, use_patch_logit=use_patch_logit,
                min_crop_frac=min_crop_frac, reward=reward, use_amp=use_amp, amp_dtype=amp_dtype,
            )
            pol.append(pf); flat.append(f); attn.append(a); oracle.append(max(f, a))
            n_zoom += int(zoomed); n_kept += nk
            for (c, adv) in survivors:
                all_conf.append(c); all_adv.append(adv)
                for d in delta_grid:
                    if c > d:
                        sweep[d].append(adv)
            captured.append(pf - a)

        def _fmt(x):
            m, mean, q1, q3 = _stats(x)
            return f'med={m:.4f} mean={mean:.4f} p25={q1:.4f} p75={q3:.4f}'
        log_line(f'[zb-eval] {source:>8} (n={len(pol)} zoomed={n_zoom}/{len(pol)} '
                 f'kept/img={n_kept / max(1, len(pol)):.2f})')
        log_line(f'[zb-eval]   policy       {_fmt(pol)}')
        log_line(f'[zb-eval]   flat         {_fmt(flat)}')
        log_line(f'[zb-eval]   attn         {_fmt(attn)}')
        log_line(f'[zb-eval]   oracle-ceil  {_fmt(oracle)}  (max(flat,attn); GT-selected, not deployable)')
        log_line(f'[zb-eval]   Δ vs attn={_stats(pol)[1] - _stats(attn)[1]:+.4f}  '
                 f'vs flat={_stats(pol)[1] - _stats(flat)[1]:+.4f}')

    if len(all_conf) >= 3 and np.std(all_conf) > 1e-9 and np.std(all_adv) > 1e-9:
        corr = float(np.corrcoef(all_conf, all_adv)[0, 1])
    else:
        corr = float('nan')
    log_line(f'[zb-eval] calibration: conf↔realized advantage corr={corr:.3f} '
             f'(n_boxes={len(all_conf)})')

    n_items = max(1, len(captured))
    sweep_s = '  '.join(
        f'δ={d:+.2f}:{(np.sum(v) / n_items):+.4f}(n={len(v)})' for d, v in sweep.items()
    )
    log_line(f'[zb-eval] δ-sweep mean captured-advantage/img:  {sweep_s}')

    cap_mean = float(np.mean(captured)) if captured else 0.0
    cm, cmean, c25, c75 = _stats(captured)
    log_line(f'[zb-eval] epoch={epoch} OVERALL captured-advantage-over-attn: med={cm:+.4f} '
             f'mean={cmean:+.4f} p25={c25:+.4f} p75={c75:+.4f} (n={len(captured)})  '
             f'operating δ={delta:+.2f}')
    return cap_mean
