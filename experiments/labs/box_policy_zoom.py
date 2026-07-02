"""experiments.labs.box_policy_zoom — RL orchestration for the learned box policy.

This composes three frozen pieces (the detector via ``model_info``, the decode
function, and the zoom geometry) with the trainable :class:`BoxPolicy` head:

    full-frame forward (frozen)  →  per-patch input [z | attn | patch_logit]
        →  BoxPolicy.act  →  set of grid-locked boxes
        →  run_bbox_zoom (frozen, on the fly)  →  union mask
        →  metric (GT, train-time only)  →  realized F1 = reward

Training is a one-step bandit (REINFORCE).  The reward is non-differentiable, so
we never backprop through the zoom — only through ``log π(action)``.  The baseline
is the attention-zoom F1 (computed once per item and cached, since the detector +
decoder are frozen and that score is static).

Design compliance:
  * the model is reached ONLY through ``model_info`` / ``run_bbox_zoom`` (I2),
  * GT is touched ONLY inside ``metric`` (I3) — at train time, to form the reward;
    inference forms its prediction with no GT,
  * geometry lives in ``lab_utils.eval.zoom`` (``grid_locked_box`` / place-back).
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution
from lab_utils.eval.fetch import ModelInfo, model_info
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.record import EvalRecord
from lab_utils.eval.zoom import attention_hot_mask, bbox_is_trivial, grid_locked_box
from lab_utils.logging.text import log_line
from lab_utils.model.box_policy import BoxPolicy
from lab_utils.train.distributed import unwrap_model

from experiments.labs.attention_zoom import (
    _resolve_decoder,
    attention_zoom_single,
    run_bbox_zoom,
)

DecodeFn = Callable[[ModelInfo], np.ndarray]


# ── per-patch input ──────────────────────────────────────────────────────────────

def policy_input_dim(info: ModelInfo, *, use_attn: bool = True, use_patch_logit: bool = True) -> int:
    """Width of the per-patch input the policy will read for this signal set."""
    if info.embeddings is None:
        raise ValueError('box_policy: embeddings required (contrastive head not enabled)')
    d = int(info.embeddings.shape[1])
    if use_attn and info.attention is not None:
        d += 1
    if use_patch_logit and info.patch_logits is not None:
        d += 1
    return d


def build_policy_input(
    info: ModelInfo,
    *,
    use_attn: bool = True,
    use_patch_logit: bool = True,
) -> np.ndarray:
    """(N, in_dim) per-patch features = z ⊕ standardized(attn) ⊕ standardized(patch_logit).

    ``z`` is already unit-norm; the scalar channels (a softmax weight, a logit)
    live on very different scales, so we per-image z-score them before
    concatenating — otherwise one channel would dominate the input.
    """
    if info.embeddings is None:
        raise ValueError('box_policy: embeddings required (contrastive head not enabled)')
    n = info.grid_hw[0] * info.grid_hw[1]
    cols = [np.asarray(info.embeddings, dtype=np.float32).reshape(n, -1)]

    def _z(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(n, 1)
        mu, sd = float(x.mean()), float(x.std())
        return (x - mu) / (sd + 1e-6)

    if use_attn and info.attention is not None:
        cols.append(_z(info.attention[:n]))
    if use_patch_logit and info.patch_logits is not None:
        cols.append(_z(info.patch_logits[:n]))
    return np.concatenate(cols, axis=1)


def candidate_mask(
    info: ModelInfo,
    decode_fn: DecodeFn,
    *,
    attn_percentile: float | str = 'otsu',
    use_decode: bool = True,
    cand_cap: int = 64,
) -> np.ndarray:
    """(N,) bool prefilter of plausible box-origin patches (variance control).

    Union of the attention hot-set and the full-frame decode mask — only places a
    box could plausibly help.  Capped at ``cand_cap`` (highest combined score) so
    the keep-sampling action space stays small.  Falls back to the top attention
    patches, then to all patches, so training always has somewhere to explore.
    """
    n = info.grid_hw[0] * info.grid_hw[1]
    score = np.zeros(n, dtype=np.float64)
    hot = np.zeros(n, dtype=bool)

    if info.attention is not None:
        a = np.asarray(info.attention, dtype=np.float64).reshape(-1)[:n]
        score += (a - a.min()) / (a.max() - a.min() + 1e-9)
        hot |= attention_hot_mask(info.attention, info.grid_hw, percentile=attn_percentile).reshape(-1)[:n]
    if use_decode:
        dm = np.asarray(decode_fn(info), dtype=bool).reshape(-1)[:n]
        score += dm.astype(np.float64)
        hot |= dm

    if not hot.any():
        if info.attention is not None:
            k = min(cand_cap, n)
            hot[np.argsort(-score)[:k]] = True
        else:
            hot[:] = True

    if hot.sum() > cand_cap:
        keep = np.argsort(-score)
        keep = keep[hot[keep]][:cand_cap]
        capped = np.zeros(n, dtype=bool)
        capped[keep] = True
        hot = capped
    return hot


def boxes_from_act(kept_idx, sizes, grid_hw) -> List[Tuple[float, float, float, float]]:
    """Convert a policy ActOutput (patch indices + (h, w) extents) to fractional boxes."""
    idx = kept_idx.detach().cpu().numpy().reshape(-1)
    sz = sizes.detach().cpu().numpy().reshape(-1, 2)
    return [grid_locked_box(int(i), float(h), float(w), grid_hw)
            for i, (h, w) in zip(idx, sz)]


# ── reward + baseline ────────────────────────────────────────────────────────────

@torch.no_grad()
def zoom_reward(
    model: torch.nn.Module,
    item: Item,
    img_pil,
    boxes: List[Tuple[float, float, float, float]],
    info1: ModelInfo,
    mask1: np.ndarray,
    res: Resolution,
    *,
    device: torch.device,
    decode_fn: DecodeFn,
    label: str,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
    min_crop_frac: float = 0.25,
) -> Tuple[float, Optional[np.ndarray], List[dict]]:
    """Realized localization F1 of zooming ``boxes`` (the reward), GT touched here only.

    Falls back to the unzoomed pass-1 decode when no box survives — so "don't
    zoom" (one near-full-frame box, or no box) is a representable, scored action.
    Returns (f1, union_mask_or_None, per_box).
    """
    union, per_box = (None, [])
    if boxes:
        union, per_box = run_bbox_zoom(
            model, img_pil, boxes, res, device=device, decode_fn=decode_fn,
            use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
        )
    pred = union if union is not None else mask1
    rec = eval_metric(pred, info1, item, decoder=label)
    return float(rec.f1), union, per_box


@torch.no_grad()
def attention_baseline_f1(
    model: torch.nn.Module,
    item: Item,
    res: Resolution,
    *,
    device: torch.device,
    decode_fn,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
) -> float:
    """Attention-zoom F1 for an item — the REINFORCE baseline (static ⇒ cacheable)."""
    rec = attention_zoom_single(
        model, item, res, device=device, use_amp=use_amp, amp_dtype=amp_dtype,
        decoder=decode_fn,
    )
    return float(rec.f1)


# ── one training item (REINFORCE) ────────────────────────────────────────────────

@dataclasses.dataclass
class StepStats:
    reward:    float
    baseline:  float
    advantage: float
    n_boxes:   int     # executed (post-cap)
    n_sampled: float   # proposed (pre-cap) — what the per-proposal cost charges
    n_cand:    int
    loss:      float


def policy_train_item(
    model: torch.nn.Module,
    policy: BoxPolicy,
    item: Item,
    res: Resolution,
    *,
    device: torch.device,
    decode_fn: DecodeFn,
    decoder_name: str,
    baselines: Dict[str, float],
    baseline_mode: str = 'flat',
    flat_cache: Optional[Dict[str, float]] = None,
    use_attn: bool = True,
    use_patch_logit: bool = True,
    attn_percentile: float | str = 'otsu',
    cand_cap: int = 64,
    max_boxes: int = 8,
    entropy_beta: float = 0.01,
    box_cost: float = 0.01,
    credit_mode: str = 'per_box',
    min_crop_frac: float = 0.25,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
) -> Optional[Tuple[torch.Tensor, StepStats]]:
    """One REINFORCE step body for a single item.

    Returns (loss_tensor, stats), or None when the item is unusable (no embeddings).
    The caller accumulates loss across a mini-batch and steps the optimizer.

    ``credit_mode`` decides how each box's score-function term is weighted:
      'per_box' — split signal per head.  keep ← (advantage + m_i − box_cost),
                  where advantage = F1(union) − F1(flat) is the GLOBAL "was zooming
                  worth it vs not zooming?" verdict (applied to EXECUTED boxes only,
                  so keep=0 candidates get NO term — no negative-advantage smear, no
                  runaway `sampled`) and m_i = F1(union) − F1(union without i) prunes
                  redundancy among placed boxes.  size ← m_i (fixes uniform sizes).
                  Without the advantage term the loss only compares the union to
                  union-minus-one-box, so a box set collectively WORSE than flat
                  still looks locally fine and the policy never escapes.  The
                  leave-one-out F1s reuse the per-box masks run_bbox_zoom already
                  computed — no extra model forwards.
      'global'  — vanilla: one advantage × Σ log π.  Cheap but smears the keep=0
                  majority and gives the size head no per-box signal.

    ``baseline_mode`` chooses the REINFORCE baseline (centers the advantage —
    affects variance, NOT the objective; the policy maximizes E[F1] either way):
      'flat'      — the no-zoom decode F1 (the policy's own "do nothing" floor).
                    Cheapest (mask1 is already decoded) and best-centered early,
                    when the policy sits near flat.  Measures "did zooming beat
                    not zooming?".  Cached in ``flat_cache`` (static per item).
      'attn_zoom' — the attention-zoom F1 (2 extra forwards/item, cached in
                    ``baselines``).  A higher bar; advantage runs more negative.

    ``box_cost`` (λ) is a per-proposal cost charged on the PRE-cap sampled count
    (minus one — the baseline already spends one box), so a box must earn ≥ λ F1
    to be worth proposing.  Charging the sampled (not executed) count is what makes
    capped-out boxes non-free: without it, nothing penalizes proposing more than
    ``max_boxes`` and the policy ratchets toward saturation.
    """
    label = f'{decoder_name}_boxpolicy'

    img_t, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    with torch.no_grad():
        info1 = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    if info1.embeddings is None:
        return None

    feats = torch.from_numpy(
        build_policy_input(info1, use_attn=use_attn, use_patch_logit=use_patch_logit)
    ).float().to(device)
    cand = candidate_mask(info1, decode_fn, attn_percentile=attn_percentile, cand_cap=cand_cap)
    cand_t = torch.from_numpy(cand).to(device)

    act = policy.act(feats, candidate_mask=cand_t, max_boxes=max_boxes, deterministic=False)
    boxes = boxes_from_act(act.kept_idx, act.sizes, info1.grid_hw)
    mask1 = decode_fn(info1)

    # Zoom every box once; align the placed masks back to box order (None for a
    # trivial/whole-frame box that run_bbox_zoom skips) so each executed box maps
    # to its own placed mask for the leave-one-out credit below.
    with torch.no_grad():
        union, per_box = run_bbox_zoom(
            model, img_pil, boxes, res, device=device, decode_fn=decode_fn,
            use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
        )
    masks_aligned: List[Optional[np.ndarray]] = []
    _pb = iter(per_box)
    for b in boxes:
        if b is None or bbox_is_trivial(b, min_crop_frac=min_crop_frac):
            masks_aligned.append(None)
        else:
            masks_aligned.append(next(_pb)['mask_px'])

    pred_union = union if union is not None else mask1
    reward = float(eval_metric(pred_union, info1, item, decoder=label).f1)

    # Baseline (per item, action-independent ⇒ unbiased; only centers variance).
    if baseline_mode == 'attn_zoom':
        base = baselines.get(item.item_id)
        if base is None:
            base = attention_baseline_f1(
                model, item, res, device=device, decode_fn=decode_fn,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            baselines[item.item_id] = base
    elif baseline_mode == 'flat':
        base = None if flat_cache is None else flat_cache.get(item.item_id)
        if base is None:
            base = float(eval_metric(mask1, info1, item, decoder=label).f1)
            if flat_cache is not None:
                flat_cache[item.item_id] = base
    else:
        raise ValueError(f"policy_train_item: unknown baseline_mode {baseline_mode!r} "
                         "(flat|attn_zoom)")

    if credit_mode == 'per_box':
        # Two signals, one per head:
        #   keep ← (advantage + m_i − box_cost): the GLOBAL advantage = F1(union) −
        #          F1(flat) says "was zooming worth it at all?" (the comparison we
        #          actually care about — full prediction vs zoom).  Without it the
        #          loss only ever compares the union to union-minus-one-box, so a
        #          set of redundant boxes that is collectively WORSE than flat still
        #          looks locally fine (every m_i ≈ 0) and the policy never escapes.
        #          advantage is applied to EXECUTED boxes only, so keep=0 candidates
        #          still get NO term (no negative-advantage smear → no inflation).
        #          The per-box m_i rides on top to still prune redundancy: among
        #          placed boxes the useful ones (m_i > 0) get more lift than the
        #          redundant ones (m_i ≈ 0, killed by box_cost).
        #   size ← m_i: the box's own leave-one-out marginal (fixes uniform sizes).
        advantage = reward - base
        K = len(boxes)
        valid = [j for j in range(K) if masks_aligned[j] is not None]
        loss = act.entropy.new_zeros(())
        for i in range(K):
            pos = act.kept_pos[i]
            if masks_aligned[i] is not None:
                others = [masks_aligned[j] for j in valid if j != i]
                u_wo = np.logical_or.reduce(others) if others else None
                pred_wo = u_wo if u_wo is not None else mask1
                f1_wo = float(eval_metric(pred_wo, info1, item, decoder=label).f1)
                m_i = reward - f1_wo
            else:
                m_i = 0.0   # trivial box contributed nothing
            loss = loss - (advantage + m_i - box_cost) * act.keep_lp[pos] - m_i * act.size_lp[pos]
        # Capped-out proposals (sampled keep=1, dropped by the cap): pay box_cost,
        # so over-proposing past max_boxes is still penalized.
        executed = torch.zeros_like(act.keep_sampled)
        if act.kept_pos.numel() > 0:
            executed[act.kept_pos] = True
        capped = act.keep_sampled & (~executed)
        if bool(capped.any()):
            loss = loss + box_cost * act.keep_lp[capped].sum()
        loss = loss - entropy_beta * act.entropy
    elif credit_mode == 'global':
        cost = box_cost * max(0.0, act.n_sampled - 1.0)
        advantage = (reward - base) - cost
        log_prob = act.keep_lp.sum() + act.size_lp[act.kept_pos].sum()
        loss = -advantage * log_prob - entropy_beta * act.entropy
    else:
        raise ValueError(f"policy_train_item: unknown credit_mode {credit_mode!r} "
                         "(per_box|global)")

    stats = StepStats(reward=reward, baseline=base, advantage=float(advantage),
                      n_boxes=len(boxes), n_sampled=act.n_sampled,
                      n_cand=int(cand.sum()), loss=float(loss.detach()))
    return loss, stats


# ── deterministic inference (eval) ───────────────────────────────────────────────

@torch.no_grad()
def box_policy_single(
    model: torch.nn.Module,
    policy: BoxPolicy,
    item: Item,
    res: Resolution,
    *,
    device: torch.device,
    decode_fn: DecodeFn,
    decoder_name: str = 'kmeans',
    use_attn: bool = True,
    use_patch_logit: bool = True,
    attn_percentile: float | str = 'otsu',
    cand_cap: int = 64,
    max_boxes: int = 8,
    min_crop_frac: float = 0.25,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
    return_debug: bool = False,
):
    """Deterministic box-policy zoom for one item (GT-free prediction).

    Emit the ``round(Σ keep_prob)`` most-confident boxes (cap ``max_boxes``), size
    = the policy mean, centers grid-locked.  Falls back to the unzoomed decode
    when the policy emits no box.
    """
    label = f'{decoder_name}_boxpolicy'
    policy.eval()

    img_t, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
    info1 = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    mask1 = decode_fn(info1)

    debug = {'boxes': [], 'mask_full': mask1, 'mask_zoom': None, 'grid_hw': info1.grid_hw,
             'zoomed': False, 'per_box': [], 'attn1': info1.attention, 'img_pil': img_pil,
             'keep_prob': None, 'candidates': None}

    if info1.embeddings is None:
        rec = eval_metric(mask1, info1, item, decoder=label)
        return (rec, debug) if return_debug else rec

    feats = torch.from_numpy(
        build_policy_input(info1, use_attn=use_attn, use_patch_logit=use_patch_logit)
    ).float().to(device)
    cand = candidate_mask(info1, decode_fn, attn_percentile=attn_percentile, cand_cap=cand_cap)
    cand_t = torch.from_numpy(cand).to(device)

    act = policy.act(feats, candidate_mask=cand_t, max_boxes=max_boxes,
                     deterministic=True)
    boxes = boxes_from_act(act.kept_idx, act.sizes, info1.grid_hw)
    debug['boxes'] = boxes
    debug['keep_prob'] = act.keep_prob.detach().cpu().numpy()
    debug['candidates'] = cand

    if not boxes:
        rec = eval_metric(mask1, info1, item, decoder=label)
        return (rec, debug) if return_debug else rec

    union, per_box = run_bbox_zoom(
        model, img_pil, boxes, res, device=device, decode_fn=decode_fn,
        use_amp=use_amp, amp_dtype=amp_dtype, min_crop_frac=min_crop_frac,
    )
    debug['per_box'] = per_box
    pred = union if union is not None else mask1
    debug.update({'mask_zoom': union, 'zoomed': union is not None})
    rec = eval_metric(pred, info1, item, decoder=label)
    return (rec, debug) if return_debug else rec


@torch.no_grad()
def box_policy_eval(
    model: torch.nn.Module,
    policy: BoxPolicy,
    items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    decode_fn: DecodeFn,
    decoder_name: str = 'kmeans',
    log_tag: str = '[boxpolicy]',
    summarize_results: bool = True,
    subgroup_key: Optional[str] = None,
    **kwargs,
) -> List[EvalRecord]:
    """Run deterministic box_policy_single over items; return EvalRecord list."""
    from lab_utils.eval.aggregate import summarize

    bare = unwrap_model(model)
    bare.eval()

    records: List[EvalRecord] = []
    for item in items:
        try:
            rec = box_policy_single(
                bare, policy, item, res, device=device, decode_fn=decode_fn,
                decoder_name=decoder_name, **kwargs,
            )
            if subgroup_key is not None:
                rec = dataclasses.replace(rec, subgroup=item.meta.get(subgroup_key))
            records.append(rec)
        except Exception as exc:
            log_line(f'{log_tag} WARN: skipped item={item.item_id}: {exc}')

    if summarize_results and records:
        summarize(records, log_tag=log_tag)
    elif not records:
        log_line(f'{log_tag} no records (n_items={len(items)})')
    return records
