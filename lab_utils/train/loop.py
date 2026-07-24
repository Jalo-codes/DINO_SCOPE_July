"""lab_utils.train.loop — epoch / step / val helpers.

Extracted from the legacy god script.  No swin (removed in rebuild).
No oracle eval.  Validation uses the canonical fetch → decode → metric →
aggregate pipeline from lab_utils.eval.

Public surface:
    build_optimizer(model, cfg) → AdamW
    build_scheduler(optimizer, *, cfg, steps_per_epoch) → SequentialLR / CosineAnnealingLR
    run_train_epoch(model, loader, optimizer, scaler, scheduler, *, epoch, cfg, device, ctx) → dict
    run_val_eval(model, val_items, res, *, device, cfg, log_tag) → (List[EvalRecord], Optional[float])
    run_epoch_viz(model, val_items, res, *, device, cfg, epoch, run_dir, n) → None
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lab_utils.compat import trapz
from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution
from lab_utils.eval.fetch import model_info
from lab_utils.eval.decode.kmeans import decode_kmeans, decode_kmeans_logit
from lab_utils.eval.decode.threshold import decode_threshold
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.record import EvalRecord
from lab_utils.errors import DataError
from lab_utils.logging.text import log_line


# ── Batch utilities ────────────────────────────────────────────────────────────

def _meta_list(batch: Dict[str, Any]) -> List[Dict]:
    meta = batch['meta']
    if isinstance(meta, list):
        return meta
    n = batch['img'].shape[0]
    return [{k: v[i] for k, v in meta.items()} for i in range(n)]


def _mask_to_patch_labels(mask_t: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(B,1,S,S) mask → (B,N) patch binary labels via avg-pool + threshold."""
    pooled = F.avg_pool2d(mask_t, kernel_size=patch_size, stride=patch_size)
    return (pooled.squeeze(1).flatten(1) > 0.5).long()


def _mask_to_patch_labels_soft_t(
    mask_t: torch.Tensor, patch_size: int, low: float, high: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched tensor twin of lab_utils.data.resolution.mask_to_patch_labels_soft.

    Same piecewise band rule (density==0 -> confident bg weight=1; 0<density<low
    -> ignore weight=0; low<=density<high -> linear ramp; density>=high ->
    confident fake weight=1), just over a (B,1,S,S) batch instead of one PIL
    mask. MUST stay bit-parity with the PIL function (see
    tests/test_equal_budget_bce.py::test_band_parity) — that function is the
    source of truth; this is a batched restatement, not an independent design.

    Returns (labels (B,N) long, weights (B,N) float).
    """
    if not (0.0 < float(low) < float(high) <= 1.0):
        raise ValueError(
            f'_mask_to_patch_labels_soft_t: need 0 < low < high <= 1, '
            f'got low={low}, high={high}'
        )
    density = F.avg_pool2d(mask_t, kernel_size=patch_size, stride=patch_size).squeeze(1).flatten(1)
    low_t, high_t = float(low), float(high)

    labels = (density >= low_t).long()
    weights = torch.zeros_like(density)
    weights[density == 0.0] = 1.0
    weights[density >= high_t] = 1.0
    ramp_mask = (density >= low_t) & (density < high_t)
    if bool(ramp_mask.any()):
        weights[ramp_mask] = (density[ramp_mask] - low_t) / (high_t - low_t)
    return labels, weights


# ── Optimizer + scheduler ──────────────────────────────────────────────────────

def build_optimizer(model: torch.nn.Module, cfg) -> torch.optim.Optimizer:
    """AdamW over all model parameters."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    cfg,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup → CosineAnnealingLR."""
    total_steps  = int(cfg.num_epochs) * steps_per_epoch
    warmup_steps = int(round(max(0.0, float(cfg.warmup_epochs)) * steps_per_epoch))
    warmup_steps = min(warmup_steps, max(0, total_steps - 1))
    eta_min      = float(cfg.lr) * 0.05

    if warmup_steps > 0:
        warmup  = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )
        log_line(
            f'[train] LR warmup={warmup_steps} steps (~{cfg.warmup_epochs:.2f} ep) '
            f'→ cosine over {total_steps - warmup_steps} steps eta_min={eta_min:.2e}'
        )
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=eta_min
        )
    return sched


# ── Training epoch ─────────────────────────────────────────────────────────────

def run_train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    epoch: int,
    cfg,
    device: torch.device,
    ctx=None,          # DistributedContext or None
    amp_dtype=None,    # torch.float16 | torch.bfloat16 | None
) -> Dict[str, float]:
    """Run one training epoch; return a loss-summary dict."""
    from lab_utils.model.losses.bce import (
        selective_bce_loss_with_diag, selective_patch_bce_loss,
        equal_budget_patch_bce_loss,
    )
    from lab_utils.model.losses.contrastive import selective_symmetric_contrastive_loss

    use_amp = cfg.use_amp and amp_dtype is not None
    bce_active  = (cfg.pool_hidden > 0 and cfg.lambda_image_bce > 0.0)
    cont_active = (cfg.contrastive_dim > 0 and cfg.lambda_contrastive > 0.0)
    patch_active = (cfg.patch_bce and cfg.lambda_patch_bce > 0.0)
    patch_balance = getattr(cfg, 'patch_balance', 'global')
    patch_band = getattr(cfg, 'patch_band', None)
    patch_k_min = float(getattr(cfg, 'patch_k_min', 4.0))
    if patch_active and patch_balance == 'per_image' and epoch == 0 \
            and float(cfg.patch_pos_weight) != 1.0:
        log_line(
            f'[train] patch_balance=per_image: --patch_pos_weight={cfg.patch_pos_weight} '
            f'is IGNORED (per-image budgets replace it wholesale)'
        )

    model.train()
    optimizer.zero_grad()

    loss_total = loss_bce = loss_cont = loss_patch = 0.0
    n_steps = 0
    # Equal-budget diagnostics (only meaningful when patch_balance == 'per_image')
    diag_P_sum = diag_Q_sum = 0.0
    diag_P_n = diag_Q_n = 0
    diag_max_patch_w = 0.0
    diag_n_no_supervision = 0

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        img  = batch['img'].to(device, non_blocking=True)
        mask = batch['mask'].to(device, non_blocking=True)   # (B,1,S,S)
        metas = _meta_list(batch)

        is_real_arr = torch.tensor(
            [bool(m.get('is_real', False)) for m in metas],
            dtype=torch.bool, device=device,
        )
        is_supervised_arr = torch.tensor(
            [bool(m.get('is_supervised', False)) for m in metas],
            dtype=torch.bool, device=device,
        )
        label = (~is_real_arr).float()       # (B,) — 1 for splice, 0 for real
        is_splice = ~is_real_arr

        # is_single_class: real images and splice crops that missed the region
        is_single = is_real_arr | (~is_supervised_arr)

        patch_labels = _mask_to_patch_labels(mask, cfg.patch_size)   # (B,N)

        ctx_amp = (
            torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype)
            if use_amp else torch.amp.autocast('cuda', enabled=False)
        )
        with ctx_amp:
            out = model(img)

        # ── BCE head ──────────────────────────────────────────────────────────
        image_logit = out.get('image_logit')
        if bce_active and image_logit is not None:
            active     = ~(is_splice & is_single)   # ignore missed-crop splices
            bce_loss, _ = selective_bce_loss_with_diag(
                image_logit.squeeze(-1) if image_logit.dim() > 1 else image_logit,
                label,
                active_mask=active,
                pos_weight=1.0,
            )
        else:
            bce_loss = torch.tensor(0.0, device=device)

        # ── Contrastive head ──────────────────────────────────────────────────
        z_contrastive = out.get('contrastive')
        if cont_active and z_contrastive is not None:
            active_cont = ~(is_splice & is_single)
            cont_loss, _ = selective_symmetric_contrastive_loss(
                z_contrastive, patch_labels, is_single,
                active_mask=active_cont,
                tau_pos=0.60,
                tau_neg=0.15,
                norm_power=0.75,
                lambda_repel=1.0,
                single_class_weight=0.05,
                area_balance_power=0.5,
            )
        else:
            cont_loss = torch.tensor(0.0, device=device)

        # ── Patch-BCE head ────────────────────────────────────────────────────
        patch_logit = out.get('patch_logit')
        if patch_active and patch_logit is not None:
            active_patch = ~(is_splice & is_single)
            if patch_band:
                bce_labels, bce_pw = _mask_to_patch_labels_soft_t(
                    mask, cfg.patch_size, float(patch_band[0]), float(patch_band[1]))
            else:
                bce_labels, bce_pw = patch_labels, None
            if patch_balance == 'per_image':
                patch_loss, patch_diag = equal_budget_patch_bce_loss(
                    patch_logit, bce_labels,
                    active_mask=active_patch,
                    k_min=patch_k_min,
                    patch_weights=bce_pw,
                )
                if not math.isnan(patch_diag['realized_P']):
                    diag_P_sum += patch_diag['realized_P']; diag_P_n += 1
                if not math.isnan(patch_diag['realized_Q']):
                    diag_Q_sum += patch_diag['realized_Q']; diag_Q_n += 1
                diag_max_patch_w = max(diag_max_patch_w, patch_diag['max_patch_w'])
                diag_n_no_supervision += patch_diag['n_no_supervision']
            else:
                patch_loss, _ = selective_patch_bce_loss(
                    patch_logit, bce_labels,
                    active_mask=active_patch,
                    pos_weight=cfg.patch_pos_weight,
                    patch_weights=bce_pw,
                )
        else:
            patch_loss = torch.tensor(0.0, device=device)

        total_loss = (
            float(cfg.lambda_image_bce)   * bce_loss  +
            float(cfg.lambda_contrastive) * cont_loss +
            float(cfg.lambda_patch_bce)   * patch_loss
        )
        scaler.scale(total_loss / cfg.grad_accum).backward()

        # ── Optimizer step every grad_accum batches ───────────────────────────
        if (step + 1) % cfg.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        loss_total += float(total_loss.detach())
        loss_bce   += float(bce_loss.detach())
        loss_cont  += float(cont_loss.detach())
        loss_patch += float(patch_loss.detach())
        n_steps    += 1

        if cfg.log_every > 0 and n_steps % cfg.log_every == 0:
            lr_now = optimizer.param_groups[0]['lr']
            log_line(
                f'[train] epoch={epoch} step={n_steps} '
                f'loss={loss_total/n_steps:.4f} '
                f'bce={loss_bce/n_steps:.4f} '
                f'cont={loss_cont/n_steps:.4f} '
                f'lr={lr_now:.2e}'
            )

    # Flush any remaining gradients in the last partial accumulation window
    if n_steps % cfg.grad_accum != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    n = max(1, n_steps)
    stats = {
        'loss':        loss_total / n,
        'loss_bce':    loss_bce   / n,
        'loss_cont':   loss_cont  / n,
        'loss_patch':  loss_patch / n,
        'n_steps':     n_steps,
    }
    if patch_active and patch_balance == 'per_image':
        stats['patch_realized_P']       = diag_P_sum / diag_P_n if diag_P_n else float('nan')
        stats['patch_realized_Q']       = diag_Q_sum / diag_Q_n if diag_Q_n else float('nan')
        stats['patch_max_patch_w']      = diag_max_patch_w
        stats['patch_n_no_supervision'] = diag_n_no_supervision
    return stats


# ── Validation eval ────────────────────────────────────────────────────────────

@torch.no_grad()
def run_val_eval(
    model: torch.nn.Module,
    val_items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    cfg,
    log_tag: str = '[eval]',
    max_items: Optional[int] = None,
    decoder: str = 'auto',
) -> Tuple[List[EvalRecord], Optional[float]]:
    """Fetch → decode → metric over val_items; return (records, image_auc).

    Args:
        model:      The (possibly DDP-wrapped) model.
        val_items:  List of Item objects from the val dataset.
        res:        Resolution (image size, patch size).
        device:     torch.device.
        cfg:        RunConfig — used to determine which heads are active.
        log_tag:    Log tag for summary output.
        max_items:  Limit items processed (for quick sanity checks).
        decoder:    'kmeans', 'threshold', 'none', or 'auto' (kmeans if
                    contrastive head enabled, threshold if patch-BCE head
                    enabled, else 'none' for image-level only).

    Returns:
        (records, image_auc) — image_auc is None when there are insufficient
        reals + splices to compute it, or when any image_score is NaN.
    """
    from lab_utils.eval.aggregate import summarize
    from lab_utils.eval.preprocess import load_image_tensor
    from lab_utils.train.distributed import unwrap_model

    bare_model = unwrap_model(model)
    bare_model.eval()

    has_contrastive = cfg.contrastive_dim > 0
    has_patch_bce   = cfg.patch_bce

    if decoder == 'auto':
        if has_contrastive:
            decoder = 'kmeans'
        elif has_patch_bce:
            decoder = 'threshold'
        else:
            decoder = 'none'

    use_amp = cfg.use_amp
    items_to_eval = val_items[:max_items] if max_items is not None else val_items

    # cfg.val_zoom: run the attention-zoom two-pass per item so the per-epoch
    # metric — and the early-stop driver — track the zoomed localization F1 we
    # actually report at eval time.  Only meaningful for localization decoders.
    # Function-local import (mirrors load_model's run_config import) keeps the
    # lazy lab_utils→experiments edge contained to this opt-in path.
    zoom_val = bool(getattr(cfg, 'val_zoom', False)) and decoder in ('kmeans', 'threshold', 'kmeans_logit')
    if zoom_val:
        from experiments.labs.attention_zoom import attention_zoom_single
        log_line(f'{log_tag} val zoom ON (two-pass, decoder={decoder})')

    import dataclasses

    def _tag_subgroup(rec, item):
        """Tag records with the cell the per-epoch summary should break out.

        TGIF: its (model|type|family) cell. full_fakes: the GENERATOR — without
        this the whole-image view has nothing to break down, since full_fakes
        items carry meta['generator'] and no tgif_subcat. Items with neither
        stay pooled."""
        sub = item.meta.get('tgif_subcat') or item.meta.get('generator')
        return dataclasses.replace(rec, subgroup=sub) if (sub and sub != 'real') else rec

    records: List[EvalRecord] = []
    for item in items_to_eval:
        try:
            if zoom_val:
                # Zoom params default to the shared DEFAULT_ZOOM operating point
                # (attention_zoom_single's signature); cfg only overrides when
                # explicitly set, so val-zoom == eval-zoom geometry by default.
                zoom_overrides = {}
                if getattr(cfg, 'val_zoom_pad_frac', None) is not None:
                    zoom_overrides['pad_side_frac'] = cfg.val_zoom_pad_frac
                if getattr(cfg, 'val_zoom_min_area', 0.0):
                    zoom_overrides['min_area_frac'] = cfg.val_zoom_min_area
                rec = attention_zoom_single(
                    bare_model, item, res,
                    device=device, use_amp=use_amp, decoder=decoder,
                    **zoom_overrides,
                )
                records.append(_tag_subgroup(rec, item))
                continue

            img_tensor = load_image_tensor(item, res, device=device)
            info = model_info(bare_model, img_tensor, device=device, amp=use_amp)

            if decoder == 'none':
                n_side = info.grid_hw[0]
                patch_mask = np.zeros((n_side, n_side), dtype=bool)
            elif decoder == 'kmeans':
                patch_mask = decode_kmeans(info)
            elif decoder == 'kmeans_logit':
                patch_mask = decode_kmeans_logit(info)
            elif decoder == 'threshold':
                patch_mask = decode_threshold(info)
            else:
                raise ValueError(f'run_val_eval: unknown decoder {decoder!r}')

            rec = eval_metric(patch_mask, info, item, decoder=decoder)
            records.append(_tag_subgroup(rec, item))
        except DataError:
            raise  # alignment/pairing bug — abort, never a skip
        except Exception as exc:
            log_line(f'{log_tag} WARN: skipped item={item.item_id}: {exc}')

    if records:
        # Sentinel-mask sources (full_fakes) make f1/iou/precision category
        # errors, not measurements (rule 2) — report separability + per-generator
        # AUROC instead of a localization block that mechanically reads ~1.0.
        from lab_utils.eval.aggregate import localization_is_meaningful
        if localization_is_meaningful(items_to_eval):
            summarize(records, log_tag=log_tag, include_sources=True)
            if any(r.subgroup is not None for r in records):
                from lab_utils.eval.aggregate import summarize_by_subgroup
                summarize_by_subgroup(records, log_tag=log_tag)
        else:
            from lab_utils.eval.aggregate import summarize_full_fakes
            summarize_full_fakes(records, log_tag=log_tag)
    else:
        log_line(f'{log_tag} no records to summarize (n_items={len(items_to_eval)})')

    # Compute image-level AUC inline (self-contained, no cross-layer import).
    image_auc = _image_auc(records)

    # Per-source AUROC breakdown.
    sources = sorted({r.source for r in records})
    if len(sources) > 1:
        for src in sources:
            src_records = [r for r in records if r.source == src]
            src_auc = _image_auc(src_records)
            if src_auc is not None:
                log_line(f'{log_tag}   {src} image_auc={src_auc:.4f} (n={len(src_records)})')

    # Per-source LOCALIZATION AUROC (threshold-free patch separability). A
    # second flat forward per item (no zoom) — gated behind --val_patch_auroc
    # since it ~doubles val cost. This is the honest per-epoch localization
    # signal under per_image, where the threshold-decode F1 above is
    # calibration-shifted (CLAUDE.md rule 1). full_fakes / pseudo-sentinel
    # items self-skip inside collect_patch_scores (rule 2).
    if getattr(cfg, 'val_patch_auroc', False) and has_patch_bce and records:
        from lab_utils.eval.patch_scores import collect_patch_scores
        band = tuple(getattr(cfg, 'patch_band', None) or (0.2, 0.8))
        pa_dtype = 'bfloat16' if getattr(cfg, 'amp_dtype', None) == 'bf16' else 'float16'
        by_src: Dict[str, List] = {}
        for it in items_to_eval:
            by_src.setdefault(it.source, []).append(it)
        for src in sorted(by_src):
            ps = collect_patch_scores(
                bare_model, by_src[src], res, device=device, use_amp=use_amp,
                amp_dtype=pa_dtype, band=band, log_tag=f'{log_tag} pa:{src}', quiet=True)
            if ps['n_items'] > 0:
                log_line(
                    f'{log_tag}   {src} patch_auroc={ps["auroc_pooled"]:.4f} '
                    f'vs_real_bg={ps["auroc_vs_real_bg"]:.4f} '
                    f'vs_splice_bg={ps["auroc_vs_splice_bg"]:.4f} (n={ps["n_items"]})')

    return records, image_auc


def _image_auc(records: List[EvalRecord]) -> Optional[float]:
    """AUC from image_score over splices + reals.  None if not computable."""
    if not records:
        return None
    scores = np.array([r.image_score for r in records], dtype=np.float64)
    labels = np.array([0 if r.is_real else 1 for r in records], dtype=np.int32)
    if np.any(np.isnan(scores)):
        return None
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-scores)
    sl    = labels[order]
    tpr   = np.cumsum(sl) / n_pos
    fpr   = np.cumsum(1 - sl) / n_neg
    auc   = float(trapz(tpr, fpr))
    return 1.0 + auc if auc < 0 else auc


def run_epoch_viz(
    model: torch.nn.Module,
    val_items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    cfg,
    epoch: int,
    run_dir: str,
    n: int = 15,
    per_source: Optional[Dict[str, int]] = None,
    seed: int = 42,
    decoder: str = 'auto',
    log_tag: str = '[viz]',
) -> None:
    """Save (and, in a notebook, inline-display) a fixed sample of splice
    items each epoch: input | predicted mask | attention | derived GT mask.

    The sample is chosen ONCE (seeded, from val_items' splice subset) and
    reused every epoch — the point is watching the same hard cases evolve
    across training, not a fresh random draw each time. Figures are written
    to run_dir/viz/epoch_{epoch:04d}/{item_id}.png regardless of frontend;
    inline display is opportunistic (no-op outside a notebook/graphics TTY).

    With per_source (e.g. {'pico_pseudo': 35}), the sample is stratified: each
    listed source gets exactly its count (capped by that source's pool size),
    and any val sources NOT listed are pooled and topped up to n. Without
    per_source, it's a flat random sample of n across all splice items.

    Uses the flat (non-zoom) single-pass prediction — mirrors run_val_eval's
    non-zoom branch, kept flat here for speed since this runs every epoch.
    """
    from pathlib import Path

    from experiments.labs.viz import display_image_inline, plot_prediction
    from lab_utils.eval.preprocess import load_image_tensor
    from lab_utils.train.distributed import unwrap_model

    bare_model = unwrap_model(model)
    bare_model.eval()

    has_contrastive = cfg.contrastive_dim > 0
    has_patch_bce   = cfg.patch_bce
    if decoder == 'auto':
        decoder = 'kmeans' if has_contrastive else ('threshold' if has_patch_bce else 'none')

    splices = [it for it in val_items if not it.is_real]
    if not splices:
        log_line(f'{log_tag} no splice items in val set — skipping')
        return

    rng = random.Random(seed)
    if per_source:
        sample = []
        stratified_ids = set()
        for src, k in per_source.items():
            pool = [it for it in splices if it.source == src]
            if len(pool) < k:
                log_line(f'{log_tag} WARNING: viz_per_source[{src}]={k} but only '
                          f'{len(pool)} splice items available — taking all of them')
            picked = rng.sample(pool, k=min(k, len(pool)))
            sample.extend(picked)
            stratified_ids.update(id(it) for it in picked)
        remainder_pool = [
            it for it in splices
            if id(it) not in stratified_ids and it.source not in per_source
        ]
        sample.extend(rng.sample(remainder_pool, k=min(n, len(remainder_pool))))
        log_line(f'{log_tag} stratified sample: {per_source} + '
                  f'{min(n, len(remainder_pool))} pooled from other sources')
    else:
        sample = rng.sample(splices, k=min(n, len(splices)))

    out_dir = Path(run_dir) / 'viz' / f'epoch_{epoch:04d}'
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    n_shown = 0
    for item in sample:
        try:
            img_tensor, img_pil = load_image_tensor(item, res, device=device, return_pil=True)
            info = model_info(bare_model, img_tensor, device=device, amp=cfg.use_amp)

            if decoder == 'none':
                n_side = info.grid_hw[0]
                patch_mask = np.zeros((n_side, n_side), dtype=bool)
            elif decoder == 'kmeans':
                patch_mask = decode_kmeans(info)
            elif decoder == 'kmeans_logit':
                patch_mask = decode_kmeans_logit(info)
            elif decoder == 'threshold':
                patch_mask = decode_threshold(info)
            else:
                raise ValueError(f'run_epoch_viz: unknown decoder {decoder!r}')

            gt_mask = None
            if item.mask is not None:
                from PIL import Image as PILImage
                gt_pil = PILImage.open(item.mask).convert('L')
                gt_mask = np.asarray(gt_pil) > 127

            fig = plot_prediction(
                img_pil, patch_mask, info,
                title=f'{item.item_id}  epoch={epoch}  decoder={decoder}',
                gt_mask=gt_mask,
            )
            safe_name = item.item_id.replace('/', '_').replace(' ', '_')
            fig.savefig(out_dir / f'{safe_name}.png', dpi=110, bbox_inches='tight')
            display_image_inline(fig)
            plt.close(fig)
            n_shown += 1
        except Exception as exc:
            log_line(f'{log_tag} WARN: skipped item={item.item_id}: {exc}')

    log_line(f'{log_tag} epoch={epoch} wrote {n_shown}/{len(sample)} figures -> {out_dir}')
