"""lab_utils.model.losses.bce — binary cross-entropy losses for forensics head.

Lifted from contrastive_test/core/harness_losses.py (BCE section).
"""

import torch
import torch.nn.functional as F


def selective_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_mask: torch.Tensor,
    pos_weight: float,
    sample_weights: torch.Tensor = None,
) -> torch.Tensor:
    """BCE with logits applied only to active (supervised) samples.

    Args:
        logits:      (B,) image-level or (B, N) per-patch raw logits.
        labels:      same shape as ``logits``, values in {0, 1}.
        active_mask: (B,) bool — which batch items to include.
        pos_weight:  BCE positive-class weight.

    Returns:
        Scalar loss tensor.  Zero-gradient zero if no active items.
    """
    active = active_mask.bool()
    if int(active.sum().item()) == 0:
        return logits.sum() * 0.0
    pos_w = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss_per_patch = F.binary_cross_entropy_with_logits(
        logits[active],
        labels[active].to(logits.dtype),
        pos_weight=pos_w,
        reduction='none',
    )
    # Image-level logits are (B,) → already per-image; per-patch logits are
    # (B, N) → average over the patch axis. Support both ranks.
    loss_per_img = loss_per_patch.mean(dim=1) if loss_per_patch.dim() > 1 else loss_per_patch
    if sample_weights is not None:
        weights = sample_weights[active].to(device=logits.device, dtype=logits.dtype)
        return (loss_per_img * weights).sum() / weights.sum().clamp(min=1.0)
    return loss_per_img.mean()


def selective_bce_loss_with_diag(
    logits: torch.Tensor,
    labels: torch.Tensor,
    active_mask: torch.Tensor,
    pos_weight: float,
    sample_weights: torch.Tensor = None,
) -> tuple[torch.Tensor, dict]:
    """BCE loss plus per-active-image diagnostics for bucketed logging."""
    active = active_mask.bool()
    if int(active.sum().item()) == 0:
        return logits.sum() * 0.0, {'per_image': {'loss': torch.empty(0), 'weight': torch.empty(0)}}
    pos_w = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss_per_patch = F.binary_cross_entropy_with_logits(
        logits[active],
        labels[active].to(logits.dtype),
        pos_weight=pos_w,
        reduction='none',
    )
    # Image-level logits are (B,) → already per-image; per-patch logits are
    # (B, N) → average over the patch axis. Support both ranks.
    loss_per_img = loss_per_patch.mean(dim=1) if loss_per_patch.dim() > 1 else loss_per_patch
    if sample_weights is not None:
        weights = sample_weights[active].to(device=logits.device, dtype=logits.dtype)
    else:
        weights = torch.ones_like(loss_per_img)
    loss = (loss_per_img * weights).sum() / weights.sum().clamp(min=1.0)
    return loss, {
        'per_image': {
            'loss': loss_per_img.detach().cpu(),
            'weight': weights.detach().cpu(),
        }
    }


def selective_patch_bce_loss(
    logits: torch.Tensor,            # (B, N) per-patch logits
    labels: torch.Tensor,            # (B, N) {0, 1} per-patch splice labels
    active_mask: torch.Tensor,       # (B,) bool — images to supervise
    pos_weight: float,
    sample_weights: torch.Tensor = None,   # (B,)
    patch_weights: torch.Tensor = None,     # (B, N) in [0,1]; None = all 1
) -> tuple[torch.Tensor, dict]:
    """Dense per-patch BCE for the supervised splice-flagging baseline.

    Unlike :func:`selective_bce_loss`, this honors a per-patch weight map
    (``patch_weights``) so the boundary ignore/soft band zeroes out ambiguous
    edge patches exactly as the contrastive loss does — keeping the two methods'
    supervision masks identical. Each image's loss is the patch-weighted mean
    over its patches; ``pos_weight`` rebalances the rare positive (splice)
    patches against the abundant clean patches.

    ``active_mask`` should include reals (all-zero labels → negative
    supervision that trains specificity) and supervised splices, and exclude
    missed-splice crops whose splice fell below the in-frame threshold (no
    reliable patch labels). Returns (scalar loss, diagnostics).
    """
    active = active_mask.bool()
    device, dtype = logits.device, logits.dtype
    if int(active.sum().item()) == 0:
        return logits.sum() * 0.0, {'patch_pos_frac': 0.0, 'pred_pos_frac': 0.0, 'n_active_img': 0}
    lg = logits[active]
    tg = labels[active].to(dtype)
    pos_w = torch.tensor(float(pos_weight), device=device, dtype=dtype)
    per_patch = F.binary_cross_entropy_with_logits(
        lg, tg, pos_weight=pos_w, reduction='none',
    )                                                              # (b, N)
    if patch_weights is not None:
        pw = patch_weights[active].to(device=device, dtype=dtype).clamp(0.0, 1.0)
    else:
        pw = torch.ones_like(per_patch)
    num = (per_patch * pw).sum(dim=1)
    den = pw.sum(dim=1).clamp(min=1.0)
    per_img = num / den                                            # (b,)
    if sample_weights is not None:
        w = sample_weights[active].to(device=device, dtype=dtype)
    else:
        w = torch.ones_like(per_img)
    loss = (per_img * w).sum() / w.sum().clamp(min=1.0)
    with torch.no_grad():
        wsum = pw.sum().clamp(min=1.0)
        patch_pos_frac = float(((tg > 0.5).to(dtype) * pw).sum().item() / wsum.item())
        pred_pos_frac  = float(((lg >= 0).to(dtype) * pw).sum().item() / wsum.item())
    return loss, {
        'patch_pos_frac': patch_pos_frac,
        'pred_pos_frac':  pred_pos_frac,
        'n_active_img':   int(active.sum().item()),
    }


def equal_budget_patch_bce_loss(
    logits: torch.Tensor,            # (B, N) per-patch logits
    labels: torch.Tensor,            # (B, N) {0, 1} per-patch labels
    active_mask: torch.Tensor,       # (B,) bool — images to supervise
    k_min: float = 4.0,
    sample_weights: torch.Tensor = None,   # (B,)
    patch_weights: torch.Tensor = None,    # (B, N) in [0,1]; None = all 1
) -> tuple[torch.Tensor, dict]:
    """Per-image equal-budget patch BCE — fixes the small-splice punishment gap.

    Under plain mean-reduced patch BCE (``selective_patch_bce_loss``), missing
    every fake patch of a k-fake-patch image costs k/N of missing every patch
    of a whole (N-patch) fake — a small splice is punished ~N/k times less for
    being TOTALLY missed, purely because it is small. The model rationally
    learns to ignore small splices.

    This loss instead gives every image a FIXED positive budget and a FIXED
    negative budget (nominally 1 unit each), split evenly over that image's
    own fake / not-fake patches:

        kp_i = (banded) fake-patch count,  kn_i = (banded) not-fake-patch count
        wpos_i = 1 / max(kp_i, k_min),     wneg_i = 1 / max(kn_i, k_min)
        L_i = sum_j patch_weight_j * [y_j*wpos_i + (1-y_j)*wneg_i] * BCE(z_j, y_j)

    The SUM (not mean) inside the image is what concentrates the budget
    instead of diluting it back out. On any single-class image (all-fake or
    all-real, band off) this collapses to exactly ``selective_patch_bce_loss``
    at pos_weight=1 — the scheme only does something on MIXED-composition
    images. ``k_min`` is a floor, not a smoothing constant: below it, a
    splice's labels are trusted less and its budget shrinks linearly
    (kp_i / k_min) rather than blowing up 1/kp_i — insurance against single-
    patch label noise dominating a whole image's gradient.

    The (adaptive, per-image) weight this replaces is a de facto
    pos_weight of kn_i/kp_i, so outputs are no longer rarity-suppressed
    posteriors but appearance likelihood-ratios: 0.5 stops being a
    calibrated decision threshold under this loss (CLAUDE.md rule 1) — any
    comparison against a ``global``-trained checkpoint MUST use a
    threshold-free metric (patch AUROC) or a re-derived threshold, never a
    fixed t.

    ``patch_weights`` (the boundary soft/ignore band) zeroes ambiguous edge
    patches out of BOTH the counts (kp/kn) and the loss sum — exactly the
    same band semantics as ``selective_patch_bce_loss``, just load-bearing
    here since it also decides what counts as "small."

    Returns:
        (scalar loss, diagnostics). Diagnostics are budget-symmetry canaries:
        realized_P/realized_Q should sit near 1.0 (their divergence from 1.0
        below k_min is expected and is exactly the muting the floor performs);
        max_patch_w is capped at 1/k_min by construction — if it exceeds that,
        the floor itself is broken.
    """
    active = active_mask.bool()
    device, dtype = logits.device, logits.dtype
    empty_diag = {
        'realized_P': float('nan'), 'realized_Q': float('nan'),
        'max_patch_w': 0.0, 'n_no_supervision': 0,
        'patch_pos_frac': 0.0, 'pred_pos_frac': 0.0,
    }
    if int(active.sum().item()) == 0:
        return logits.sum() * 0.0, empty_diag

    lg = logits[active]
    tg = labels[active].to(dtype)
    if patch_weights is not None:
        pw = patch_weights[active].to(device=device, dtype=dtype).clamp(0.0, 1.0)
    else:
        pw = torch.ones_like(lg)

    with torch.no_grad():
        kp = (pw * tg).sum(dim=1)                       # (b,)
        kn = (pw * (1.0 - tg)).sum(dim=1)                # (b,)
        k_min_t = torch.tensor(float(k_min), device=device, dtype=dtype)
        wpos = 1.0 / torch.clamp(kp, min=k_min_t)
        wneg = 1.0 / torch.clamp(kn, min=k_min_t)
        has_supervision = (kp + kn) > 0

    per_patch_bce = F.binary_cross_entropy_with_logits(lg, tg, reduction='none')  # (b, N)
    per_patch_w = pw * (tg * wpos.unsqueeze(1) + (1.0 - tg) * wneg.unsqueeze(1))
    per_img = (per_patch_w * per_patch_bce).sum(dim=1)   # (b,)

    if sample_weights is not None:
        s = sample_weights[active].to(device=device, dtype=dtype)
    else:
        s = torch.ones_like(per_img)
    s = s * has_supervision.to(dtype)

    denom = s.sum().clamp(min=1.0)
    loss = (per_img * s).sum() / denom

    with torch.no_grad():
        sup = has_supervision
        n_sup = int(sup.sum().item())
        if n_sup > 0:
            has_pos = sup & (kp > 0)
            has_neg = sup & (kn > 0)
            realized_P = float((kp[has_pos] * wpos[has_pos]).mean().item()) if bool(has_pos.any()) else float('nan')
            realized_Q = float((kn[has_neg] * wneg[has_neg]).mean().item()) if bool(has_neg.any()) else float('nan')
            max_patch_w = float(torch.maximum(wpos[sup].max(), wneg[sup].max()).item()) if n_sup > 0 else 0.0
        else:
            realized_P = realized_Q = float('nan')
            max_patch_w = 0.0
        wsum = pw.sum().clamp(min=1.0)
        patch_pos_frac = float((tg * pw).sum().item() / wsum.item())
        pred_pos_frac  = float(((lg >= 0).to(dtype) * pw).sum().item() / wsum.item())

    diag = {
        'realized_P': realized_P,
        'realized_Q': realized_Q,
        'max_patch_w': max_patch_w,
        'n_no_supervision': int((~has_supervision).sum().item()),
        'patch_pos_frac': patch_pos_frac,
        'pred_pos_frac': pred_pos_frac,
    }
    return loss, diag


def logit_consistency_loss(
    clean_logits: torch.Tensor,
    aug_logits: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    """MSE between sigmoid probabilities on clean vs augmented views.

    Encourages the BCE head to be invariant to light augmentations.

    Args:
        clean_logits: (B, N) logits on clean image.
        aug_logits:   (B, N) logits on augmented image.
        active_mask:  (B,) bool.

    Returns:
        Scalar loss tensor.
    """
    active = active_mask.bool()
    if int(active.sum().item()) == 0:
        return clean_logits.sum() * 0.0
    clean_prob = torch.sigmoid(clean_logits[active])
    aug_prob   = torch.sigmoid(aug_logits[active])
    return F.mse_loss(clean_prob, aug_prob)
