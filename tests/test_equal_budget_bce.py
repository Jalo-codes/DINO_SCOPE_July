"""tests.test_equal_budget_bce — equal_budget_patch_bce_loss + its band twin.

Pins the properties docs/equal_budget_bce_spec.md Part A1 promises: flat
per-image FN budget, single-class identity with the current (global) loss,
the k_min clamp, symmetric FP pricing, and exact parity between the batched
tensor band twin (lab_utils.train.loop._mask_to_patch_labels_soft_t) and its
PIL source of truth (lab_utils.data.resolution.mask_to_patch_labels_soft).
"""

import numpy as np
import pytest

pytest.importorskip('torch')  # every case here needs real tensors + autograd

import torch
from PIL import Image

from lab_utils.data.resolution import Resolution, mask_to_patch_labels_soft
from lab_utils.model.losses.bce import equal_budget_patch_bce_loss, selective_patch_bce_loss
from lab_utils.train.loop import _mask_to_patch_labels_soft_t


# ── 1. single-class identity ──────────────────────────────────────────────────

def test_single_class_identity_all_fake():
    torch.manual_seed(0)
    N = 16
    logits = torch.randn(1, N)
    labels = torch.ones(1, N)
    active = torch.tensor([True])

    loss_eb, _ = equal_budget_patch_bce_loss(logits, labels, active, k_min=4.0)
    loss_van, _ = selective_patch_bce_loss(logits, labels, active, pos_weight=1.0)

    assert torch.allclose(loss_eb, loss_van, atol=1e-6)


def test_single_class_identity_all_real():
    torch.manual_seed(1)
    N = 16
    logits = torch.randn(1, N)
    labels = torch.zeros(1, N)
    active = torch.tensor([True])

    loss_eb, _ = equal_budget_patch_bce_loss(logits, labels, active, k_min=4.0)
    loss_van, _ = selective_patch_bce_loss(logits, labels, active, pos_weight=1.0)

    assert torch.allclose(loss_eb, loss_van, atol=1e-6)


# ── 2. flat budget: a small splice and a whole fake cost the SAME for total blindness ──

def test_flat_budget_small_splice_equals_whole_fake():
    N = 100
    logits = torch.full((1, N), -10.0)  # confidently predicts "not fake" everywhere

    labels_small = torch.zeros(1, N)
    labels_small[0, :7] = 1.0            # kp = 7

    labels_full = torch.ones(1, N)       # kp = 100 (whole fake)

    active = torch.tensor([True])
    loss_small, _ = equal_budget_patch_bce_loss(logits, labels_small, active, k_min=4.0)
    loss_full, _ = equal_budget_patch_bce_loss(logits, labels_full, active, k_min=4.0)

    assert torch.allclose(loss_small, loss_full, atol=1e-3)

    # the vanilla loss this replaces punishes the two wildly differently —
    # confirms the test is actually exercising the fix, not a vacuous equality.
    loss_van_small, _ = selective_patch_bce_loss(logits, labels_small, active, pos_weight=1.0)
    loss_van_full, _ = selective_patch_bce_loss(logits, labels_full, active, pos_weight=1.0)
    assert loss_van_small.item() < 0.5 * loss_van_full.item()


# ── 3. clamp: below k_min, budget shrinks linearly (kp / k_min) ───────────────

def test_clamp_halves_budget_at_half_k_min():
    N = 100
    logits = torch.full((1, N), -10.0)
    active = torch.tensor([True])

    labels_kp2 = torch.zeros(1, N); labels_kp2[0, :2] = 1.0
    labels_kp4 = torch.zeros(1, N); labels_kp4[0, :4] = 1.0

    loss_kp2, _ = equal_budget_patch_bce_loss(logits, labels_kp2, active, k_min=4.0)
    loss_kp4, _ = equal_budget_patch_bce_loss(logits, labels_kp4, active, k_min=4.0)

    assert loss_kp2.item() == pytest.approx(0.5 * loss_kp4.item(), rel=1e-3)


# ── 4. symmetric FP pricing: scarce background is expensive ──────────────────

def test_false_positive_pricing_scales_with_kn():
    def _one_false_alarm(n_bg: int) -> float:
        logits = torch.full((1, n_bg), -10.0)  # confidently "not fake" everywhere...
        logits[0, 0] = 10.0                     # ...except one confident false alarm
        labels = torch.zeros(1, n_bg)           # all-real supervision
        active = torch.tensor([True])
        loss, _ = equal_budget_patch_bce_loss(logits, labels, active, k_min=1.0)
        return loss.item()

    loss_kn10 = _one_false_alarm(10)
    loss_kn50 = _one_false_alarm(50)
    assert loss_kn10 == pytest.approx(5.0 * loss_kn50, rel=1e-2)


# ── 5. band parity: the tensor twin must match the PIL source of truth exactly ──

def test_band_parity_against_pil_reference():
    patch_size = 16
    n_side = 4
    S = n_side * patch_size
    res = Resolution(image_size=S, patch_size=patch_size)

    # Each patch is a UNIFORM 8-bit gray level spanning every band regime
    # relative to (low=0.2, high=0.8): exact 0, sub-low, in-ramp, above-high,
    # exact 255. Uniform-per-patch means avg_pool2d recovers exactly this
    # value with no partial-coverage ambiguity.
    values_255 = np.array([
        [0,   12,  128, 230],
        [255, 38,  89,  166],
        [0,   201, 204, 207],
        [48,  51,  54,  255],
    ], dtype=np.uint8)
    mask_np = np.repeat(np.repeat(values_255, patch_size, axis=0), patch_size, axis=1)
    mask_pil = Image.fromarray(mask_np, mode='L')
    assert mask_pil.size == (S, S)

    from torchvision.transforms import functional as TF
    # Same quantized pixel data feeds BOTH paths, so any divergence is real.
    mask_t = TF.to_tensor(mask_pil).unsqueeze(0)  # (1, 1, S, S)

    labels_t, weights_t = _mask_to_patch_labels_soft_t(mask_t, patch_size, 0.2, 0.8)
    labels_ref, weights_ref = mask_to_patch_labels_soft(mask_pil, res, low=0.2, high=0.8)

    assert torch.equal(labels_t.reshape(-1), labels_ref)
    assert torch.allclose(weights_t.reshape(-1), weights_ref, atol=1e-6)


def test_band_parity_random_masks():
    """Same parity check over several random uniform-per-patch density grids."""
    patch_size = 16
    n_side = 6
    S = n_side * patch_size
    res = Resolution(image_size=S, patch_size=patch_size)
    rng = np.random.default_rng(42)

    from torchvision.transforms import functional as TF

    for _ in range(5):
        values_255 = rng.integers(0, 256, size=(n_side, n_side)).astype(np.uint8)
        mask_np = np.repeat(np.repeat(values_255, patch_size, axis=0), patch_size, axis=1)
        mask_pil = Image.fromarray(mask_np, mode='L')
        mask_t = TF.to_tensor(mask_pil).unsqueeze(0)

        labels_t, weights_t = _mask_to_patch_labels_soft_t(mask_t, patch_size, 0.2, 0.8)
        labels_ref, weights_ref = mask_to_patch_labels_soft(mask_pil, res, low=0.2, high=0.8)

        assert torch.equal(labels_t.reshape(-1), labels_ref)
        assert torch.allclose(weights_t.reshape(-1), weights_ref, atol=1e-6)


def test_band_invalid_thresholds_raise():
    mask_t = torch.zeros(1, 1, 32, 32)
    with pytest.raises(ValueError):
        _mask_to_patch_labels_soft_t(mask_t, 16, 0.8, 0.2)   # low > high
    with pytest.raises(ValueError):
        _mask_to_patch_labels_soft_t(mask_t, 16, 0.0, 0.8)   # low == 0 not allowed


# ── 6. degenerate cases: no NaN, grad-safe zero ───────────────────────────────

def test_degenerate_all_patches_banded_out():
    N = 8
    logits = torch.randn(1, N, requires_grad=True)
    labels = torch.ones(1, N)
    weights = torch.zeros(1, N)   # entire image banded out -> no supervision
    active = torch.tensor([True])

    loss, diag = equal_budget_patch_bce_loss(
        logits, labels, active, k_min=4.0, patch_weights=weights)

    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert diag['n_no_supervision'] == 1

    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_degenerate_empty_active_mask():
    N = 8
    logits = torch.randn(2, N, requires_grad=True)
    labels = torch.zeros(2, N)
    active = torch.tensor([False, False])

    loss, diag = equal_budget_patch_bce_loss(logits, labels, active, k_min=4.0)

    assert loss.item() == 0.0
    assert np.isnan(diag['realized_P'])
    assert np.isnan(diag['realized_Q'])

    loss.backward()
    assert torch.isfinite(logits.grad).all()


# ── 7. gradients finite on a general mixed batch ──────────────────────────────

def test_gradients_finite_mixed_batch():
    torch.manual_seed(3)
    N = 32
    logits = torch.randn(4, N, requires_grad=True)
    labels = (torch.rand(4, N) > 0.7).float()
    active = torch.tensor([True, True, True, False])

    loss, diag = equal_budget_patch_bce_loss(logits, labels, active, k_min=4.0)
    loss.backward()

    assert torch.isfinite(logits.grad).all()
    assert diag['max_patch_w'] <= (1.0 / 4.0) + 1e-6   # capped at 1/k_min by construction
