"""lab_utils.model.box_policy — learned grid-locked box proposer (RL head).

A small TRAINABLE head that reads the FROZEN detector's per-patch signal and
emits a *set* of zoom boxes.  The detector (DINOv3 + LoRA + contrastive /
attention heads) is never touched here — this head consumes only the per-patch
vectors it produces (`embeddings` ⊕ `attention` ⊕ `patch_logit`), so it is
information-bottlenecked away from the raw backbone and cannot relearn the
noise/boundary shortcut the contrastive objective was trained to suppress.

Output contract — a set of boxes, each LOCKED to a patch cell:
    box center = the patch's own grid center (never a continuous offset),
    box extent = a predicted (h, w) fraction of the frame.
The only continuous outputs are the extents; positions are read off the grid.

Training is a one-step bandit (REINFORCE): sample a set of boxes, zoom them with
the frozen decoder on the fly, score the realized localization F1, and nudge the
policy toward sets that scored above a baseline.  The reward is non-differentiable
(crop + decode + F1), so we never backprop through it — only through the
log-probability of the sampled action.  All of that orchestration lives in
``experiments.labs.box_policy_zoom``; this module owns the network + the action
distributions and nothing else.

Capacity split (by design, per the design discussion):
    * the shared transformer ENCODER carries the capacity,
    * the keep / size HEADS are single Linear layers, so box decisions cannot
      stray far from the encoder representation.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal


@dataclasses.dataclass
class ActOutput:
    """One sampled (or deterministic) action from the policy.

    kept_idx:  (K,) long — flat patch indices that became boxes (K ≤ max_boxes).
    sizes:     (K, 2) float — (h, w) fractions of the frame for each kept box.
    keep_prob: (N,) float — per-patch keep probability (for viz / inspection).
    entropy:   scalar — keep-Bernoulli entropy over candidates (exploration term).
    n_sampled: float — count of keep=1 BEFORE the cap.

    Per-candidate pieces — the CREDIT seam.  The caller decides how to weight the
    score-function terms: one global advantage (vanilla REINFORCE — smears a
    persistently-negative advantage over the keep=0 majority and inflates them) or
    a per-box difference reward (each box judged by its own marginal; keep=0 gets
    NO term, so no inflation).  All indexed over the C candidate patches:
      cand_idx:     (C,) long — global patch index of each candidate.
      keep_lp:      (C,) — log π(sampled keep) per candidate.
      size_lp:      (C,) — log π(sampled size) per candidate (summed over h, w).
      keep_sampled: (C,) bool — the sampled keep value (True incl. capped-out).
      kept_pos:     (K,) long — candidate positions that became boxes (executed),
                    in the SAME order as kept_idx / sizes.
    """
    kept_idx:     torch.Tensor
    sizes:        torch.Tensor
    keep_prob:    torch.Tensor
    entropy:      torch.Tensor
    n_sampled:    float
    cand_idx:     torch.Tensor
    keep_lp:      torch.Tensor
    size_lp:      torch.Tensor
    keep_sampled: torch.Tensor
    kept_pos:     torch.Tensor


class BoxPolicy(nn.Module):
    """Grid-locked set-of-boxes policy over frozen per-patch features.

    Args:
        in_dim:        Per-patch input width (e.g. contrastive_dim + attn + patch_logit).
        width:         Transformer / hidden width.
        depth:         Number of transformer encoder layers (the capacity).
        n_heads:       Attention heads.
        max_positions: Upper bound on patch count for the learned position table.
        size_init:     Initial mean box extent (fraction); the size head bias is
                       set so exp(mean) ≈ size_init at init.
        size_log_std_init: Initial log-std of the (log-space) size distribution.
        min_log_std:   Floor on the size log-std so exploration never collapses.
        keep_bias_init: Initial keep-head bias; negative ⇒ the policy starts
                        SPARSE (few boxes) so step 1 does not zoom hundreds.
        size_min_frac / size_max_frac: clamp on the realized box extent (per side,
                        as a fraction of the frame).  The MINIMUM matters: a crop
                        re-fed at the model's input size is upscaled by ~1/size, so
                        too-small a box (≈ a few patches) becomes mostly
                        interpolation — an over-magnified garbage zoom.  The default
                        ~0.18 ≈ 5 patches at 448/16, in the spirit of the
                        attention-zoom ``min_box_size=8`` convention while still
                        allowing tight small-splice crops.
        dropout:       Encoder dropout.
    """

    def __init__(
        self,
        in_dim: int,
        *,
        width: int = 128,
        depth: int = 2,
        n_heads: int = 4,
        max_positions: int = 1024,
        size_init: float = 0.30,
        size_log_std_init: float = -1.2,
        min_log_std: float = -2.3,
        keep_bias_init: float = -3.0,
        size_min_frac: float = 0.18,
        size_max_frac: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.min_log_std = float(min_log_std)
        self.size_min_frac = float(size_min_frac)
        self.size_max_frac = float(size_max_frac)

        self.in_proj = nn.Linear(in_dim, width)
        # Learned positional table — the policy must know WHERE each patch sits to
        # place a box there; DINO patch order is row-major so a per-position
        # embedding (sliced to N) suffices.
        self.pos = nn.Parameter(torch.zeros(max_positions, width))
        nn.init.normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads, dim_feedforward=width * 2,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

        # Low-capacity heads — single Linear each (cannot stray from the encoder).
        self.keep_head = nn.Linear(width, 1)
        self.size_head = nn.Linear(width, 2)   # mean of (log h, log w)
        self.size_log_std = nn.Parameter(torch.full((2,), float(size_log_std_init)))

        # Sparse / sane-extent initialisation.  The BIAS sets the operating point
        # (sparse keep via keep_bias_init; box extent ≈ size_init), while SMALL
        # RANDOM weights let the heads respond to per-patch features from step 1.
        # Zero-init weights would make every patch identical (same keep-logit, and
        # — visible in the deterministic viz — the SAME box size everywhere), since
        # only the bias speaks until gradients slowly move the weights off zero.
        import math
        nn.init.normal_(self.keep_head.weight, std=0.02)
        nn.init.constant_(self.keep_head.bias, float(keep_bias_init))
        nn.init.normal_(self.size_head.weight, std=0.02)
        nn.init.constant_(self.size_head.bias, math.log(max(1e-3, float(size_init))))

    # ── network forward ──────────────────────────────────────────────────────────

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(N, in_dim) per-patch features → (keep_logit (N,), size_mean (N,2), log_std (2,)).

        feats is a single image's patch grid (no batch dim); positions 0..N-1 map
        to row-major patch order.
        """
        if feats.dim() != 2:
            raise ValueError(f'BoxPolicy.forward expects (N, in_dim), got {tuple(feats.shape)}')
        n = feats.shape[0]
        if n > self.pos.shape[0]:
            raise ValueError(f'BoxPolicy: N={n} exceeds max_positions={self.pos.shape[0]}')

        h = self.in_proj(feats) + self.pos[:n]            # (N, width)
        h = self.encoder(h.unsqueeze(0)).squeeze(0)        # (N, width)
        keep_logit = self.keep_head(h).squeeze(-1)         # (N,)
        size_mean  = self.size_head(h)                     # (N, 2) = (log h, log w)
        log_std    = self.size_log_std.clamp(min=self.min_log_std)
        return keep_logit, size_mean, log_std

    # ── policy action (sample or deterministic) ──────────────────────────────────

    def act(
        self,
        feats: torch.Tensor,
        *,
        candidate_mask: torch.Tensor,
        max_boxes: int = 8,
        deterministic: bool = False,
    ) -> ActOutput:
        """Produce a set of boxes from the per-patch features.

        Boxes are only ever placed at *candidate* patches (``candidate_mask`` —
        the variance-controlling prefilter, e.g. attention-hot ∪ decode-mask).
        Among the kept candidates we cap to ``max_boxes`` by keep-probability.

        Train (``deterministic=False``): sample keep ~ Bernoulli and size ~ Normal
        in log-space; ``log_prob`` is the REINFORCE score function over the
        executed action.  Inference (``deterministic=True``): emit the
        ``round(Σ keep_prob)`` highest-probability candidates — the EXPECTED number
        of boxes the stochastic policy would sample, picking the most confident.
        This is self-calibrating: a fixed threshold τ instead silently emits ZERO
        boxes whenever every keep-prob is below τ (common with the sparse prior,
        even when Σ p — and training — want several), collapsing eval to the flat
        decode.  The per-candidate log-probs are returned but not meaningful here.
        """
        keep_logit, size_mean, log_std = self.forward(feats)
        n = keep_logit.shape[0]
        keep_prob = torch.sigmoid(keep_logit)

        cand = candidate_mask.to(keep_logit.device).bool().reshape(-1)[:n]
        cand_idx = torch.nonzero(cand, as_tuple=False).reshape(-1)
        if cand_idx.numel() == 0:
            zero = keep_logit.sum() * 0.0
            empty_l = cand_idx.new_zeros((0,), dtype=torch.long)
            empty_f = keep_logit.new_zeros((0,))
            empty_b = torch.zeros((0,), dtype=torch.bool, device=keep_logit.device)
            empty_sz = size_mean.new_zeros((0, 2))
            return ActOutput(
                kept_idx=empty_l, sizes=empty_sz, keep_prob=keep_prob.detach(),
                entropy=zero, n_sampled=0.0, cand_idx=empty_l, keep_lp=empty_f,
                size_lp=empty_f, keep_sampled=empty_b, kept_pos=empty_l,
            )

        k_logit_c = keep_logit[cand_idx]                 # (C,)
        mean_c    = size_mean[cand_idx]                  # (C, 2)
        kdist = Bernoulli(logits=k_logit_c)
        sdist = Normal(mean_c, torch.exp(log_std))

        if deterministic:
            probs_c = torch.sigmoid(k_logit_c)
            k_det = max(0, min(int(round(float(probs_c.sum().item()))), max_boxes))
            keep_c = torch.zeros_like(probs_c, dtype=torch.bool)
            if k_det > 0:
                top = torch.topk(probs_c, k=min(k_det, probs_c.numel())).indices
                keep_c[top] = True
            size_raw = mean_c
        else:
            keep_c = kdist.sample().bool()
            size_raw = sdist.sample()

        # Hard cap at max_boxes — keep the highest-probability candidates.
        kept_pos = torch.nonzero(keep_c, as_tuple=False).reshape(-1)
        if kept_pos.numel() > max_boxes:
            probs = torch.sigmoid(k_logit_c[kept_pos])
            top = torch.topk(probs, k=max_boxes).indices
            kept_pos = kept_pos[top]

        # Per-candidate score-function terms.  We return them UN-summed so the
        # caller can weight each box by its own credit (difference reward) instead
        # of smearing one global advantage across the keep=0 majority.
        keep_lp = kdist.log_prob(keep_c.float())             # (C,)
        size_lp = sdist.log_prob(size_raw).sum(dim=-1)       # (C,)
        entropy = kdist.entropy().sum()

        sizes = torch.exp(size_raw[kept_pos]).clamp(self.size_min_frac, self.size_max_frac)
        kept_idx = cand_idx[kept_pos]
        n_sampled = float(keep_c.sum().item())
        return ActOutput(
            kept_idx=kept_idx, sizes=sizes, keep_prob=keep_prob.detach(),
            entropy=entropy, n_sampled=n_sampled, cand_idx=cand_idx,
            keep_lp=keep_lp, size_lp=size_lp, keep_sampled=keep_c, kept_pos=kept_pos,
        )


def build_box_policy(in_dim: int, *, device=None, **kwargs) -> BoxPolicy:
    """Construct a BoxPolicy and (optionally) move it to ``device``."""
    policy = BoxPolicy(in_dim, **kwargs)
    if device is not None:
        policy = policy.to(device)
    return policy
