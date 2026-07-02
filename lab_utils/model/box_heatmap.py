"""lab_utils.model.box_heatmap — single-box heatmap head (supervised MVP).

The simplest thing that tests the core hypothesis behind the box policy: can a
small TRAINABLE head, reading the FROZEN detector's per-patch signal
(`z` ⊕ attention ⊕ patch_logit), predict *where to zoom*?

Output is a per-patch logit — a heatmap.  Train it with plain weighted BCE
toward a binary box mask (1 inside the padded GT box, 0 outside; all-0 for large
splices that should not zoom).  At eval the heatmap is thresholded, the bounding
rectangle of the ON patches is read off as THE zoom box, and the realized F1
tells you whether it helped.  No RL, no sampling, no set/union credit — just a
supervised heatmap, trainable on cached vectors in seconds.

This deliberately mirrors :class:`BoxPolicy`'s encoder (the shared transformer
carries the capacity; the head is a single Linear) but emits one logit per patch
instead of keep/size distributions.  It does NOT touch the RL modules.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class BoxHeatmap(nn.Module):
    """Per-patch box-membership logit over frozen per-patch features.

    Args:
        in_dim:        Per-patch input width (contrastive_dim + attn + patch_logit).
        width:         Transformer / hidden width.
        depth:         Number of transformer encoder layers (the capacity).
        n_heads:       Attention heads — the cross-patch sharing that lets a patch
                       know whether it sits inside the box.
        max_positions: Upper bound on patch count for the learned position table.
        bias_init:     Initial head bias; negative ⇒ the heatmap starts mostly-OFF,
                       matching the sparse (few-1s) target.
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
        bias_init: float = -2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)

        self.in_proj = nn.Linear(in_dim, width)
        # Learned positional table — the head must know WHERE each patch sits to
        # decide box membership; DINO patch order is row-major so a per-position
        # embedding (sliced to N) suffices.
        self.pos = nn.Parameter(torch.zeros(max_positions, width))
        nn.init.normal_(self.pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads, dim_feedforward=width * 2,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

        self.head = nn.Linear(width, 1)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.constant_(self.head.bias, float(bias_init))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """(N, in_dim) per-patch features → (N,) per-patch box-membership logit.

        feats is a single image's patch grid (no batch dim); positions 0..N-1 map
        to row-major patch order.
        """
        if feats.dim() != 2:
            raise ValueError(f'BoxHeatmap.forward expects (N, in_dim), got {tuple(feats.shape)}')
        n = feats.shape[0]
        if n > self.pos.shape[0]:
            raise ValueError(f'BoxHeatmap: N={n} exceeds max_positions={self.pos.shape[0]}')

        h = self.in_proj(feats) + self.pos[:n]            # (N, width)
        h = self.encoder(h.unsqueeze(0)).squeeze(0)        # (N, width)
        return self.head(h).squeeze(-1)                    # (N,)


def build_box_heatmap(in_dim: int, *, device=None, **kwargs) -> BoxHeatmap:
    """Construct a BoxHeatmap and (optionally) move it to ``device``."""
    head = BoxHeatmap(in_dim, **kwargs)
    if device is not None:
        head = head.to(device)
    return head
