"""lab_utils.model.zoom_head — learned zoom head (projection + per-cluster value).

Implements the locked design in docs/zoom_head_spec.md. Two LIGHT learned heads
over a FROZEN detector's per-patch signal:

  * ProjectionHead  z (frozen contrastive, 64-d) → z' (low-d, L2-normed).
        A small MLP (NOT linear — depth>=2 by default), reducing dimension to a
        zoom-coherent space where HDBSCAN is clean (density estimation degrades
        in 64-d). Trained on a GT-instance metric loss (pull within-instance,
        push across) — see zoom_cluster_lab.projection_instance_loss.
  * ZoomValueHead   per-patch features [z|attn|patch_logit] → per-patch scalar.
        Aggregated per HDBSCAN region → predicted zoom-ADVANTAGE for that region
        (regressed toward realized F1-improvement-over-baseline). Reuses the
        BoxHeatmap transformer trunk (raw scalar output, no sigmoid — advantage
        is a real number, not a probability).

The clustering, advantage scoring, gate (δ), and training/eval orchestration live
in experiments/labs/zoom_cluster_lab.py. This module is just the parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lab_utils.model.box_heatmap import BoxHeatmap


class ProjectionHead(nn.Module):
    """Frozen contrastive `z` (in_dim) → zoom-clustering space `z'` (out_dim), L2-normed.

    Args:
        in_dim:   input width (the detector's contrastive_dim, e.g. 64).
        out_dim:  projected width (default 32 — kept above the 8-16 that proved
                  too shallow; HDBSCAN wants a few stable density dims).
        hidden:   hidden width.
        depth:    number of hidden layers (>=1 ⇒ a real MLP; 0 ⇒ bare linear).
        dropout:  hidden dropout.
    """

    def __init__(self, in_dim: int, *, out_dim: int = 32, hidden: int = 128,
                 depth: int = 2, dropout: float = 0.0):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        layers = []
        d = int(in_dim)
        for _ in range(max(0, int(depth))):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = hidden
        layers += [nn.Linear(d, int(out_dim))]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(N, in_dim) → (N, out_dim), L2-normalized per patch."""
        if z.dim() != 2:
            raise ValueError(f'ProjectionHead expects (N, in_dim), got {tuple(z.shape)}')
        return F.normalize(self.net(z), p=2, dim=-1)


class ZoomHead(nn.Module):
    """Container: a ProjectionHead (z→z' for clustering) + a BoxHeatmap value head
    (per-patch advantage scalar). The two are trained on different objectives
    (projection: instance metric loss; value: per-region advantage regression),
    so callers invoke them separately via .project() / .value_logit().
    """

    def __init__(
        self,
        emb_dim: int,
        value_in_dim: int,
        *,
        proj_dim: int = 32,
        proj_hidden: int = 128,
        proj_depth: int = 2,
        proj_dropout: float = 0.0,
        value_width: int = 128,
        value_depth: int = 2,
        value_heads: int = 4,
        value_dropout: float = 0.1,
        value_bias_init: float = 0.0,   # advantage ~centered at 0 (not a sparse box logit)
        max_positions: int = 4096,      # >= patch count at 688 res (43*43=1849)
    ):
        super().__init__()
        self.emb_dim = int(emb_dim)
        self.value_in_dim = int(value_in_dim)
        self.proj = ProjectionHead(emb_dim, out_dim=proj_dim, hidden=proj_hidden,
                                   depth=proj_depth, dropout=proj_dropout)
        self.value = BoxHeatmap(value_in_dim, width=value_width, depth=value_depth,
                                n_heads=value_heads, max_positions=max_positions,
                                bias_init=value_bias_init, dropout=value_dropout)

    def project(self, emb: torch.Tensor) -> torch.Tensor:
        """(N, emb_dim) frozen contrastive z → (N, proj_dim) L2-normed z'."""
        return self.proj(emb)

    def value_logit(self, feats: torch.Tensor) -> torch.Tensor:
        """(N, value_in_dim) per-patch features → (N,) per-patch advantage scalar."""
        return self.value(feats)


def build_zoom_head(emb_dim: int, value_in_dim: int, *, device=None, **kwargs) -> ZoomHead:
    """Construct a ZoomHead and (optionally) move it to ``device``."""
    head = ZoomHead(emb_dim, value_in_dim, **kwargs)
    if device is not None:
        head = head.to(device)
    return head
