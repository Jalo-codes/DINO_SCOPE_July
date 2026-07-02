"""lab_utils.model.zoom_box_head — dense per-patch box + confidence head (FCOS-style),
trained as a contextual bandit.  See docs/zoom_box_spec.md.

A self-attention encoder over the frozen per-patch features ``[z | attn | patch_logit]``
gives each patch enough global context to perceive a splice's *extent* (a per-patch
scalar field could only say "hot/not" — it cannot regress a box).  Two per-patch heads
sit on the encoded tokens:

  * box  → 4 NON-NEGATIVE distances (top, left, bottom, right) in FRACTION units, anchored
           at the patch centre (FCOS parametrisation) → a fractional bbox (y0,x0,y1,x1).
           Neighbouring patches over one splice see similar context ⇒ regress nearly the
           SAME absolute box (consensus, not contradiction — DINO-native).
  * conf → a per-patch scalar = predicted zoom-ADVANTAGE of that box.  Drives the
           NMS ranking and the δ-gate at decode.

By default the box and conf heads use SEPARATE encoder trunks (``shared_trunk=False``).
This is deliberate: the confidence target jumps scale at the warm-start→bandit hand-off
(BCE-ish logits → small advantages), so a shared trunk lets the (large) conf gradient
corrupt the (delicate) box features and collapse the boxes.  Separate trunks isolate them.

The bandit reward, AWR candidate search, NMS decode, and eval live in
experiments/labs/zoom_box_lab.py.  This module is just the parameters + the box geometry.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def patch_centers(grid_hw: Tuple[int, int], *, device=None, dtype=torch.float32) -> torch.Tensor:
    """(N, 2) fractional (y, x) centre of each row-major patch on a ``grid_hw`` grid."""
    n_rows, n_cols = int(grid_hw[0]), int(grid_hw[1])
    ys = (torch.arange(n_rows, device=device, dtype=dtype) + 0.5) / n_rows
    xs = (torch.arange(n_cols, device=device, dtype=dtype) + 0.5) / n_cols
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)          # (N, 2)


def boxes_from_distances(dist: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """(N, 4) non-negative distances (dt, dl, db, dr) + (N, 2) centres → (N, 4) frac boxes
    (y0, x0, y1, x1), clamped to [0, 1] and side-ordered."""
    cy, cx = centers[:, 0], centers[:, 1]
    dt, dl, db, dr = dist[:, 0], dist[:, 1], dist[:, 2], dist[:, 3]
    y0 = (cy - dt).clamp(0.0, 1.0)
    x0 = (cx - dl).clamp(0.0, 1.0)
    y1 = (cy + db).clamp(0.0, 1.0)
    x1 = (cx + dr).clamp(0.0, 1.0)
    return torch.stack([torch.minimum(y0, y1), torch.minimum(x0, x1),
                        torch.maximum(y0, y1), torch.maximum(x0, x1)], dim=-1)


class _Trunk(nn.Module):
    """Per-patch self-attention encoder (mirrors the BoxHeatmap trunk)."""

    def __init__(self, in_dim, width, depth, n_heads, max_positions, dropout):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, width)
        self.pos = nn.Parameter(torch.zeros(max_positions, width))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=n_heads, dim_feedforward=width * 2,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.dim() != 2:
            raise ValueError(f'_Trunk expects (N, in_dim), got {tuple(feats.shape)}')
        n = feats.shape[0]
        if n > self.pos.shape[0]:
            raise ValueError(f'_Trunk: N={n} exceeds max_positions={self.pos.shape[0]}')
        h = self.in_proj(feats) + self.pos[:n]                 # (N, width)
        return self.encoder(h.unsqueeze(0)).squeeze(0)         # (N, width)


class ZoomBoxHead(nn.Module):
    """Encoder trunk(s) + a 4-d box head and a 1-d confidence head.

    Args:
        in_dim:        per-patch input width (contrastive_dim + attn + patch_logit).
        width/depth/n_heads: transformer trunk size.
        max_positions: upper bound on patch count for the learned position table.
        dist_bias:     box-head bias (pre-sigmoid); the cold-start per-side distance is
                       min + (max−min)·sigmoid(dist_bias).  Default -1.0 ⇒ ~0.27 of the
                       range ⇒ a small-ish starting box.
        conf_bias:     confidence-head bias (a raw advantage scalar, starts ~0).
        min_box_half / max_box_half:  per-side distance is BOUNDED to [min, max] frac via
                       a sigmoid.  This keeps every box a GENUINE zoom window — it can
                       neither collapse to ~0 nor inflate to the whole frame (the no-op
                       escape).  "Whether to zoom" is the gate's job, not the geometry's.
        shared_trunk:  if True, box and conf share one encoder (legacy; risks the conf
                       gradient corrupting box features).  Default False ⇒ separate.
    """

    def __init__(
        self,
        in_dim: int,
        *,
        width: int = 128,
        depth: int = 2,
        n_heads: int = 4,
        max_positions: int = 2048,
        dropout: float = 0.0,
        dist_bias: float = -1.0,
        conf_bias: float = 0.0,
        min_box_half: float = 0.04,
        max_box_half: float = 0.35,
        shared_trunk: bool = False,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.width = int(width)
        self.min_box_half = float(min_box_half)
        self.max_box_half = float(max_box_half)
        if not self.max_box_half > self.min_box_half:
            raise ValueError(f'max_box_half ({max_box_half}) must exceed min_box_half ({min_box_half})')
        self.shared_trunk = bool(shared_trunk)

        self.box_trunk = _Trunk(in_dim, width, depth, n_heads, max_positions, dropout)
        self.conf_trunk = self.box_trunk if shared_trunk else _Trunk(
            in_dim, width, depth, n_heads, max_positions, dropout)

        self.box_head = nn.Linear(width, 4)
        nn.init.normal_(self.box_head.weight, std=0.02)
        nn.init.constant_(self.box_head.bias, float(dist_bias))

        self.conf_head = nn.Linear(width, 1)
        nn.init.normal_(self.conf_head.weight, std=0.02)
        nn.init.constant_(self.conf_head.bias, float(conf_bias))

    def distances(self, raw: torch.Tensor) -> torch.Tensor:
        """Raw box logits → per-side distances bounded to [min_box_half, max_box_half]
        via a sigmoid (a genuine zoom window — no collapse, no whole-frame inflation)."""
        return self.min_box_half + (self.max_box_half - self.min_box_half) * torch.sigmoid(raw)

    def box_logits(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(N, in_dim) → (raw_dist (N,4) PRE-softplus, conf (N,)).

        Pre-softplus distances are returned so the caller can add exploration noise in
        the unconstrained space and re-apply softplus (keeps samples ≥ 0)."""
        raw = self.box_head(self.box_trunk(feats))
        conf = self.conf_head(self.conf_trunk(feats)).squeeze(-1)
        return raw, conf

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(N, in_dim) → (dist (N,4) ≥ 0 frac distances, conf (N,))."""
        raw, conf = self.box_logits(feats)
        return self.distances(raw), conf


def build_zoom_box_head(in_dim: int, *, device=None, **kwargs) -> ZoomBoxHead:
    """Construct a ZoomBoxHead and (optionally) move it to ``device``."""
    head = ZoomBoxHead(in_dim, **kwargs)
    if device is not None:
        head = head.to(device)
    return head
