"""lab_utils.eval.decode.graph — calibrated similarity-graph decode.

Pure, silent, GT-free.  Requires embeddings (contrastive head).

Ported from legacy/lab_utils/eval/partition.py:graph_components_decode.
Thresholds pairwise similarity inside the trained dead band [tau_neg, tau_pos]
and takes connected components — multi-region, natural abstention (zero-mass
prediction when nothing clears the bar), no RNG, no restarts, fully deterministic.

Polarity: background = largest component; attention_polarity selects the
low-attention component as background in the >50%-splice regime.
"""

import dataclasses
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from lab_utils.eval.fetch import ModelInfo


# ── Graph spec ─────────────────────────────────────────────────────────────────

@dataclasses.dataclass(frozen=True)
class GraphSpec:
    """Parameters for the graph-components decode."""
    tau_pos:          float           = 0.55
    tau_neg:          float           = 0.20
    s_edge:           Optional[float] = None   # default = (tau_pos+tau_neg)/2
    mutual_knn_k:     int             = 10
    r_spatial:        Optional[int]   = None
    m_min:            int             = 4
    theta_w:          Optional[float] = None   # default = tau_pos - 0.05
    theta_x:          Optional[float] = None   # default = (tau_pos+tau_neg)/2
    attention_polarity: bool          = False
    min_patches:      int             = 16


# ── Internal helpers (ported from legacy) ─────────────────────────────────────

def _infer_grid_hw(n: int) -> Optional[Tuple[int, int]]:
    s = int(round(math.sqrt(n)))
    return (s, s) if s * s == n else None


def _union_find(adj: np.ndarray) -> np.ndarray:
    n      = adj.shape[0]
    parent = np.arange(n, dtype=np.int64)

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    src, dst = np.where(np.triu(adj, k=1))
    for a, b in zip(src.tolist(), dst.tolist()):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int64)


def graph_components_decode(
    z: np.ndarray,
    *,
    tau_pos: float,
    tau_neg: float,
    grid_hw: Optional[Tuple[int, int]] = None,
    s_edge: Optional[float] = None,
    mutual_knn_k: int = 10,
    r_spatial: Optional[int] = None,
    m_min: int = 4,
    theta_w: Optional[float] = None,
    theta_x: Optional[float] = None,
    attention: Optional[np.ndarray] = None,
    attention_polarity: bool = False,
    min_patches: int = 16,
) -> Tuple[np.ndarray, dict]:
    """Connected-components decode over a calibrated similarity graph.

    Returns:
        (mask, info) — mask is (N,) int {0,1}; 1 = accepted foreground (splice).
        all-zeros = abstain.
    """
    z       = np.ascontiguousarray(z, dtype=np.float32)
    n       = z.shape[0]
    s_edge  = float((tau_pos + tau_neg) / 2.0) if s_edge is None else float(s_edge)
    theta_w = float(tau_pos - 0.05)            if theta_w is None else float(theta_w)
    theta_x = float((tau_pos + tau_neg) / 2.0) if theta_x is None else float(theta_x)

    base_info: Dict = {
        'method': 'graph', 'abstained': True, 'n_components': 0,
        'background_size': 0, 'components': [],
        's_edge': s_edge, 'theta_w': theta_w, 'theta_x': theta_x,
        'mutual_knn_k': int(mutual_knn_k), 'r_spatial': r_spatial, 'm_min': int(m_min),
    }
    if n < int(min_patches):
        return np.zeros(n, dtype=np.int64), base_info

    norms = np.linalg.norm(z, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError(
            f'graph_components_decode: z rows must be L2-normalised '
            f'(norm range [{norms.min():.4f}, {norms.max():.4f}]).'
        )

    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)

    k = max(1, min(int(mutual_knn_k), n - 1))
    knn_idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    knn     = np.zeros((n, n), dtype=bool)
    rows    = np.repeat(np.arange(n), k)
    knn[rows, knn_idx.reshape(-1)] = True
    mutual  = knn & knn.T

    np.fill_diagonal(sim, 1.0)
    edges = (sim >= s_edge) & mutual
    np.fill_diagonal(edges, False)

    if r_spatial is not None:
        hw = grid_hw or _infer_grid_hw(n)
        if hw is None:
            raise ValueError(
                f'graph_components_decode: r_spatial set but grid_hw is None '
                f'and N={n} is not square — pass grid_hw explicitly.'
            )
        h, w = hw
        rr   = np.repeat(np.arange(h), w)
        cc   = np.tile(np.arange(w), h)
        cheb = np.maximum(np.abs(rr[:, None] - rr[None, :]),
                          np.abs(cc[:, None] - cc[None, :]))
        edges &= (cheb <= int(r_spatial))

    comp_labels   = _union_find(edges)
    comp_ids, comp_sizes = np.unique(comp_labels, return_counts=True)

    max_size = int(comp_sizes.max())
    tied     = [int(c) for c, s in zip(comp_ids, comp_sizes) if int(s) == max_size]
    bg_id    = tied[0] if len(tied) == 1 else min(
        tied, key=lambda c: int(np.where(comp_labels == c)[0].min())
    )

    if attention_polarity and attention is not None:
        a   = np.asarray(attention, dtype=np.float64).reshape(-1)
        big = [int(c) for c, s in zip(comp_ids, comp_sizes) if int(s) >= 0.2 * n]
        if len(big) >= 2 and a.shape[0] == n:
            bg_id = min(big, key=lambda c: float(a[comp_labels == c].mean()))

    bg_mask = comp_labels == bg_id
    bg_size = int(bg_mask.sum())
    bg_idx  = np.where(bg_mask)[0]

    def _within(idx: np.ndarray) -> float:
        if idx.size < 2:
            return 1.0
        sub = sim[np.ix_(idx, idx)]
        iu  = np.triu_indices(idx.size, k=1)
        return float(sub[iu].mean())

    components: List[Dict] = []
    accept = np.zeros(n, dtype=bool)
    for c, sz in zip(comp_ids.tolist(), comp_sizes.tolist()):
        if c == bg_id:
            continue
        idx    = np.where(comp_labels == c)[0]
        if idx.size < int(m_min):
            continue
        within = _within(idx)
        cross  = float(sim[np.ix_(idx, bg_idx)].mean()) if bg_idx.size else 0.0
        ok     = bool(within >= theta_w and cross <= theta_x)
        components.append({
            'comp_id': int(c), 'size': int(idx.size),
            'within': within, 'cross': cross, 'margin': within - cross, 'accepted': ok,
        })
        if ok:
            accept[idx] = True

    # fallback: accept the most-isolated candidate when nothing passed
    if accept.sum() == 0 and components:
        best = min(components, key=lambda d: d['cross'])
        idx  = np.where(comp_labels == best['comp_id'])[0]
        accept[idx] = True
        best['accepted'] = True

    mask = accept.astype(np.int64)
    info = dict(base_info)
    info.update({
        'abstained':       bool(mask.sum() == 0),
        'n_components':    int(comp_ids.size),
        'background_size': bg_size,
        'background_id':   int(bg_id),
        'labels':          comp_labels.copy(),
        'components':      components,
        'n_accepted':      int(sum(c['accepted'] for c in components)),
    })
    return mask, info


# ── Public decode function ─────────────────────────────────────────────────────

def decode_graph(info: ModelInfo, *, spec: Optional[GraphSpec] = None) -> np.ndarray:
    """Graph-components decode on embeddings → (n_side, n_side) bool mask.

    Args:
        info: ModelInfo.  embeddings must not be None.
        spec: GraphSpec (defaults to GraphSpec() with mid-band thresholds).

    Returns:
        (n_side, n_side) bool array — True = predicted-splice patch.

    Raises:
        ValueError: if embeddings is None (contrastive head not enabled).
    """
    if info.embeddings is None:
        raise ValueError(
            'decode_graph: ModelInfo.embeddings is None '
            '(contrastive head not enabled in this model).'
        )
    if spec is None:
        spec = GraphSpec()
    z      = np.ascontiguousarray(info.embeddings, dtype=np.float32)
    n_side = info.grid_hw[0]
    mask, _ = graph_components_decode(
        z,
        tau_pos=spec.tau_pos,
        tau_neg=spec.tau_neg,
        grid_hw=info.grid_hw,
        s_edge=spec.s_edge,
        mutual_knn_k=spec.mutual_knn_k,
        r_spatial=spec.r_spatial,
        m_min=spec.m_min,
        theta_w=spec.theta_w,
        theta_x=spec.theta_x,
        attention=info.attention,
        attention_polarity=spec.attention_polarity,
        min_patches=spec.min_patches,
    )
    return mask.astype(bool).reshape(n_side, n_side)
