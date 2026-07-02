"""lab_utils.eval.decode.hdbscan — HDBSCAN density-cluster decode (main decode set).

Pure, silent, GT-free.  Requires embeddings (contrastive head).
Requires scikit-learn >= 1.3 (sklearn.cluster.HDBSCAN) or the standalone
'hdbscan' package.  hdbscan_available() returns False if neither is installed.

Ported from legacy/graph_lab/hdbscan_decode.py.

Distance note: unit-norm embeddings → Euclidean ≡ cosine distance (monotonic),
so HDBSCAN's default metric is cosine-equivalent.  No precomputed matrix needed.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from lab_utils.eval.fetch import ModelInfo


# ── Backend loader ─────────────────────────────────────────────────────────────

def _load_hdbscan():
    """Return (backend_name, HDBSCAN_class) or (None, None)."""
    try:
        from sklearn.cluster import HDBSCAN
        return 'sklearn', HDBSCAN
    except Exception:
        pass
    try:
        import hdbscan
        return 'hdbscan', hdbscan.HDBSCAN
    except Exception:
        return None, None


def hdbscan_available() -> bool:
    """True if a suitable HDBSCAN backend is installed."""
    return _load_hdbscan()[0] is not None


# ── Core decode (ported from legacy) ──────────────────────────────────────────

def _infer_square(n: int) -> Tuple[int, int]:
    s = int(round(n ** 0.5))
    if s * s != n:
        raise ValueError(
            f'hdbscan_decode: N={n} is not a perfect square; '
            'pass grid_hw explicitly for spatial features.'
        )
    return s, s


def hdbscan_decode(
    z: np.ndarray,
    *,
    attention: Optional[np.ndarray] = None,
    grid_hw: Optional[Tuple[int, int]] = None,
    min_cluster_size: int = 8,
    min_samples: Optional[int] = None,
    spatial_weight: float = 0.0,
    theta_x: float = 0.5,
    polarity: str = 'attention',
    min_patches: int = 16,
) -> Tuple[np.ndarray, dict]:
    """HDBSCAN decode on L2-normalised embeddings.

    Args:
        z:                (N, D) L2-normalised per-patch embeddings.
        attention:        (N,) per-patch attention weights, or None.
        grid_hw:          (h, w) patch grid; inferred as square when None.
        min_cluster_size: Minimum cluster size for HDBSCAN.
        min_samples:      min_samples for HDBSCAN (None = default).
        spatial_weight:   > 0 → append scaled grid coordinates to features.
        theta_x:          Accept a non-bg cluster iff cross-similarity ≤ theta_x.
        polarity:         'size' (largest = background) or 'attention' (lowest
                          mean attention = background, handles >50%-splice).
        min_patches:      Abstain outright when N < this.

    Returns:
        (mask_bool, info) — same contract as decode_graph.
    """
    n = int(z.shape[0])
    base_info: Dict = {
        'method': 'hdbscan', 'abstained': True, 'n_clusters': 0, 'n_noise': n,
        'components': [], 'm_min': 0, 'theta_x': float(theta_x),
        'min_cluster_size': int(min_cluster_size),
        'min_samples': (None if min_samples is None else int(min_samples)),
        'spatial_weight': float(spatial_weight), 'polarity': str(polarity),
    }
    if n < int(min_patches):
        return np.zeros(n, dtype=bool), base_info

    backend, HDB = _load_hdbscan()
    if backend is None:
        raise ImportError(
            'hdbscan_decode requires scikit-learn >= 1.3 (sklearn.cluster.HDBSCAN) '
            "or the standalone 'hdbscan' package."
        )

    feats = np.ascontiguousarray(z, dtype=np.float64)
    if spatial_weight > 0.0:
        h, w = grid_hw or _infer_square(n)
        rr    = np.repeat(np.arange(h), w).astype(np.float64) / max(1, h - 1)
        cc    = np.tile(np.arange(w), h).astype(np.float64) / max(1, w - 1)
        feats = np.concatenate([feats, np.stack([rr, cc], axis=1) * spatial_weight], axis=1)

    kw: Dict = dict(min_cluster_size=int(max(2, min_cluster_size)))
    if min_samples is not None:
        kw['min_samples'] = int(min_samples)
    if backend == 'sklearn':
        kw['copy'] = False
        kw['algorithm'] = 'brute'
        kw['n_jobs'] = -1
    elif backend == 'hdbscan':
        kw['algorithm'] = 'generic'
        kw['core_dist_n_jobs'] = -1
    clusterer = HDB(**kw)
    labels = np.asarray(clusterer.fit_predict(feats), dtype=np.int64)

    uniq    = [int(c) for c in np.unique(labels) if c != -1]
    n_noise = int((labels == -1).sum())
    if not uniq:
        info = dict(base_info)
        info.update({'labels': labels.copy(), 'n_noise': n_noise})
        return np.zeros(n, dtype=bool), info

    sim   = z @ z.T
    sizes = {c: int((labels == c).sum()) for c in uniq}
    a     = (None if attention is None
             else np.asarray(attention, dtype=np.float64).reshape(-1))

    if polarity == 'attention' and a is not None:
        bg = min(uniq, key=lambda c: float(a[labels == c].mean()))
    else:
        bg = max(uniq, key=lambda c: sizes[c])
    bg_idx = np.where(labels == bg)[0]

    persistence          = getattr(clusterer, 'cluster_persistence_', None)
    components: List[Dict] = []
    accept = np.zeros(n, dtype=bool)
    for c in uniq:
        if c == bg:
            continue
        idx    = np.where(labels == c)[0]
        cross  = float(sim[np.ix_(idx, bg_idx)].mean()) if bg_idx.size else 0.0
        within = (float(sim[np.ix_(idx, idx)][np.triu_indices(idx.size, 1)].mean())
                  if idx.size > 1 else 1.0)
        mean_att = float(a[idx].mean()) if a is not None else float('nan')
        accepted = bool(cross <= float(theta_x))
        stab     = (float(persistence[c])
                    if persistence is not None and 0 <= c < len(persistence)
                    else float('nan'))
        components.append({
            'comp_id': int(c), 'size': int(idx.size), 'within': within,
            'cross': cross, 'margin': within - cross,
            'mean_attention': mean_att, 'persistence': stab, 'accepted': accepted,
        })
        if accepted:
            accept[idx] = True

    if accept.sum() == 0 and len(uniq) > 1:
        best = min(components, key=lambda d: d['cross'])
        idx  = np.where(labels == best['comp_id'])[0]
        accept[idx] = True
        best['accepted'] = True

    info = dict(base_info)
    info.update({
        'abstained':       bool(accept.sum() == 0),
        'n_clusters':      len(uniq),
        'n_noise':         n_noise,
        'background_id':   int(bg),
        'background_size': int(bg_idx.size),
        'labels':          labels.copy(),
        'components':      components,
        'n_accepted':      int(sum(c['accepted'] for c in components)),
    })
    return accept, info


# ── Public decode function ─────────────────────────────────────────────────────

def decode_hdbscan(
    info: ModelInfo,
    *,
    min_cluster_size: int = 8,
    min_samples: Optional[int] = None,
    spatial_weight: float = 0.0,
    theta_x: float = 0.5,
    polarity: str = 'attention',
) -> np.ndarray:
    """HDBSCAN decode on embeddings → (n_side, n_side) bool mask.

    Args:
        info: ModelInfo.  embeddings must not be None.

    Returns:
        (n_side, n_side) bool array — True = predicted-splice patch.

    Raises:
        ValueError:  if embeddings is None (contrastive head not enabled).
        ImportError: if no HDBSCAN backend is available.
    """
    if info.embeddings is None:
        raise ValueError(
            'decode_hdbscan: ModelInfo.embeddings is None '
            '(contrastive head not enabled in this model).'
        )
    z      = np.ascontiguousarray(info.embeddings, dtype=np.float32)
    n_side = info.grid_hw[0]
    mask, _ = hdbscan_decode(
        z,
        attention=info.attention,
        grid_hw=info.grid_hw,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        spatial_weight=spatial_weight,
        theta_x=theta_x,
        polarity=polarity,
    )
    return np.asarray(mask, dtype=bool).reshape(n_side, n_side)
