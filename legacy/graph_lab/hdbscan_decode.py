"""graph_lab.hdbscan_decode — density-based (HDBSCAN) decode for the sandbox.

Sandbox-only. Uses scikit-learn's HDBSCAN (a dependency the shipped numpy-only
decode forbids — keep this out of the production path). The idea, matching the
diagnosis we landed on:

  * cluster the L2-normalized patch embeddings by *mutual-reachability* density,
    which inflates distances through sparse boundary patches → resists the
    slow-gradient chaining that flat-threshold connected-components leaks through;
  * patches in no dense cluster get the noise label (-1) → natural abstention;
  * the background falls out as the largest (or lowest-attention) stable cluster
    — NOT a fragile largest-connected-component, and the muddy shards get
    re-absorbed into it by the hierarchy instead of becoming 300 fragments;
  * a non-background cluster is emitted as splice when its mean cosine similarity
    to the background (``cross``) is low enough (``theta_x``) — the one signal the
    granular dump showed actually separates.

Returns the SAME ``(mask_bool, info)`` contract as the graph decode so the
sandbox's component overlay and IoU helpers work unchanged.

Distance note: the embeddings are unit-norm, so Euclidean distance is monotonic
with cosine (||a-b||² = 2 - 2cos) — HDBSCAN's default metric is therefore
cosine-equivalent here, no precomputed matrix needed.
"""

from typing import Optional, Tuple

import numpy as np


def _load_hdbscan():
    """Return (backend_name, HDBSCAN_class) or (None, None) if unavailable."""
    try:
        from sklearn.cluster import HDBSCAN  # sklearn >= 1.3
        return 'sklearn', HDBSCAN
    except Exception:
        pass
    try:
        import hdbscan  # the standalone package
        return 'hdbscan', hdbscan.HDBSCAN
    except Exception:
        return None, None


def hdbscan_available() -> bool:
    return _load_hdbscan()[0] is not None


def _infer_square(n: int) -> Tuple[int, int]:
    s = int(round(n ** 0.5))
    if s * s != n:
        raise ValueError(f'hdbscan_decode: N={n} is not square; pass grid_hw for spatial features.')
    return s, s


def hdbscan_decode(
    z: np.ndarray,                          # (N, D) L2-normalized
    *,
    attention: Optional[np.ndarray] = None,
    grid_hw: Optional[Tuple[int, int]] = None,
    min_cluster_size: int = 8,
    min_samples: Optional[int] = None,
    spatial_weight: float = 0.0,            # >0 → append scaled (row,col) so adjacency matters
    theta_x: float = 0.5,                   # accept a non-bg cluster iff cross ≤ theta_x
    polarity: str = 'size',                 # 'size' | 'attention' — how to pick background
    min_patches: int = 16,
) -> Tuple[np.ndarray, dict]:
    n = int(z.shape[0])
    base_info = {
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
            'hdbscan_decode needs scikit-learn>=1.3 (sklearn.cluster.HDBSCAN) '
            "or the 'hdbscan' package. In Colab: already present with recent sklearn."
        )

    # Features: unit embeddings (Euclidean ≈ cosine). Optionally fold in grid
    # coordinates so spatially-adjacent patches are pulled together.
    feats = np.ascontiguousarray(z, dtype=np.float64)
    if spatial_weight and spatial_weight > 0.0:
        h, w = grid_hw or _infer_square(n)
        rr = (np.repeat(np.arange(h), w).astype(np.float64) / max(1, h - 1))
        cc = (np.tile(np.arange(w), h).astype(np.float64) / max(1, w - 1))
        coords = np.stack([rr, cc], axis=1) * float(spatial_weight)
        feats = np.concatenate([feats, coords], axis=1)

    kw = dict(min_cluster_size=int(max(2, min_cluster_size)))
    if min_samples is not None:
        kw['min_samples'] = int(min_samples)
    clusterer = HDB(**kw)
    labels = np.asarray(clusterer.fit_predict(feats), dtype=np.int64)

    uniq = [int(c) for c in np.unique(labels) if c != -1]
    n_noise = int((labels == -1).sum())
    if not uniq:
        info = dict(base_info)
        info.update({'labels': labels.copy(), 'n_noise': n_noise})
        return np.zeros(n, dtype=bool), info

    sim = z @ z.T                                    # cosine on the ORIGINAL embeddings
    sizes = {c: int((labels == c).sum()) for c in uniq}
    a = None if attention is None else np.asarray(attention, dtype=np.float64).reshape(-1)

    # Background = largest cluster, or (attention polarity) the cluster with the
    # lowest mean BCE attention — handles the >50%-splice regime on zoom crops.
    if polarity == 'attention' and a is not None:
        bg = min(uniq, key=lambda c: float(a[labels == c].mean()))
    else:
        bg = max(uniq, key=lambda c: sizes[c])
    bg_idx = np.where(labels == bg)[0]

    persistence = getattr(clusterer, 'cluster_persistence_', None)
    components, accept = [], np.zeros(n, dtype=bool)
    for c in uniq:
        if c == bg:
            continue
        idx = np.where(labels == c)[0]
        cross = float(sim[np.ix_(idx, bg_idx)].mean()) if bg_idx.size else 0.0
        within = (float(sim[np.ix_(idx, idx)][np.triu_indices(idx.size, 1)].mean())
                  if idx.size > 1 else 1.0)
        mean_att = float(a[idx].mean()) if a is not None else float('nan')
        accepted = bool(cross <= float(theta_x))
        stab = (float(persistence[c]) if (persistence is not None
                and 0 <= c < len(persistence)) else float('nan'))
        components.append({
            'comp_id': int(c), 'size': int(idx.size), 'within': within,
            'cross': cross, 'margin': within - cross, 'mean_attention': mean_att,
            'persistence': stab, 'accepted': accepted,
        })
        if accepted:
            accept[idx] = True

    # Fallback: if nothing was accepted but candidate clusters exist, accept the most probable one (lowest cross similarity)
    if accept.sum() == 0 and len(uniq) > 1:
        best_c = None
        min_cross = np.inf
        for comp in components:
            if comp['cross'] < min_cross:
                min_cross = comp['cross']
                best_c = comp['comp_id']
        if best_c is not None:
            idx = np.where(labels == best_c)[0]
            accept[idx] = True
            for comp in components:
                if comp['comp_id'] == best_c:
                    comp['accepted'] = True
                    break

    info = dict(base_info)
    info.update({
        'abstained': bool(accept.sum() == 0),
        'n_clusters': len(uniq), 'n_noise': n_noise,
        'background_id': int(bg), 'background_size': int(bg_idx.size),
        'labels': labels.copy(), 'components': components,
        'n_accepted': int(sum(c['accepted'] for c in components)),
    })
    return accept, info
