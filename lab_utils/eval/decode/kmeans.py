"""lab_utils.eval.decode.kmeans — spherical k-means (k=2) + attention polarity.

Pure, silent, GT-free.  Requires embeddings (contrastive head).

Polarity rule: the cluster with higher mean attention weight is the splice
prediction.  Falls back to the smaller-cluster rule when attention is None.
No oracle, no GT (decode_oracle_labels does not exist in this tree).
"""

from typing import Optional, Tuple

import numpy as np

from lab_utils.eval.fetch import ModelInfo


# ── Core clustering (ported from legacy/lab_utils/eval/partition.py) ──────────

def _init_centroids(z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n  = z.shape[0]
    i0 = int(rng.integers(0, n))
    c0 = z[i0]
    d  = np.clip(1.0 - z @ c0, a_min=0.0, a_max=None)
    if d.sum() <= 0:
        i1 = int(rng.integers(0, n))
    else:
        i1 = int(rng.choice(n, p=d / d.sum()))
    return np.stack([z[i0], z[i1]], axis=0)


def spherical_kmeans2(
    z: np.ndarray,
    n_init: int = 4,
    n_iters: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """Spherical k-means (k=2) on L2-normalised embeddings.

    Returns:
        (labels, inertia) — labels ∈ {0,1}^N, inertia = sum(1 − cos to centroid).
    """
    z = np.ascontiguousarray(z, dtype=np.float32)
    best_labels:  Optional[np.ndarray] = None
    best_inertia: float = np.inf

    for run in range(int(n_init)):
        rng       = np.random.default_rng(seed + run)
        centroids = _init_centroids(z, rng)
        labels    = np.zeros(z.shape[0], dtype=np.int64)

        for _ in range(int(n_iters)):
            sim        = z @ centroids.T
            new_labels = np.argmax(sim, axis=1)
            if (new_labels == labels).all():
                labels = new_labels
                break
            labels        = new_labels
            new_centroids = np.zeros_like(centroids)
            for k in (0, 1):
                mask = labels == k
                if mask.sum() == 0:
                    other = centroids[1 - k]
                    far   = int(np.argmin(z @ other))
                    new_centroids[k] = z[far]
                else:
                    mean = z[mask].mean(axis=0)
                    n    = np.linalg.norm(mean) + 1e-12
                    new_centroids[k] = mean / n
            centroids = new_centroids

        sim     = z @ centroids.T
        labels  = np.argmax(sim, axis=1)
        inertia = float((1.0 - sim[np.arange(z.shape[0]), labels]).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels  = labels

    return best_labels, best_inertia


def polarity_attn(
    raw_labels: np.ndarray,
    attention: Optional[np.ndarray],
) -> np.ndarray:
    """Pick the splice cluster by attention.  Falls back to smaller-cluster rule.

    Returns:
        (N,) bool — True for predicted-splice patches.
    """
    raw = np.asarray(raw_labels).reshape(-1)
    n0  = int((raw == 0).sum())
    n1  = int((raw == 1).sum())
    if attention is None:
        chosen = 0 if n0 <= n1 else 1
        return (raw == chosen)
    att   = np.asarray(attention).reshape(-1)
    mean0 = float(att[raw == 0].mean()) if n0 else float('-inf')
    mean1 = float(att[raw == 1].mean()) if n1 else float('-inf')
    chosen = 0 if mean0 >= mean1 else 1
    return (raw == chosen)


# ── Public decode function ─────────────────────────────────────────────────────

def decode_kmeans(info: ModelInfo, *, n_init: int = 4) -> np.ndarray:
    """Spherical k-means on embeddings; polarity by attention → (n_side, n_side) bool.

    Args:
        info:   ModelInfo.  embeddings must not be None.
        n_init: Number of k-means random restarts.

    Returns:
        (n_side, n_side) bool array — True = predicted-splice patch.

    Raises:
        ValueError: if embeddings is None (contrastive head not enabled).
    """
    if info.embeddings is None:
        raise ValueError(
            'decode_kmeans: ModelInfo.embeddings is None '
            '(contrastive head not enabled in this model).'
        )
    z      = np.ascontiguousarray(info.embeddings, dtype=np.float32)
    n_side = info.grid_hw[0]
    raw_labels, _ = spherical_kmeans2(z, n_init=n_init)
    mask = polarity_attn(raw_labels, info.attention)
    return mask.reshape(n_side, n_side)
