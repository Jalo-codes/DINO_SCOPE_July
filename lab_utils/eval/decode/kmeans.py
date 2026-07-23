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


# ── 1-D two-means (exact, deterministic) ───────────────────────────────────────

def _kmeans2_1d(x: np.ndarray) -> np.ndarray:
    """Exact 1-D k=2 clustering.

    The optimal 1-D 2-means partition is a threshold between two adjacent
    sorted values that minimises total within-cluster SSE (equivalently the
    Jenks/Otsu natural break).  Solved in closed form via prefix sums — no
    random restarts, fully deterministic.

    Returns:
        (N,) int labels in {0, 1}; cluster 1 is the UPPER (larger-value) group.
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.int64)
    order = np.argsort(x, kind='mergesort')
    xs    = x[order]
    csum  = np.cumsum(xs)
    csum2 = np.cumsum(xs * xs)
    total, total2 = csum[-1], csum2[-1]
    # split after sorted index i (i in 1..n-1): lower = xs[:i], upper = xs[i:]
    i     = np.arange(1, n)
    n_lo  = i.astype(np.float64)
    n_hi  = (n - i).astype(np.float64)
    s_lo  = csum[:-1]
    s2_lo = csum2[:-1]
    s_hi  = total - s_lo
    s2_hi = total2 - s2_lo
    sse   = (s2_lo - s_lo * s_lo / n_lo) + (s2_hi - s_hi * s_hi / n_hi)
    cut   = int(np.argmin(sse)) + 1          # size of lower group
    labels_sorted            = np.zeros(n, dtype=np.int64)
    labels_sorted[cut:]      = 1
    labels                   = np.empty(n, dtype=np.int64)
    labels[order]            = labels_sorted
    return labels


# ── Public decode functions ────────────────────────────────────────────────────

def decode_kmeans_logit(info: ModelInfo) -> np.ndarray:
    """k=2 clustering on the BCE head's own per-patch logits ("learned vector").

    Clusters ``patch_logits`` (the scalar ``w·f + b`` the trained patch head
    emits) into two groups and picks the splice cluster by attention polarity —
    the same GT-free polarity rule as ``decode_kmeans``.  This is the
    self-calibrating (per-crop adaptive) counterpart to ``decode_threshold``:
    identical axis, but the split is chosen by the data instead of a fixed t.

    Requires the patch-BCE head (``patch_logits`` not None).
    """
    if info.patch_logits is None:
        raise ValueError(
            'decode_kmeans_logit: ModelInfo.patch_logits is None '
            '(patch-BCE head not enabled in this model).'
        )
    x          = np.asarray(info.patch_logits, dtype=np.float64).reshape(-1)
    n_side     = info.grid_hw[0]
    raw_labels = _kmeans2_1d(x)
    mask       = polarity_attn(raw_labels, info.attention)
    return mask.reshape(n_side, n_side)


def decode_kmeans_feats(info: ModelInfo, *, n_init: int = 4) -> np.ndarray:
    """Spherical k=2 clustering on the raw backbone patch features ("whole vector").

    Same clustering + attention-polarity as ``decode_kmeans``, but over the
    L2-normalised raw ``patch_feats`` (the full ViT feature vector) instead of
    the contrastive projector output.  For checkpoints trained WITHOUT a
    contrastive head (``contrastive_dim=0``) this reads the trained backbone
    geometry directly, with no random-projector bottleneck.

    Requires ``model_info(..., return_feats=True)`` so ``patch_feats`` is set.
    """
    if info.patch_feats is None:
        raise ValueError(
            'decode_kmeans_feats: ModelInfo.patch_feats is None '
            '(call model_info(..., return_feats=True)).'
        )
    z          = np.ascontiguousarray(info.patch_feats, dtype=np.float32)
    norms      = np.linalg.norm(z, axis=1, keepdims=True)
    z          = z / np.clip(norms, a_min=1e-12, a_max=None)
    n_side     = info.grid_hw[0]
    raw_labels, _ = spherical_kmeans2(z, n_init=n_init)
    mask       = polarity_attn(raw_labels, info.attention)
    return mask.reshape(n_side, n_side)


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
