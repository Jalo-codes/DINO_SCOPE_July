"""lab_utils.data.sampling — deterministic subsets and per-item sample weights.

TORCH-FREE (GAMEPLAN C3). Ported from legacy/lab_utils/data/sampling.py.

Key change from legacy: operates on ``list[Item]`` (using Item.is_real,
Item.source, Item.item_id, Item.meta) instead of ``list[dict]`` with
free-form 'kind' / 'case_id' / 'img' keys.  The 'kind' string is replaced
by the binary ``item.is_real`` predicate and ``item.source``.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

from lab_utils.data.item import Item


# ---------------------------------------------------------------------------
# Sort key
# ---------------------------------------------------------------------------

def stable_item_sort_key(item: Item) -> str:
    """Deterministic sort key. Uses item_id which is already a stable MD5."""
    return item.item_id


# ---------------------------------------------------------------------------
# Balance weights
# ---------------------------------------------------------------------------

def splice_balance_weights(
    items: Sequence[Item],
    *,
    target_splice_frac: float = 0.5,
    return_stats: bool = False,
):
    """Per-item sampler weights balancing splice vs real items.

    The default (``target_splice_frac=0.5``) gives equal total weight to each
    class.  Pass a higher value to bias toward splice positives.

    Returns:
        weights (list[float]) or (weights, stats) when return_stats=True.
    """
    target     = float(min(1.0, max(0.0, target_splice_frac)))
    single_frac = 1.0 - target

    buckets = ["splice" if not item.is_real else "real" for item in items]
    counts  = Counter(buckets)
    n_splice = counts.get("splice", 0)
    n_real   = counts.get("real",   0)

    if n_splice == 0 or n_real == 0:
        weights = [1.0] * len(items)
    else:
        class_mass = {"splice": target, "real": single_frac}
        weights    = [class_mass[b] / max(1, counts[b]) for b in buckets]

    if not return_stats:
        return weights

    stats = dict(sorted(counts.items()))
    stats["target_splice_frac"] = target
    stats["target_real_frac"]   = single_frac
    return weights, stats


def source_splice_balance_weights(
    items: Sequence[Item],
    source_fracs: Dict[str, float],
    *,
    target_splice_frac: float = 0.5,
) -> Tuple[List[float], Dict[str, float]]:
    """Sampler weights controlling splice mix by source.

    Two-level allocation:
      1. Splice items receive ``target_splice_frac`` of total draw mass.
      2. Within splice, mass is split across sources by ``source_fracs``
         (normalized to sum 1).  A source absent from source_fracs gets 0.

    Real items share uniform weight summing to ``1 - target_splice_frac``.

    Returns:
        (weights, stats)
    """
    target = float(min(1.0, max(0.0, target_splice_frac)))

    total_frac = sum(max(0.0, f) for f in source_fracs.values())
    fracs = (
        {s: max(0.0, f) / total_frac for s, f in source_fracs.items()}
        if total_frac > 0 else {}
    )

    splice_idx_by_src: Dict[str, List[int]] = {}
    real_idx: List[int] = []
    for i, item in enumerate(items):
        if not item.is_real:
            splice_idx_by_src.setdefault(item.source, []).append(i)
        else:
            real_idx.append(i)

    weights = [0.0] * len(items)

    for src, idxs in splice_idx_by_src.items():
        f = fracs.get(src, 0.0)
        if f <= 0.0 or not idxs:
            continue
        w = (target * f) / len(idxs)
        for i in idxs:
            weights[i] = w

    if real_idx:
        wr = (1.0 - target) / len(real_idx)
        for i in real_idx:
            weights[i] = wr

    if not any(w > 0.0 for w in weights):
        weights = [1.0] * len(items)

    excluded = sorted(s for s in splice_idx_by_src if fracs.get(s, 0.0) <= 0.0)
    stats: Dict[str, float] = {
        "target_splice_frac": target,
        "n_real": float(len(real_idx)),
    }
    for src in sorted(splice_idx_by_src):
        stats[f"n_splice[{src}]"]  = float(len(splice_idx_by_src[src]))
        stats[f"frac[{src}]"]      = float(fracs.get(src, 0.0))
    if excluded:
        stats["excluded_sources"] = ",".join(excluded)  # type: ignore[assignment]
    return weights, stats


# ---------------------------------------------------------------------------
# Val-set counters
# ---------------------------------------------------------------------------

def val_mix_counts(items: Sequence[Item]) -> Dict[Tuple[str, str], int]:
    """Counter over (source, 'real'|'splice') pairs, sorted for stable logging."""
    return dict(sorted(
        Counter(
            (item.source, "real" if item.is_real else "splice")
            for item in items
        ).items(),
        key=lambda kv: str(kv[0]),
    ))


def val_source_counts(items: Sequence[Item]) -> Dict[str, int]:
    """Counter over source, sorted for stable logging."""
    return dict(sorted(
        Counter(item.source for item in items).items(),
        key=lambda kv: str(kv[0]),
    ))


def items_for_source(items: Iterable[Item], source: str) -> List[Item]:
    """Filter to items whose source matches."""
    return [item for item in items if item.source == source]


# ---------------------------------------------------------------------------
# Val subset builders
# ---------------------------------------------------------------------------

def build_quick_val_items(
    items: Sequence[Item],
    cap: int,
) -> List[Item]:
    """Deterministic stratified val subset over (source, real/splice).

    Groups by (source, 'real'|'splice'), allocates quotas proportionally,
    then deterministically picks via stable_item_sort_key.  Pass cap <= 0
    to disable capping.
    """
    if cap <= 0 or len(items) <= cap:
        return list(items)

    groups: Dict[Tuple[str, str], List[Item]] = {}
    for item in items:
        key = (item.source, "real" if item.is_real else "splice")
        groups.setdefault(key, []).append(item)

    total    = len(items)
    targets: Dict[Tuple[str, str], int] = {}
    remainders: List[Tuple[float, Tuple[str, str]]] = []
    assigned = 0
    for key, group_items in groups.items():
        exact = float(cap) * float(len(group_items)) / float(total)
        take  = min(len(group_items), int(exact))
        if take == 0 and len(group_items) > 0:
            take = 1
        targets[key] = take
        assigned    += take
        remainders.append((exact - int(exact), key))

    if assigned > cap:
        for _frac, key in sorted(remainders, key=lambda x: (x[0], str(x[1]))):
            if assigned <= cap:
                break
            if targets[key] > 1:
                targets[key] -= 1
                assigned -= 1

    if assigned < cap:
        for _frac, key in sorted(remainders, key=lambda x: (-x[0], str(x[1]))):
            if assigned >= cap:
                break
            room = len(groups[key]) - targets[key]
            if room <= 0:
                continue
            add = min(room, cap - assigned)
            targets[key] += add
            assigned     += add

    chosen: List[Item] = []
    for key, group_items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        ordered = sorted(group_items, key=stable_item_sort_key)
        chosen.extend(ordered[:targets[key]])

    return sorted(chosen, key=stable_item_sort_key)[:cap]


def build_case_balanced_quick_val_items(
    items: Sequence[Item],
    *,
    sources_and_caps: Dict[str, int],
) -> List[Item]:
    """Quick val with N items per source, balanced real/splice within each source.

    For each source in sources_and_caps, selects up to cap items split evenly
    between reals and splices.  Items are chosen deterministically via
    stable_item_sort_key.  Sources absent from the list are excluded.

    Args:
        items:           Full item list.
        sources_and_caps: e.g. {'imd2020': 200, 'casia': 100}

    Returns:
        Sorted (stable_item_sort_key) list of selected items.
    """
    chosen: List[Item] = []
    for source, cap in sources_and_caps.items():
        src_items = items_for_source(items, source)
        reals   = sorted([it for it in src_items if it.is_real],     key=stable_item_sort_key)
        splices = sorted([it for it in src_items if not it.is_real],  key=stable_item_sort_key)
        half = max(1, cap // 2)
        chosen.extend(reals[:half])
        chosen.extend(splices[:cap - min(len(reals), half)])
    return sorted(chosen, key=stable_item_sort_key)


# ---------------------------------------------------------------------------
# Deterministic subsamplers
# ---------------------------------------------------------------------------

def deterministic_subsample(
    items: Sequence[Item],
    n: int,
    *,
    seed: str,
) -> List[Item]:
    """Deterministic subsample by hashing (seed, item_id).

    Returns ``items`` truncated to ``n``.  When ``len(items) <= n``,
    returns the full list unchanged.  Stable across runs with the same seed.
    """
    if not items or len(items) <= n:
        return list(items)

    def _key(item: Item) -> str:
        return hashlib.md5(f"{seed}|{item.item_id}".encode("utf-8")).hexdigest()

    return sorted(items, key=_key)[:n]


def reals_subsample(
    items: Sequence[Item],
    rate: float,
    *,
    seed: str,
) -> List[Item]:
    """Keep approximately ``rate`` fraction of items deterministically.

    ``rate >= 1.0`` returns the full list; ``rate <= 0.0`` returns one item
    (preserving legacy behavior of not returning empty lists).
    """
    if not items or rate >= 1.0:
        return list(items)
    target = max(1, int(round(len(items) * float(rate))))
    return deterministic_subsample(items, target, seed=seed + "|reals")
