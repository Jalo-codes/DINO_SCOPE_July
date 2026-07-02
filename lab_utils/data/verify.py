"""lab_utils.data.verify — verifier gate between dataset pairing and use.

TORCH-FREE (GAMEPLAN C3). Uses only PIL and numpy.

Policy: drop-and-log (DESIGN_GUIDE §9.2). Failed items are filtered with a
logged reason + aggregate counts. A shape mismatch at __getitem__ time is the
only hard DataError; collection-time failures are soft drops.
"""

from __future__ import annotations

import dataclasses
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from lab_utils.data.item import Item
from lab_utils.logging.text import log_line

SKIP_VERIFY = object()  # sentinel: pass verify_policy=SKIP_VERIFY to skip all checks


@dataclasses.dataclass(frozen=True)
class Rejection:
    """Record of a dropped Item and the reason it was rejected."""
    item_id:  str
    source:   str
    path:     str
    reason:   str


@dataclasses.dataclass(frozen=True)
class VerifyPolicy:
    """Tunable thresholds for the verifier.

    Attributes:
        min_mask_area:  Minimum foreground fraction for a valid splice mask.
                        Items below this (too tiny to supervise) are dropped.
        max_mask_area:  Maximum foreground fraction — almost-all-white masks
                        usually indicate a labelling error.
        variance_floor: Minimum per-channel pixel variance (0-255 scale
                        squared) below which an image is flagged as
                        all-white/all-black/degenerate.
    """
    min_mask_area:  float = 0.001
    max_mask_area:  float = 0.99
    variance_floor: float = 25.0


DEFAULT_POLICY = VerifyPolicy()


def verify(
    item: Item,
    *,
    policy: Optional[VerifyPolicy] = DEFAULT_POLICY,
) -> Optional[str]:
    """Check one Item for validity.

    Returns:
        None if the item passes.
        A short reason string describing the first failure.
    """
    from PIL import Image, UnidentifiedImageError

    if policy is None:                  # dataset builders pass through None
        policy = DEFAULT_POLICY

    # 1. Manipulated image must exist.
    if not Path(item.image).exists():
        return "manipulated_missing"

    # 2. Image must load and decode without error.
    try:
        img = Image.open(item.image).convert('RGB')
        arr = np.asarray(img, dtype=np.float32)
    except (UnidentifiedImageError, OSError, Exception):
        return "image_corrupt"

    # 3. Image must not be degenerate (all-white / all-black / constant).
    if float(arr.var()) < float(policy.variance_floor):
        return "image_degenerate"

    # 4. Splice items must have a valid, non-empty mask.
    if not item.is_real:
        if item.mask is None or not Path(item.mask).exists():
            return "mask_missing"

        try:
            mask_pil = Image.open(item.mask).convert('L')
            mask_arr = np.asarray(mask_pil, dtype=np.uint8)
        except Exception:
            return "mask_corrupt"

        area = float((mask_arr > 0).mean())
        if area < float(policy.min_mask_area):
            return "mask_area_too_small"
        if area > float(policy.max_mask_area):
            return "mask_area_too_large"

    return None


def verify_all(
    items: List[Item],
    *,
    policy: Optional[VerifyPolicy] = DEFAULT_POLICY,
    log_tag: str = '[verify]',
    max_workers: Optional[int] = None,
) -> Tuple[List[Item], List[Rejection]]:
    """Gate an item list, dropping and logging invalid entries.

    Args:
        items:   List of Items to verify.
        policy:  Tunable thresholds.  Pass SKIP_VERIFY to bypass all checks
                 (useful in eval labs where the inference loop handles errors).
        log_tag: Prefix for log_line output.
        max_workers: Thread count for the (decode-bound) verify pass. Each
                 ``verify`` is independent and spends its time in PIL decode +
                 numpy, both of which release the GIL, so this scales with cores.
                 Defaults to ~2× CPU count (capped at 16). Pass 1 to force the
                 old serial path.

    Returns:
        (kept, rejected) — disjoint lists. The kept list preserves ordering.
    """
    if policy is SKIP_VERIFY:
        log_line(f'{log_tag} kept={len(items)} dropped=0/{len(items)} (verify skipped)')
        return list(items), []

    kept:     List[Item]      = []
    rejected: List[Rejection] = []

    reason_counts: dict = {}

    if max_workers is None:
        max_workers = min(16, (os.cpu_count() or 4) * 2)

    # executor.map preserves input order, so the kept list stays deterministic.
    if max_workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            reasons = list(ex.map(lambda it: verify(it, policy=policy), items))
    else:
        reasons = [verify(it, policy=policy) for it in items]

    for item, reason in zip(items, reasons):
        if reason is None:
            kept.append(item)
        else:
            rejected.append(Rejection(
                item_id=item.item_id,
                source=item.source,
                path=str(item.image),
                reason=reason,
            ))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

    log_line(
        f'{log_tag} kept={len(kept)} dropped={len(rejected)}/{len(items)}'
        + ('' if not reason_counts else
           ' reasons=' + ','.join(f'{r}:{c}' for r, c in sorted(reason_counts.items())))
    )
    return kept, rejected
