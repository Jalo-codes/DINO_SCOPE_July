"""lab_utils.data.datasets.full_fakes — whole-image ("full fake") generation eval set.

Layout::

    root/real/           pristine photos (real negatives)
    root/<generator>/    fully AI-generated images from that generator — the
                          ENTIRE frame is synthetic; there is no splice
                          boundary and no source photo to point authentic at
                          (e.g. root/sdxl-juggernaut/0162.png)

Every non-real subfolder is one generator's output pool; its name becomes
Item.meta['generator']. experiments/scripts/eval.py already reads
meta['generator'] to set EvalRecord.subgroup (same convention TGIF uses), so
--subgroup filtering and summarize_by_subgroup() work with zero extra code.

NO ground-truth masks ship with this dataset — there is no localized region,
the whole frame IS the label. Item.is_real is defined as ``mask is None``
(lab_utils/data/item.py), so a fake item naively indexed with mask=None would
be silently mislabeled real. Every fake item instead gets a synthetic
full-frame mask (one shared all-white PNG, same trick as pico_banana.py's
sentinel), purely to keep is_real=False correct. meta['gt_mask_reliable'] =
False flags this as geometry-free (lab_utils/data/verify.py,
lab_utils/eval/metric.py both key off it) so patch-level precision/f1 on this
source are NOT meaningful (mechanically ~1.0 / high given an all-true GT).
recall and iou ARE meaningful despite the sentinel: since GT is all-true,
recall == iou == the predicted-positive pixel fraction — the localization
distribution this eval set exists to measure (how much of a wholly-fake
frame the model's patch head lights up). image_score / AUC is the real/fake
separability number.

Eval-only (mirrors sagid/pico_banana/unpaired): returns
(empty_train_dataset, val_dataset).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from lab_utils.data.item import Item, make_item_id
from lab_utils.data.dataset import Dataset
from lab_utils.data.verify import VerifyPolicy, verify_all
from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'})
_REAL_DIR_NAMES = {'real', 'reals'}

_SYNTHETIC_MASK_PATH = Path(tempfile.gettempdir()) / 'dino_scope_full_fakes_full_mask.png'


def _synthetic_full_mask() -> Path:
    """One shared all-white PNG, reused as every fake item's mask (see module docstring)."""
    if not _SYNTHETIC_MASK_PATH.exists():
        Image.new('L', (32, 32), 255).save(_SYNTHETIC_MASK_PATH)
    return _SYNTHETIC_MASK_PATH


def build(
    root: Path,
    *,
    res: Resolution,
    source: str = 'full_fakes',
    verify_policy: Optional[VerifyPolicy] = None,
    valid_exts: Optional[frozenset] = None,
) -> Tuple[Dataset, Dataset]:
    """Discover root/real/ + root/<generator>/ whole-image pools.

    verify_policy: the DEFAULT_POLICY rejects any mask covering >99% of the
    image (max_mask_area). Every fake item here carries the synthetic
    full-frame sentinel mask (100% coverage) by design, so when the caller
    does not pass an explicit policy this builder relaxes max_mask_area to
    1.0 so the sentinel survives verify_all.
    """
    root = Path(root)
    exts = valid_exts or _VALID_EXTS

    if not root.is_dir():
        log_line(f'[data] WARN: {source} root not found: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not subdirs:
        log_line(f'[data] WARN: {source} no subfolders found under {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    mask_path = _synthetic_full_mask()
    log_line(f'[data] Indexing {source} ({len(subdirs)} subfolders): {root}')

    items = []
    n_generators = 0
    for sub in subdirs:
        is_real_dir = sub.name.lower() in _REAL_DIR_NAMES
        if not is_real_dir:
            n_generators += 1
        files = sorted(f for f in sub.iterdir() if f.is_file() and f.suffix.lower() in exts)
        for f in files:
            if is_real_dir:
                items.append(Item(
                    image=f, authentic=None, mask=None, source=source,
                    item_id=make_item_id(source, f),
                    meta={'case_id': f'real_{f.stem}'},
                ))
            else:
                items.append(Item(
                    image=f, authentic=None, mask=mask_path, source=source,
                    item_id=make_item_id(source, f),
                    meta={'case_id': f'{sub.name}_{f.stem}', 'generator': sub.name,
                          'gt_mask_reliable': False},
                ))

    effective_policy = (
        verify_policy if verify_policy is not None else VerifyPolicy(max_mask_area=1.0)
    )
    kept, _ = verify_all(items, policy=effective_policy, log_tag=f'[data] {source}')

    n_real = sum(1 for it in kept if it.is_real)
    log_line(
        f'[data] {source}: loaded {len(kept)} items '
        f'(real={n_real} fake={len(kept) - n_real} across {n_generators} generators) '
        f'| NO GT masks (synthetic full-frame sentinel) — precision/f1 not meaningful; '
        f'use image_score/AUC and recall/iou (== predicted-positive pixel fraction)'
    )

    val_ds = Dataset(kept, res=res, augment=False)
    train_ds = Dataset([], res=res, augment=True)
    return train_ds, val_ds
