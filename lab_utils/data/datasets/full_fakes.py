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

Eval-only BY DEFAULT (``val_split=1.0``, the same idiom region_probes' sp_*
wrappers use), which is how this builder began life. Pass ``val_split=0.0`` to
index a root entirely as TRAIN — used to train on whole-image fakes (the
OpenFake train split). Intermediate values give a stratified internal split.
"""

from __future__ import annotations

import random
import tempfile
from collections import defaultdict
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
    val_split: float = 1.0,
    split_seed: int = 42,
    val_per_pool: Optional[int] = None,
    val_real_cap: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """Discover root/real/ + root/<generator>/ whole-image pools.

    verify_policy: the DEFAULT_POLICY rejects any mask covering >99% of the
    image (max_mask_area). Every fake item here carries the synthetic
    full-frame sentinel mask (100% coverage) by design, so when the caller
    does not pass an explicit policy this builder relaxes max_mask_area to
    1.0 so the sentinel survives verify_all.

    val_split: fraction routed to the VAL dataset. Defaults to 1.0 —
    eval-only, this builder's original and still most common use. 0.0 sends
    everything to TRAIN, which is the mode that matters for OpenFake, whose
    train / validation / test splits are SEPARATE DOWNLOADS carrying semantics
    an internal random split cannot reproduce (validation = held-out images
    from the training generators; test = held-out GENERATORS). Point one root
    per split. Intermediate values split internally, stratified by generator
    so a small val slice cannot silently drop a generator entirely.

    val_per_pool / val_real_cap: cap the VAL side to N items per generator pool
    and M reals. Pools are wildly uneven (200 images down to 3), so an uncapped
    val is both slow and dominated by whichever generators happen to be large.
    Applied AFTER the split and selected deterministically from split_seed, so
    the val set is identical every epoch and across runs — a moving eval set
    would make epoch-to-epoch deltas unreadable. A pool smaller than its cap is
    taken whole, never padded.
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

    frac = float(val_split)
    if frac >= 1.0:
        train_split, val_split_items = [], list(kept)
    elif frac <= 0.0:
        train_split, val_split_items = list(kept), []
    else:
        # Stratify by generator ('real' is its own stratum) so a small val
        # slice cannot miss a generator entirely — the pools are per-generator
        # and can be small.
        by_gen = defaultdict(list)
        for it in kept:
            by_gen[it.meta.get('generator') or 'real'].append(it)
        split_rng = random.Random(int(split_seed))
        train_split, val_split_items = [], []
        for gen in sorted(by_gen):
            group = list(by_gen[gen])
            split_rng.shuffle(group)
            n_val = int(len(group) * frac)
            val_split_items.extend(group[:n_val])
            train_split.extend(group[n_val:])
        log_line(
            f'[data] {source}: internal split val_split={frac} seed={split_seed} '
            f'-> train={len(train_split)} val={len(val_split_items)} '
            f'(stratified over {len(by_gen)} strata)'
        )

    if val_per_pool is not None or val_real_cap is not None:
        cap_rng = random.Random(int(split_seed) + 1)   # +1: independent of the split draw
        by_pool = defaultdict(list)
        for it in val_split_items:
            by_pool[it.meta.get('generator') or 'real'].append(it)
        capped, dropped = [], 0
        for pool in sorted(by_pool):
            group = list(by_pool[pool])
            cap = val_real_cap if pool == 'real' else val_per_pool
            if cap is not None and len(group) > int(cap):
                group.sort(key=lambda i: i.item_id)   # stable order before the draw
                cap_rng.shuffle(group)
                dropped += len(group) - int(cap)
                group = group[:int(cap)]
            capped.extend(group)
        log_line(
            f'[data] {source}: val capped per_pool={val_per_pool} reals={val_real_cap} '
            f'-> {len(capped)} items across {len(by_pool)} pools ({dropped} dropped)'
        )
        val_split_items = capped

    val_ds = Dataset(val_split_items, res=res, augment=False)
    train_ds = Dataset(train_split, res=res, augment=True)
    return train_ds, val_ds
