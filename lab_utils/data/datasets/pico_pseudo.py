"""lab_utils.data.datasets.pico_pseudo — Gemini full-re-render pseudo-mask triplets.

Consumes the v2 output of experiments/scripts/export_pico_masks.py::

    root/modified/<case_id>.png      edited image, border-crop BAKED IN, lossless PNG
    root/original/<case_id>.png      source image, identically cropped
    root/mask/<case_id>_mask.png     pseudo-mask at the SAME cropped geometry
    root/export_format.json          {'version': 2, 'crop_baked_in': True, ...}

Why this is its own builder and not an ``inpaint`` registry alias:

1. NO PASTE-BACK, STRUCTURALLY. The "fake" here is a whole-frame Gemini
   re-render — the generator shifts composition/camera beyond the mask, so
   the fake is NOT pixel-aligned with the original outside the mask.
   ``meta['real_path']`` is never set, so ``Dataset.paste_background`` can
   never composite two misaligned frames and manufacture a seam at the mask
   boundary (a trivial cue that competes with real edit-boundary learning).
   This is a property of the source, not a tunable.

2. FORMAT GATE. A populated root without the v2 marker is a v1 export
   (full-frame images + zero-border masks) whose geometry no longer matches
   what training expects — that raises DataError loudly instead of training
   on misaligned pairs. A missing/empty root just warns and returns empty
   (missing-mount convention, same as every other builder).

3. ALIGNMENT HARD-CHECK. Every kept pair must satisfy image.size ==
   mask.size (cheap PIL header reads at index time); mismatches are dropped
   and counted. With a v2 export this should always be 0 — a nonzero count
   means a corrupted/mixed export.

Masks are pseudo-labels (raw-DINO feature diff, decisiveness-filtered at
export) — real localization supervision, flagged ``meta['pseudo_mask']=True``
for reporting.

Returns (train_dataset, val_dataset) with the same seeded val split as the
inpaint builder.
"""

import json
import random
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from lab_utils.data.item import Item, make_item_id
from lab_utils.data.dataset import Dataset
from lab_utils.data.verify import VerifyPolicy, verify_all
from lab_utils.data.resolution import Resolution
from lab_utils.data.datasets.inpaint import index_dir
from lab_utils.errors import DataError
from lab_utils.logging.text import log_line

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff'})


def _check_format(root: Path, mask_dir: Path, source: str) -> None:
    """Raise DataError on a v1 (full-frame) export; accept v2 or empty."""
    fmt_path = root / 'export_format.json'
    if fmt_path.exists():
        with open(fmt_path) as f:
            fmt = json.load(f)
        if fmt.get('version') == 2 and fmt.get('crop_baked_in'):
            log_line(f'[data] {source}: export format v2 '
                     f'(crop_frac={fmt.get("crop_frac")} baked into files)')
            return
        raise DataError(
            f'{source}: {root} has export_format.json={fmt}, expected '
            f'version=2 with crop_baked_in — re-run export_pico_masks.'
        )
    if mask_dir.is_dir() and any(mask_dir.iterdir()):
        raise DataError(
            f'{source}: {root} contains masks but no export_format.json — '
            f'this is a v1 export (full-frame images, zero-border masks), '
            f'whose geometry no longer matches training. Discard it and '
            f're-run export_pico_masks (v2 bakes the crop into the files).'
        )


def build(
    root: Path,
    *,
    res: Resolution,
    source: str = 'pico_pseudo',
    verify_policy: Optional[VerifyPolicy] = None,
    val_split: float = 0.10,
    split_seed: int = 42,
    valid_exts: Optional[frozenset] = None,
) -> Tuple[Dataset, Dataset]:
    """Discover pico pseudo-mask triplets; return (train_dataset, val_dataset)."""
    root = Path(root)
    exts = valid_exts or _VALID_EXTS

    if not root.is_dir():
        log_line(f'[data] WARN: {source} root not found: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    mod_dir, orig_dir, mask_dir = root / 'modified', root / 'original', root / 'mask'
    for label, d in (('modified', mod_dir), ('original', orig_dir), ('mask', mask_dir)):
        if not d.is_dir():
            log_line(f'[data] WARN: {source} missing {label}/ dir: {d}')
            empty = Dataset([], res=res, augment=False)
            return empty, empty

    _check_format(root, mask_dir, source)
    log_line(f'[data] Indexing {source}: {root}')

    mods  = index_dir(mod_dir,  exts)
    origs = index_dir(orig_dir, exts)
    masks = index_dir(mask_dir, frozenset(exts | {'.png'}))

    bases = sorted(set(mods) & set(origs) & set(masks))
    n_unmatched = len(mods) - len(bases)
    if not bases:
        log_line(f'[data] {source}: no complete triplets '
                 f'(mods={len(mods)} origs={len(origs)} masks={len(masks)})')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    # Alignment hard-check: exported image and mask must agree in native size
    # (v2 export guarantees this; a mismatch means a corrupted/mixed dir).
    # PIL .size on an opened-not-loaded file is a header read — cheap.
    aligned: list = []
    n_misaligned = 0
    for base in bases:
        with Image.open(mods[base]) as im, Image.open(masks[base]) as mk:
            if im.size == mk.size:
                aligned.append(base)
            else:
                n_misaligned += 1
                log_line(f'[data] {source} WARN: dropping {base}: image '
                         f'{im.size} != mask {mk.size}')
    if n_misaligned:
        log_line(f'[data] {source} WARN: dropped {n_misaligned} misaligned '
                 f'pairs — the export dir looks corrupted/mixed, consider '
                 f're-exporting')
    bases = aligned

    split_rng = random.Random(int(split_seed))
    shuffled  = list(bases)
    split_rng.shuffle(shuffled)
    val_bases = set(shuffled[: int(len(shuffled) * float(val_split))])

    train_items: list = []
    val_items:   list = []
    for base in bases:
        bucket = val_items if base in val_bases else train_items
        case_id = f'{source}_{base}'
        bucket.append(Item(
            image=origs[base],
            authentic=None,
            mask=None,
            source=source,
            item_id=make_item_id(source, origs[base]),
            meta={'case_id': case_id},
        ))
        # No 'real_path' — paste-back is structurally impossible for this
        # source (full re-render, not pixel-aligned; see module docstring).
        bucket.append(Item(
            image=mods[base],
            authentic=origs[base],
            mask=masks[base],
            source=source,
            item_id=make_item_id(source, mods[base]),
            meta={'case_id': case_id, 'pseudo_mask': True},
        ))

    train_kept, _ = verify_all(train_items, policy=verify_policy,
                               log_tag=f'[data] {source} train')
    val_kept,   _ = verify_all(val_items,   policy=verify_policy,
                               log_tag=f'[data] {source} val')
    log_line(
        f'[data] {source}: train={len(train_kept)} val={len(val_kept)} '
        f'triplets={len(bases)} unmatched_mods={n_unmatched} | pseudo-masks, '
        f'crop baked into files, paste-back structurally disabled'
    )
    return (
        Dataset(train_kept, res=res, augment=True),
        Dataset(val_kept,   res=res, augment=False),
    )
