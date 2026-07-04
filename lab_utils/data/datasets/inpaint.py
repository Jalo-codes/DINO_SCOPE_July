"""lab_utils.data.datasets.inpaint — inpainting-triplet dataset builder.

Ported from legacy/lab_utils/data/indexer.py:index_inpaint_triplet.

Handles SD-inpaint, COCO-inpaint, SAGID, and any other dataset following the
three-folder layout::

    root/<modified_subdir>/   inpainted fakes
    root/<original_subdir>/   pristine originals (pre-inpaint)
    root/<mask_subdir>/       inpaint-region masks

Files are matched by cleaned basename (extension + common suffixes stripped).
Each matched triplet yields:
    - real negative: image=original, mask=None
    - splice positive: image=modified, authentic=original, mask=mask,
                       meta['real_path']=original (drives paste_background in Dataset)

paste_back=False omits meta['real_path'], disabling the paste. REQUIRED for
full-re-render sources (pico_pseudo): the "fake" is a whole-frame regeneration,
not pixel-aligned with the original outside the mask, so pasting the original
back would composite two misaligned frames — an artificial seam at the mask
boundary that trains the model on a trivial cue. Paste is only valid for true
inpainting sources (SAGI-D, COCO-inpaint) where pixels outside the mask are
byte-identical to the original.

Returns (train_dataset, val_dataset).
"""

import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

from lab_utils.data.item import Item, make_item_id
from lab_utils.data.dataset import Dataset
from lab_utils.data.verify import VerifyPolicy, verify_all
from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff'})


def _clean_name(filename: str) -> str:
    """Strip extension and common modified/original/mask suffixes for matching."""
    stem = os.path.splitext(filename)[0]
    for suf in ('_modified', '_original', '_orig', '_mask', '_fake', '_real',
                '_inpainted', '_gt'):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def _index_dir(folder: Path, exts: frozenset) -> Dict[str, Path]:
    """Map cleaned basename → path for all image files in a folder."""
    out: Dict[str, Path] = {}
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            out[_clean_name(f.name)] = f
    return out


def build(
    root: Path,
    *,
    res: Resolution,
    source: str,
    modified_subdir: str = 'modified',
    original_subdir: str = 'original',
    mask_subdir: str = 'mask',
    verify_policy: Optional[VerifyPolicy] = None,
    val_split: float = 0.10,
    split_seed: int = 42,
    valid_exts: Optional[frozenset] = None,
    paste_back: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Discover and pair inpaint-triplet items; return (train_dataset, val_dataset).

    Args:
        root:            Dataset root (must contain modified/, original/, mask/).
        res:             Resolution for the Dataset.
        source:          Source name for Items (e.g. 'coco_inpaint', 'sagid').
        modified_subdir: Subdirectory containing inpainted images.
        original_subdir: Subdirectory containing pristine originals.
        mask_subdir:     Subdirectory containing inpaint-region masks.
        paste_back:      Set meta['real_path'] on fakes (enables Dataset's
                         paste_background). Must be False for full-re-render
                         sources — see module docstring.
    """
    root = Path(root)
    exts = valid_exts or _VALID_EXTS
    mask_exts = frozenset(exts | {'.png'})

    if not root.is_dir():
        log_line(f'[data] WARN: inpaint root not found ({source}): {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    mod_dir  = root / modified_subdir
    orig_dir = root / original_subdir
    mask_dir = root / mask_subdir

    for label, d in (('modified', mod_dir), ('original', orig_dir), ('mask', mask_dir)):
        if not d.is_dir():
            log_line(f'[data] WARN: inpaint {source} missing {label}/ dir: {d}')
            empty = Dataset([], res=res, augment=False)
            return empty, empty

    log_line(f'[data] Indexing inpaint ({source}): {root}')

    mods  = _index_dir(mod_dir,  exts)
    origs = _index_dir(orig_dir, exts)
    masks = _index_dir(mask_dir, mask_exts)

    bases = sorted(set(mods) & set(origs) & set(masks))
    n_missing = len(mods) - len(bases)
    if not bases:
        log_line(
            f'[data] inpaint {source}: no complete triplets '
            f'(mods={len(mods)} origs={len(origs)} masks={len(masks)})'
        )
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    split_rng = random.Random(int(split_seed))
    shuffled  = list(bases)
    split_rng.shuffle(shuffled)
    n_val     = int(len(shuffled) * float(val_split))
    val_bases = set(shuffled[:n_val])

    train_items: list = []
    val_items:   list = []

    for base in bases:
        bucket = val_items if base in val_bases else train_items
        orig_path = origs[base]
        mod_path  = mods[base]
        mask_path = masks[base]
        case_id   = f'{source}_{base}'
        bucket.append(Item(
            image=orig_path,
            authentic=None,
            mask=None,
            source=source,
            item_id=make_item_id(source, orig_path),
            meta={'case_id': case_id},
        ))
        fake_meta = {'case_id': case_id}
        if paste_back:
            fake_meta['real_path'] = orig_path
        bucket.append(Item(
            image=mod_path,
            authentic=orig_path,
            mask=mask_path,
            source=source,
            item_id=make_item_id(source, mod_path),
            meta=fake_meta,
        ))

    train_kept, _ = verify_all(train_items, policy=verify_policy,
                                log_tag=f'[data] {source} train')
    val_kept,   _ = verify_all(val_items,   policy=verify_policy,
                                log_tag=f'[data] {source} val')
    log_line(
        f'[data] inpaint {source}: train={len(train_kept)} val={len(val_kept)} '
        f'triplets={len(bases)} unmatched_mods={n_missing}'
    )
    return (
        Dataset(train_kept, res=res, augment=True),
        Dataset(val_kept,   res=res, augment=False),
    )
