"""lab_utils.data.datasets.unpaired — generic unpaired dataset builder.

Discovers images in a two-folder structure:
    root/images/[name]_real.[ext] -> authentic image, mask=None
    root/images/[name]_fake.[ext] -> manipulated image, mask=root/masks/[name]_fake.png (or same extension)

Returns (empty_train_dataset, val_dataset) containing all discovered items.
This is designed specifically for evaluation-only datasets.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

from lab_utils.data.item import Item, make_item_id
from lab_utils.data.dataset import Dataset
from lab_utils.data.verify import VerifyPolicy, verify_all
from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff'})


def build(
    root: Path,
    *,
    res: Resolution,
    source: str,
    verify_policy: Optional[VerifyPolicy] = None,
    valid_exts: Optional[frozenset] = None,
) -> Tuple[Dataset, Dataset]:
    """Discover unpaired evaluation items; return (empty_train_dataset, val_dataset)."""
    root = Path(root)
    exts = valid_exts or _VALID_EXTS

    if not root.is_dir():
        log_line(f'[data] WARN: {source} root not found: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    img_dir  = root / 'images'
    mask_dir = root / 'masks'

    if not img_dir.is_dir():
        log_line(f'[data] WARN: {source} missing images/ dir: {img_dir}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    log_line(f'[data] Indexing unpaired ({source}): {root}')

    items = []
    for f in sorted(img_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        stem = f.stem.lower()
        if stem.endswith('_real'):
            items.append(Item(
                image=f,
                authentic=None,
                mask=None,
                source=source,
                item_id=make_item_id(source, f),
                meta={'case_id': f.stem},
            ))
        elif stem.endswith('_fake'):
            mask_path = None
            if mask_dir.is_dir():
                # Look for mask with same name but potential extension match
                for m_ext in ['.png', f.suffix]:
                    cand = mask_dir / f"{f.name[:-len(f.suffix)]}{m_ext}"
                    if cand.is_file():
                        mask_path = cand
                        break
            items.append(Item(
                image=f,
                authentic=None,
                mask=mask_path,
                source=source,
                item_id=make_item_id(source, f),
                meta={'case_id': f.stem},
            ))

    kept, _ = verify_all(items, policy=verify_policy, log_tag=f'[data] {source}')

    # Since this is an evaluation registry, all items go to validation
    val_ds = Dataset(kept, res=res, augment=False)
    train_ds = Dataset([], res=res, augment=True)

    log_line(f'[data] {source}: loaded {len(kept)} items for evaluation')
    return train_ds, val_ds
