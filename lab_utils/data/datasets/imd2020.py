"""lab_utils.data.datasets.imd2020 — IMD2020 dataset builder.

Ported from legacy/lab_utils/data/indexer.py:index_imd2020.

Layout::
    root/<case_id>/<case_id>_orig.jpg  — pristine original (the real negative)
    root/<case_id>/<stem>.jpg          — spliced versions
    root/<case_id>/<stem>_mask.png     — GT binary mask for each splice

Returns (train_dataset, val_dataset).
"""

import random
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
    verify_policy: Optional[VerifyPolicy] = None,
    val_split: float = 0.10,
    split_seed: int = 42,
    valid_exts: Optional[frozenset] = None,
) -> Tuple[Dataset, Dataset]:
    """Discover and pair IMD2020 items; return (train_dataset, val_dataset)."""
    root = Path(root)
    exts = valid_exts or _VALID_EXTS

    if not root.exists():
        log_line(f'[data] WARN: IMD2020 root not found: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    log_line(f'[data] Indexing IMD2020: {root}')

    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    split_rng = random.Random(int(split_seed))
    shuffled = list(subdirs)
    split_rng.shuffle(shuffled)
    n_val = int(len(shuffled) * float(val_split))
    val_dirs = {d.name for d in shuffled[:n_val]}

    train_items: list = []
    val_items:   list = []
    n_real = n_splice = n_skipped = 0

    for case_dir in subdirs:
        files = sorted(case_dir.iterdir())
        file_names = [f.name for f in files]
        orig_file = next(
            (f for f in files
             if '_orig' in f.stem and f.suffix.lower() in exts),
            None,
        )
        if orig_file is None:
            continue

        bucket = val_items if case_dir.name in val_dirs else train_items
        bucket.append(Item(
            image=orig_file,
            authentic=None,
            mask=None,
            source='imd2020',
            item_id=make_item_id('imd2020', orig_file),
            meta={'case_id': case_dir.name},
        ))
        n_real += 1

        masks = {
            f.stem.replace('_mask', ''): f
            for f in files if f.name.endswith('_mask.png')
        }
        for f in files:
            if f.suffix.lower() not in exts:
                continue
            if f == orig_file or f.name.endswith('_mask.png'):
                continue
            mask_path = masks.get(f.stem)
            if mask_path is None:
                n_skipped += 1
                continue
            bucket.append(Item(
                image=f,
                authentic=orig_file,
                mask=mask_path,
                source='imd2020',
                item_id=make_item_id('imd2020', f),
                meta={'case_id': case_dir.name},
            ))
            n_splice += 1

    train_kept, _ = verify_all(train_items, policy=verify_policy, log_tag='[data] imd2020 train')
    val_kept,   _ = verify_all(val_items,   policy=verify_policy, log_tag='[data] imd2020 val')
    log_line(
        f'[data] IMD2020: train={len(train_kept)} val={len(val_kept)} '
        f'real_cases={n_real} splices={n_splice} skipped_unmasked={n_skipped}'
    )
    return (
        Dataset(train_kept, res=res, augment=True),
        Dataset(val_kept,   res=res, augment=False),
    )
