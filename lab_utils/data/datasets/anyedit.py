"""lab_utils.data.datasets.anyedit — AnyEdit dataset builder.

Ported from legacy/lab_utils/data/indexer.py:index_anyedit.

Layout::
    root/images/  real + edited pairs; real files end with '_real'/'_original',
                  edited files end with '_fake'/'_modified'.
    root/masks/   binary masks matched to edited image by cleaned base name.

Each matched (real, fake) pair yields:
    - real negative: image=real, mask=None
    - splice positive: image=fake, authentic=real, mask=mask

AnyEdit applies localised edits (no full-image VAE fingerprint), so no
paste_background is needed and meta['real_path'] is not set.

Returns (train_dataset, val_dataset).
"""

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

from lab_utils.data.item import Item, make_item_id
from lab_utils.data.dataset import Dataset
from lab_utils.data.verify import VerifyPolicy, verify_all
from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line
from lab_utils.data.datasets.inpaint import _clean_name

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff'})


def build(
    root: Path,
    *,
    res: Resolution,
    source: str = 'anyedit',
    verify_policy: Optional[VerifyPolicy] = None,
    val_split: float = 0.10,
    split_seed: int = 42,
    valid_exts: Optional[frozenset] = None,
) -> Tuple[Dataset, Dataset]:
    """Discover and pair AnyEdit items; return (train_dataset, val_dataset)."""
    root = Path(root)
    exts = valid_exts or _VALID_EXTS
    mask_exts = frozenset(exts | {'.png'})

    if not root.is_dir():
        log_line(f'[data] WARN: anyedit root not found: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    img_dir  = root / 'images'
    mask_dir = root / 'masks'
    for label, d in (('images', img_dir), ('masks', mask_dir)):
        if not d.is_dir():
            log_line(f'[data] WARN: anyedit missing {label}/ dir: {d}')
            empty = Dataset([], res=res, augment=False)
            return empty, empty

    log_line(f'[data] Indexing AnyEdit: {root}')

    reals: Dict[str, Path] = {}
    fakes: Dict[str, Path] = {}
    for f in sorted(img_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        stem = f.stem.lower()
        base = _clean_name(f.name)
        if stem.endswith('_real') or stem.endswith('_original'):
            reals[base] = f
        elif stem.endswith('_fake') or stem.endswith('_modified'):
            fakes[base] = f

    masks: Dict[str, Path] = {}
    for f in sorted(mask_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in mask_exts:
            masks[_clean_name(f.name)] = f

    bases = sorted(set(reals) & set(fakes))
    if not bases:
        log_line(
            f'[data] anyedit: no matched (real, fake) pairs '
            f'(reals={len(reals)} fakes={len(fakes)})'
        )
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    split_rng = random.Random(int(split_seed))
    shuffled  = list(bases)
    split_rng.shuffle(shuffled)
    val_bases = set(shuffled[: int(len(shuffled) * float(val_split))])

    train_items: list = []
    val_items:   list = []
    n_masked = 0

    for base in bases:
        bucket    = val_items if base in val_bases else train_items
        mask_path = masks.get(base)
        if mask_path is not None:
            n_masked += 1
        real_path = reals[base]
        fake_path = fakes[base]
        case_id   = f'{source}_{base}'
        bucket.append(Item(
            image=real_path,
            authentic=None,
            mask=None,
            source=source,
            item_id=make_item_id(source, real_path),
            meta={'case_id': case_id},
        ))
        bucket.append(Item(
            image=fake_path,
            authentic=real_path,
            mask=mask_path,
            source=source,
            item_id=make_item_id(source, fake_path),
            meta={'case_id': case_id},
        ))

    train_kept, _ = verify_all(train_items, policy=verify_policy,
                                log_tag=f'[data] {source} train')
    val_kept,   _ = verify_all(val_items,   policy=verify_policy,
                                log_tag=f'[data] {source} val')
    log_line(
        f'[data] anyedit: train={len(train_kept)} val={len(val_kept)} '
        f'pairs={len(bases)} masked={n_masked}'
    )
    if n_masked < len(bases):
        log_line(f'[data] anyedit WARN: {len(bases) - n_masked} pairs missing a mask')
    return (
        Dataset(train_kept, res=res, augment=True),
        Dataset(val_kept,   res=res, augment=False),
    )
