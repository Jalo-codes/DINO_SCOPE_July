"""lab_utils.data.datasets.casia — CASIA exported dataset builder.

Ported from legacy/lab_utils/data/indexer.py:index_casia_exported.

Layout::
    root/images/<base>_real.<ext>  — pristine original
    root/images/<base>_fake.<ext>  — spliced version
    root/masks/<base>_mask.png     — GT binary mask

Train/val split is by bg_id and fg_id (encoded in the base name), so a
real/fake pair never straddles splits. Returns (train_dataset, val_dataset).
"""

import random
from pathlib import Path
from typing import Dict, Optional, Tuple

from lab_utils.data.item import Item, make_item_id
from lab_utils.data.dataset import Dataset
from lab_utils.data.verify import VerifyPolicy, verify_all
from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff'})


def _parse_base_ids(base: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (bg_id, fg_id) from 'casia_<bg_id>_<fg_id>'."""
    if not base.startswith('casia_'):
        return None, None
    stem = base[len('casia_'):]
    parts = stem.split('_', 1)
    if len(parts) != 2:
        return None, None
    return parts[0], parts[1]


def build(
    root: Path,
    *,
    res: Resolution,
    verify_policy: Optional[VerifyPolicy] = None,
    val_split: float = 0.15,
    split_seed: int = 42,
    valid_exts: Optional[frozenset] = None,
) -> Tuple[Dataset, Dataset]:
    """Discover and pair CASIA items; return (train_dataset, val_dataset)."""
    root = Path(root)
    exts = valid_exts or _VALID_EXTS

    if not root.exists():
        log_line(f'[data] WARN: CASIA root not found: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    img_dir  = root / 'images'
    mask_dir = root / 'masks'
    if not img_dir.is_dir() or not mask_dir.is_dir():
        log_line(f'[data] WARN: CASIA requires images/ and masks/ dirs: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    log_line(f'[data] Indexing CASIA: {root}')

    reals: Dict[str, Path] = {}
    fakes: Dict[str, Path] = {}
    for f in sorted(img_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() not in exts:
            continue
        stem = f.stem
        if stem.endswith('_real'):
            reals[stem[:-5]] = f
        elif stem.endswith('_fake'):
            fakes[stem[:-5]] = f

    masks: Dict[str, Path] = {}
    for f in sorted(mask_dir.iterdir()):
        if not f.is_file() or f.suffix.lower() != '.png':
            continue
        stem = f.stem
        if stem.endswith('_mask'):
            masks[stem[:-5]] = f

    pair_bases = sorted(set(reals) & set(fakes) & set(masks))
    if not pair_bases:
        log_line('[data] CASIA: no complete (real, fake, mask) triplets found')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    pairs = []
    malformed = 0
    for base in pair_bases:
        bg_id, fg_id = _parse_base_ids(base)
        if not bg_id or not fg_id:
            malformed += 1
            continue
        pairs.append({'base': base, 'bg_id': bg_id, 'fg_id': fg_id,
                      'real_path': reals[base], 'fake_path': fakes[base],
                      'mask_path': masks[base]})

    all_bgs = sorted({p['bg_id'] for p in pairs})
    all_fgs = sorted({p['fg_id'] for p in pairs})
    split_rng = random.Random(int(split_seed))
    split_rng.shuffle(all_bgs)
    split_rng.shuffle(all_fgs)
    n_val_bgs = int(len(all_bgs) * float(val_split))
    n_val_fgs = int(len(all_fgs) * float(val_split))
    val_bgs   = set(all_bgs[:n_val_bgs])
    train_bgs = set(all_bgs[n_val_bgs:])
    val_fgs   = set(all_fgs[:n_val_fgs])
    train_fgs = set(all_fgs[n_val_fgs:])

    train_items: list = []
    val_items:   list = []
    discarded = 0
    for p in pairs:
        bg_id, fg_id = p['bg_id'], p['fg_id']
        if bg_id in train_bgs and fg_id in train_fgs:
            bucket = train_items
        elif bg_id in val_bgs and fg_id in val_fgs:
            bucket = val_items
        else:
            discarded += 1
            continue
        meta = {'case_id': p['base'], 'bg_id': bg_id, 'fg_id': fg_id}
        bucket.append(Item(
            image=p['real_path'],
            authentic=None,
            mask=None,
            source='casia',
            item_id=make_item_id('casia', p['real_path']),
            meta=meta,
        ))
        bucket.append(Item(
            image=p['fake_path'],
            authentic=p['real_path'],
            mask=p['mask_path'],
            source='casia',
            item_id=make_item_id('casia', p['fake_path']),
            meta=meta,
        ))

    train_kept, _ = verify_all(train_items, policy=verify_policy, log_tag='[data] casia train')
    val_kept,   _ = verify_all(val_items,   policy=verify_policy, log_tag='[data] casia val')
    log_line(
        f'[data] CASIA: train={len(train_kept)} val={len(val_kept)} '
        f'triplets={len(pairs)} discarded={discarded} malformed={malformed}'
    )
    return (
        Dataset(train_kept, res=res, augment=True),
        Dataset(val_kept,   res=res, augment=False),
    )
