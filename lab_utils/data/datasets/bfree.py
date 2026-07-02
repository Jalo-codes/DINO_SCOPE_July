"""lab_utils.data.datasets.bfree — BFree COCO-anchored SD2.1 inpainting builder.

Ported from legacy/lab_utils/data/indexer.py:index_bfree + _resolve_bfree_root.

Layout (under root, or a parent containing it)::
    COCO_real_512/              anchor real images
    SD2.1_inpainted_diffcat/    fakes — different semantic category
    SD2.1_inpainted_samecat/    fakes — same semantic category
    masks/ (or mask/)           exact segmentation masks
    bbox/                       bounding-box masks

Mask policy (mirrors legacy v6):
    diffcat → bbox mask (falls back to exact segmentation)
    samecat → exact segmentation mask

Each anchor yields one real negative plus up to two splice positives.
Fakes carry meta['real_path'] to trigger paste_background in Dataset.

Returns (train_dataset, val_dataset).
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lab_utils.data.item import Item, make_item_id
from lab_utils.data.dataset import Dataset
from lab_utils.data.verify import VerifyPolicy, verify_all
from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line
from lab_utils.data.datasets.inpaint import _clean_name

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff'})


def _normalize(name: str) -> str:
    base = name.lower().rstrip('/\\')
    return base[:-4] if base.endswith('.zip') else base


def _resolve_named_subdir(root: Path, desired_names) -> Optional[Path]:
    desired = {_normalize(n) for n in desired_names}
    if not root.is_dir():
        return None
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and _normalize(entry.name) in desired:
            return entry
    return None


def _resolve_bfree_root(root: Path) -> Path:
    if not root.is_dir():
        return root
    required = {'coco_real_512', 'sd2.1_inpainted_diffcat',
                'sd2.1_inpainted_samecat', 'bbox'}
    mask_names = {'masks', 'mask'}

    def _entries(d: Path):
        return {_normalize(e.name) for e in d.iterdir() if e.is_dir()}

    ents = _entries(root)
    if required.issubset(ents) and ents & mask_names:
        return root
    for entry in sorted(root.iterdir()):
        if entry.is_dir():
            sub = _entries(entry)
            if required.issubset(sub) and sub & mask_names:
                return entry
    return root


def _file_dict(folder: Optional[Path], exts: frozenset) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if folder is None or not folder.is_dir():
        return out
    for root_d, _, files in os.walk(folder):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in exts:
                out[_clean_name(f).lower()] = Path(root_d) / f
    return out


def build(
    root: Path,
    *,
    res: Resolution,
    source: str = 'bfree',
    verify_policy: Optional[VerifyPolicy] = None,
    val_split: float = 0.10,
    split_seed: int = 42,
    valid_exts: Optional[frozenset] = None,
) -> Tuple[Dataset, Dataset]:
    """Discover and pair BFree items; return (train_dataset, val_dataset)."""
    root = Path(root)
    exts = valid_exts or _VALID_EXTS
    mask_exts = frozenset(exts | {'.png'})

    if not root.is_dir():
        log_line(f'[data] WARN: bfree root not found: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    root = _resolve_bfree_root(root)
    anchor_dir = _resolve_named_subdir(root, ['COCO_real_512'])
    mask_dir   = _resolve_named_subdir(root, ['masks', 'mask'])
    bbox_dir   = _resolve_named_subdir(root, ['bbox'])

    if anchor_dir is None:
        log_line(f'[data] WARN: bfree anchor dir (COCO_real_512) not found under {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    target_specs = [
        ('diffcat', _resolve_named_subdir(root, ['SD2.1_inpainted_diffcat'])),
        ('samecat', _resolve_named_subdir(root, ['SD2.1_inpainted_samecat'])),
    ]

    log_line(f'[data] Indexing BFree: {root}')

    anchor_d = _file_dict(anchor_dir, exts)
    mask_d   = _file_dict(mask_dir,   mask_exts)
    bbox_d   = _file_dict(bbox_dir,   mask_exts)

    fakes: List[Tuple[str, str, Path, Optional[Path]]] = []
    for variant, target_dir in target_specs:
        if target_dir is None:
            log_line(f'[data] bfree: missing target dir for {variant}')
            continue
        for base, fake_path in _file_dict(target_dir, exts).items():
            if base not in anchor_d:
                continue
            exact = mask_d.get(base)
            bbox  = bbox_d.get(base) if variant == 'diffcat' else None
            fakes.append((base, variant, fake_path, bbox or exact))

    bases = sorted({b for b, _, _, _ in fakes})
    if not bases:
        log_line(f'[data] bfree: no matched (anchor, fake) pairs under {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    split_rng = random.Random(int(split_seed))
    shuffled  = list(bases)
    split_rng.shuffle(shuffled)
    val_bases = set(shuffled[: int(len(shuffled) * float(val_split))])

    train_items: list = []
    val_items:   list = []
    seen_real: set = set()
    n_masked = 0

    for base, variant, fake_path, mask_path in fakes:
        bucket  = val_items if base in val_bases else train_items
        anchor  = anchor_d[base]
        case_id = f'{source}_{base}'
        if mask_path is not None:
            n_masked += 1
        if base not in seen_real:
            bucket.append(Item(
                image=anchor,
                authentic=None,
                mask=None,
                source=source,
                item_id=make_item_id(source, anchor),
                meta={'case_id': case_id},
            ))
            seen_real.add(base)
        bucket.append(Item(
            image=fake_path,
            authentic=anchor,
            mask=mask_path,
            source=source,
            item_id=make_item_id(source, fake_path),
            meta={'case_id': f'{case_id}_{variant}', 'variant': variant,
                  'real_path': anchor},
        ))

    train_kept, _ = verify_all(train_items, policy=verify_policy,
                                log_tag=f'[data] {source} train')
    val_kept,   _ = verify_all(val_items,   policy=verify_policy,
                                log_tag=f'[data] {source} val')
    log_line(
        f'[data] bfree: train={len(train_kept)} val={len(val_kept)} '
        f'bases={len(bases)} fakes={len(fakes)} masked={n_masked}'
    )
    if n_masked < len(fakes):
        log_line(f'[data] bfree WARN: {len(fakes) - n_masked} fakes missing a mask')
    return (
        Dataset(train_kept, res=res, augment=True),
        Dataset(val_kept,   res=res, augment=False),
    )
