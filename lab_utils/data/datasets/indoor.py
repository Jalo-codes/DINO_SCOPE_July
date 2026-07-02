"""lab_utils.data.datasets.indoor — indoor real-image dataset builder.

Ported from legacy/lab_utils/data/indexer.py:index_indoor_dataset.

Accepts three layouts:
    1. Single image file.
    2. Manifest file (.txt) — one image path per line (abs or relative to
       the manifest's directory). Lines starting with # are comments.
    3. Directory tree — recursively discovers image files.

An optional holdout_subdir marks images in that subdirectory as validation.

Returns (train_dataset, val_dataset) of reals (no mask, is_real=True).
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from lab_utils.data.item import Item, make_item_id
from lab_utils.data.dataset import Dataset
from lab_utils.data.verify import VerifyPolicy, verify_all
from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff'})


def _is_valid(path: str, exts: frozenset) -> bool:
    return os.path.splitext(path)[1].lower() in exts


def _from_manifest(path: Path, holdout_subdir: str, exts: frozenset):
    train, val = [], []
    base_dir = path.parent
    try:
        raw = path.read_bytes()
        text = raw.decode('utf-8', errors='ignore')
    except OSError:
        return train, val
    for line in text.replace('\x00', '\n').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        cand = line if os.path.isabs(line) else str(base_dir / line)
        cand = os.path.abspath(cand)
        if not os.path.exists(cand) or not _is_valid(cand, exts):
            continue
        parts = set(os.path.normpath(cand).split(os.sep))
        (val if holdout_subdir and holdout_subdir in parts else train).append(cand)
    return sorted(set(train)), sorted(set(val))


def _from_tree(root: Path, holdout_subdir: str, exts: frozenset):
    train, val = [], []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if not _is_valid(full, exts):
                continue
            rel_parts = set(os.path.relpath(full, root).split(os.sep))
            (val if holdout_subdir and holdout_subdir in rel_parts else train).append(full)
    return sorted(set(train)), sorted(set(val))


def _discover(root: Path, holdout_subdir: str, exts: frozenset):
    """Return (train_paths, val_paths) using whichever layout applies."""
    if root.is_file():
        if _is_valid(str(root), exts):
            return [str(root.resolve())], []
        return _from_manifest(root, holdout_subdir, exts)

    direct_imgs = [
        f for f in sorted(root.iterdir())
        if f.is_file() and _is_valid(f.name, exts)
    ]
    if direct_imgs:
        return _from_tree(root, holdout_subdir, exts)

    train, val = [], []
    for subdir in sorted(root.iterdir()):
        if not subdir.is_dir():
            continue
        img_dir = subdir / 'images' if (subdir / 'images').is_dir() else subdir
        imgs = [str(f) for f in sorted(img_dir.iterdir())
                if f.is_file() and _is_valid(f.name, exts)]
        if not imgs:
            continue
        if subdir.name == holdout_subdir:
            val.extend(imgs)
        else:
            train.extend(imgs)

    if not train and not val:
        return _from_tree(root, holdout_subdir, exts)
    return train, val


def _paths_to_items(paths: List[str], source: str = 'indoor') -> List[Item]:
    return [
        Item(
            image=Path(p),
            authentic=None,
            mask=None,
            source=source,
            item_id=make_item_id(source, p),
            meta={'case_id': os.path.splitext(os.path.basename(p))[0]},
        )
        for p in paths
    ]


def build(
    root: Path,
    *,
    res: Resolution,
    source: str = 'indoor',
    holdout_subdir: str = '',
    verify_policy: Optional[VerifyPolicy] = None,
    valid_exts: Optional[frozenset] = None,
) -> Tuple[Dataset, Dataset]:
    """Discover indoor real images; return (train_dataset, val_dataset).

    Args:
        root:           Dataset root (file, manifest, or directory tree).
        res:            Resolution for the Dataset.
        source:         Source label on the items (default 'indoor').
        holdout_subdir: Subdirectory name whose images become the val set.
    """
    root = Path(root)
    exts = valid_exts or _VALID_EXTS

    if not root.exists():
        log_line(f'[data] WARN: indoor root not found ({source}): {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    log_line(f'[data] Indexing indoor ({source}): {root}')
    train_paths, val_paths = _discover(root, holdout_subdir, exts)

    train_items = _paths_to_items(train_paths, source)
    val_items   = _paths_to_items(val_paths,   source)

    train_kept, _ = verify_all(train_items, policy=verify_policy,
                                log_tag=f'[data] {source} train')
    val_kept,   _ = verify_all(val_items,   policy=verify_policy,
                                log_tag=f'[data] {source} val')
    log_line(
        f'[data] indoor ({source}): train={len(train_kept)} val={len(val_kept)}'
    )
    return (
        Dataset(train_kept, res=res, augment=True),
        Dataset(val_kept,   res=res, augment=False),
    )
