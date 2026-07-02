"""lab_utils.data.datasets.opensdi — OpenSDI generator-nested eval dataset builder.

Discovers images in a per-generator subfolder structure:
    root/[generator]/images/[category]_[name]_real.png -> mask=None
    root/[generator]/images/partial_[name]_fake.png -> mask=root/[generator]/masks/partial_[name]_fake.png

Only reals and partial fakes are loaded. All items are interleaved by generator and label
to prevent evaluation bias.
"""

from collections import defaultdict
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    source: str = 'opensdi',
    verify_policy: Optional[VerifyPolicy] = None,
    valid_exts: Optional[frozenset] = None,
) -> Tuple[Dataset, Dataset]:
    """Discover OpenSDI evaluation items by generator subfolder; return (empty_train, val_ds)."""
    root = Path(root)
    exts = valid_exts or _VALID_EXTS

    if not root.is_dir():
        log_line(f'[data] WARN: {source} root not found: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    # Dynamically discover generator folders
    generators = []
    gen_dirs = {}
    for sub in sorted(root.iterdir()):
        if sub.is_dir() and (sub / 'images').is_dir():
            generators.append(sub.name)
            gen_dirs[sub.name] = sub

    # Fallback to root if no subdirectories with images/ are found
    if not generators and (root / 'images').is_dir():
        generators = [root.name]
        gen_dirs[root.name] = root

    if not generators:
        log_line(f'[data] WARN: {source} no generator directories found in: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    log_line(f'[data] Indexing OpenSDI ({len(generators)} generators): {root}')

    # Group items by generator and label to support interleaving
    by_gen_label: Dict[str, Dict[str, List[Item]]] = defaultdict(lambda: defaultdict(list))

    for gen in generators:
        gen_dir = gen_dirs[gen]
        img_dir = gen_dir / 'images'
        mask_dir = gen_dir / 'masks'

        for f in sorted(img_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in exts:
                continue
            
            stem = f.stem.lower()
            if stem.endswith('_real'):
                # Extract category prefix (e.g. 'partial' or 'entire')
                content = stem[:-5]  # strip '_real'
                parts = content.split('_', 1)
                category = parts[0] if len(parts) > 1 else 'unknown'

                item = Item(
                    image=f,
                    authentic=None,
                    mask=None,
                    source=source,
                    item_id=make_item_id(source, f),
                    meta={
                        'case_id': f.stem,
                        'generator': gen,
                        'category': category,
                    },
                )
                by_gen_label[gen]['real'].append(item)

            elif stem.endswith('_fake'):
                # Extract category prefix (e.g. 'partial')
                content = stem[:-5]  # strip '_fake'
                parts = content.split('_', 1)
                category = parts[0] if len(parts) > 1 else 'unknown'

                mask_path = None
                if mask_dir.is_dir():
                    for m_ext in ['.png', f.suffix]:
                        cand = mask_dir / f"{f.name[:-len(f.suffix)]}{m_ext}"
                        if cand.is_file():
                            mask_path = cand
                            break

                item = Item(
                    image=f,
                    authentic=None,
                    mask=mask_path,
                    source=source,
                    item_id=make_item_id(source, f),
                    meta={
                        'case_id': f.stem,
                        'generator': gen,
                        'category': category,
                    },
                )
                by_gen_label[gen]['fake'].append(item)

    # Interleave items across generators and labels (real / fake)
    # to maintain strict balance when sub-slicing (e.g. max_items)
    max_len = 0
    for gen in generators:
        for label in ['real', 'fake']:
            max_len = max(max_len, len(by_gen_label[gen][label]))

    interleaved_items = []
    for i in range(max_len):
        for gen in sorted(generators):
            for label in ['real', 'fake']:
                if i < len(by_gen_label[gen][label]):
                    interleaved_items.append(by_gen_label[gen][label][i])

    # Verify paths exist and are readable
    kept, _ = verify_all(interleaved_items, policy=verify_policy, log_tag=f'[data] {source}')

    val_ds = Dataset(kept, res=res, augment=False)
    train_ds = Dataset([], res=res, augment=True)

    log_line(f'[data] {source}: loaded {len(kept)} items for evaluation')
    return train_ds, val_ds
