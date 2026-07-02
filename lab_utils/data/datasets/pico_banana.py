"""lab_utils.data.datasets.pico_banana — PicoBanana native-S3 edit builder.

Layout (per edit-category subfolder, e.g. "Remove_an_existing_object")::

    root/<category>/<real_subdir>/   pristine originals   (e.g. originals/)
    root/<category>/<fake_subdir>/   AI-edited images      (e.g. modified/)

Also accepts a flat root (no category nesting) with the same two subfolders
directly under root. Real/fake subfolder names are matched case-insensitively
against a small set of aliases (originals/original/real/reals,
modified/modifications/fake/fakes/edited).

NO ground-truth masks ship with this dataset. Item.is_real is defined as
``mask is None`` (lab_utils/data/item.py), so a fake item naively indexed
with mask=None would be silently mislabeled real for image-level
supervision and eval AUC. Every fake item here instead gets a synthetic
full-frame mask (one shared all-white PNG, cached in the system temp dir)
purely to keep is_real=False correct — it carries NO localization signal.
``meta['gt_mask_reliable'] = False`` flags this so patch-level IoU/F1 on
this source can be recognized as meaningless; only image-level
classification (image_score / AUC) should be trusted. The edit operation
is recorded in ``meta['category']`` for per-category breakdown.

Eval-only (mirrors opensdi/unpaired): returns (empty_train_dataset, val_dataset).

Indexing walks every category's originals/ and modified/ folders and runs
verify_all (PIL-decodes every image) — the expensive part at ~15k images.
By default the result is cached to a JSON file next to root
(<root>.<source>_index_cache.json) and reused on the next build() call,
skipping discovery + verify entirely. Pass use_cache=False or
force_rebuild=True to bypass; delete the cache file (or bump `source`)
after the on-disk dataset changes.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from lab_utils.data.item import Item, make_item_id
from lab_utils.data.dataset import Dataset
from lab_utils.data.verify import VerifyPolicy, verify_all
from lab_utils.data.resolution import Resolution
from lab_utils.data.datasets.inpaint import _clean_name
from lab_utils.logging.text import log_line

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff'})
_REAL_DIR_NAMES = ('originals', 'original', 'reals', 'real')
_FAKE_DIR_NAMES = ('modified', 'modifications', 'fakes', 'fake', 'edited')

_SYNTHETIC_MASK_PATH = Path(tempfile.gettempdir()) / 'dino_scope_pico_banana_full_mask.png'


def _synthetic_full_mask() -> Path:
    """One shared all-white PNG, reused as every fake item's mask.

    Item.load()/load_image_tensor resize any mask to the model resolution,
    so a single tiny file suffices. This exists purely to make is_real=False
    correct (Item.is_real == mask is None) — it is NOT a real localization
    target (see module docstring).
    """
    if not _SYNTHETIC_MASK_PATH.exists():
        Image.new('L', (32, 32), 255).save(_SYNTHETIC_MASK_PATH)
    return _SYNTHETIC_MASK_PATH


def _default_cache_path(root: Path, source: str) -> Path:
    return root.parent / f'.{root.name}.{source}_index_cache.json'


def _meta_to_json(meta: dict) -> dict:
    out = dict(meta)
    if out.get('real_path') is not None:
        out['real_path'] = str(out['real_path'])
    return out


def _meta_from_json(meta: dict) -> dict:
    out = dict(meta)
    if out.get('real_path') is not None:
        out['real_path'] = Path(out['real_path'])
    return out


def _items_to_cache(items: List[Item]) -> List[dict]:
    return [
        {
            'image': str(it.image),
            'authentic': str(it.authentic) if it.authentic is not None else None,
            'mask': str(it.mask) if it.mask is not None else None,
            'source': it.source,
            'item_id': it.item_id,
            'meta': _meta_to_json(it.meta),
        }
        for it in items
    ]


def _items_from_cache(records: List[dict]) -> List[Item]:
    return [
        Item(
            image=Path(r['image']),
            authentic=Path(r['authentic']) if r['authentic'] is not None else None,
            mask=Path(r['mask']) if r['mask'] is not None else None,
            source=r['source'],
            item_id=r['item_id'],
            meta=_meta_from_json(r['meta']),
        )
        for r in records
    ]


def _find_pair_dirs(folder: Path) -> Optional[Tuple[Path, Path]]:
    """Return (real_dir, fake_dir) if folder directly contains both, else None."""
    if not folder.is_dir():
        return None
    entries = {p.name.lower(): p for p in folder.iterdir() if p.is_dir()}
    real_dir = next((entries[n] for n in _REAL_DIR_NAMES if n in entries), None)
    fake_dir = next((entries[n] for n in _FAKE_DIR_NAMES if n in entries), None)
    if real_dir is not None and fake_dir is not None:
        return real_dir, fake_dir
    return None


def _index_dir(folder: Path, exts: frozenset) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            out[_clean_name(f.name)] = f
    return out


def _discover_categories(root: Path) -> List[Tuple[str, Path, Path]]:
    """Return [(category, real_dir, fake_dir), ...] — root itself, or one per subfolder."""
    direct = _find_pair_dirs(root)
    if direct is not None:
        return [(root.name, direct[0], direct[1])]

    out = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        pair = _find_pair_dirs(sub)
        if pair is not None:
            out.append((sub.name, pair[0], pair[1]))
    return out


def build(
    root: Path,
    *,
    res: Resolution,
    source: str = 'pico_banana',
    verify_policy: Optional[VerifyPolicy] = None,
    valid_exts: Optional[frozenset] = None,
    use_cache: bool = True,
    cache_path: Optional[Path] = None,
    force_rebuild: bool = False,
) -> Tuple[Dataset, Dataset]:
    """Discover PicoBanana real/modified pairs across edit-category subfolders.

    verify_policy: the DEFAULT_POLICY rejects any mask covering >99% of the
    image (max_mask_area — a legitimate tripwire for labeling bugs elsewhere
    in the codebase). Every fake item here carries the synthetic full-frame
    sentinel mask (see _synthetic_full_mask) by design, which is 100%
    coverage, so applying that default would silently drop every fake. When
    the caller does not pass an explicit policy, this builder relaxes
    max_mask_area to 1.0 so the sentinel survives verify_all; pass your own
    VerifyPolicy (or verify.SKIP_VERIFY) to override.

    use_cache/cache_path/force_rebuild: the discovery+verify_all pass
    PIL-decodes every image (the expensive part). By default the resulting
    item list is cached to JSON at `cache_path` (default: next to root,
    `<root>.<source>_index_cache.json`) and reused verbatim on later calls
    — no filesystem walk, no re-verify. Pass force_rebuild=True (or delete
    the cache file) after the on-disk dataset changes; use_cache=False
    disables caching entirely.
    """
    root = Path(root)
    exts = valid_exts or _VALID_EXTS
    cache_file = Path(cache_path) if cache_path is not None else _default_cache_path(root, source)

    # Regenerate unconditionally (idempotent, cheap) — a cached Item may
    # reference this mask path from a prior process/session, so it must
    # exist before we return cached items too.
    mask_path = _synthetic_full_mask()

    if use_cache and not force_rebuild and cache_file.exists():
        try:
            with open(cache_file) as f:
                cached = json.load(f)
            kept = _items_from_cache(cached['items'])
            n_real = sum(1 for it in kept if it.is_real)
            log_line(
                f'[data] {source}: loaded {len(kept)} items '
                f'(real={n_real} fake={len(kept) - n_real}) from index cache {cache_file}'
            )
            val_ds = Dataset(kept, res=res, augment=False)
            train_ds = Dataset([], res=res, augment=True)
            return train_ds, val_ds
        except Exception as exc:
            log_line(f'[data] {source} WARN: failed to read index cache {cache_file} ({exc}); rebuilding')

    if not root.is_dir():
        log_line(f'[data] WARN: {source} root not found: {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    categories = _discover_categories(root)
    if not categories:
        log_line(f'[data] WARN: {source} no real/modified folder pairs found under {root}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    log_line(f'[data] Indexing {source} ({len(categories)} categories): {root}')

    by_cat_label: Dict[str, Dict[str, List[Item]]] = {}
    n_unmatched = 0

    for category, real_dir, fake_dir in categories:
        reals = _index_dir(real_dir, exts)
        fakes = _index_dir(fake_dir, exts)
        bases = sorted(set(reals) & set(fakes))
        n_unmatched += (len(reals) - len(bases)) + (len(fakes) - len(bases))
        if not bases:
            log_line(f'[data] {source}: no matched pairs in category {category!r}')
            continue

        real_items: List[Item] = []
        fake_items: List[Item] = []
        for base in bases:
            case_id = f'{category}_{base}'
            real_items.append(Item(
                image=reals[base],
                authentic=None,
                mask=None,
                source=source,
                item_id=make_item_id(source, reals[base]),
                meta={'case_id': case_id, 'category': category},
            ))
            fake_items.append(Item(
                image=fakes[base],
                authentic=reals[base],
                mask=mask_path,
                source=source,
                item_id=make_item_id(source, fakes[base]),
                meta={'case_id': case_id, 'category': category,
                      'real_path': reals[base], 'gt_mask_reliable': False},
            ))
        by_cat_label[category] = {'real': real_items, 'fake': fake_items}

    if not by_cat_label:
        log_line(f'[data] {source}: no category yielded any matched pairs')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    # Interleave categories and real/fake so slicing (e.g. --max_items) stays balanced.
    max_len = max(
        len(items) for cat_items in by_cat_label.values() for items in cat_items.values()
    )
    interleaved: List[Item] = []
    for i in range(max_len):
        for category in sorted(by_cat_label):
            for label in ('real', 'fake'):
                items = by_cat_label[category][label]
                if i < len(items):
                    interleaved.append(items[i])

    effective_policy = (
        verify_policy if verify_policy is not None else VerifyPolicy(max_mask_area=1.0)
    )
    kept, _ = verify_all(interleaved, policy=effective_policy, log_tag=f'[data] {source}')

    n_real = sum(1 for it in kept if it.is_real)
    log_line(
        f'[data] {source}: loaded {len(kept)} items '
        f'(real={n_real} fake={len(kept) - n_real}) across {len(by_cat_label)} categories '
        f'| unmatched={n_unmatched} | NO GT masks (synthetic full-frame sentinel) — '
        f'localization metrics on this source are not meaningful, image-level only'
    )

    if use_cache:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump({'items': _items_to_cache(kept)}, f)
            log_line(f'[data] {source}: wrote index cache to {cache_file} ({len(kept)} items)')
        except Exception as exc:
            log_line(f'[data] {source} WARN: failed to write index cache {cache_file} ({exc})')

    val_ds = Dataset(kept, res=res, augment=False)
    train_ds = Dataset([], res=res, augment=True)
    return train_ds, val_ds
