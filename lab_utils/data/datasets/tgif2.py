"""lab_utils.data.datasets.tgif2 — TGIF2 FLUX OOD eval dataset builder.

Ported from legacy/contrastive_inpainting_v1/experiments/tgif2_flux.py.

Reads a pre-normalized ``tgif2_index.json``
(per-coco_id → {category, original_512, masks{type_res}, manipulations[...]})
and builds Items for splice / real evaluation.

TGIF2 is a *pure OOD probe* — the model never trains on diffusion inpainting
from this source.  The default build() call therefore returns an empty train
dataset and a val dataset containing all discovered items.

Half-split mode (train_frac > 0.0) reserves that fraction of coco_ids for
fine-tuning.  The split is deterministic at the coco_id level so no content
leaks between the two halves.

Manipulation types:
    'sp'  paste-back (inpainting over the original background)
    'fr'  full re-encode (whole image passed through diffusion)

meta keys for splice items:
    case_id, tgif_coco_id, tgif_category, tgif_type, tgif_model,
    tgif_mask_type, tgif_mask_family, tgif_var_id

meta keys for real items:
    case_id, tgif_coco_id, tgif_category

NOTE: 'real_path' is intentionally NOT set — TGIF2 is evaluated on actual
generated images (not background-pasted).  paste_real_background is for
training inpaint datasets only.

Returns (train_dataset, val_dataset).
"""

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from lab_utils.data.item import Item, make_item_id
from lab_utils.data.dataset import Dataset
from lab_utils.data.verify import VerifyPolicy, verify_all
from lab_utils.data.resolution import Resolution
from lab_utils.data.sampling import deterministic_subsample
from lab_utils.logging.text import log_line


SP, FR = 'sp', 'fr'


def _mask_type_short(mask_used: str) -> str:
    return str(mask_used).split('_', 1)[0]


def _mask_family(mask_type: str) -> str:
    return 'random' if mask_type == 'random' else 'semantic'


def _split_coco_ids(
    index: Dict,
    train_frac: float,
    seed: str,
) -> Tuple[Set[str], Set[str]]:
    """Deterministic (train_ids, eval_ids) partition at the coco_id level.

    Splitting at coco_id keeps every manipulation variant AND the pristine
    original on the same side — no content leakage.
    """
    ids = sorted(index.keys())
    ranked = sorted(
        ids, key=lambda cid: hashlib.md5(f'{seed}|{cid}'.encode('utf-8')).hexdigest()
    )
    n_train = int(round(len(ranked) * train_frac))
    return set(ranked[:n_train]), set(ranked[n_train:])


def _coco_id_cell_counts(entry: Dict, types: Optional[Set[str]]) -> Dict[Tuple[str, str, str], int]:
    """Per-(model, type, mask_family) item count a single coco_id contributes."""
    counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    for man in entry.get('manipulations', []):
        type_ = man.get('type', '')
        if types is not None and type_ not in types:
            continue
        model = man.get('model', '')
        mtype = _mask_type_short(man.get('mask_used', ''))
        counts[(model, type_, _mask_family(mtype))] += 1
    return counts


def _split_coco_ids_by_cell(
    index: Dict,
    eval_per_cell: int,
    seed: str,
    types: Optional[Set[str]],
) -> Tuple[Set[str], Set[str]]:
    """Leakage-free (train_ids, eval_ids) split that fills a per-cell eval quota.

    Cells are the (model, type, mask_family) subcategories.  Iterating coco_ids
    in deterministic hash order, a coco_id joins the EVAL side while ANY cell it
    contributes to is still under ``eval_per_cell``; otherwise TRAIN.  Whole
    coco_ids stay together so no scene leaks across the split.

    The per-cell counter is an upper bound on available eval items (some
    manipulations are later dropped for missing files); the caller subsamples
    each eval cell down to exactly ``eval_per_cell``.
    """
    ids = sorted(index.keys())
    ranked = sorted(
        ids, key=lambda cid: hashlib.md5(f'{seed}|{cid}'.encode('utf-8')).hexdigest()
    )
    cell_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    eval_ids: Set[str] = set()
    train_ids: Set[str] = set()
    for cid in ranked:
        cells = _coco_id_cell_counts(index[cid], types)
        if any(cell_counts[c] < eval_per_cell for c in cells):
            eval_ids.add(cid)
            for c, k in cells.items():
                cell_counts[c] += k
        else:
            train_ids.add(cid)
    return train_ids, eval_ids


def _cap_eval_cells(items: List[Item], cap: int, seed: str) -> List[Item]:
    """Subsample splice items to ``cap`` per (model, type, family) cell.

    Reals pass through in full (they have no cell and serve as eval negatives).
    Deterministic via per-cell seeded ``deterministic_subsample``.
    """
    fakes = [it for it in items if not it.is_real]
    reals = [it for it in items if it.is_real]
    by_cell: Dict[Tuple[str, str, str], List[Item]] = defaultdict(list)
    for it in fakes:
        ck = (
            it.meta.get('tgif_model', ''),
            it.meta.get('tgif_type', ''),
            it.meta.get('tgif_mask_family', ''),
        )
        by_cell[ck].append(it)
    capped: List[Item] = []
    for ck in sorted(by_cell):
        cell_seed = f'{seed}:eval:{ck[0]}:{ck[1]}:{ck[2]}'
        capped.extend(deterministic_subsample(by_cell[ck], cap, seed=cell_seed))
    return capped + reals


def _build_items(
    root: str,
    index: Dict,
    *,
    source: str,
    coco_ids: Optional[Set[str]],
    include_reals: bool,
    types: Optional[Set[str]],
) -> Tuple[List[Item], List[Item]]:
    """Return (fake_items, real_items) as Item objects."""
    by_cell: Dict[Tuple[str, str, str], List[Item]] = defaultdict(list)
    real_items: List[Item] = []
    n_missing_fake = n_missing_mask = 0

    for coco_id, entry in index.items():
        if coco_ids is not None and str(coco_id) not in coco_ids:
            continue
        category = entry.get('category', '')
        masks    = entry.get('masks', {})
        orig_rel = entry.get('original_512')
        real_path = Path(os.path.join(root, orig_rel)) if orig_rel else None

        if include_reals and real_path is not None and real_path.exists():
            real_items.append(Item(
                image=real_path,
                authentic=None,
                mask=None,
                source=source,
                item_id=make_item_id(source, real_path),
                meta={
                    'case_id':       str(coco_id),
                    'tgif_coco_id':  str(coco_id),
                    'tgif_category': category,
                    'tgif_subcat':   'real',
                },
            ))

        for man in entry.get('manipulations', []):
            type_ = man.get('type', '')
            if types is not None and type_ not in types:
                continue
            fake_rel = man.get('fake_path', '')
            fake_path = Path(os.path.join(root, fake_rel))
            mask_used = man.get('mask_used', '')
            mask_rel  = masks.get(mask_used)

            if not mask_rel:
                n_missing_mask += 1
                continue
            mask_path = Path(os.path.join(root, mask_rel))

            if not fake_path.exists():
                n_missing_fake += 1
                continue
            if not mask_path.exists():
                n_missing_mask += 1
                continue

            model   = man.get('model', '')
            mtype   = _mask_type_short(mask_used)
            family  = _mask_family(mtype)
            var_id  = int(man.get('variation_id', 0))
            case_id = f'{coco_id}_{model}_{type_}_{mtype}_v{var_id}'
            # Subcategory cell for the eval partition: (model, type, mask_family).
            # 3 models × {sp,fr} × {semantic,random} = the 12 held-out cells.
            subcat  = f'{model}|{type_}|{family}'

            item = Item(
                image=fake_path,
                authentic=real_path,
                mask=mask_path,
                source=source,
                item_id=make_item_id(source, fake_path),
                meta={
                    'case_id':          case_id,
                    'tgif_coco_id':     str(coco_id),
                    'tgif_category':    category,
                    'tgif_type':        type_,
                    'tgif_model':       model,
                    'tgif_mask_type':   mtype,
                    'tgif_mask_family': family,
                    'tgif_subcat':      subcat,
                    'tgif_var_id':      var_id,
                },
            )
            by_cell[(model, type_, mtype)].append(item)

    if n_missing_fake or n_missing_mask:
        log_line(
            f'[data] tgif2 WARNING: '
            f'missing_fake={n_missing_fake} missing_mask={n_missing_mask}'
        )

    fake_items: List[Item] = []
    # Interleave items across cells so that slicing the first N items (e.g. --max_items)
    # distributes them evenly across all subgroups instead of getting stuck in one cell.
    max_len = max((len(by_cell[ck]) for ck in by_cell), default=0)
    for i in range(max_len):
        for ck in sorted(by_cell):
            if i < len(by_cell[ck]):
                fake_items.append(by_cell[ck][i])

    return fake_items, real_items


def build(
    root: Path,
    *,
    res: Resolution,
    source: str = 'tgif2',
    index_path: Optional[Path] = None,
    verify_policy: Optional[VerifyPolicy] = None,
    train_frac: float = 0.0,
    split_seed: str = 'tgif_fr_half',
    max_per_cell: Optional[int] = None,
    eval_per_cell: Optional[int] = None,
    include_reals: bool = True,
    types: Optional[Set[str]] = None,
    build_train_side: bool = True,
) -> Tuple[Dataset, Dataset]:
    """Build (train_dataset, val_dataset) from a tgif2_index.json.

    Args:
        root:          Dataset root; index-relative paths are resolved against it.
        res:           Resolution for the Datasets.
        source:        Source label on Items (default 'tgif2').
        index_path:    Path to tgif2_index.json (default <root>/tgif2_index.json).
        verify_policy: Override the default drop-and-log verify policy.
        train_frac:    Fraction of coco_ids to reserve for training.  0.0 (default)
                       returns an empty train_dataset (pure eval probe).
        split_seed:    Hash seed for the deterministic coco_id split.
        max_per_cell:  If set, deterministically subsample each
                       (model, type, mask_type) cell to this many fakes.
        eval_per_cell: TGIF-FINETUNE holdout mode (overrides train_frac).  When
                       set, split coco_ids leakage-free so the VAL side holds up
                       to this many splices per (model, type, mask_family) cell
                       and TRAIN gets every remaining coco_id.  Reals ride with
                       their coco_id's side.
        include_reals: Also build one real item per coco_id.
        types:         If set, restrict to these manipulation types ('sp', 'fr').
        build_train_side: Set False when the caller discards the train dataset
                       (e.g. train.py's per-epoch TGIF2 val-only probe — "TGIF
                       is never added to train_items").  Skips collecting AND
                       verifying the train-side coco_ids entirely, so a bad
                       mask/image pairing in an item nobody will ever use can't
                       crash a run.  train_tgif.py (the TGIF-FINETUNE recipe)
                       actually trains on this split and must leave it True.
    """
    root = Path(root)
    idx_path = Path(index_path) if index_path is not None else root / 'tgif2_index.json'

    if not idx_path.exists():
        log_line(f'[data] WARN: tgif2 index not found: {idx_path}')
        empty = Dataset([], res=res, augment=False)
        return empty, empty

    with open(idx_path) as f:
        index = json.load(f)

    log_line(f'[data] Indexing tgif2 ({len(index)} coco_ids): {idx_path}')

    if eval_per_cell is not None:
        train_coco_ids, val_coco_ids = _split_coco_ids_by_cell(
            index, eval_per_cell, split_seed, types
        )
    elif train_frac > 0.0:
        train_coco_ids, val_coco_ids = _split_coco_ids(index, train_frac, split_seed)
    else:
        train_coco_ids = set()
        val_coco_ids   = set(index.keys())

    root_str = str(root)

    def _collect(coco_ids: Set[str]) -> List[Item]:
        if not coco_ids:
            return []
        fake_items, real_items = _build_items(
            root_str, index,
            source=source,
            coco_ids=coco_ids,
            include_reals=include_reals,
            types=types,
        )
        if max_per_cell is not None:
            by_cell: Dict[Tuple[str, str, str], List[Item]] = defaultdict(list)
            for it in fake_items:
                ck = (
                    it.meta.get('tgif_model', ''),
                    it.meta.get('tgif_type', ''),
                    it.meta.get('tgif_mask_type', ''),
                )
                by_cell[ck].append(it)
            fake_items = []
            for ck in sorted(by_cell):
                cell_seed = f'{split_seed}:{ck[0]}:{ck[1]}:{ck[2]}'
                fake_items.extend(
                    deterministic_subsample(by_cell[ck], max_per_cell, seed=cell_seed)
                )
        return fake_items + real_items

    train_all = _collect(train_coco_ids) if build_train_side else []
    val_all   = _collect(val_coco_ids)

    # Holdout mode: cap the VAL side to eval_per_cell splices per (model, type,
    # family) cell.  TRAIN keeps every remaining coco_id uncapped.
    if eval_per_cell is not None:
        val_all = _cap_eval_cells(val_all, eval_per_cell, split_seed)

    if build_train_side:
        train_kept, _ = verify_all(train_all, policy=verify_policy,
                                    log_tag=f'[data] {source} train')
    else:
        train_kept = []
    val_kept, _ = verify_all(val_all, policy=verify_policy,
                              log_tag=f'[data] {source} val')

    log_line(
        f'[data] tgif2: train={len(train_kept)} val={len(val_kept)} '
        f'coco_ids={len(index)} train_frac={train_frac} '
        f'max_per_cell={max_per_cell} eval_per_cell={eval_per_cell}'
        + (f' types={sorted(types)}' if types is not None else '')
    )
    return (
        Dataset(train_kept, res=res, augment=True),
        Dataset(val_kept,   res=res, augment=False),
    )
