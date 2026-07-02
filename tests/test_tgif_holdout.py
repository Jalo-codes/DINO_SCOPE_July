"""tests.test_tgif_holdout — TGIF finetune holdout split (leakage-free, per-cell).

Pins the ``eval_per_cell`` holdout added for the TGIF continued-finetune recipe:
the coco_id-level split fills a per-(model|type|family) eval quota, is
deterministic, and never puts a coco_id on both sides (no scene leakage).

``tgif2`` pulls torch transitively (Dataset), so this is gated on torch being
installed — it runs on the GPU box and skips on the torch-free dev box.
"""

import pytest

pytest.importorskip('torch')

from collections import Counter
from pathlib import Path

from lab_utils.data.item import Item
from lab_utils.data.datasets import tgif2


MODELS = ['flux-dev', 'sd3', 'kandinsky']
TYPES  = ['sp', 'fr']
MASKS  = ['bbox_512', 'random_512']   # bbox→semantic, random→random


def _synthetic_index(n_coco: int) -> dict:
    idx = {}
    for cid in range(n_coco):
        mans = [
            {'fake_path': f'f/{cid}_{m}_{t}_{mu}.png', 'mask_used': mu,
             'model': m, 'type': t, 'variation_id': 0}
            for m in MODELS for t in TYPES for mu in MASKS
        ]
        idx[str(cid)] = {
            'category': 'cat', 'original_512': f'o/{cid}.png',
            'masks': {'bbox_512': f'm/{cid}_b.png', 'random_512': f'm/{cid}_r.png'},
            'manipulations': mans,
        }
    return idx


def test_split_is_leakage_free_and_per_cell():
    idx = _synthetic_index(200)
    train_ids, eval_ids = tgif2._split_coco_ids_by_cell(idx, 30, 'seedX', None)

    # No coco_id on both sides.
    assert not (train_ids & eval_ids)
    # Together they cover everything.
    assert train_ids | eval_ids == set(idx.keys())

    # Each of the 12 cells reaches the quota on the eval side (each coco_id
    # contributes exactly one item per cell here).
    cc = Counter()
    for cid in eval_ids:
        for cell, k in tgif2._coco_id_cell_counts(idx[cid], None).items():
            cc[cell] += k
    assert len(cc) == 12
    assert min(cc.values()) >= 30


def test_split_is_deterministic():
    idx = _synthetic_index(120)
    a = tgif2._split_coco_ids_by_cell(idx, 20, 'seedX', None)
    b = tgif2._split_coco_ids_by_cell(idx, 20, 'seedX', None)
    assert a == b


def test_type_filter_restricts_cells():
    idx = _synthetic_index(10)
    cells = tgif2._coco_id_cell_counts(idx['0'], {'sp'})
    # only sp survives → 3 models × 1 type × 2 families = 6 cells
    assert len(cells) == 6
    assert all(t == 'sp' for (_, t, _) in cells)


def _fake_item(model, type_, family, i):
    return Item(
        image=Path(f'/x/{model}_{type_}_{family}_{i}.png'),
        authentic=None, mask=Path('/x/m.png'), source='tgif2',
        item_id=f'{model}|{type_}|{family}|{i}',
        meta={'tgif_model': model, 'tgif_type': type_,
              'tgif_mask_family': family, 'tgif_subcat': f'{model}|{type_}|{family}'},
    )


def _real_item(i):
    return Item(image=Path(f'/x/real_{i}.png'), authentic=None, mask=None,
                source='tgif2', item_id=f'real|{i}', meta={'tgif_subcat': 'real'})


def test_cap_eval_cells_caps_splices_keeps_reals():
    items = (
        [_fake_item('flux-dev', 'sp', 'semantic', i) for i in range(50)]
        + [_fake_item('sd3', 'fr', 'random', i) for i in range(50)]
        + [_real_item(i) for i in range(7)]
    )
    capped = tgif2._cap_eval_cells(items, 30, 'seedY')

    by_cell = Counter(
        (it.meta['tgif_model'], it.meta['tgif_type'], it.meta['tgif_mask_family'])
        for it in capped if not it.is_real
    )
    assert by_cell[('flux-dev', 'sp', 'semantic')] == 30
    assert by_cell[('sd3', 'fr', 'random')] == 30
    # all reals survive
    assert sum(it.is_real for it in capped) == 7
    # deterministic
    assert [it.item_id for it in tgif2._cap_eval_cells(items, 30, 'seedY')] == \
           [it.item_id for it in capped]
