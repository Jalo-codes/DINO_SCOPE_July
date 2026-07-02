"""tests.test_aggregate_subgroup — by_subgroup / summarize_by_subgroup (torch-free).

Pins the eval-only subgroup reporting dim added for the TGIF finetune recipe:
records group by their caller-assigned ``subgroup`` label, reals pool separately,
and records with no subgroup are excluded from the partition.
"""

import numpy as np

from lab_utils.eval.record import EvalRecord
from lab_utils.eval.aggregate import by_subgroup, summarize_by_subgroup


def _rec(*, subgroup, is_real=False, f1=0.5, source='tgif2', item_id='x'):
    z = np.zeros((2, 2), dtype=bool)
    return EvalRecord(
        item_id=item_id, is_real=is_real, source=source, decoder='kmeans_flat',
        gt_mask=z, pred_mask=z, attention=None, image_score=float('nan'),
        f1=f1, iou=f1, precision=f1, recall=f1, accuracy=1.0 if is_real else f1,
        mask_area=0.0 if is_real else 0.1, bucket='small', subgroup=subgroup,
    )


def test_subgroup_defaults_to_none():
    z = np.zeros((2, 2), dtype=bool)
    r = EvalRecord(
        item_id='x', is_real=False, source='imd2020', decoder='kmeans_flat',
        gt_mask=z, pred_mask=z, attention=None, image_score=float('nan'),
        f1=0.4, iou=0.4, precision=0.4, recall=0.4, accuracy=0.4,
        mask_area=0.1, bucket='small',
    )
    assert r.subgroup is None


def test_by_subgroup_groups_and_drops_none():
    records = [
        _rec(subgroup='flux|sp|semantic', f1=0.6),
        _rec(subgroup='flux|sp|semantic', f1=0.8),
        _rec(subgroup='flux|fr|random', f1=0.2),
        _rec(subgroup=None, f1=0.9),                 # opted out → dropped
        _rec(subgroup='real', is_real=True),
    ]
    groups = by_subgroup(records)
    assert set(groups) == {'flux|sp|semantic', 'flux|fr|random', 'real'}
    assert len(groups['flux|sp|semantic']) == 2
    # the None-subgroup record is excluded entirely
    assert sum(len(v) for v in groups.values()) == 4


def test_summarize_by_subgroup_pools_reals_separately():
    records = [
        _rec(subgroup='flux|sp|semantic', f1=0.6),
        _rec(subgroup='flux|sp|semantic', f1=0.8),
        _rec(subgroup='flux|sp|semantic', is_real=True),
    ]
    out = summarize_by_subgroup(records, log_tag='[eval]')
    cell = out['flux|sp|semantic']
    assert cell['n_splice'] == 2
    assert cell['n_real'] == 1
    # median of {0.6, 0.8} splice F1
    assert abs(cell['splices']['f1']['median'] - 0.7) < 1e-9


def test_summarize_by_subgroup_empty_is_safe():
    out = summarize_by_subgroup([_rec(subgroup=None)], log_tag='[eval]')
    assert out == {}
