"""tests.test_full_fakes_view — the whole-image eval view + its trigger.

full_fakes fakes carry an all-white sentinel mask, so f1/iou/precision are
category errors rather than measurements (CLAUDE.md rule 2). These cover the
detector for that condition and the AUROC-based view that replaces the
localization block. Torch-free: aggregate.py is numpy + dataclasses only.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_utils.eval.aggregate import (                          # noqa: E402
    localization_is_meaningful,
    summarize_full_fakes,
)
from lab_utils.eval.record import EvalRecord                     # noqa: E402


class _Item:
    """Minimal stand-in for a dataset Item (only is_real + meta are read)."""

    def __init__(self, is_real, sentinel=False):
        self.is_real = is_real
        self.meta = {} if is_real else ({'gt_mask_reliable': False} if sentinel else {})


def _rec(is_real, score, lit=0.5, gen=None):
    return EvalRecord(
        item_id='x', is_real=is_real, source='full_fakes', decoder='threshold',
        gt_mask=np.zeros((2, 2), bool), pred_mask=np.zeros((2, 2), bool),
        attention=None, image_score=score,
        f1=1.0, iou=lit, precision=1.0, recall=lit, accuracy=1.0 - lit,
        mask_area=0.0 if is_real else 1.0, bucket='large', subgroup=gen,
    )


class TestLocalizationIsMeaningful:
    def test_false_when_every_fake_is_sentinel(self):
        items = [_Item(True), _Item(False, sentinel=True), _Item(False, sentinel=True)]
        assert localization_is_meaningful(items) is False

    def test_true_with_real_gt_masks(self):
        assert localization_is_meaningful([_Item(True), _Item(False)]) is True

    def test_true_when_mixed(self):
        # A real-GT source present alongside full_fakes: localization still means
        # something for that source, so the standard view must stay.
        items = [_Item(True), _Item(False, sentinel=True), _Item(False)]
        assert localization_is_meaningful(items) is True

    def test_false_with_no_fakes(self):
        assert localization_is_meaningful([_Item(True)]) is False

    def test_false_on_empty(self):
        assert localization_is_meaningful([]) is False


class TestSummarizeFullFakes:
    def _mixed(self):
        rng = np.random.default_rng(0)
        recs = [_rec(True, float(rng.uniform(0.0, 0.4)), 0.2) for _ in range(50)]
        recs += [_rec(False, float(rng.uniform(0.7, 1.0)), 0.95, 'easy') for _ in range(20)]
        recs += [_rec(False, float(rng.uniform(0.2, 0.6)), 0.95, 'hard') for _ in range(20)]
        return recs

    def test_reports_auc_and_counts(self):
        out = summarize_full_fakes(self._mixed())
        assert out['n_fake'] == 40 and out['n_real'] == 50
        assert 0.5 < out['image_auc'] <= 1.0

    def test_reports_no_localization_f1(self):
        out = summarize_full_fakes(self._mixed())
        for banned in ('f1', 'iou', 'precision'):
            assert banned not in out, f'{banned} must not appear in the full-fakes view'

    def test_per_generator_auc_separates_easy_from_hard(self):
        out = summarize_full_fakes(self._mixed())
        gens = out['generators']
        assert set(gens) == {'easy', 'hard'}
        assert gens['easy']['image_auc'] > gens['hard']['image_auc']
        assert gens['easy']['n'] == 20

    def test_thin_generators_are_pooled_not_reported(self):
        # A 3-image pool posts a meaningless auc=1.000 if reported on its own.
        recs = self._mixed()
        recs += [_rec(False, 0.99, 0.95, 'tiny-pool') for _ in range(3)]
        out = summarize_full_fakes(recs, min_n=5)
        assert 'tiny-pool' not in out['generators'], 'thin pool reported individually'
        assert 'tiny-pool' in out['generators']['(thin)']['members']
        assert out['generators']['(thin)']['n'] == 3

    def test_min_n_of_one_reports_everything(self):
        recs = self._mixed()
        recs += [_rec(False, 0.99, 0.95, 'tiny-pool') for _ in range(3)]
        out = summarize_full_fakes(recs, min_n=1)
        assert 'tiny-pool' in out['generators']
        assert '(thin)' not in out['generators']

    def test_lit_and_false_lit_are_reported(self):
        # lit == predicted-positive fraction on fakes; false_lit == 1 - accuracy
        # on reals. These are the honest replacements for recall/precision here.
        # reals are built with lit=0.2 -> accuracy=0.8 -> false_lit=0.2
        out = summarize_full_fakes(self._mixed())
        assert out['lit']['mean'] == 0.95
        assert abs(out['false_lit']['mean'] - 0.2) < 1e-9

    def test_survives_records_with_no_subgroup(self):
        recs = [_rec(True, 0.1), _rec(False, 0.9)]
        out = summarize_full_fakes(recs)
        assert out['generators'] == {}
