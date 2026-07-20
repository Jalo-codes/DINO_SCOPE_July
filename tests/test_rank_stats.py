"""tests.test_rank_stats — lab_utils.eval.rank_stats (rank_auc / stats).

Shared by analysis/probe_contrasts.py and
analysis/full_fakes_report.py. No torch, no GPU; numpy only.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lab_utils.eval.rank_stats import rank_auc, stats          # noqa: E402


class TestRankAuc:
    def test_perfectly_separable_is_one(self):
        pos = np.array([0.9, 0.8, 0.7])
        neg = np.array([0.3, 0.2, 0.1])
        assert rank_auc(pos, neg) == 1.0

    def test_perfectly_reversed_is_zero(self):
        pos = np.array([0.1, 0.2, 0.3])
        neg = np.array([0.7, 0.8, 0.9])
        assert rank_auc(pos, neg) == 0.0

    def test_identical_distributions_is_half(self):
        pos = np.array([0.5, 0.5, 0.5, 0.5])
        neg = np.array([0.5, 0.5, 0.5, 0.5])
        assert rank_auc(pos, neg) == 0.5

    def test_empty_pos_or_neg_is_nan(self):
        assert np.isnan(rank_auc(np.array([]), np.array([0.1, 0.2])))
        assert np.isnan(rank_auc(np.array([0.1, 0.2]), np.array([])))

    def test_nan_scores_are_dropped(self):
        pos = np.array([0.9, np.nan])
        neg = np.array([0.1])
        assert rank_auc(pos, neg) == 1.0

    def test_matches_manual_tie_count(self):
        # 2 pos, 2 neg; one exact tie. Manual Mann-Whitney U:
        # pos=[0.5, 0.8], neg=[0.5, 0.2] -> pairs: (0.5,0.5)=0.5, (0.5,0.2)=1,
        # (0.8,0.5)=1, (0.8,0.2)=1 -> U=3.5 -> AUC=3.5/4=0.875
        pos = np.array([0.5, 0.8])
        neg = np.array([0.5, 0.2])
        assert rank_auc(pos, neg) == 0.875


class TestStats:
    def test_basic_stats(self):
        s = stats([1.0, 2.0, 3.0, 4.0])
        assert s['n'] == 4
        assert s['median'] == 2.5
        assert s['mean'] == 2.5

    def test_empty_returns_nan_block(self):
        s = stats([])
        assert s['n'] == 0
        assert np.isnan(s['median'])
        assert np.isnan(s['mean'])

    def test_nan_values_are_dropped(self):
        s = stats([1.0, float('nan'), 3.0])
        assert s['n'] == 2
        assert s['mean'] == 2.0
