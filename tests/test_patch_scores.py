"""tests.test_patch_scores — lab_utils.eval.patch_scores.weighted_auroc.

Pure numpy, no torch — runs on any machine, including a torch-less laptop.
collect_patch_scores (needs the full data/model stack) is tested separately
in tests/test_patch_scores_collect.py: pytest imports a test file as ONE
module at collection time, so a pytest.importorskip('torch') partway through
this file would abort collection of the whole module — including these
torch-free tests — when torch is missing, not just the tests below it.
"""

import numpy as np
import pytest

from lab_utils.eval.patch_scores import weighted_auroc


# ── weighted_auroc: pure numpy, no torch needed ───────────────────────────────

def test_perfect_separation():
    scores = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
    labels = [0, 0, 0, 1, 1, 1]
    assert weighted_auroc(scores, labels) == pytest.approx(1.0)


def test_anti_separation():
    scores = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
    labels = [0, 0, 0, 1, 1, 1]
    assert weighted_auroc(scores, labels) == pytest.approx(0.0)


def test_tie_gets_half_credit():
    assert weighted_auroc([0.5, 0.5], [0, 1]) == pytest.approx(0.5)


def test_random_scores_near_half():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=4000)
    labels = rng.integers(0, 2, size=4000)
    auc = weighted_auroc(scores, labels)
    assert 0.47 < auc < 0.53


def test_weight_two_equals_two_duplicates():
    scores = [0.1, 0.2, 0.3, 0.9]
    labels = [0, 0, 0, 1]
    weights = [1, 1, 2, 1]
    scores_dup = [0.1, 0.2, 0.3, 0.3, 0.9]
    labels_dup = [0, 0, 0, 0, 1]

    auc_weighted = weighted_auroc(scores, labels, weights)
    auc_duplicated = weighted_auroc(scores_dup, labels_dup)
    assert auc_weighted == pytest.approx(auc_duplicated)


def test_one_class_returns_nan():
    assert np.isnan(weighted_auroc([0.1, 0.2, 0.3], [0, 0, 0]))
    assert np.isnan(weighted_auroc([0.1, 0.2, 0.3], [1, 1, 1]))


def test_empty_returns_nan():
    assert np.isnan(weighted_auroc([], []))


def test_matches_brute_force_pairwise():
    """Cross-check against an O(n^2) reference on a tie-heavy small set."""
    def brute_auc(s, y):
        s = np.asarray(s, dtype=float)
        y = np.asarray(y)
        pos, neg = s[y == 1], s[y == 0]
        total = 0.0
        for p in pos:
            for n in neg:
                if p > n:
                    total += 1.0
                elif p == n:
                    total += 0.5
        return total / (len(pos) * len(neg))

    rng = np.random.default_rng(7)
    scores = rng.integers(0, 5, size=24).astype(float)   # heavy ties
    labels = rng.integers(0, 2, size=24)
    assert weighted_auroc(scores, labels) == pytest.approx(brute_auc(scores, labels))


def test_weighted_matches_weighted_brute_force():
    """Same cross-check but with non-uniform weights (Mann-Whitney generalization)."""
    def brute_weighted_auc(s, y, w):
        s, y, w = np.asarray(s, float), np.asarray(y), np.asarray(w, float)
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        num, den = 0.0, 0.0
        for i in pos_idx:
            for j in neg_idx:
                wij = w[i] * w[j]
                den += wij
                if s[i] > s[j]:
                    num += wij
                elif s[i] == s[j]:
                    num += 0.5 * wij
        return num / den

    rng = np.random.default_rng(9)
    scores = rng.integers(0, 6, size=20).astype(float)
    labels = rng.integers(0, 2, size=20)
    weights = rng.uniform(0.2, 3.0, size=20)
    assert weighted_auroc(scores, labels, weights) == pytest.approx(
        brute_weighted_auc(scores, labels, weights))
