"""tests.test_mil_masking — AttentionPool keep_mask (MIL-level patch hiding).

Torch-dependent; skipped automatically where torch is unavailable (e.g. the
bare editing venv).  Run on the training box.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip('torch')

from lab_utils.errors import DataError  # noqa: E402
from lab_utils.model.image_bce_detector import AttentionPool  # noqa: E402


def _pool_and_feats(n=16, d=8, seed=0):
    torch.manual_seed(seed)
    pool = AttentionPool(d_in=d, d_hidden=8).eval()
    x = torch.randn(1, n, d)
    return pool, x


def test_keep_mask_none_matches_all_true():
    pool, x = _pool_and_feats()
    with torch.no_grad():
        logit0, attn0 = pool(x, return_attention=True)
        keep = torch.ones(x.shape[1], dtype=torch.bool)
        logit1, attn1 = pool(x, return_attention=True, keep_mask=keep)
    assert torch.allclose(attn0, attn1, atol=1e-5)
    assert torch.allclose(logit0, logit1, atol=1e-5)


def test_hidden_patches_get_zero_attention_and_rest_sums_to_one():
    pool, x = _pool_and_feats()
    n = x.shape[1]
    keep = torch.ones(n, dtype=torch.bool)
    keep[:4] = False  # hide the first 4 patches
    with torch.no_grad():
        _, attn = pool(x, return_attention=True, keep_mask=keep)
    attn = attn.squeeze(0)
    assert torch.allclose(attn[:4], torch.zeros(4), atol=1e-6)
    assert attn[4:].sum().item() == pytest.approx(1.0, abs=1e-5)


def test_batched_keep_mask_shape():
    torch.manual_seed(1)
    pool = AttentionPool(d_in=8, d_hidden=8).eval()
    x = torch.randn(3, 16, 8)
    keep = torch.ones(3, 16, dtype=torch.bool)
    keep[0, :8] = False
    with torch.no_grad():
        logit, attn = pool(x, return_attention=True, keep_mask=keep)
    assert attn.shape == (3, 16)
    assert torch.allclose(attn[0, :8], torch.zeros(8), atol=1e-6)


def test_all_hidden_raises():
    pool, x = _pool_and_feats()
    keep = torch.zeros(x.shape[1], dtype=torch.bool)
    with pytest.raises(DataError):
        pool(x, return_attention=True, keep_mask=keep)
