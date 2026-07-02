"""lab_utils.eval.decode.threshold — sigmoid-threshold decode on patch_logits.

Pure, silent, GT-free.  Requires the patch-BCE head (ModelInfo.patch_logits).
"""

import numpy as np

from lab_utils.eval.fetch import ModelInfo


def decode_threshold(info: ModelInfo, *, t: float = 0.5) -> np.ndarray:
    """Threshold sigmoid(patch_logits) → (n_side, n_side) bool mask.

    Args:
        info: ModelInfo.  patch_logits must not be None.
        t:    Decision threshold on sigmoid-probabilities in [0, 1].

    Returns:
        (n_side, n_side) bool array — True = predicted-splice patch.

    Raises:
        ValueError: if patch_logits is None (patch-BCE head not enabled).
    """
    if info.patch_logits is None:
        raise ValueError(
            'decode_threshold: ModelInfo.patch_logits is None '
            '(patch-BCE head not enabled in this model).'
        )
    logits = np.asarray(info.patch_logits, dtype=np.float64).reshape(-1)
    probs  = 1.0 / (1.0 + np.exp(-logits))
    n_side = info.grid_hw[0]
    return (probs >= float(t)).reshape(n_side, n_side)
