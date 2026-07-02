"""lab_utils.data.augment — augmentation building blocks.

TORCH-FREE at import time (GAMEPLAN C3). The AppliedOp and AugmentResult
types are pure Python / PIL. The augmentation PIPELINE (ordering + gating)
is owned by the Dataset (lab_utils/data/dataset.py), not by this package.
"""

import dataclasses
from typing import Optional, Tuple

from PIL import Image


@dataclasses.dataclass(frozen=True)
class AppliedOp:
    """Record of a single applied augmentation step.

    Attributes:
        name:     Short identifier, e.g. 'jpeg', 'gaussian_noise', 'flip_h'.
        params:   Exact parameters used (quality=70, std=0.05, …).
        severity: Normalized severity in [0, 1]. 0 = no effect.
    """
    name:     str
    params:   dict
    severity: float


@dataclasses.dataclass
class AugmentResult:
    """Return type for every public augmentation function.

    Attributes:
        image:   Augmented PIL RGB image.
        mask:    Augmented PIL 'L' mask if applicable. Geometric ops update
                 this; appearance ops pass it through unchanged.
        applied: Tuple of AppliedOp records for every step that ran.
    """
    image:   Image.Image
    mask:    Optional[Image.Image]
    applied: Tuple[AppliedOp, ...]


__all__ = ['AppliedOp', 'AugmentResult']
