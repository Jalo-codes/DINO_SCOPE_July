"""lab_utils.data.item — Item, the uniform triplet member class.

Every dataset is reduced to a list of Items. Reals are the degenerate case
(authentic == manipulated, mask is None). Downstream code never branches on
which dataset an item came from; it operates on Items.

This module is TORCH-FREE at import time (GAMEPLAN C3). The load() method
lazily imports torch only when called.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any, Optional


@dataclasses.dataclass
class Item:
    """Uniform image record: (image, authentic, mask).

    Reals: image is the authentic photo; authentic is None; mask is None.
    Splices: image is the forged image fed to the model; authentic is the
             pre-manipulation source (None when unavailable); mask is the
             GT binary mask of the manipulated region.

    item_id is the single source of determinism for subsampling, sorting,
    and seeding. It is derived from (source, image path) — stable
    across runs as long as the file tree doesn't change.
    """

    image:       Path             # image fed to the model
    authentic:   Optional[Path]   # pre-manipulation source; None for reals and unknown-source splices
    mask:        Optional[Path]   # GT mask; None ⇒ real (no manipulation)
    source:      str              # 'imd2020' | 'casia' | 'tgif2' | 'inpaint' | ...
    item_id:     str              # stable deterministic id
    meta:        dict             # dataset-specific extras (category, model, ...)

    @property
    def is_real(self) -> bool:
        """True when this item is an authentic (negative) image."""
        return self.mask is None

    def mask_area(self, res: Any) -> float:
        """Fraction of pixels that are forged, evaluated at the given resolution.

        Uses PIL (no torch). Returns 0.0 for real items. The resolution arg
        is accepted for API uniformity but not needed for area computation —
        we measure at the mask's native size and report the pixel fraction
        (area fraction is resolution-invariant).
        """
        if self.mask is None:
            return 0.0
        try:
            import numpy as np
            from PIL import Image
            m = np.asarray(Image.open(self.mask).convert('L'), dtype=np.uint8)
            return float((m > 0).mean())
        except Exception:
            return 0.0

    def load(self, res: Any) -> tuple:
        """Load and return (img_tensor, mask_tensor) at the target resolution.

        Lazily imports torch; this is the only method in this module that
        touches tensors. Returns FloatTensor (3,S,S) and FloatTensor (1,S,S)
        where S = res.image_size. For real items the mask tensor is all zeros.
        """
        import torch
        from PIL import Image
        from torchvision.transforms import functional as TF

        S = res.image_size

        img_pil = Image.open(self.image).convert('RGB')
        if img_pil.size != (S, S):
            img_pil = img_pil.resize((S, S), Image.BILINEAR)
        img_t = TF.to_tensor(img_pil)  # (3, S, S) in [0, 1]

        if self.mask is not None:
            mask_pil = Image.open(self.mask).convert('L')
            if mask_pil.size != (S, S):
                mask_pil = mask_pil.resize((S, S), Image.NEAREST)
            mask_t = TF.to_tensor(mask_pil)  # (1, S, S) in [0, 1]
        else:
            mask_t = torch.zeros(1, S, S)

        return img_t, mask_t


# Alias so both names resolve to the same class throughout the codebase.
# DESIGN_GUIDE calls the class Item; type annotations also use ImageTriplet.
ImageTriplet = Item


def make_item_id(source: str, image_path: Any) -> str:
    """Stable deterministic item_id from (source, image path).

    Args:
        source:     Dataset source string, e.g. 'imd2020'.
        image_path: Path to the model-input image (forged for splices, real for reals).

    Returns:
        32-char hex MD5 digest that is stable across runs.
    """
    raw = f"{source}|{str(image_path)}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()
