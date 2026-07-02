"""lab_utils.data.resolution — single source of truth for image/patch geometry.

``Resolution`` is the one place where image_size, patch_size, and num_patches
live.  Pass a Resolution everywhere instead of individual size integers.

Crop helpers (``random_resized_crop_pair``) are torch-free at import time
(GAMEPLAN C3) — they use PIL and stdlib random.  ``mask_to_patch_labels`` and
``mask_to_patch_labels_soft`` lazy-import torch so they are only evaluated at
Dataset __getitem__ time when torch is guaranteed to be present.

oracle_mask_crop has been moved to lab_utils/data/augment/crop.py (I1 rule).
"""

import dataclasses
import math
import random
from typing import Optional, Tuple

from PIL import Image

from lab_utils.errors import ConfigError, DataError


@dataclasses.dataclass(frozen=True)
class Resolution:
    """Immutable description of a ViT-compatible image resolution.

    Args:
        image_size: Square image side length in pixels.
        patch_size: ViT patch side length in pixels.

    Raises:
        ConfigError: If image_size is not divisible by patch_size.
    """
    image_size: int
    patch_size: int

    def __post_init__(self):
        if self.image_size <= 0:
            raise ConfigError(f"Resolution.image_size must be > 0, got {self.image_size}")
        if self.patch_size <= 0:
            raise ConfigError(f"Resolution.patch_size must be > 0, got {self.patch_size}")
        if self.image_size % self.patch_size != 0:
            raise ConfigError(
                f"image_size={self.image_size} must be divisible by "
                f"patch_size={self.patch_size}."
            )

    @property
    def num_patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        n = self.num_patches_per_side
        return n * n

    def __str__(self) -> str:
        return (
            f"Resolution(image_size={self.image_size}, "
            f"patch_size={self.patch_size}, "
            f"num_patches={self.num_patches})"
        )


@dataclasses.dataclass
class CropResult:
    """Paired (image, mask) after a crop operation.

    mask is None when the source had no mask.

    ``mode`` records how the crop was produced:
        'random'       — standard random-resized crop
        'oracle'       — oracle_mask_crop (training only, I1 controlled)
        'oracle_empty' — oracle called on an empty mask; sample should be dropped
        'resize'       — plain resize fallback
        'center'       — center-crop fallback
    ``coverage`` is the realized splice pixel-fraction inside the crop
    (0.0 for maskless / empty).
    """
    image:          Image.Image
    mask:           Optional[Image.Image]
    valid:          bool = True
    fallback_used:  bool = False
    chosen_params:  Optional[Tuple[int, int, int, int]] = None
    mode:           str = 'random'
    coverage:       float = 0.0


# ---------------------------------------------------------------------------
# PIL-native crop helpers (torch-free)
# ---------------------------------------------------------------------------

def resize_only(img: Image.Image, res: Resolution) -> Image.Image:
    """Resize img to (res.image_size, res.image_size) without cropping."""
    s = res.image_size
    return img.resize((s, s), Image.BILINEAR)


def resize_only_mask(mask: Image.Image, res: Resolution) -> Image.Image:
    """Resize a mask to resolution using NEAREST to preserve hard edges."""
    s = res.image_size
    return mask.resize((s, s), Image.NEAREST)


def center_crop_resize(img: Image.Image, res: Resolution) -> Image.Image:
    """Center-square-crop then resize to resolution."""
    s = res.image_size
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((s, s), Image.BILINEAR)


def _get_rrc_params(
    img: Image.Image,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    max_tries: int = 10,
) -> Tuple[int, int, int, int]:
    """PIL-equivalent of torchvision.transforms.RandomResizedCrop.get_params."""
    width, height = img.size
    area = width * height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(max_tries):
        target_area = area * random.uniform(scale[0], scale[1])
        aspect = math.exp(random.uniform(*log_ratio))
        w = int(round(math.sqrt(target_area * aspect)))
        h = int(round(math.sqrt(target_area / aspect)))
        if 0 < w <= width and 0 < h <= height:
            i = random.randint(0, height - h)
            j = random.randint(0, width  - w)
            return i, j, h, w
    # Fallback: center crop
    side = min(width, height)
    i = (height - side) // 2
    j = (width  - side) // 2
    return i, j, side, side


def _pil_crop(img: Image.Image, i: int, j: int, h: int, w: int) -> Image.Image:
    """PIL crop from top-left (i, j) with height h, width w."""
    return img.crop((j, i, j + w, i + h))


def random_resized_crop(
    img: Image.Image,
    res: Resolution,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
) -> Image.Image:
    """Random-resized-crop an image to resolution (no paired mask)."""
    i, j, h, w = _get_rrc_params(img, scale, ratio)
    return _pil_crop(img, i, j, h, w).resize((res.image_size, res.image_size), Image.BILINEAR)


def random_resized_crop_pair(
    img: Image.Image,
    mask: Optional[Image.Image],
    res: Resolution,
    scale: Tuple[float, float],
    ratio: Tuple[float, float],
    max_tries: int = 24,
    min_mask_area_frac: float = 0.0,
) -> CropResult:
    """Random-resized-crop with the same params applied to both img and mask.

    Falls back to center-crop if all tries produce a mask that is either empty
    or below min_mask_area_frac.  The area check uses numpy (torch-free) so
    this function can run on the dev box without torch installed.

    Args:
        img:               PIL RGB image.
        mask:              PIL 'L' mask, or None for real items.
        res:               Target resolution.
        scale:             (min_scale, max_scale) for area fraction.
        ratio:             (min_ratio, max_ratio) for aspect ratio.
        max_tries:         Number of random attempts before falling back.
        min_mask_area_frac: Minimum fraction of crop pixels that must be
                            foreground for the attempt to be accepted.

    Returns:
        CropResult.
    """
    import numpy as np

    S = res.image_size

    if mask is None:
        i, j, h, w = _get_rrc_params(img, scale, ratio)
        return CropResult(
            image=_pil_crop(img, i, j, h, w).resize((S, S), Image.BILINEAR),
            mask=None,
            valid=True,
            fallback_used=False,
            chosen_params=(i, j, h, w),
            mode='random',
        )

    for _ in range(max_tries):
        i, j, h, w = _get_rrc_params(img, scale, ratio)
        cropped_img  = _pil_crop(img,  i, j, h, w).resize((S, S), Image.BILINEAR)
        cropped_mask = _pil_crop(mask, i, j, h, w).resize((S, S), Image.NEAREST)
        arr = np.asarray(cropped_mask, dtype=np.uint8)
        frac = float((arr > 0).mean())
        if frac > 0.0 and frac >= float(min_mask_area_frac):
            return CropResult(
                image=cropped_img,
                mask=cropped_mask,
                valid=True,
                fallback_used=False,
                chosen_params=(i, j, h, w),
                mode='random',
            )

    # Fallback: center crop.
    width, height = img.size
    side = min(width, height)
    fi = (height - side) // 2
    fj = (width  - side) // 2
    return CropResult(
        image =_pil_crop(img,  fi, fj, side, side).resize((S, S), Image.BILINEAR),
        mask  =_pil_crop(mask, fi, fj, side, side).resize((S, S), Image.NEAREST),
        valid=False,
        fallback_used=True,
        chosen_params=None,
        mode='center',
    )


# ---------------------------------------------------------------------------
# Patch-label helpers (lazy-torch — only called at Dataset.__getitem__ time)
# ---------------------------------------------------------------------------

def mask_to_patch_labels(
    mask: Image.Image,
    res: Resolution,
    threshold: float = 0.15,
):
    """Convert a PIL 'L' mask (image_size × image_size) to per-patch binary labels.

    A patch is labelled 1 if the mean foreground density inside it exceeds
    ``threshold``.

    Returns:
        1D LongTensor of shape (res.num_patches,).

    Raises:
        DataError: If mask size does not match resolution.
    """
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import functional as TF

    w, h = mask.size
    if w != res.image_size or h != res.image_size:
        raise DataError(
            f"mask_to_patch_labels: mask size ({w}×{h}) does not match "
            f"resolution.image_size={res.image_size}.  Resize before calling."
        )
    mask_t  = TF.to_tensor(mask)
    density = F.avg_pool2d(mask_t, res.patch_size, res.patch_size).flatten()
    return (density > threshold).long()


def mask_to_patch_labels_soft(
    mask: Image.Image,
    res: Resolution,
    low: float = 0.02,
    high: float = 0.06,
):
    """Soft per-patch labels with an ignore band near the splice boundary.

    Returns (labels, weights) of shape (res.num_patches,) each:
        density == 0.0           → label=0, weight=1.0  (confident background)
        0 < density < low        → label=0, weight=0.0  (IGNORE — boundary noise)
        low <= density < high    → label=1, weight=ramp (linear 0→1 over band)
        density >= high          → label=1, weight=1.0  (confident splice)

    Returns:
        (labels: LongTensor, weights: FloatTensor) both (num_patches,).

    Raises:
        DataError:  If mask size does not match resolution.
        ValueError: If band thresholds are invalid.
    """
    import torch
    import torch.nn.functional as F
    from torchvision.transforms import functional as TF

    if not (0.0 < float(low) < float(high) <= 1.0):
        raise ValueError(
            f'mask_to_patch_labels_soft: need 0 < low < high <= 1, '
            f'got low={low}, high={high}'
        )
    w, h = mask.size
    if w != res.image_size or h != res.image_size:
        raise DataError(
            f"mask_to_patch_labels_soft: mask size ({w}×{h}) does not match "
            f"resolution.image_size={res.image_size}."
        )
    mask_t  = TF.to_tensor(mask)
    density = F.avg_pool2d(mask_t, res.patch_size, res.patch_size).flatten()

    low_t  = float(low)
    high_t = float(high)

    labels  = (density >= low_t).long()
    weights = torch.zeros_like(density)
    weights[density == 0.0]  = 1.0
    weights[density >= high_t] = 1.0
    ramp_mask = (density >= low_t) & (density < high_t)
    if bool(ramp_mask.any()):
        weights[ramp_mask] = (density[ramp_mask] - low_t) / (high_t - low_t)
    return labels, weights
