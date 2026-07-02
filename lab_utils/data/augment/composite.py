"""lab_utils.data.augment.composite — regional compositing for splice simulation.

TORCH-FREE (GAMEPLAN C3). Ported from legacy/lab_utils/data/paste.py.

Two entry points:
    paste_regional_ae          — paste AE-reconstructed pixels into a region
    paste_regional_degradation — paste algorithmically-degraded pixels into a region

Both accept a soft float mask (values in [0, 1]) so experiments can use the
raw output of generate_blob_mask without thresholding, preserving edge realism.
A hard binary PIL 'L' mask is also accepted (auto-converted).
"""

from typing import Callable, Union

import numpy as np
from PIL import Image

from lab_utils.errors import DataError
from lab_utils.data.augment.blob import paste_soft_alpha


def _to_float_alpha(mask: Union[Image.Image, np.ndarray],
                    expected_hw: tuple) -> np.ndarray:
    """Normalise mask to float32 (H, W) array in [0, 1]."""
    if isinstance(mask, Image.Image):
        arr = np.array(mask, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
    else:
        arr = np.asarray(mask, dtype=np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0

    if arr.ndim == 3:
        arr = arr[..., 0]

    if arr.shape != expected_hw:
        # Mask saved at a different resolution than the image — resize to match.
        mask_pil = Image.fromarray((arr * 255).astype(np.uint8), mode='L')
        mask_pil = mask_pil.resize((expected_hw[1], expected_hw[0]), Image.NEAREST)
        arr = np.array(mask_pil, dtype=np.float32) / 255.0
    return arr


def paste_regional_ae(
    img: Image.Image,
    ae_recon: Image.Image,
    mask: Union[Image.Image, np.ndarray],
) -> Image.Image:
    """Composite AE-reconstructed pixels into img within the mask region.

    Pixels where mask=1 come from ae_recon; mask=0 keeps the original img.
    Soft mask values (0 < alpha < 1) blend the boundary.
    """
    if img.size != ae_recon.size:
        raise DataError(
            f"paste_regional_ae: img.size={img.size} != ae_recon.size={ae_recon.size}."
        )
    H, W = img.size[1], img.size[0]
    alpha = _to_float_alpha(mask, (H, W))
    composited = paste_soft_alpha(
        background=np.array(img,      dtype=np.uint8),
        foreground=np.array(ae_recon, dtype=np.uint8),
        alpha=alpha,
    )
    return Image.fromarray(composited)


def paste_regional_degradation(
    img: Image.Image,
    mask: Union[Image.Image, np.ndarray],
    degradation_fn: Callable[[Image.Image], Image.Image],
) -> Image.Image:
    """Apply degradation_fn to img then composite result into img within mask.

    degradation_fn is applied to the entire image first (so global frequency
    statistics are intact), then only the masked region is kept.
    """
    H, W = img.size[1], img.size[0]
    alpha = _to_float_alpha(mask, (H, W))

    degraded = degradation_fn(img)
    if degraded.size != img.size:
        raise DataError(
            f"paste_regional_degradation: degradation_fn changed size "
            f"{img.size} → {degraded.size}. Must preserve size."
        )

    composited = paste_soft_alpha(
        background=np.array(img,      dtype=np.uint8),
        foreground=np.array(degraded, dtype=np.uint8),
        alpha=alpha,
    )
    return Image.fromarray(composited)


def paste_real_background(
    manipulated: Image.Image,
    real: Image.Image,
    mask: Union[Image.Image, np.ndarray],
) -> Image.Image:
    """Paste the pristine original over the un-masked region of a manipulated image.

    Used for inpaint-family items (SD-inpaint, COCO-inpaint) where the VAE
    applies a global fingerprint. Pasting the original background restricts the
    signal to the inpainted blob, making the item behave like a true splice.

    Args:
        manipulated: The inpainted/forged image (model input without paste).
        real:        The pristine original. Resized to manipulated.size if needed.
        mask:        Hard or soft mask of the FORGED region. mask=1 → keep
                     manipulated; mask=0 → paste real.
    """
    if real.size != manipulated.size:
        real = real.resize(manipulated.size, Image.BICUBIC)
    H, W = manipulated.size[1], manipulated.size[0]
    alpha = _to_float_alpha(mask, (H, W))
    # alpha=1 → forged region (keep manipulated); alpha=0 → real background
    composited = paste_soft_alpha(
        background=np.array(real,        dtype=np.uint8),
        foreground=np.array(manipulated, dtype=np.uint8),
        alpha=alpha,
    )
    return Image.fromarray(composited)
