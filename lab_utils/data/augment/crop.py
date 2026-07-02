"""lab_utils.data.augment.crop — oracle mask crop (I1 tripwire).

TORCH-FREE (GAMEPLAN C3). Uses numpy + PIL only.

I1 — "oracle" token is allowed ONLY in this file.  Dataset code may import
oracle_mask_crop but must never name the token elsewhere; that naming is the
signal used to audit for eval leakage.

oracle_mask_crop is a training-only augmentation.  It must NOT be called from
any eval path (fetch, decode, metric).
"""

from typing import Optional, Tuple

import numpy as np
from PIL import Image

from lab_utils.data.resolution import CropResult


def oracle_mask_crop(
    img: Image.Image,
    mask: Image.Image,
    res: object,
    *,
    target_cov_range: Tuple[float, float] = (0.10, 0.40),
    jitter_frac: float = 0.25,
    rng=None,
) -> CropResult:
    """Mask-centered zoom crop that guarantees the splice stays in frame.

    The fix for small/off-center splices that random-resized crops can't
    surface.  Picks a SQUARE window centered on the mask centroid, sized so
    the splice covers roughly ``target_cov_range`` of the window area —
    zooming a tiny splice up to a learnable scale — always containing the
    full mask bounding box, with optional center jitter for variety.

    Returns ``valid=True`` with ``mode='oracle'`` on success.  Returns
    ``valid=False`` with ``mode='oracle_empty'`` ONLY when the mask is empty
    (no splice pixels); the caller should then DROP the sample rather than
    feed a splice-free crop with a splice label.

    Args:
        img:               PIL RGB image.
        mask:              PIL 'L' mask of the manipulated region.
        res:               Resolution (must have .image_size int attribute).
        target_cov_range:  (lo, hi) target splice coverage fraction in the crop.
        jitter_frac:       Centroid jitter as a fraction of the crop side.
        rng:               Optional numpy RNG (np.random by default).

    Returns:
        CropResult with mode='oracle' or mode='oracle_empty'.
    """
    if rng is None:
        rng = np.random

    image_size = int(res.image_size)

    m = np.asarray(mask.convert('L'), dtype=np.uint8) > 0
    H, W = m.shape
    if not m.any():
        resized = img.resize((image_size, image_size), Image.BILINEAR)
        return CropResult(
            image=resized,
            mask=None,
            valid=False,
            fallback_used=True,
            mode='oracle_empty',
            coverage=0.0,
        )

    ys, xs = np.where(m)
    r0, r1 = int(ys.min()), int(ys.max()) + 1
    c0, c1 = int(xs.min()), int(xs.max()) + 1
    bbox_h, bbox_w = r1 - r0, c1 - c0
    cy, cx = float(ys.mean()), float(xs.mean())
    splice_px = float(m.sum())

    lo, hi = float(target_cov_range[0]), float(target_cov_range[1])
    tcov = float(rng.uniform(lo, hi))
    side_for_cov = (splice_px / max(tcov, 1e-6)) ** 0.5
    side = int(round(max(side_for_cov, float(bbox_h), float(bbox_w))))
    side = max(8, min(side, H, W))

    jit = int(round(float(jitter_frac) * side))
    dy = int(round(rng.uniform(-jit, jit))) if jit > 0 else 0
    dx = int(round(rng.uniform(-jit, jit))) if jit > 0 else 0
    top  = int(round(cy - side / 2.0)) + dy
    left = int(round(cx - side / 2.0)) + dx
    # Clamp so the whole bbox stays in frame AND the crop stays within bounds.
    top  = max(max(0, r1 - side), min(top,  min(r0, H - side)))
    left = max(max(0, c1 - side), min(left, min(c0, W - side)))

    coverage = float(m[top:top + side, left:left + side].sum()) / float(side * side)

    crop_img  = img.crop( (left, top, left + side, top + side)).resize(
        (image_size, image_size), Image.BILINEAR)
    crop_mask = mask.crop((left, top, left + side, top + side)).resize(
        (image_size, image_size), Image.NEAREST)
    return CropResult(
        image=crop_img,
        mask=crop_mask,
        valid=True,
        fallback_used=False,
        chosen_params=(top, left, side, side),
        mode='oracle',
        coverage=coverage,
    )
