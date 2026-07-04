"""lab_utils.eval.metric — the sole GT touch in the eval pipeline (I3).

metric() is the ONLY function that reads triplet.mask / triplet.mask_area.
Everything above it (fetch, decode) is GT-free; everything below it
(aggregate, robustness, labs) consumes EvalRecords only.

Pipeline contract:
    ModelInfo  = fetch.model_info(model, image)
    patch_mask = decode_*(info, ...)
    record     = metric(patch_mask, info, triplet)
"""

import math
from typing import Optional

import numpy as np
from PIL import Image

from lab_utils.eval.fetch import ModelInfo
from lab_utils.eval.record import EvalRecord
from lab_utils.eval.buckets import area_to_bucket
from lab_utils.data.item import Item
from lab_utils.data.dataset import _crop_edges


# ── Binary scoring helpers ─────────────────────────────────────────────────────

def _binary_scores(
    pred: np.ndarray,
    gt: np.ndarray,
) -> dict:
    """Patch-level binary scores: f1, iou, precision, recall, accuracy."""
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = int((pred & gt).sum())
    p_n   = int(pred.sum())
    g_n   = int(gt.sum())
    union = int((pred | gt).sum())
    n     = int(pred.size)
    return {
        'f1':        (2.0 * inter / (p_n + g_n)) if (p_n + g_n) > 0 else 0.0,
        'iou':       (inter / union)              if union > 0         else 0.0,
        'precision': (inter / p_n)                if p_n > 0           else 0.0,
        'recall':    (inter / g_n)                if g_n > 0           else 0.0,
        'accuracy':  float((pred == gt).mean()),
    }


def _load_gt_pixels(
    mask_path,
    threshold: float = 0.5,
    *,
    edge_crop_frac: float = 0.0,
    image_path=None,
) -> Optional[np.ndarray]:
    """Load GT mask at its NATIVE pixel resolution; binarise.

    Returns (H, W) bool at the mask's real size, or None when mask_path is None
    (real item).  Eval is always per-pixel — the patch grid is never used for
    scoring; the prediction is upsampled to meet the GT here.

    When edge_crop_frac is set, the model was fed a border-cropped image (the
    same _crop_edges training/inference use), so the GT must be cropped by the
    IDENTICAL fraction or the upsampled prediction (which only covers the
    cropped frame) gets compared against a still-full-frame mask — a silent
    spatial misalignment, not a real accuracy signal. image_path lets us hard-
    check the image and mask agree in native size before cropping both by the
    same fraction; a mismatch means the crop can't be trusted to land the same
    place on both and is treated as a hard error, not silently ignored.
    """
    if mask_path is None:
        return None
    pil = Image.open(mask_path).convert('L')
    if edge_crop_frac:
        if image_path is not None:
            img_size = Image.open(image_path).size
            if img_size != pil.size:
                raise ValueError(
                    f'_load_gt_pixels: image/mask size mismatch, cannot apply a '
                    f'consistent edge crop — image={image_path} size={img_size}, '
                    f'mask={mask_path} size={pil.size}'
                )
        pil = _crop_edges(pil, edge_crop_frac)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return arr >= threshold


def _upsample_pred_to(
    patch_mask: np.ndarray,
    hw: tuple,
) -> np.ndarray:
    """Nearest-upsample an (n_side, n_side) patch mask to (H, W) pixels.

    NEAREST inverts the square resize the loader applied, so each patch maps
    back to the block of pixels it covered in the model's input frame.
    """
    H, W = hw
    pil = Image.fromarray(patch_mask.astype(np.uint8) * 255, mode='L')
    if pil.size != (W, H):
        pil = pil.resize((W, H), Image.NEAREST)
    return np.asarray(pil) > 127


def _image_score(image_logit: Optional[float]) -> float:
    """sigmoid(image_logit), or NaN when the head is disabled."""
    if image_logit is None or not math.isfinite(image_logit):
        return float('nan')
    return float(1.0 / (1.0 + math.exp(-image_logit)))


# ── Public function ────────────────────────────────────────────────────────────

def metric(
    patch_mask: np.ndarray,
    info: ModelInfo,
    triplet: Item,
    *,
    decoder: str = 'unknown',
    gt_threshold: float = 0.5,
    subgroup: Optional[str] = None,
    edge_crop_frac: float = 0.0,
) -> EvalRecord:
    """Package a decode output into a scored EvalRecord.

    This is the only function that reads triplet.mask and triplet.mask_area.
    Everything above it is GT-free; this function closes the loop.

    Scoring is ALWAYS per-pixel: GT is loaded at its native pixel size and the
    patch-grid prediction is nearest-upsampled to meet it.  No score is ever
    computed on the patch grid.  For real items (no GT) the comparison frame is
    the model's square input resolution.

    Args:
        patch_mask:   (n_side, n_side) bool array — the committed decode output.
        info:         ModelInfo from the same forward pass.
        triplet:      Item — provides GT mask path, mask_area, item_id, source.
        decoder:      String label for the decode method used.
        gt_threshold: Binarise GT mask at this threshold (default 0.5).
        subgroup:     Optional GT-free reporting label (caller-chosen from
                      Item.meta) stored verbatim on the record for by_subgroup().
        edge_crop_frac: MUST match whatever border crop the caller applied to
                      the image before the forward pass (0.0 = none, the old
                      behaviour). The GT mask is cropped by the identical
                      fraction so it lines up with what the model actually
                      saw — passing an image crop without this would silently
                      score a cropped-frame prediction against a full-frame
                      mask (see _load_gt_pixels).

    Returns:
        EvalRecord with all scores pre-computed (gt_mask / pred_mask are pixel-res).
    """
    # The prediction may arrive flat (N,), at the patch grid (n_side, n_side),
    # or already at a finer 2-D resolution (the zoom path places its mask back
    # at pixel resolution to keep the detail zooming bought).  Accept all three;
    # only reshape when it is flat.  Scoring upsamples whatever we get to GT res.
    n_side  = info.grid_hw[0]
    pred_in = np.asarray(patch_mask, dtype=bool)
    pred_patch = pred_in.reshape(n_side, n_side) if pred_in.ndim == 1 else pred_in

    # GT at native pixel resolution (sole touch of triplet.mask).
    gt = _load_gt_pixels(
        triplet.mask, threshold=gt_threshold,
        edge_crop_frac=edge_crop_frac, image_path=triplet.image,
    )
    if gt is not None:
        pred = _upsample_pred_to(pred_patch, gt.shape)          # → (H_gt, W_gt)
    else:
        # Real item: no GT mask.  Score on the square input frame (all-zero GT).
        S    = int(info.res.image_size)
        gt   = np.zeros((S, S), dtype=bool)
        pred = _upsample_pred_to(pred_patch, (S, S))

    # Scores (per-pixel)
    scores = _binary_scores(pred, gt)

    # Mask area and bucket (I5: derived from Item.mask_area, not from gt directly
    # — UNLESS edge_crop_frac is set, in which case Item.mask_area would read
    # the full-frame mask again and disagree with the cropped gt actually
    # scored above; reuse the already-cropped array for area/bucket instead).
    mask_area = float(gt.mean()) if (edge_crop_frac and triplet.mask is not None) else triplet.mask_area(info.res)
    bucket    = area_to_bucket(mask_area)

    return EvalRecord(
        item_id=triplet.item_id,
        is_real=triplet.is_real,
        source=triplet.source,
        decoder=decoder,
        gt_mask=gt,
        pred_mask=pred,
        attention=info.attention,
        image_score=_image_score(info.image_logit),
        f1=scores['f1'],
        iou=scores['iou'],
        precision=scores['precision'],
        recall=scores['recall'],
        accuracy=scores['accuracy'],
        mask_area=float(mask_area),
        bucket=bucket,
        subgroup=subgroup,
    )
