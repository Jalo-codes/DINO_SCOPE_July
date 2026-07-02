"""lab_utils.eval.fetch — the sole model entry point (I2).

model_info() is the ONLY function in the codebase that calls the model forward
pass.  Every eval path (decoders, labs, scripts) goes through this and only this.
No GT parameter exists on this function — that is structurally enforced.
"""

import dataclasses
from typing import Optional, Tuple

import numpy as np

from lab_utils.data.resolution import Resolution


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    """Raw model signal packaged for downstream decode and metric.

    Maps directly onto MultiHeadDetector.forward() output.
    Disabled heads yield None.  grid_hw and res carry the geometry.

    GT-free by construction — this dataclass has no mask / label fields.
    """
    patch_logits: Optional[np.ndarray]   # (N,)    per-patch BCE logits (patch-BCE head)
    attention:    Optional[np.ndarray]   # (N,)    per-patch pool attention weights
    embeddings:   Optional[np.ndarray]   # (N, D)  L2-normalised contrastive embeddings
    image_logit:  Optional[float]        # scalar  image-level logit (None = head disabled)
    grid_hw:      Tuple[int, int]        # (n_side, n_side) patch grid shape
    res:          Resolution             # geometry for patch→pixel projection
    patch_feats:  Optional[np.ndarray] = None  # (N, D) raw patch features; only set when
                                               # model_info(return_feats=True). Feeds
                                               # repool_hidden (MIL re-pool). Never cached.


def model_info(
    model,
    image_tensor,
    *,
    device,
    amp: bool = True,
    amp_dtype: str = 'float16',
    return_feats: bool = False,
) -> ModelInfo:
    """Single forward pass → packaged ModelInfo.  The ONLY model call site.

    Args:
        model:        MultiHeadDetector instance (or duck-typed equivalent with
                      .res: Resolution and a forward that returns the expected dict).
        image_tensor: FloatTensor (1, 3, S, S) normalised to ImageNet stats.
        device:       torch.device to run on.
        amp:          Use torch.autocast for the forward (faster on GPU; set False
                      to debug numerical issues).
        amp_dtype:    Data type for mixed precision ('float16' or 'bfloat16').
        return_feats: Also extract raw patch features into ModelInfo.patch_feats.
                      Off by default (extra memory); the second-best / re-pool
                      path turns it on so repool_hidden can reuse pass-1 features.

    Returns:
        ModelInfo with numpy arrays extracted from the forward output.
    """
    import contextlib
    import torch

    res: Resolution = model.res
    n_side = res.num_patches_per_side

    with torch.no_grad():
        img = image_tensor.to(device, non_blocking=True)
        if amp:
            device_type = str(device).split(':')[0]
            dtype = torch.bfloat16 if amp_dtype == 'bfloat16' else torch.float16
            ctx = torch.autocast(device_type=device_type, dtype=dtype)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            out = model(img)

    def _to_np(t) -> Optional[np.ndarray]:
        if t is None:
            return None
        arr = t.detach().cpu().float().numpy()
        if arr.ndim >= 2 and arr.shape[0] == 1:
            arr = arr[0]   # squeeze batch dim: (1, N) → (N,), (1, N, D) → (N, D)
        return arr

    img_logit_t  = out.get('image_logit')
    image_logit: Optional[float] = (
        None if img_logit_t is None
        else float(img_logit_t.detach().cpu().float().item())
    )

    return ModelInfo(
        patch_logits=_to_np(out.get('patch_logit')),
        attention=_to_np(out.get('pool_attention')),
        embeddings=_to_np(out.get('contrastive')),
        image_logit=image_logit,
        grid_hw=(n_side, n_side),
        res=res,
        patch_feats=_to_np(out.get('patch_feats')) if return_feats else None,
    )


def repool_hidden(model, info: ModelInfo, hide_mask) -> ModelInfo:
    """Re-run the MIL attention pool with a set of patches hidden — NO backbone.

    Reuses the pass-1 patch features (``info.patch_feats``, populated via
    ``model_info(..., return_feats=True)``).  Hidden patches are excluded BEFORE
    the pool softmax, so the returned attention is renormalized over the
    surviving patches (0 at hidden positions) and the image logit reflects the
    re-pool.

    This is the MIL-level hide.  A future backbone-level hide would instead
    re-run ``model_info`` with a token mask; this same pool-exclusion then
    applies on top.  Lives in fetch (the model-entry module) so all
    model-touching stays centralized (I2 spirit); it does not call ``model_info``
    or the backbone forward.

    Args:
        model:     MultiHeadDetector with a ``.pool`` head.
        info:      pass-1 ModelInfo with ``patch_feats`` set.
        hide_mask: (N,) bool over patches, True = hide.

    Returns:
        A new ModelInfo with updated ``attention`` and ``image_logit``; all other
        fields (including ``patch_feats``) carried unchanged.
    """
    import torch

    if info.patch_feats is None:
        raise ValueError(
            'repool_hidden: info.patch_feats is None — call model_info(..., return_feats=True)'
        )
    if getattr(model, 'pool', None) is None:
        raise ValueError('repool_hidden: model has no BCE attention pool')

    feats_np = np.asarray(info.patch_feats)
    n = feats_np.shape[0]
    hide = np.asarray(hide_mask, dtype=bool).reshape(-1)[:n]
    keep = ~hide

    pool_device = next(model.pool.parameters()).device
    feats  = torch.from_numpy(feats_np).float().unsqueeze(0).to(pool_device)  # (1, N, D)
    keep_t = torch.from_numpy(keep).to(pool_device)                           # (N,)

    with torch.no_grad():
        logit_t, attn_t = model.pool(feats, return_attention=True, keep_mask=keep_t)

    image_logit = float(logit_t.detach().cpu().float().item())
    attention = attn_t.detach().cpu().float().numpy()
    if attention.ndim >= 2 and attention.shape[0] == 1:
        attention = attention[0]

    return dataclasses.replace(info, attention=attention, image_logit=image_logit)
