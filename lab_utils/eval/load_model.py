"""lab_utils.eval.load_model — checkpoint .pt → ready-to-eval model.

The "read cfg from the checkpoint, rebuild the model with those hyperparams,
load_state_dict, return (model, cfg, res)" sequence was inline in scripts/eval.py.
The HDBSCAN lab needs the exact same thing, so it lives here once.

Legacy checkpoints (contrastive_inpainting_v1) were saved as just
``{'model', 'epoch', 'optimizer'}`` — no ``cfg`` slot.  For those we cannot
read hyperparameters from the file, so we:
  * take the backbone identity (model_name / image_size / patch_size / LoRA)
    from explicit overrides, defaulting to this project's real training config
    (dinov3-vith16plus @ 448/16), NOT a generic HF default; and
  * infer the head configuration (contrastive_dim / pool_hidden / patch_bce)
    directly from the state_dict, so any head combination loads exactly.

New rebuild checkpoints that DO carry a ``cfg`` slot keep working unchanged;
explicit overrides always win over both cfg and inference.

Torch-bound; imported lazily by eval scripts/labs, not by eval/__init__.
"""

from __future__ import annotations

from typing import Optional, Tuple

from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line

# This project trains exclusively on DINOv3 ViT-H/16+ at 448/16.  Legacy
# checkpoints carry no cfg, so these are the load-time fallbacks.
_DEFAULT_MODEL_NAME = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
_DEFAULT_IMAGE_SIZE = 448
_DEFAULT_PATCH_SIZE = 16
_DEFAULT_LORA_RANK = 32
_DEFAULT_LORA_ALPHA = 64
_DEFAULT_LORA_DROPOUT = 0.1


def _infer_heads(model_sd: dict) -> Tuple[int, int, bool]:
    """Infer (contrastive_dim, pool_hidden, patch_bce) from a model state_dict.

    Reads head presence + width straight off the saved tensors so we don't
    depend on a cfg slot the legacy checkpoints never wrote:
      * contrastive_proj.weight  → contrastive_dim = shape[0]   (0 if absent)
      * pool.V.weight            → pool_hidden    = shape[0]    (0 if absent)
      * patch_head.weight        → patch_bce      = present
    """
    def _find(suffix: str):
        for k, v in model_sd.items():
            if k.endswith(suffix):
                return v
        return None

    cw = _find('contrastive_proj.weight')
    contrastive_dim = int(cw.shape[0]) if cw is not None else 0

    pv = _find('pool.V.weight')
    pool_hidden = int(pv.shape[0]) if pv is not None else 0

    patch_bce = _find('patch_head.weight') is not None
    return contrastive_dim, pool_hidden, patch_bce


def load_eval_model(
    checkpoint_path: str,
    *,
    device=None,
    strict: bool = True,
    model_name: Optional[str] = None,
    base_dtype: Optional[str] = None,
    image_size: Optional[int] = None,
    patch_size: Optional[int] = None,
    lora_rank: Optional[int] = None,
    lora_alpha: Optional[int] = None,
    lora_dropout: Optional[float] = None,
    contrastive_dim: Optional[int] = None,
    pool_hidden: Optional[int] = None,
    patch_bce: Optional[bool] = None,
) -> Tuple[object, object, Resolution]:
    """Load a checkpoint and return (model, run_config_or_None, resolution).

    Resolution order for every hyperparameter:
        explicit override  >  checkpoint cfg slot  >  inferred / project default

    Head dims (contrastive_dim / pool_hidden / patch_bce) fall back to being
    *inferred from the state_dict* when neither an override nor a cfg slot
    supplies them, so legacy cfg-less checkpoints rebuild exactly.

    Args:
        checkpoint_path: Path to the .pt checkpoint.
        device:          torch device (None → cpu).
        strict:          Pass-through to load_state_dict.
        model_name, image_size, patch_size, lora_*, contrastive_dim,
        pool_hidden, patch_bce: explicit architecture overrides (None → auto).

    Returns:
        (model, cfg, res) — cfg is a RunConfig or None if the slot was empty.
    """
    import torch
    from experiments.configs.run_config import from_dict
    from lab_utils.model.multi_head_detector import build_multi_head_detector
    from lab_utils.train.checkpoint import load as load_ckpt

    state = load_ckpt(checkpoint_path)
    cfg_d = state.get('cfg', {})
    cfg   = from_dict(cfg_d) if cfg_d else None
    model_sd = state['model']

    def _pick(override, cfg_attr, default):
        if override is not None:
            return override
        if cfg is not None:
            return getattr(cfg, cfg_attr, default)
        return default

    # ── Backbone identity (cfg-less checkpoints → project defaults) ────────────
    r_model_name = _pick(model_name, 'model_name', _DEFAULT_MODEL_NAME)
    # base_dtype defaults to the saved cfg slot ('fp32' for legacy checkpoints),
    # so eval VRAM/numerics match how the run was trained. Override to 'bf16' to
    # evaluate the undistilled ViT-7B on a 24 GB GPU.
    r_base_dtype = _pick(base_dtype, 'base_dtype', 'fp32')
    r_image_size = _pick(image_size, 'image_size', _DEFAULT_IMAGE_SIZE)
    r_patch_size = _pick(patch_size, 'patch_size', _DEFAULT_PATCH_SIZE)
    r_lora_rank  = _pick(lora_rank,  'lora_rank',  _DEFAULT_LORA_RANK)
    r_lora_alpha = _pick(lora_alpha, 'lora_alpha', _DEFAULT_LORA_ALPHA)
    r_lora_drop  = _pick(lora_dropout, 'lora_dropout', _DEFAULT_LORA_DROPOUT)

    # ── Heads: override > cfg > inferred-from-state_dict ───────────────────────
    inf_contrastive, inf_pool, inf_patch = _infer_heads(model_sd)
    r_contrastive = _pick(contrastive_dim, 'contrastive_dim', inf_contrastive)
    r_pool_hidden = _pick(pool_hidden,     'pool_hidden',     inf_pool)
    r_patch_bce   = _pick(patch_bce,       'patch_bce',       inf_patch)

    res = Resolution(image_size=r_image_size, patch_size=r_patch_size)

    if cfg is None:
        log_line(
            f'[ckpt] no cfg slot — backbone={r_model_name} @ {r_image_size}/{r_patch_size} '
            f'lora_rank={r_lora_rank}, '
            f'heads(inferred): contrastive_dim={r_contrastive} '
            f'pool_hidden={r_pool_hidden} patch_bce={r_patch_bce}'
        )

    model = build_multi_head_detector(
        model_name=r_model_name,
        base_dtype=r_base_dtype,
        resolution=res,
        lora_rank=r_lora_rank,
        lora_alpha=r_lora_alpha,
        lora_dropout=r_lora_drop,
        contrastive_dim=r_contrastive,
        pool_hidden=r_pool_hidden,
        patch_bce=r_patch_bce,
        device=device,
    )
    model.load_state_dict(model_sd, strict=strict)
    model.eval()

    # Traceability: record the EXACT model identity every eval path ran on, so
    # eval.log / sweep.log are self-describing for apples-to-apples comparison.
    _epoch = state.get('epoch')
    log_line(
        f'[ckpt] identity | checkpoint={checkpoint_path}'
        + (f' epoch={_epoch}' if _epoch is not None else '')
        + f' | backbone={r_model_name} @ {r_image_size}/{r_patch_size}'
          f' lora_rank={r_lora_rank} lora_alpha={r_lora_alpha}'
        + f' | heads: contrastive_dim={r_contrastive} pool_hidden={r_pool_hidden}'
          f' patch_bce={r_patch_bce} | base_dtype={r_base_dtype}'
    )
    return model, cfg, res
