"""experiments.configs.run_config — frozen RunConfig + resolve_config (C2).

One resolved config object drives the entire run — model build, data build,
train loop, and eval.  Nothing downstream reads raw argparse Namespace objects.

The settings printout is auto-rendered from RunConfig fields by log_run_config
(lab_utils/logging/run_config.py).  A new field prints automatically; a printed
value equals the value that actually runs.

RunConfig is serialized into the checkpoint `cfg` slot and written to the run
directory — so printed settings == saved settings == resumed settings.

Swin/sliding-window fields are EXCLUDED (removed in the rebuild).
Oracle-soft-label fields are EXCLUDED (no oracle in eval, I1).
"""

import dataclasses
from pathlib import Path
from typing import Dict, Optional, Tuple

from lab_utils.train.hardware import HardwareInfo


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """Resolved, immutable configuration for one training run.

    Built once by resolve_config(); passed to model build, data build, loop,
    and eval.  Serializable to/from dict for checkpoint embedding (C1).
    """

    # ── Dataset roots ──────────────────────────────────────────────────────────
    imd2020_root:       Optional[str] = None
    casia_root:         Optional[str] = None
    indoor_root:        Optional[str] = None
    coco_inpaint_root:  Optional[str] = None
    sagid_root:         Optional[str] = None
    bfree_root:         Optional[str] = None
    anyedit_root:       Optional[str] = None
    tgif2_root:         Optional[str] = None

    # ── Checkpoint / run dir ──────────────────────────────────────────────────
    run_dir:            Optional[str] = None
    resume:             Optional[str] = None
    init_weights:       Optional[str] = None
    seed:               int           = 42
    log_every:          int           = 20

    # ── Training loop ─────────────────────────────────────────────────────────
    num_epochs:              int   = 10
    warmup_epochs:           float = 1.0
    early_stop_patience:     int   = 3
    early_stop_min_delta:    float = 0.002
    min_epochs:              int   = 0       # floor: never early-stop before this many epochs
    max_train_epochs:        Optional[int] = None  # hard cap on the training loop; LR schedule horizon stays num_epochs
    early_stop_reduce:       str   = 'median'  # 'median' | 'mean' — reduction over per-splice loc F1
    batch_size:              int   = 8
    grad_accum:              int   = 4
    lr:                      float = 2e-4
    weight_decay:            float = 1e-4
    train_samples:           int   = 2000
    num_workers:             int   = 0
    persistent_workers:      bool  = False
    prefetch_factor:         Optional[int] = None

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name:          str   = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'
    base_dtype:          str   = 'fp32'  # frozen-backbone load dtype: 'fp32' (default, legacy) | 'bf16' | 'fp16'.
                                         # 'bf16' halves backbone VRAM (~27→~13.4 GB for ViT-7B) so the 7B fits a 24 GB L4;
                                         # trainable LoRA/head params are kept in fp32 for stable optimization regardless.
    image_size:          int   = 448
    patch_size:          int   = 16
    lora_rank:           int   = 32   # LoRA rank; 0 = no LoRA (fully frozen backbone, heads-only training)
    lora_alpha:          int   = 64
    lora_dropout:        float = 0.1
    lora_block_start:    Optional[int] = None  # adapt only blocks with index >= this (None = from block 0)
    lora_block_end:      Optional[int] = None  # adapt only blocks with index <  this (None = through last); half-open [start, end)
    contrastive_dim:     int   = 128
    pool_hidden:         int   = 256
    patch_bce:           bool  = False
    no_grad_checkpoint:  bool  = False  # True = disable backbone gradient checkpointing (faster backward, more VRAM)

    # ── Loss lambdas ──────────────────────────────────────────────────────────
    lambda_image_bce:     float = 1.0
    lambda_contrastive:   float = 2.0
    lambda_patch_bce:     float = 1.0
    patch_pos_weight:     float = 10.0

    # ── Data / sampling ───────────────────────────────────────────────────────
    splice_mix:          Optional[Dict[str, float]] = None   # {source: frac}
    casia_train:         bool  = False
    imd_val_only:        bool  = False
    imd_val_split:       Optional[float] = None  # override IMD val_split (1.0 = full IMD set); None = dataset default (0.10)

    # ── Augmentation ──────────────────────────────────────────────────────────
    train_crop_min:          float = 0.18
    train_crop_max:          float = 1.00
    train_crop_ratio_min:    float = 0.60
    train_crop_ratio_max:    float = 1.70
    use_splice_degradation:  bool  = False
    use_real_degradation:    Optional[bool] = None
    paste_frac:              float = 0.40   # inpaint paste-back prob == sp share (rest = fr)
    noise_prob:              Optional[float] = None
    jpeg_prob:               Optional[float] = None
    whole_corrupt_prob:      float = 0.0
    oracle_crop:             bool  = False

    # ── Recipe (which train harness produced this run) ─────────────────────────
    # 'standard' = experiments/scripts/train.py.  'tgif_finetune' = the isolated
    # warm-start-on-TGIF harness (experiments/scripts/train_tgif.py).  Tagged so
    # a finetune checkpoint can never be confused with a standard OOD run.
    recipe:              str           = 'standard'
    init_checkpoint:     Optional[str] = None   # warm-start weights (model only)
    eval_per_cell:       int           = 500    # TGIF holdout: eval splices / cell
    val_per_cell:        Optional[int] = None   # per-epoch scoring cap / cell
    imd_max_items:       Optional[int] = None   # per-epoch IMD scoring cap
    val_zoom:            bool          = True   # per-epoch val uses attention-zoom two-pass (default on)
    val_zoom_pad_frac:   Optional[float] = None  # area-based crop pad (frame fraction/side); None = legacy patch pad
    val_zoom_min_area:   float         = 0.0    # with val_zoom_pad_frac: floor padded crop to this frame-area fraction
    tgif_val_models:     Optional[Tuple[str, ...]] = None  # restrict tgif per-epoch val to these generators
    tgif_types:          Optional[tuple] = None # restrict TGIF to these manip types ('sp','fr'); None = all
    tgif_eval_decoders:  Tuple[str, ...] = ('kmeans', 'hdbscan')
    primary_surface:     str           = 'imd'  # early-stop driver surface: 'imd' (OOD) or 'tgif' (in-domain)

    # ── Hardware (resolved by resolve_config, embedded for logging) ────────────
    device:              str           = 'cuda'
    use_amp:             bool          = False
    amp_dtype:           Optional[str] = None   # 'fp16' | 'bf16' | None
    world_size:          int           = 1
    rank:                int           = 0
    gpu_name:            Optional[str] = None
    compute_cap:         Optional[str] = None


def resolve_config(args, *, hw: Optional[HardwareInfo] = None) -> RunConfig:
    """Collapse CLI args + resolved hardware into one frozen RunConfig.

    Args:
        args: argparse.Namespace from the train script's parser.
        hw:   Pre-resolved HardwareInfo (or None to use cpu/no-amp defaults).

    Returns:
        Frozen RunConfig — the single source of truth for the entire run.
    """
    splice_mix: Optional[Dict[str, float]] = None
    if getattr(args, 'splice_mix', None):
        pairs = args.splice_mix if isinstance(args.splice_mix, list) else [args.splice_mix]
        splice_mix = {}
        for tok in pairs:
            src, _, frac = tok.partition('=')
            splice_mix[src.strip()] = float(frac)

    hw_device    = hw.device      if hw else getattr(args, 'device', 'cpu')
    hw_use_amp   = hw.use_amp     if hw else False
    hw_amp_dtype = hw.amp_dtype   if hw else None
    # Explicit --amp_dtype overrides hardware autodetection (e.g. pin fp16 on a
    # bf16-capable L4/A100 to keep precision constant across a sweep + resume).
    _amp_override = getattr(args, 'amp_dtype', None)
    if _amp_override:
        hw_amp_dtype = _amp_override
        hw_use_amp   = True
    hw_world     = hw.world_size  if hw else 1
    hw_rank      = hw.rank        if hw else 0
    hw_gpu_name  = hw.gpu_name    if hw else None
    hw_cc        = hw.compute_cap if hw else None

    return RunConfig(
        # dataset roots
        imd2020_root=getattr(args, 'imd2020_root', None),
        casia_root=getattr(args, 'casia_root', None),
        indoor_root=getattr(args, 'indoor_root', None),
        coco_inpaint_root=getattr(args, 'coco_inpaint_root', None),
        sagid_root=getattr(args, 'sagid_root', None),
        bfree_root=getattr(args, 'bfree_root', None),
        anyedit_root=getattr(args, 'anyedit_root', None),
        tgif2_root=getattr(args, 'tgif2_root', None),
        # ckpt/run dir
        run_dir=getattr(args, 'checkpoint_root', None) or getattr(args, 'run_dir', None),
        resume=getattr(args, 'resume', None),
        init_weights=getattr(args, 'init_weights', None),
        seed=getattr(args, 'seed', 42),
        log_every=getattr(args, 'log_every', 20),
        # training loop
        num_epochs=getattr(args, 'num_epochs', 10),
        warmup_epochs=getattr(args, 'warmup_epochs', 1.0),
        early_stop_patience=getattr(args, 'early_stop_patience', 3),
        early_stop_min_delta=getattr(args, 'early_stop_min_delta', 0.002),
        min_epochs=getattr(args, 'min_epochs', 0),
        max_train_epochs=getattr(args, 'max_train_epochs', None),
        early_stop_reduce=getattr(args, 'early_stop_reduce', 'median'),
        batch_size=getattr(args, 'batch_size', 8),
        grad_accum=getattr(args, 'grad_accum', 4),
        lr=getattr(args, 'lr', 2e-4),
        weight_decay=getattr(args, 'weight_decay', 1e-4),
        train_samples=getattr(args, 'train_samples', 2000),
        num_workers=getattr(args, 'num_workers', 0),
        persistent_workers=getattr(args, 'persistent_workers', False),
        prefetch_factor=getattr(args, 'prefetch_factor', None),
        # model
        model_name=getattr(args, 'model_name', 'facebook/dinov3-vith16plus-pretrain-lvd1689m'),
        base_dtype=getattr(args, 'base_dtype', 'fp32'),
        image_size=getattr(args, 'image_size', 448),
        patch_size=getattr(args, 'patch_size', 16),
        lora_rank=getattr(args, 'lora_rank', 32),
        lora_alpha=getattr(args, 'lora_alpha', 64),
        lora_dropout=getattr(args, 'lora_dropout', 0.1),
        lora_block_start=getattr(args, 'lora_block_start', None),
        lora_block_end=getattr(args, 'lora_block_end', None),
        contrastive_dim=getattr(args, 'contrastive_dim', 128),
        pool_hidden=getattr(args, 'pool_hidden', 256),
        patch_bce=getattr(args, 'patch_bce', False),
        no_grad_checkpoint=getattr(args, 'no_grad_checkpoint', False),
        # loss
        lambda_image_bce=getattr(args, 'lambda_image_bce', 1.0),
        lambda_contrastive=getattr(args, 'lambda_contrastive', 2.0),
        lambda_patch_bce=getattr(args, 'lambda_patch_bce', 1.0),
        patch_pos_weight=getattr(args, 'patch_pos_weight', 10.0),
        # data/sampling
        splice_mix=splice_mix,
        casia_train=getattr(args, 'casia_train', False),
        imd_val_only=getattr(args, 'imd_val_only', False),
        imd_val_split=getattr(args, 'imd_val_split', None),
        # augmentation
        train_crop_min=getattr(args, 'train_crop_min', 0.18),
        train_crop_max=getattr(args, 'train_crop_max', 1.00),
        train_crop_ratio_min=getattr(args, 'train_crop_ratio_min', 0.60),
        train_crop_ratio_max=getattr(args, 'train_crop_ratio_max', 1.70),
        use_splice_degradation=getattr(args, 'use_splice_degradation', False),
        use_real_degradation=getattr(args, 'use_real_degradation', None),
        paste_frac=getattr(args, 'paste_frac', 0.40),
        noise_prob=getattr(args, 'noise_prob', None),
        jpeg_prob=getattr(args, 'jpeg_prob', None),
        whole_corrupt_prob=getattr(args, 'whole_corrupt_prob', 0.0),
        oracle_crop=getattr(args, 'oracle_crop', False),
        # recipe / tgif-finetune
        recipe=getattr(args, 'recipe', 'standard'),
        init_checkpoint=getattr(args, 'init_checkpoint', None),
        eval_per_cell=getattr(args, 'eval_per_cell', 500),
        val_per_cell=getattr(args, 'val_per_cell', None),
        imd_max_items=getattr(args, 'imd_max_items', None),
        val_zoom=getattr(args, 'val_zoom', True),
        val_zoom_pad_frac=getattr(args, 'val_zoom_pad_frac', None),
        val_zoom_min_area=getattr(args, 'val_zoom_min_area', 0.0),
        tgif_val_models=(
            tuple(s.strip() for s in args.tgif_val_models.split(',') if s.strip())
            if getattr(args, 'tgif_val_models', None) else None
        ),
        tgif_types=(tuple(getattr(args, 'tgif_types')) if getattr(args, 'tgif_types', None) else None),
        primary_surface=getattr(args, 'primary_surface', 'imd'),
        tgif_eval_decoders=tuple(getattr(args, 'val_decoders', None) or ('kmeans', 'hdbscan')),
        # hardware (from resolved HardwareInfo)
        device=hw_device,
        use_amp=hw_use_amp,
        amp_dtype=hw_amp_dtype,
        world_size=hw_world,
        rank=hw_rank,
        gpu_name=hw_gpu_name,
        compute_cap=hw_cc,
    )


def to_dict(cfg: RunConfig) -> dict:
    """Serialize RunConfig to a JSON-compatible dict (for checkpoint cfg slot)."""
    return dataclasses.asdict(cfg)


def from_dict(d: dict) -> RunConfig:
    """Reconstruct RunConfig from a serialized dict (from checkpoint cfg slot).

    Unknown keys in d are silently ignored so old checkpoints survive field
    additions (C1 forward compatibility).
    """
    known = {f.name for f in dataclasses.fields(RunConfig)}
    filtered = {k: v for k, v in d.items() if k in known}
    # Convert None splice_mix to None (not an empty dict)
    if 'splice_mix' in filtered and filtered['splice_mix'] == {}:
        filtered['splice_mix'] = None
    return RunConfig(**filtered)
