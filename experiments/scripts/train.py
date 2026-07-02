"""experiments.scripts.train — DINO_SCOPE_final training entry point.

Usage (single GPU):
    python -m experiments.scripts.train \\
        --imd2020_root /data/imd2020 \\
        --casia_root   /data/casia \\
        --run_dir      /runs/exp01 \\
        --num_epochs 20

Usage (DDP, 2 GPUs):
    torchrun --nproc_per_node=2 -m experiments.scripts.train \\
        --imd2020_root /data/imd2020 ...

This script owns ONLY: argument parsing, hardware setup, model/data wiring,
checkpoint I/O, and the epoch loop.  No eval metric bodies, no decode logic,
no loss definitions — those live in lab_utils.{eval,train,model}.
"""

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from experiments.configs.run_config import from_dict, resolve_config, to_dict
from lab_utils.data.dataset import Dataset, lab_collate_fn
from lab_utils.data.datasets.registry import REGISTRY
from lab_utils.data.resolution import Resolution
from lab_utils.logging.run_config import log_run_config
from lab_utils.logging.text import log_line
from lab_utils.model.multi_head_detector import build_multi_head_detector
from lab_utils.train.checkpoint import find_latest_checkpoint, load, save
from lab_utils.train.distributed import DistributedContext, barrier, cleanup, unwrap_model, wrap_model
from lab_utils.train.hardware import resolve_hardware
from lab_utils.train.loop import (
    build_optimizer,
    build_scheduler,
    run_train_epoch,
    run_val_eval,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='train',
        description='Train a DINO_SCOPE_final detector.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # dataset roots
    g = p.add_argument_group('dataset roots')
    g.add_argument('--imd2020_root',      default=None)
    g.add_argument('--casia_root',        default=None)
    g.add_argument('--indoor_root',       default=None)
    g.add_argument('--coco_inpaint_root', default=None)
    g.add_argument('--sagid_root',        default=None)
    g.add_argument('--bfree_root',        default=None)
    g.add_argument('--anyedit_root',      default=None)
    g.add_argument('--tgif2_root',        default=None)
    g.add_argument('--pico_pseudo_root',  default=None,
                   help='PicoBanana pseudo-mask inpaint-triplet root '
                        '(experiments/scripts/export_pico_masks.py output)')

    # run management
    g = p.add_argument_group('run management')
    g.add_argument('--run_dir', default=None, help='Checkpoint and log directory')
    g.add_argument('--checkpoint_root', default=None,
                   help='Alias for --run_dir injected by the sweep orchestrator '
                        '(experiments.scripts.orchestrate); takes precedence over --run_dir')
    g.add_argument('--resume',  default=None,  help='Checkpoint path to resume from (full training state; same-codebase, strict)')
    g.add_argument('--init_weights', default=None,
                   help='Warm-start model weights only (strict=False, fresh optimizer/sched/scaler, '
                        'epoch 0). Use for legacy / cross-codebase checkpoints. Ignored if a resumable '
                        'checkpoint already exists in run_dir.')
    g.add_argument('--seed',    type=int, default=42)
    g.add_argument('--log_every', type=int, default=20)

    # training loop
    g = p.add_argument_group('training loop')
    g.add_argument('--num_epochs',           type=int,   default=10)
    g.add_argument('--warmup_epochs',        type=float, default=1.0)
    g.add_argument('--early_stop_patience',  type=int,   default=3)
    g.add_argument('--early_stop_min_delta', type=float, default=0.002)
    g.add_argument('--min_epochs',           type=int,   default=0,
                   help='Floor: never early-stop before this many epochs have run')
    g.add_argument('--max_train_epochs',     type=int,   default=None,
                   help='Hard cap on the training loop. The LR schedule horizon '
                        'stays --num_epochs, so a checkpoint at epoch N is identical '
                        'to a full --num_epochs run truncated at N (use to harvest a '
                        'fixed epoch on a fixed schedule without wasting later epochs)')
    g.add_argument('--early_stop_reduce',    choices=['median', 'mean'], default='median',
                   help='Reduction over per-splice localization F1 for the early-stop metric')
    g.add_argument('--batch_size',           type=int,   default=8)
    g.add_argument('--grad_accum',           type=int,   default=4)
    g.add_argument('--lr',                   type=float, default=2e-4)
    g.add_argument('--weight_decay',         type=float, default=1e-4)
    g.add_argument('--train_samples',        type=int,   default=2000)
    g.add_argument('--num_workers',          type=int,   default=0)
    g.add_argument('--persistent_workers',   action='store_true')
    g.add_argument('--prefetch_factor',      type=int,   default=None)

    # model
    g = p.add_argument_group('model')
    g.add_argument('--model_name',      default='facebook/dinov3-vith16plus-pretrain-lvd1689m')
    g.add_argument('--base_dtype',      choices=['fp32', 'bf16', 'fp16'], default='fp32',
                   help='Frozen backbone load dtype. fp32 = legacy default; bf16 halves backbone '
                        'VRAM so the undistilled ViT-7B fits a 24 GB L4 (trainable params stay fp32).')
    g.add_argument('--no_grad_checkpoint', action='store_true',
                   help='Disable backbone gradient checkpointing (~20-30%% faster backward, '
                        'much higher activation VRAM). Safe on a 48 GB Ada at batch_size 2; '
                        'do NOT combine with a large micro-batch — the val-zoom pass will OOM.')
    g.add_argument('--image_size',      type=int,   default=448)
    g.add_argument('--patch_size',      type=int,   default=16)
    g.add_argument('--lora_rank',       type=int,   default=32,
                   help='LoRA rank. 0 disables LoRA (fully frozen backbone, heads-only).')
    g.add_argument('--lora_alpha',      type=int,   default=64)
    g.add_argument('--lora_dropout',    type=float, default=0.1)
    g.add_argument('--lora_block_start', type=int,  default=None,
                   help='Adapt only transformer blocks with index >= this (None = from 0)')
    g.add_argument('--lora_block_end',   type=int,  default=None,
                   help='Adapt only blocks with index < this (None = through last); half-open [start,end)')
    g.add_argument('--contrastive_dim', type=int,   default=64)
    g.add_argument('--pool_hidden',     type=int,   default=256)
    g.add_argument('--patch_bce',       action='store_true')

    # loss
    g = p.add_argument_group('loss')
    g.add_argument('--lambda_image_bce',   type=float, default=1.0)
    g.add_argument('--lambda_contrastive', type=float, default=2.0)
    g.add_argument('--lambda_patch_bce',   type=float, default=1.0)
    g.add_argument('--patch_pos_weight',   type=float, default=10.0)

    # data / sampling
    g = p.add_argument_group('data / sampling')
    g.add_argument('--splice_mix',     nargs='*', default=None,
                   metavar='source=frac', help='e.g. imd2020=0.6 casia=0.4')
    g.add_argument('--casia_train',    action='store_true')
    g.add_argument('--imd_val_only',   action='store_true')
    g.add_argument('--imd_val_split',  type=float, default=None,
                   help='Override IMD2020 val_split fraction for the per-epoch val '
                        '(use 1.0 with --imd_val_only to validate on the full IMD set)')

    # augmentation
    g = p.add_argument_group('augmentation')
    g.add_argument('--train_crop_min',         type=float, default=0.18)
    g.add_argument('--train_crop_max',         type=float, default=1.00)
    g.add_argument('--train_crop_ratio_min',   type=float, default=0.60)
    g.add_argument('--train_crop_ratio_max',   type=float, default=1.70)
    g.add_argument('--use_splice_degradation', action='store_true')
    g.add_argument('--use_real_degradation',   type=lambda x: x.lower() == 'true',
                   default=None)
    g.add_argument('--paste_frac',             type=float, default=0.40,
                   help='Per-item paste-back probability for inpaint items == the '
                        '"sp" share; the rest keep the whole-image diffusion '
                        'fingerprint (fr). Default 0.40 → 40%% sp / 60%% fr.')
    g.add_argument('--noise_prob',             type=float, default=None)
    g.add_argument('--jpeg_prob',              type=float, default=None)
    g.add_argument('--whole_corrupt_prob',     type=float, default=0.0)
    g.add_argument('--oracle_crop',            action='store_true')

    # hardware
    g = p.add_argument_group('hardware')
    g.add_argument('--device',    default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp',    action='store_true')
    g.add_argument('--amp_dtype', choices=['fp16', 'bf16'], default=None,
                   help='Override autodetected AMP precision. Default: hardware-'
                        'detected (bf16 on compute-capability >=8 e.g. L4/A100, '
                        'fp16 otherwise). Pin fp16 to stay comparable across GPUs.')

    # eval
    g = p.add_argument_group('eval')
    g.add_argument('--val_decoder', default='auto',
                   choices=['auto', 'kmeans', 'threshold'])
    g.add_argument('--val_max_items', type=int, default=None,
                   help='Limit val items per epoch (for quick smoke tests)')
    g.add_argument('--val_zoom', action=argparse.BooleanOptionalAction, default=True,
                   help='Per-epoch val runs attention-zoom two-pass; early-stop '
                        'then tracks the zoomed localization F1 (default on; '
                        'use --no-val_zoom to score the flat single pass)')
    g.add_argument('--val_zoom_pad_frac', type=float, default=None,
                   help='Area-based zoom-crop padding for val: pad each side by this '
                        'fraction of the frame (resolution-invariant). None = legacy patch pad')
    g.add_argument('--val_zoom_min_area', type=float, default=0.0,
                   help='With --val_zoom_pad_frac: floor the padded crop to this '
                        'fraction of the frame area')
    g.add_argument('--val_per_cell', type=int, default=100,
                   help='TGIF per-epoch val: held-out splices per (model,type,family) cell')
    g.add_argument('--tgif_val_models', default=None,
                   help='Comma-separated generators to keep in TGIF per-epoch val '
                        "(e.g. 'flux1dev,flux1filldev'); reals always kept. Default: all")

    return p


# ── Seeding ────────────────────────────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Dataset wiring ─────────────────────────────────────────────────────────────

def _root(cfg, attr: str) -> Optional[Path]:
    val = getattr(cfg, attr, None)
    return Path(val) if val else None


def _build_datasets(cfg, res: Resolution):
    """Build and merge train + val datasets from all configured sources."""
    from lab_utils.data.datasets import imd2020 as _imd2020_mod
    from lab_utils.data.datasets import casia   as _casia_mod
    from lab_utils.data.item import Item

    train_items, val_items = [], []
    source_map = {
        'imd2020':      ('imd2020_root', {}),
        'casia':        ('casia_root',   {}),
        'coco_inpaint': ('coco_inpaint_root', {}),
        'sagid':        ('sagid_root',   {}),
        'bfree':        ('bfree_root',   {}),
        'anyedit':      ('anyedit_root', {}),
        'indoor':       ('indoor_root',  {}),
        'pico_pseudo':  ('pico_pseudo_root', {}),
    }

    for source, (root_attr, kwargs) in source_map.items():
        root = _root(cfg, root_attr)
        if root is None:
            continue  # source not requested
        if not root.exists():
            # A configured root that isn't on disk is almost always a missing
            # mount / un-downloaded dataset on a fresh VM. Skipping silently here
            # surfaces later as a cryptic empty-DataLoader 'num_samples=0', so
            # make it loud instead.
            log_line(f'[data] WARNING: --{root_attr} = {root} does not exist — skipping {source}')
            continue

        # Source-specific train/val override flags
        if source == 'casia' and not cfg.casia_train:
            continue
        if source == 'imd2020' and cfg.imd_val_only:
            # val only — still add to val_items
            imd_kwargs = dict(kwargs)
            if cfg.imd_val_split is not None:
                imd_kwargs['val_split'] = cfg.imd_val_split
            _, val_ds = REGISTRY[source](root, res=res, **imd_kwargs)
            val_items.extend(val_ds.items)
            continue

        train_ds, val_ds = REGISTRY[source](root, res=res, **kwargs)
        train_items.extend(train_ds.items)
        val_items.extend(val_ds.items)

    # TGIF2 → per-epoch VAL only.  Uses the leakage-free held-out partition
    # (eval_per_cell) and optionally filters to a subset of generator models, so
    # the per-epoch eval scores the same partitioned TGIF cells the standalone
    # eval reports.  TGIF is never added to train_items.
    tgif_root = _root(cfg, 'tgif2_root')
    if tgif_root is not None and tgif_root.exists():
        from lab_utils.data.datasets import tgif2 as _tgif2_mod
        per_cell = cfg.val_per_cell or 100
        _, tg_val = _tgif2_mod.build(
            str(tgif_root), res=res, eval_per_cell=per_cell, include_reals=True,
        )
        keep_models = set(cfg.tgif_val_models or ())
        tg_items = [
            it for it in tg_val.items
            if it.is_real or not keep_models or it.meta.get('tgif_model') in keep_models
        ]
        cells  = sorted({it.meta.get('tgif_subcat') for it in tg_items if not it.is_real})
        n_real = sum(1 for it in tg_items if it.is_real)
        val_items.extend(tg_items)
        log_line(
            f'[data] tgif2 → val: {len(tg_items)} items ({n_real} real) '
            f'models={sorted(keep_models) or "all"} per_cell={per_cell} cells={cells}'
        )

    if not train_items:
        raise RuntimeError(
            'train.py: training set is empty. Every configured train source was '
            'skipped (missing on disk) or routed to val only (e.g. --imd_val_only '
            'sends IMD to val). Check the [data] WARNING lines above and confirm '
            'at least one non-val-only root (--casia_root, --bfree_root, '
            '--coco_inpaint_root, --sagid_root, ...) exists on disk.'
        )

    train_light_aug = {}
    if cfg.noise_prob is not None:
        train_light_aug['noise_prob'] = cfg.noise_prob
    if cfg.jpeg_prob is not None:
        train_light_aug['jpeg_prob'] = cfg.jpeg_prob
    if cfg.whole_corrupt_prob > 0:
        train_light_aug['whole_corrupt_prob'] = cfg.whole_corrupt_prob

    train_ds = Dataset(
        train_items,
        res,
        augment=True,
        crop_scale=(cfg.train_crop_min, cfg.train_crop_max),
        crop_ratio=(cfg.train_crop_ratio_min, cfg.train_crop_ratio_max),
        oracle_crop=cfg.oracle_crop,
        paste_frac=cfg.paste_frac,
        light_aug_kwargs=train_light_aug if train_light_aug else None,
    )
    # val keeps paste_frac=1.0 (always paste → all-sp) so per-source val metrics
    # stay comparable across runs and aren't perturbed by the fr/sp mix knob.
    val_ds = Dataset(val_items, res, augment=False)
    return train_ds, val_ds


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _save_ckpt(model, optimizer, scaler, scheduler, *, epoch, cfg, best_metric, run_dir, is_main):
    if not is_main:
        return
    # LoRA: the backbone is frozen and identical to the pretrained weights the
    # model is rebuilt from at startup, so only the trainable params (LoRA
    # adapters + attention-pool head) need saving. This turns a ~13-27 GB/epoch
    # dump of the full frozen backbone (catastrophic for the 7B) into ~100s of MB.
    # All load paths are non-strict, so the rebuilt backbone fills the missing keys.
    trainable_sd = {
        n: p.detach().cpu()
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    state = {
        'epoch':       epoch,
        'model':       trainable_sd,
        'optimizer':   optimizer.state_dict(),
        'scaler':      scaler.state_dict(),
        'scheduler':   scheduler.state_dict(),
        'best_metric': best_metric,
        'cfg':         to_dict(cfg),
        'meta':        {},
    }
    ckpt_path = Path(run_dir) / f'epoch_{epoch:04d}.pt'
    save(state, ckpt_path, is_main=is_main)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser  = _build_parser()
    args    = parser.parse_args()

    hw  = resolve_hardware(
        device=args.device,
        want_amp=not args.no_amp,
        dist_backend='nccl',
    )
    cfg = resolve_config(args, hw=hw)
    if not cfg.run_dir:
        parser.error('one of --run_dir or --checkpoint_root is required')
    _seed_everything(cfg.seed + hw.rank)

    if hw.is_main:
        log_run_config(cfg)
        run_dir = Path(cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'run_config.json').write_text(json.dumps(to_dict(cfg), indent=2))

    # ── Resolution ────────────────────────────────────────────────────────────
    res = Resolution(image_size=cfg.image_size, patch_size=cfg.patch_size)
    device = torch.device(hw.device)

    # ── AMP dtype ─────────────────────────────────────────────────────────────
    amp_dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16}
    amp_dtype = amp_dtype_map.get(cfg.amp_dtype) if cfg.amp_dtype else None

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_multi_head_detector(
        model_name=cfg.model_name,
        base_dtype=cfg.base_dtype,
        resolution=res,
        lora_rank=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        lora_block_start=cfg.lora_block_start,
        lora_block_end=cfg.lora_block_end,
        contrastive_dim=cfg.contrastive_dim,
        pool_hidden=cfg.pool_hidden,
        patch_bce=cfg.patch_bce,
        grad_checkpointing=not cfg.no_grad_checkpoint,
        device=device,
    )
    # Rebuild the DistributedContext from HardwareInfo for the DDP wrap (no-op
    # when world_size == 1). resolve_hardware returns HardwareInfo, so we
    # reconstruct the ctx wrap_model expects.
    dist_ctx = DistributedContext(
        is_distributed=hw.world_size > 1,
        rank=hw.rank, world_size=hw.world_size,
        local_rank=hw.local_rank, is_main=hw.is_main,
    )
    model = wrap_model(model, dist_ctx, device=device)

    # ── Optimizer / scheduler / scaler ────────────────────────────────────────
    optimizer = build_optimizer(model, cfg)
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds, val_ds = _build_datasets(cfg, res)

    train_sampler = None
    cap = cfg.train_samples
    if hw.world_size > 1:
        train_sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=hw.world_size, rank=hw.rank, shuffle=True,
        )
        if 0 < cap < len(train_ds) and hw.is_main:
            log_line(f'[data] NOTE: --train_samples={cap} ignored under DDP '
                     f'(DistributedSampler uses full dataset)')
    elif 0 < cap < len(train_ds):
        train_sampler = torch.utils.data.RandomSampler(
            train_ds, replacement=False, num_samples=cap,
        )
        log_line(f'[data] train epoch capped at {cap}/{len(train_ds)} samples')
    loader_kw = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        pin_memory=(device.type == 'cuda'),
        collate_fn=lab_collate_fn,
    )
    if cfg.prefetch_factor is not None and cfg.num_workers > 0:
        loader_kw['prefetch_factor'] = cfg.prefetch_factor

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        **loader_kw,
    )

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum)
    scheduler = build_scheduler(optimizer, cfg=cfg, steps_per_epoch=steps_per_epoch)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch   = 0
    best_metric   = 0.0
    patience_left = cfg.early_stop_patience

    resume_path = cfg.resume or find_latest_checkpoint(cfg.run_dir)
    if resume_path:
        log_line(f'[ckpt] resuming from {resume_path}')
        state = load(resume_path)
        # strict=False: checkpoints now store only trainable params (LoRA + head);
        # the frozen backbone is rebuilt from the base model, so its keys are
        # legitimately "missing" here. Still loads full legacy checkpoints fine.
        incompat = model.load_state_dict(state['model'], strict=False)
        if incompat.unexpected_keys:
            log_line(f'[ckpt] resume: {len(incompat.unexpected_keys)} unexpected keys dropped '
                     f'(e.g. {incompat.unexpected_keys[:6]})')
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = int(state.get('epoch', 0)) + 1
        best_metric = float(state.get('best_metric', 0.0))
    elif cfg.init_weights:
        # Weights-only warm start from a legacy / cross-codebase checkpoint.
        # Unlike --resume this tolerates a missing optimizer/scheduler/scaler/cfg
        # and renamed or sparse state_dict keys (strict=False) — the eval loader
        # already reads these checkpoints non-strict. Optimizer, scheduler and
        # scaler stay freshly initialised and we start at epoch 0; the legacy
        # optimizer state is not portable across the refactor anyway.
        log_line(f'[ckpt] init weights (no optim/sched/scaler) from {cfg.init_weights}')
        state = load(cfg.init_weights)
        incompat = unwrap_model(model).load_state_dict(state['model'], strict=False)
        if incompat.missing_keys:
            log_line(f'[ckpt] init_weights: {len(incompat.missing_keys)} MISSING keys left at init '
                     f'(e.g. {incompat.missing_keys[:6]})')
        if incompat.unexpected_keys:
            log_line(f'[ckpt] init_weights: {len(incompat.unexpected_keys)} UNEXPECTED keys dropped '
                     f'(e.g. {incompat.unexpected_keys[:6]})')
        if not incompat.missing_keys and not incompat.unexpected_keys:
            log_line('[ckpt] init_weights: exact key match (would have loaded strict)')

    # ── Training loop ─────────────────────────────────────────────────────────
    # LR schedule horizon is num_epochs (build_scheduler); max_train_epochs only
    # truncates the loop, so a checkpoint here matches a full-schedule run cut short.
    stop_epoch = cfg.num_epochs if cfg.max_train_epochs is None \
        else min(cfg.num_epochs, cfg.max_train_epochs)
    for epoch in range(start_epoch, stop_epoch):
        if train_sampler is not None and hasattr(train_sampler, 'set_epoch'):
            train_sampler.set_epoch(epoch)

        barrier()

        loss_stats = run_train_epoch(
            model, train_loader, optimizer, scaler, scheduler,
            epoch=epoch, cfg=cfg, device=device, amp_dtype=amp_dtype,
        )

        if hw.is_main:
            log_line(
                f'[train] epoch={epoch} '
                f'loss={loss_stats["loss"]:.4f} '
                f'bce={loss_stats["loss_bce"]:.4f} '
                f'cont={loss_stats["loss_cont"]:.4f} '
                f'patch={loss_stats["loss_patch"]:.4f}'
            )

        # ── Per-epoch val eval (main rank only) ───────────────────────────────
        val_metric = best_metric
        mil_only = (cfg.contrastive_dim <= 0 and not cfg.patch_bce)
        metric_label = 'val_image_auc' if mil_only else f'val_f1_{cfg.early_stop_reduce}'
        if hw.is_main and val_ds.items:
            records, image_auc = run_val_eval(
                model,
                val_ds.items,
                res,
                device=device,
                cfg=cfg,
                log_tag='[eval]',
                max_items=args.val_max_items,
                decoder=args.val_decoder,
            )
            if mil_only:
                # MIL-only: early-stop on image-level AUC
                if image_auc is not None and not math.isnan(image_auc):
                    val_metric = image_auc
                    log_line(f'[eval] epoch={epoch} {metric_label}={val_metric:.4f}')
            else:
                # Standard: early-stop on the chosen reduction of per-splice loc F1
                splice_records = [r for r in records if not r.is_real]
                if splice_records:
                    reduce_fn = np.mean if cfg.early_stop_reduce == 'mean' else np.median
                    val_metric = float(reduce_fn([r.f1 for r in splice_records]))
                    log_line(f'[eval] epoch={epoch} {metric_label}={val_metric:.4f}')

        # ── Checkpoint + early stop ───────────────────────────────────────────
        is_best = val_metric >= best_metric + cfg.early_stop_min_delta

        if hw.is_main:
            _save_ckpt(
                model, optimizer, scaler, scheduler,
                epoch=epoch, cfg=cfg, best_metric=max(val_metric, best_metric),
                run_dir=cfg.run_dir, is_main=True,
            )
            if is_best:
                best_metric   = val_metric
                patience_left = cfg.early_stop_patience
                import shutil
                shutil.copy(
                    Path(cfg.run_dir) / f'epoch_{epoch:04d}.pt',
                    Path(cfg.run_dir) / 'best.pt',
                )
                log_line(f'[ckpt] best model saved  {metric_label}={best_metric:.4f}')
            elif (epoch + 1) < cfg.min_epochs:
                # Below the min-epoch floor: hold patience, never early-stop yet.
                patience_left = cfg.early_stop_patience
                log_line(
                    f'[ckpt] no improvement (epoch {epoch} < min_epochs={cfg.min_epochs}); '
                    f'patience held at {patience_left}  best={best_metric:.4f}'
                )
            else:
                patience_left -= 1
                log_line(
                    f'[ckpt] no improvement  patience_left={patience_left}'
                    f'  best={best_metric:.4f}'
                )
                if patience_left <= 0:
                    log_line(
                        f'[train] early stop at epoch={epoch}'
                        f'  best_{metric_label}={best_metric:.4f}'
                    )
                    break

    barrier()
    cleanup()
    log_line(f'[train] done  best_{metric_label}={best_metric:.4f}')


if __name__ == '__main__':
    main()
