"""experiments.scripts.train_tgif — ISOLATED TGIF continued-finetune harness.

This is an EXCEPTIONAL recipe, intentionally separate from the standard
``experiments/scripts/train.py`` (DESIGN_GUIDE I7 sanctioned exception): it
warm-starts an existing checkpoint and continues training on **TGIF data only**.
TGIF is normally a pure OOD probe (excluded from train.py's source map), so this
recipe must never leak into the standard runs that measure OOD numbers — hence a
distinct entry point, a ``recipe='tgif_finetune'`` tag on every checkpoint, and
its own run dir.

What it does differently from train.py:
  * Warm-start: ``--init_checkpoint`` builds the model with the checkpoint's
    architecture and loads its weights, then starts a FRESH optimizer/scheduler
    at a finetune LR.  (``--resume`` from the run dir still does full-state resume.)
  * Train data: the TGIF train split ONLY (the complement of the held-out cells).
  * Eval each epoch: IMD is the primary OOD metric (drives best.pt / early-stop),
    PLUS the TGIF held-out set broken into its (model|type|family) subcategory
    cells — each under 4 readouts (kmeans/hdbscan × flat/zoom).  See
    experiments/labs/tgif_finetune_eval.py.

All eval logic lives in lab_utils / labs; this script only parses, wires, and
runs the loop.

Usage (single GPU):
    python -m experiments.scripts.train_tgif \\
        --init_checkpoint /runs/base/best.pt \\
        --tgif2_root   /data/tgif2 \\
        --imd2020_root /data/imd2020 \\
        --run_dir      /runs/tgif_ft01 \\
        --num_epochs 5 --lr 2e-5 --eval_per_cell 500 --val_per_cell 100
"""

import argparse
import dataclasses
import json
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from experiments.configs.run_config import resolve_config, to_dict
from experiments.labs.tgif_finetune_eval import run_tgif_finetune_eval
from lab_utils.data.dataset import Dataset, lab_collate_fn
from lab_utils.data.datasets import tgif2
from lab_utils.data.datasets.registry import REGISTRY
from lab_utils.data.resolution import Resolution
from lab_utils.eval.load_model import load_eval_model
from lab_utils.logging.run_config import log_run_config
from lab_utils.logging.text import log_line
from lab_utils.train.checkpoint import find_latest_checkpoint, load, save
from lab_utils.train.distributed import DistributedContext, barrier, cleanup, wrap_model
from lab_utils.train.hardware import resolve_hardware
from lab_utils.train.loop import build_optimizer, build_scheduler, run_train_epoch


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='train_tgif',
        description='Continue-finetune a checkpoint on TGIF data only (isolated recipe).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group('warm start + data')
    g.add_argument('--init_checkpoint', default=None,
                   help='Checkpoint to warm-start from (model weights only). '
                        'Architecture is taken from this checkpoint.')
    g.add_argument('--tgif2_root',   required=True, help='TGIF dataset root (with tgif2_index.json)')
    g.add_argument('--tgif_types',   nargs='+', default=None, choices=['sp', 'fr'],
                   help="Restrict TGIF train+holdout to these manipulation types: "
                        "'sp' (local splice), 'fr' (full re-encode). Default: all.")
    g.add_argument('--imd2020_root', required=True, help='IMD2020 root (primary OOD metric)')

    g = p.add_argument_group('run management')
    g.add_argument('--run_dir', required=True, help='Checkpoint and log directory')
    g.add_argument('--resume',  default=None, help='Full-state resume of an in-progress finetune')
    g.add_argument('--seed',    type=int, default=42)
    g.add_argument('--log_every', type=int, default=20)

    g = p.add_argument_group('training loop (finetune defaults)')
    g.add_argument('--num_epochs',           type=int,   default=5)
    g.add_argument('--warmup_epochs',        type=float, default=0.5)
    g.add_argument('--early_stop_patience',  type=int,   default=3)
    g.add_argument('--early_stop_min_delta', type=float, default=0.002)
    g.add_argument('--batch_size',           type=int,   default=8)
    g.add_argument('--grad_accum',           type=int,   default=4)
    g.add_argument('--lr',                   type=float, default=2e-5)
    g.add_argument('--weight_decay',         type=float, default=1e-4)
    g.add_argument('--num_workers',          type=int,   default=0)
    g.add_argument('--persistent_workers',   action='store_true')
    g.add_argument('--prefetch_factor',      type=int,   default=None)

    g = p.add_argument_group('loss (defaults match the standard recipe)')
    g.add_argument('--lambda_image_bce',   type=float, default=1.0)
    g.add_argument('--lambda_contrastive', type=float, default=2.0)
    g.add_argument('--lambda_patch_bce',   type=float, default=1.0)
    g.add_argument('--patch_pos_weight',   type=float, default=10.0)

    g = p.add_argument_group('augmentation')
    g.add_argument('--train_crop_min',       type=float, default=0.18)
    g.add_argument('--train_crop_max',       type=float, default=1.00)
    g.add_argument('--train_crop_ratio_min', type=float, default=0.60)
    g.add_argument('--train_crop_ratio_max', type=float, default=1.70)
    g.add_argument('--noise_prob',           type=float, default=None)
    g.add_argument('--jpeg_prob',            type=float, default=None)
    g.add_argument('--whole_corrupt_prob',   type=float, default=0.0)
    g.add_argument('--oracle_crop',          action='store_true')

    g = p.add_argument_group('tgif holdout + eval')
    g.add_argument('--eval_per_cell', type=int, default=500,
                   help='Hold out this many splices per (model|type|family) cell for eval.')
    g.add_argument('--val_per_cell', type=int, default=None,
                   help='Per-epoch: score at most this many held-out splices per cell '
                        '(compute control; None = score all eval_per_cell).')
    g.add_argument('--imd_max_items', type=int, default=None,
                   help='Per-epoch: cap IMD items scored (compute control).')
    g.add_argument('--val_decoders', nargs='+', default=['kmeans', 'hdbscan'],
                   choices=['kmeans', 'hdbscan', 'threshold'],
                   help='Decoders run BOTH flat and attention-zoom each eval.')
    g.add_argument('--primary_surface', choices=['imd', 'tgif'], default='tgif',
                   help="Surface that drives best.pt / early-stop. 'tgif' (default for "
                        'this in-domain finetune) = TGIF held-out median splice F1; '
                        "'imd' = IMD OOD median splice F1. Both surfaces are always reported.")

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')

    return p


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Warm-start: model arch from checkpoint, weights loaded, fresh optimizer ──────

_ARCH_FIELDS = (
    'model_name', 'image_size', 'patch_size',
    'lora_rank', 'lora_alpha', 'lora_dropout',
    'contrastive_dim', 'pool_hidden', 'patch_bce',
)


def _align_cfg_to_checkpoint(cfg, ckpt_cfg, res: Resolution):
    """Override the finetune cfg's architecture fields to match the loaded model.

    The model is BUILT from the checkpoint, so the cfg slot we save (and log)
    must describe that same architecture — not the finetune parser defaults.
    Falls back to the actual Resolution for image/patch size when the warm-start
    checkpoint carried no cfg slot (legacy).
    """
    overrides = {'image_size': res.image_size, 'patch_size': res.patch_size}
    if ckpt_cfg is not None:
        for f in _ARCH_FIELDS:
            overrides[f] = getattr(ckpt_cfg, f, getattr(cfg, f))
    return dataclasses.replace(cfg, **overrides)


# ── Checkpoint save ──────────────────────────────────────────────────────────────

def _save_ckpt(model, optimizer, scaler, scheduler, *, epoch, cfg, best_metric, run_dir):
    state = {
        'epoch':       epoch,
        'model':       model.state_dict(),
        'optimizer':   optimizer.state_dict(),
        'scaler':      scaler.state_dict(),
        'scheduler':   scheduler.state_dict(),
        'best_metric': best_metric,
        'cfg':         to_dict(cfg),
        'meta':        {'recipe': cfg.recipe},
    }
    ckpt_path = Path(run_dir) / f'epoch_{epoch:04d}.pt'
    save(state, ckpt_path, is_main=True)
    return ckpt_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()
    args.recipe = 'tgif_finetune'   # stamp the recipe onto the resolved cfg

    hw  = resolve_hardware(device=args.device, want_amp=not args.no_amp, dist_backend='nccl')
    cfg = resolve_config(args, hw=hw)
    _seed_everything(cfg.seed + hw.rank)

    device = torch.device(hw.device)

    # ── Warm-start source: arch + weights from init_checkpoint (or a resume) ────
    resume_path = cfg.resume or find_latest_checkpoint(cfg.run_dir)
    arch_ckpt   = cfg.init_checkpoint or resume_path
    if arch_ckpt is None:
        raise RuntimeError(
            'train_tgif: nothing to warm-start from. Pass --init_checkpoint '
            '(or --resume / a checkpoint already in --run_dir).'
        )

    log_line(f'[ft] warm-start architecture+weights from: {arch_ckpt}')
    model, ckpt_cfg, res = load_eval_model(arch_ckpt, device=device, strict=False)
    cfg = _align_cfg_to_checkpoint(cfg, ckpt_cfg, res)

    if hw.is_main:
        log_run_config(cfg)
        run_dir = Path(cfg.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / 'run_config.json').write_text(json.dumps(to_dict(cfg), indent=2))

    model.train()
    # Rebuild the DistributedContext from HardwareInfo for the DDP wrap (no-op
    # when world_size == 1).  resolve_hardware sets up the process group but
    # returns HardwareInfo, so we reconstruct the ctx wrap_model expects.
    dist_ctx = DistributedContext(
        is_distributed=hw.world_size > 1,
        rank=hw.rank, world_size=hw.world_size,
        local_rank=hw.local_rank, is_main=hw.is_main,
    )
    model = wrap_model(model, dist_ctx, device=device)

    amp_dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16}
    amp_dtype = amp_dtype_map.get(cfg.amp_dtype) if cfg.amp_dtype else None
    # model_info / fetch take the spelled-out dtype name ('float16'|'bfloat16').
    eval_amp_dtype = {'fp16': 'float16', 'bf16': 'bfloat16'}.get(cfg.amp_dtype, 'float16')

    # ── Data: TGIF train (only), TGIF held-out, IMD val ────────────────────────
    tgif_types = set(cfg.tgif_types) if cfg.tgif_types else None
    if tgif_types is not None:
        log_line(f'[ft] TGIF restricted to manipulation types={sorted(tgif_types)}')
    tgif_train_ds, tgif_val_ds = tgif2.build(
        Path(cfg.tgif2_root), res=res, eval_per_cell=cfg.eval_per_cell,
        types=tgif_types,
    )
    if not tgif_train_ds.items:
        raise RuntimeError('train_tgif: TGIF train split is empty — check --tgif2_root / index.')

    light_aug = {}
    if cfg.noise_prob is not None:
        light_aug['noise_prob'] = cfg.noise_prob
    if cfg.jpeg_prob is not None:
        light_aug['jpeg_prob'] = cfg.jpeg_prob
    if cfg.whole_corrupt_prob > 0:
        light_aug['whole_corrupt_prob'] = cfg.whole_corrupt_prob

    train_ds = Dataset(
        tgif_train_ds.items, res, augment=True,
        crop_scale=(cfg.train_crop_min, cfg.train_crop_max),
        crop_ratio=(cfg.train_crop_ratio_min, cfg.train_crop_ratio_max),
        oracle_crop=cfg.oracle_crop,
        light_aug_kwargs=light_aug or None,
    )

    _, imd_val_ds = REGISTRY['imd2020'](Path(cfg.imd2020_root), res=res)
    imd_items     = imd_val_ds.items

    # hdbscan needs contrastive embeddings — drop it cleanly if the head is off.
    decoders = list(cfg.tgif_eval_decoders)
    if 'hdbscan' in decoders and cfg.contrastive_dim <= 0:
        log_line('[ft] WARN: contrastive head disabled → dropping hdbscan from eval decoders')
        decoders = [d for d in decoders if d != 'hdbscan']

    log_line(
        f'[ft] data: tgif_train={len(train_ds.items)} '
        f'tgif_holdout={len(tgif_val_ds.items)} imd_val={len(imd_items)}'
    )

    # ── Loaders ────────────────────────────────────────────────────────────────
    train_sampler = (
        torch.utils.data.DistributedSampler(
            train_ds, num_replicas=hw.world_size, rank=hw.rank, shuffle=True)
        if hw.world_size > 1 else None
    )
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
        train_ds, sampler=train_sampler,
        shuffle=(train_sampler is None), **loader_kw,
    )

    # ── Optimizer / scheduler / scaler (FRESH for finetune) ────────────────────
    optimizer = build_optimizer(model, cfg)
    scaler    = torch.cuda.amp.GradScaler(enabled=cfg.use_amp)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, cfg=cfg, steps_per_epoch=steps_per_epoch)

    # ── Full-state resume (only when continuing an in-progress finetune) ───────
    start_epoch   = 0
    best_metric   = 0.0
    patience_left = cfg.early_stop_patience
    if resume_path:
        log_line(f'[ft] full-state resume from {resume_path}')
        state = load(resume_path)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch = int(state.get('epoch', 0)) + 1
        best_metric = float(state.get('best_metric', 0.0))
        # If the early-stop surface changed since the checkpoint (e.g. resuming an
        # IMD-driven run to continue under the TGIF metric), the stored best_metric
        # is on the old scale and incomparable — reset the early-stop bookkeeping so
        # patience tracks the new metric from scratch.
        prev_surface = str(state.get('cfg', {}).get('primary_surface', 'imd'))
        if prev_surface != cfg.primary_surface:
            log_line(f'[ft] early-stop surface changed {prev_surface!r} → '
                     f'{cfg.primary_surface!r}; resetting best_metric/patience')
            best_metric   = 0.0
            patience_left = cfg.early_stop_patience

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        barrier()

        loss_stats = run_train_epoch(
            model, train_loader, optimizer, scaler, scheduler,
            epoch=epoch, cfg=cfg, device=device, amp_dtype=amp_dtype,
        )
        if hw.is_main:
            log_line(
                f'[train] epoch={epoch} loss={loss_stats["loss"]:.4f} '
                f'bce={loss_stats["loss_bce"]:.4f} cont={loss_stats["loss_cont"]:.4f} '
                f'patch={loss_stats["loss_patch"]:.4f}'
            )

        # ── Per-epoch eval: IMD primary + TGIF by subcategory (main rank) ──────
        val_metric = best_metric
        if hw.is_main:
            primary = run_tgif_finetune_eval(
                model, imd_items, tgif_val_ds.items, res,
                device=device, use_amp=cfg.use_amp, amp_dtype=eval_amp_dtype,
                decoders=tuple(decoders),
                val_per_cell=cfg.val_per_cell,
                imd_max_items=cfg.imd_max_items,
                primary_decoder='kmeans', primary_mode='zoom',
                primary_surface=cfg.primary_surface,
            )
            if primary == primary:   # not NaN
                val_metric = float(primary)
                log_line(f'[ft] epoch={epoch} primary_{cfg.primary_surface}_kmeans_zoom_f1={val_metric:.4f}')

        # ── Checkpoint + early stop on the configured primary metric ───────────
        if hw.is_main:
            ckpt_path = _save_ckpt(
                model, optimizer, scaler, scheduler,
                epoch=epoch, cfg=cfg, best_metric=max(val_metric, best_metric),
                run_dir=cfg.run_dir,
            )
            is_best = val_metric >= best_metric + cfg.early_stop_min_delta
            if is_best:
                best_metric   = val_metric
                patience_left = cfg.early_stop_patience
                shutil.copy(ckpt_path, Path(cfg.run_dir) / 'best.pt')
                log_line(f'[ft] best model saved  primary_f1={best_metric:.4f}')
            else:
                patience_left -= 1
                log_line(f'[ft] no improvement  patience_left={patience_left}  best={best_metric:.4f}')
                if patience_left <= 0:
                    log_line(f'[ft] early stop at epoch={epoch}  best_primary_f1={best_metric:.4f}')
                    break

    barrier()
    cleanup()
    log_line(f'[ft] done  best_primary_f1={best_metric:.4f}')


if __name__ == '__main__':
    main()
