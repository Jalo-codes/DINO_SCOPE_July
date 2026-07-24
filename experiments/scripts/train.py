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
from lab_utils.data.augment.degradation import resolve_severity
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
    run_epoch_viz,
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
    g.add_argument('--full_fakes_root',     default=None,
                   help='Whole-image fakes TRAIN root, full_fakes layout '
                        '(root/real/ + root/<generator>/). Indexed entirely as '
                        'train (val_split=0.0); pair with --full_fakes_val_root')
    g.add_argument('--full_fakes_val_root', default=None,
                   help='Whole-image fakes VAL root — a SEPARATE download from '
                        '--full_fakes_root. OpenFake validation = held-out images '
                        'from the TRAINING generators; test = held-out GENERATORS. '
                        'Those semantics cannot be reproduced by splitting one root, '
                        'hence two flags. Indexed val-only (val_split=1.0)')
    g.add_argument('--full_fakes_val_per_pool', type=int, default=None,
                   help='Cap the full_fakes VAL set to N items per generator pool. '
                        'Pools range 200 images down to 3, so an uncapped val is slow '
                        'and dominated by the large generators. Deterministic per seed')
    g.add_argument('--full_fakes_val_reals', type=int, default=None,
                   help='Cap the full_fakes VAL real pool to N items (paired with '
                        '--full_fakes_val_per_pool)')

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
    g.add_argument('--lora_rank',       type=int,   default=16,
                   help='LoRA rank. 0 disables LoRA (fully frozen backbone, heads-only). '
                        'Default 16 = the ablation-winning "optimal" rank (optimal_h16plus_688_r16).')
    g.add_argument('--lora_alpha',      type=int,   default=32,
                   help='2x lora_rank by convention (matches the ablation sweep).')
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
    g.add_argument('--patch_balance', choices=['global', 'per_image'], default='global',
                   help="'global' (default) reproduces current behavior exactly: a flat "
                        "mean-reduced patch BCE at --patch_pos_weight. 'per_image' switches "
                        "to equal-budget patch BCE (lab_utils.model.losses.bce."
                        "equal_budget_patch_bce_loss): every image gets a fixed positive and "
                        "negative loss budget split evenly over its own fake/not-fake "
                        "patches, fixing the small-splice punishment gap (a k-fake-patch "
                        "splice is punished k/N as hard as a whole N-patch fake for being "
                        "totally missed, under 'global'). --patch_pos_weight is IGNORED "
                        "under 'per_image'. Outputs are no longer calibrated at t=0.5 under "
                        "'per_image' (CLAUDE.md rule 1) — compare checkpoints via patch AUROC "
                        "(eval_numbers --patch_auroc), never at a fixed decode threshold.")
    g.add_argument('--patch_k_min', type=float, default=4.0,
                   help="Only used when --patch_balance per_image. Floor on the banded "
                        "fake/not-fake patch count before its budget is treated as full: "
                        "an image with fewer than this many supervised fake (or "
                        "not-fake) patches gets a linearly-shrunk budget (count/k_min) "
                        "instead of the 1/count blowup — insurance against one noisy "
                        "patch label owning a whole image's gradient.")
    g.add_argument('--patch_band', type=float, nargs=2, default=None, metavar=('LOW', 'HIGH'),
                   help="Ignore-band thresholds on per-patch mask coverage, matching "
                        "lab_utils.data.resolution.mask_to_patch_labels_soft: density==0 -> "
                        "confident not-fake; 0<density<LOW -> ignored (boundary noise); "
                        "LOW<=density<HIGH -> linear ramp weight; density>=HIGH -> confident "
                        "fake. Default None reproduces today's hard binarize at 0.5 exactly "
                        "(no ignore band). Only affects the patch-BCE head's labels/weights — "
                        "the contrastive head always sees the hard-binarized labels. "
                        "Suggested for --patch_balance per_image: 0.2 0.8.")

    # data / sampling
    g = p.add_argument_group('data / sampling')
    g.add_argument('--splice_mix',     nargs='*', default=None,
                   metavar='source=frac', help='e.g. imd2020=0.6 casia=0.4')
    g.add_argument('--balance_real_fake', action='store_true',
                   help='Sample each epoch 50/50 real/fake. --splice_mix cannot do '
                        'this: it weights by SOURCE, and a source holds both classes '
                        '(all full_fakes items share source=full_fakes), so the class '
                        'ratio falls out of pool sizes. Composes with --splice_mix; '
                        'ignored under DDP')
    g.add_argument('--casia_train',    action='store_true')
    g.add_argument('--imd_val_only',   action='store_true')
    g.add_argument('--imd_val_split',  type=float, default=None,
                   help='Override IMD2020 val_split fraction for the per-epoch val '
                        '(use 1.0 with --imd_val_only to validate on the full IMD set)')
    g.add_argument('--pico_pseudo_val_only', action='store_true',
                   help='Route --pico_pseudo_root to the per-epoch VAL only — index pico '
                        'but never add it to TRAIN. Lets a no-pico arm still monitor pico '
                        'localization each epoch (the B arm of the pico-label-quality '
                        'probe). Uses pico\'s default 90/10 split, so the val 10% matches '
                        'the held-out 10% an arm that DID train on pico evaluates on. '
                        'Do NOT combine with a pico weight in --splice_mix.')

    # augmentation — THREE knobs (I7): the severity preset, the sp/fr paste
    # share, and the oracle-crop regime switch. Crop geometry, jpeg/noise
    # probabilities etc. are RunConfig/Dataset defaults, not CLI surface —
    # change them in code (with a commit) rather than per-run.
    g = p.add_argument_group('augmentation')
    g.add_argument('--aug_severity',           choices=['light', 'medium', 'heavy', 'extreme'],
                   default='light',
                   help='Preset bundling BOTH how often the appearance stage '
                        'is replaced by heavy multi-region corruption (jpeg/'
                        'gaussian/resize/poisson) AND how strong that '
                        'corruption is, applied identically to real and '
                        'splice/fake items. light=off (today\'s mild jpeg/'
                        'noise/resize jitter only, prob=0.0); medium/heavy/'
                        'extreme progressively raise both fire-probability '
                        '(0.35/0.65/0.90) and corruption strength. See '
                        'lab_utils/data/augment/degradation.py SEVERITY_TIERS.')
    g.add_argument('--paste_frac',             type=float, default=0.40,
                   help='Per-item paste-back probability for inpaint items == the '
                        '"sp" share; the rest keep the whole-image diffusion '
                        'fingerprint (fr). Default 0.40 → 40%% sp / 60%% fr.')
    g.add_argument('--fr_bg_negative_prob',    type=float, default=0.0,
                   help='For UN-pasted inpaint items (fr-style), probability of '
                        'training on a window fully outside the edit mask served '
                        'as a clean negative (mask=zeros, label real). Teaches '
                        'regen-texture-without-edit = clean, isolating the '
                        'decoder fingerprint away from the semantic heads. '
                        '0.0 (default) = off.')
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
                   choices=['auto', 'kmeans', 'kmeans_logit', 'threshold', 'none'],
                   help="'kmeans_logit' = adaptive per-image split on the patch-BCE "
                        "logits (Otsu); the representative per-epoch localization metric "
                        "for a BCE head — 'threshold' (fixed t=0.5) collapses on small "
                        "masks and mis-selects best.pt. 'none' = image-level only, no "
                        'mask decode. Correct for an image-head-only run.')
    g.add_argument('--val_max_items', type=int, default=None,
                   help='Limit val items per epoch (for quick smoke tests). Truncates the '
                        'FLAT source-ordered list — takes whole sources in order and drops '
                        'the rest; use --val_per_source for a balanced multi-source cap.')
    g.add_argument('--val_patch_auroc', action=argparse.BooleanOptionalAction, default=False,
                   help='Also report threshold-free per-source LOCALIZATION AUROC each '
                        'epoch (patch sigmoid vs banded GT). The honest localization signal '
                        'under --patch_balance per_image, where the val F1 above is '
                        'calibration-shifted (t=0.5 no longer optimal). Costs a second flat '
                        'forward per val item. full_fakes/pseudo-sentinel items self-skip.')
    g.add_argument('--val_per_source', type=int, default=None,
                   help='Cap the per-epoch val to N items PER SOURCE (deterministic '
                        'subsample, seeded from --seed). Unlike --val_max_items this keeps '
                        'the eval balanced across every source, so a condensed per-epoch '
                        'eval still reports every held-out set (in-domain + pico + imd) '
                        'rather than whichever sources sort first. tgif2 and full_fakes are '
                        'EXEMPT (they self-cap via --val_per_cell / --full_fakes_val_per_pool, '
                        'which keep their cell/pool balance a flat draw would destroy).')
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
    g.add_argument('--tgif_val_reals', type=int, default=None,
                   help='Cap the TGIF per-epoch val reals to N (deterministic subsample, '
                        'seeded from --seed). TGIF keeps ONE real per val coco_id and '
                        '--tgif_val_models never filters reals, so without this the real '
                        'pool is the whole val split — far more than a condensed eval or a '
                        'balanced image-AUROC needs. --val_per_cell bounds the fakes.')
    g.add_argument('--tgif_types', nargs='*', default=None, choices=['sp', 'fr'],
                   help="Restrict TGIF per-epoch val to these manipulation types "
                        "(e.g. --tgif_types sp fr keeps both; omit for all). Combine "
                        "with --tgif_val_models restricted to ONE generator to get "
                        "exactly its 4 held-out cells (sp/fr x semantic/random).")
    g.add_argument('--viz_every_epoch', action=argparse.BooleanOptionalAction, default=False,
                   help='Save (+ inline-display in a notebook) input/prediction/attention/GT '
                        'figures for a fixed splice-item sample every epoch (lab_utils.train.loop.'
                        'run_epoch_viz). Off by default — meant for small exploratory runs, not '
                        'full sweeps, where per-epoch figure I/O would just be noise.')
    g.add_argument('--viz_n', type=int, default=15,
                   help='Number of (fixed, seeded) splice items to visualize per epoch. '
                        'With --viz_per_source set, this is instead the cap for the pooled '
                        '"everything else" remainder (sources not explicitly listed).')
    g.add_argument('--viz_per_source', nargs='*', default=None,
                   metavar='source=n', help='Stratified per-epoch viz counts, e.g. '
                        '--viz_per_source pico_pseudo=35 sagid=10. Sources not listed '
                        'here are pooled and sampled up to --viz_n.')

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


def _log_data_diet(train_items, val_items, cfg) -> None:
    """One aligned [data] table: what this run actually trains and scores on.

    Per source: train/val counts split real|fake, the fakes' mask kind
    (gt / pseudo / sentinel), how many train fakes are paste-eligible
    (meta['real_path'] present -> Dataset.paste_background can fire at
    paste_frac), and the requested splice_mix weight. This is the one place
    to catch a wrong diet before burning an epoch on it.
    """
    sources = sorted({it.source for it in train_items} | {it.source for it in val_items})
    mix = cfg.splice_mix or {}

    def _counts(items):
        n_real = sum(1 for it in items if it.is_real)
        return n_real, len(items) - n_real

    def _mask_kind(fakes):
        if not fakes:
            return '-'
        if all(it.meta.get('pseudo_mask') for it in fakes):
            return 'pseudo'
        if all(it.meta.get('gt_mask_reliable') is False for it in fakes):
            return 'sentinel'
        return 'gt'

    header = (f'{"source":<14} {"train r|f":>12} {"val r|f":>12} '
              f'{"mask":>8} {"paste-elig":>10} {"mix":>6}')
    log_line(f'[data] ── data diet ── (paste_frac={cfg.paste_frac}, '
             f'aug_severity={cfg.aug_severity}, oracle_crop={cfg.oracle_crop})')
    log_line(f'[data]   {header}')
    for src in sources:
        tr = [it for it in train_items if it.source == src]
        va = [it for it in val_items if it.source == src]
        tr_fakes = [it for it in tr if not it.is_real]
        tr_r, tr_f = _counts(tr)
        va_r, va_f = _counts(va)
        n_paste = sum(1 for it in tr_fakes if it.meta.get('real_path') is not None)
        kind = _mask_kind(tr_fakes or [it for it in va if not it.is_real])
        mix_s = f'{mix[src]:.2f}' if src in mix else '-'
        log_line(f'[data]   {src:<14} {f"{tr_r}|{tr_f}":>12} {f"{va_r}|{va_f}":>12} '
                 f'{kind:>8} {f"{n_paste}/{tr_f}":>10} {mix_s:>6}')


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
        # Whole-image fakes, TRAIN side only — val comes from a separate root
        # (--full_fakes_val_root, handled below), so this root is indexed
        # entirely as train.
        'full_fakes':   ('full_fakes_root', {'val_split': 0.0}),
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
        if source == 'pico_pseudo' and cfg.pico_pseudo_val_only:
            # val only — monitor pico each epoch WITHOUT training on it (the
            # no-pico arm of the pico-label-quality probe). Default 90/10 split,
            # so this val 10% == the held-out 10% a pico-trained arm evaluates on.
            _, val_ds = REGISTRY[source](root, res=res, **kwargs)
            val_items.extend(val_ds.items)
            continue

        train_ds, val_ds = REGISTRY[source](root, res=res, **kwargs)
        train_items.extend(train_ds.items)
        val_items.extend(val_ds.items)

    # full_fakes VAL root → per-epoch VAL only.  A SEPARATE download from the
    # train root: OpenFake's validation split is held-out IMAGES from the
    # training generators, its test split is held-out GENERATORS, and neither
    # is reproducible by randomly splitting the train root — so the split
    # boundary is expressed as two roots rather than a ratio.
    ff_val_root = _root(cfg, 'full_fakes_val_root')
    if ff_val_root is not None:
        if not ff_val_root.exists():
            log_line(f'[data] WARNING: --full_fakes_val_root = {ff_val_root} '
                     f'does not exist — skipping full_fakes val')
        else:
            _, ff_val = REGISTRY['full_fakes'](
                ff_val_root, res=res, val_split=1.0,
                val_per_pool=cfg.full_fakes_val_per_pool,
                val_real_cap=cfg.full_fakes_val_reals,
                split_seed=cfg.seed,
            )
            val_items.extend(ff_val.items)

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
            types=set(cfg.tgif_types) if cfg.tgif_types else None,
            build_train_side=False,
        )
        keep_models = set(cfg.tgif_val_models or ())
        tg_items = [
            it for it in tg_val.items
            if it.is_real or not keep_models or it.meta.get('tgif_model') in keep_models
        ]
        if cfg.tgif_val_reals is not None:
            reals = [it for it in tg_items if it.is_real]
            fakes = [it for it in tg_items if not it.is_real]
            if len(reals) > cfg.tgif_val_reals:
                reals.sort(key=lambda i: i.item_id)          # stable order before the draw
                random.Random(cfg.seed + 11).shuffle(reals)  # +11: independent of other draws
                reals = reals[:cfg.tgif_val_reals]
            tg_items = fakes + reals
        cells  = sorted({it.meta.get('tgif_subcat') for it in tg_items if not it.is_real})
        n_real = sum(1 for it in tg_items if it.is_real)
        val_items.extend(tg_items)
        log_line(
            f'[data] tgif2 → val: {len(tg_items)} items ({n_real} real) '
            f'models={sorted(keep_models) or "all"} per_cell={per_cell} '
            f'reals_cap={cfg.tgif_val_reals} cells={cells}'
        )

    if not train_items:
        raise RuntimeError(
            'train.py: training set is empty. Every configured train source was '
            'skipped (missing on disk) or routed to val only (e.g. --imd_val_only '
            'sends IMD to val). Check the [data] WARNING lines above and confirm '
            'at least one non-val-only root (--casia_root, --bfree_root, '
            '--coco_inpaint_root, --sagid_root, ...) exists on disk.'
        )

    if cfg.val_per_source is not None and val_items:
        from collections import defaultdict as _defaultdict
        # tgif2 (per-CELL cap, val_per_cell) and full_fakes (per-POOL cap,
        # val_per_pool) already carry bespoke BALANCED per-epoch caps; a flat
        # per-source draw would flatten their cell/pool structure (e.g. lose
        # tgif's sp/fr and per-generator balance). Exempt them — they pass
        # through whole.
        _SELF_CAPPED = {'tgif2', 'full_fakes'}
        by_src = _defaultdict(list)
        for it in val_items:
            by_src[it.source].append(it)
        cap_rng = random.Random(cfg.seed + 7)   # +7: independent of any other draw
        capped, n_dropped = [], 0
        for src in sorted(by_src):
            group = list(by_src[src])
            if src in _SELF_CAPPED:
                capped.extend(group)
                continue
            if len(group) > cfg.val_per_source:
                group.sort(key=lambda i: i.item_id)   # stable order before the draw
                cap_rng.shuffle(group)
                n_dropped += len(group) - cfg.val_per_source
                group = group[:cfg.val_per_source]
            capped.extend(group)
        log_line(f'[data] val_per_source={cfg.val_per_source} → {len(capped)} val items '
                 f'across {len(by_src)} sources ({n_dropped} dropped; '
                 f'tgif2/full_fakes exempt — self-capped)')
        val_items = capped

    _log_data_diet(train_items, val_items, cfg)

    train_ds = Dataset(
        train_items,
        res,
        augment=True,
        crop_scale=(cfg.train_crop_min, cfg.train_crop_max),
        crop_ratio=(cfg.train_crop_ratio_min, cfg.train_crop_ratio_max),
        oracle_crop=cfg.oracle_crop,
        paste_frac=cfg.paste_frac,
        fr_bg_negative_prob=cfg.fr_bg_negative_prob,
        aug_severity=cfg.aug_severity,
    )
    if cfg.aug_severity != 'light':
        _sev_prob, _ = resolve_severity(cfg.aug_severity)
        log_line(
            f'[data] aug_severity={cfg.aug_severity} prob={_sev_prob:.2f} '
            f'(replaces light appearance stage per-item when it fires, '
            f'same treatment for real and splice/fake items)'
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
    if cfg.patch_band is not None:
        lo, hi = cfg.patch_band
        if not (0.0 < lo < hi <= 1.0):
            parser.error(f'--patch_band needs 0 < LOW < HIGH <= 1, got {lo} {hi}')
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
        if cfg.splice_mix and hw.is_main:
            log_line(f'[data] NOTE: --splice_mix={cfg.splice_mix} ignored under DDP '
                     f'(DistributedSampler uses full dataset, unweighted)')
        if 0 < cap < len(train_ds) and hw.is_main:
            log_line(f'[data] NOTE: --train_samples={cap} ignored under DDP '
                     f'(DistributedSampler uses full dataset)')
        if cfg.balance_real_fake and hw.is_main:
            log_line('[data] NOTE: --balance_real_fake ignored under DDP '
                     '(DistributedSampler uses full dataset, unweighted)')
    elif cfg.balance_real_fake:
        # 50/50 real/fake per epoch. --splice_mix cannot express this: it
        # weights by SOURCE, and a source is real+fake together (every
        # full_fakes item, real or fake, has source='full_fakes'), so the
        # class ratio just falls out of the pool — 10000 reals vs 6409 fakes
        # is 61/39, not 50/50.
        #
        # Composes with --splice_mix rather than replacing it:
        #   weight(item) = source_frac[src] * 0.5 / count[(src, is_real)]
        # so each source keeps its requested share AND each source's share is
        # split evenly between its reals and its fakes. Groups that don't
        # exist (a fakes-only source has no real group) contribute nothing and
        # the remaining mass renormalises — WeightedRandomSampler normalises
        # internally, so absent groups cannot silently steal probability.
        from collections import Counter
        grp_counts = Counter((it.source, it.is_real) for it in train_ds.items)
        src_counts = Counter(it.source for it in train_ds.items)
        if cfg.splice_mix:
            unknown = sorted(set(cfg.splice_mix) - set(src_counts))
            if unknown:
                log_line(f'[data] WARNING: --splice_mix sources not present in '
                         f'train set: {unknown}')
            src_frac = {s: cfg.splice_mix.get(s, 0.0) for s in src_counts}
        else:
            src_frac = {s: 1.0 / len(src_counts) for s in src_counts}

        weights = [
            src_frac.get(it.source, 0.0) * 0.5 / grp_counts[(it.source, it.is_real)]
            for it in train_ds.items
        ]
        if sum(weights) <= 0:
            raise RuntimeError(
                'train.py: --balance_real_fake resolved to all-zero sampling '
                f'weights; loaded sources {sorted(src_counts)}, '
                f'splice_mix={cfg.splice_mix or "(none)"}'
            )
        n_samples = cap if cap > 0 else len(train_ds)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=n_samples, replacement=True,
        )
        n_real = sum(1 for it in train_ds.items if it.is_real)
        grp_desc = ', '.join(
            '{}/{}={}'.format(s, 'real' if r else 'fake', n)
            for (s, r), n in sorted(grp_counts.items())
        )
        log_line(
            f'[data] balance_real_fake: 50/50 per epoch, num_samples={n_samples} '
            f'(pool is {n_real} real / {len(train_ds) - n_real} fake) '
            f'groups={{{grp_desc}}}'
        )
    elif cfg.splice_mix:
        # Per-source weighted sampling so the mix ratio (e.g. sagid=0.33
        # pico_pseudo=0.33 casia=0.34) is exact regardless of each source's
        # pool size, instead of falling out of the concatenated pool's natural
        # proportions. weight(item) = target_frac[source] / pool_size[source],
        # so every item in a source shares that source's total probability mass
        # evenly. replacement=True since a requested share can exceed a small
        # source's pool size (e.g. 1000/epoch from a 600-item pico_pseudo set).
        from collections import Counter
        src_counts = Counter(it.source for it in train_ds.items)
        unknown = sorted(set(cfg.splice_mix) - set(src_counts))
        if unknown:
            log_line(f'[data] WARNING: --splice_mix sources not present in '
                      f'train set (no items loaded for them): {unknown}')
        weights = [
            cfg.splice_mix.get(it.source, 0.0) / src_counts[it.source]
            for it in train_ds.items
        ]
        if sum(weights) <= 0:
            raise RuntimeError(
                'train.py: --splice_mix resolved to all-zero sampling weights — '
                f'requested sources {sorted(cfg.splice_mix)} vs. loaded sources '
                f'{sorted(src_counts)}; check source names match dataset roots.'
            )
        n_samples = cap if cap > 0 else len(train_ds)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=n_samples, replacement=True,
        )
        log_line(f'[data] splice_mix={cfg.splice_mix} weighted sampler '
                  f'num_samples={n_samples} pool_sizes={dict(src_counts)}')
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
            if 'patch_realized_P' in loss_stats:
                log_line(
                    f'[train] epoch={epoch} patch_balance=per_image diag: '
                    f'P={loss_stats["patch_realized_P"]:.3f} '
                    f'Q={loss_stats["patch_realized_Q"]:.3f} '
                    f'max_patch_w={loss_stats["patch_max_patch_w"]:.3f} '
                    f'n_no_supervision={loss_stats["patch_n_no_supervision"]}'
                )

        # ── Per-epoch val eval (main rank only) ───────────────────────────────
        val_metric = best_metric
        mil_only = (cfg.contrastive_dim <= 0 and not cfg.patch_bce)
        # A val set of sentinel masks (full_fakes) makes loc F1 mechanically
        # ~1.0, so selecting on it pins best.pt to epoch 0 forever
        # (is_best needs f1 >= 1.0 + min_delta) and burns early-stop patience
        # while the model is still improving. Select on image AUC instead.
        from lab_utils.eval.aggregate import localization_is_meaningful
        loc_ok = localization_is_meaningful(val_ds.items)
        use_auc = mil_only or not loc_ok
        if not loc_ok and not mil_only and epoch == 0:
            log_line('[eval] val localization is sentinel-only (rule 2) — '
                     'early-stop metric switched to image AUC')
        metric_label = 'val_image_auc' if use_auc else f'val_f1_{cfg.early_stop_reduce}'
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
            if use_auc:
                # MIL-only, or localization is sentinel-only: early-stop on
                # image-level AUC
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

            if cfg.viz_every_epoch:
                run_epoch_viz(
                    model, val_ds.items, res,
                    device=device, cfg=cfg, epoch=epoch,
                    run_dir=cfg.run_dir, n=cfg.viz_n,
                    per_source=cfg.viz_per_source,
                )

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
