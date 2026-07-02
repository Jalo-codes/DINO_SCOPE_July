"""experiments.scripts.train_zoom_head — train the learned zoom head.

ISOLATED recipe (frozen detector): load a trained detector, FREEZE it, and train
two light heads — a projection `z→z'` (HDBSCAN-clusterable space) and a per-patch
value head (regressed to per-cluster zoom-ADVANTAGE). See docs/zoom_head_spec.md.

    frozen forward → z, feats
      ProjectionHead z→z' → HDBSCAN + CC → REGIONS
      ValueHead feats→scalar → region-mean = predicted advantage
    reward(region) = F1(zoom→region) − F1(no-zoom)        (frozen, queryable)
    train: value regresses region-mean → realized advantage; projection by a
           GT-instance metric loss. Gate at inference: predicted advantage > δ.

Train on casia + sagid splices (1k/epoch); eval on 150 imd2020 + 150 sagid with
policy-F1-vs-flat/attn references, predicted↔realized advantage calibration, and a
δ-sweep to pick the operating point.

    python -m experiments.scripts.train_zoom_head \
        --init_checkpoint /runs/r032/best.pt --run_dir /runs/zoom_head/r032 \
        --casia_root … --sagid_root … --imd2020_root … --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from lab_utils.data.sampling import deterministic_subsample
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.fetch import model_info
from lab_utils.logging.text import log_line
from lab_utils.model.zoom_head import build_zoom_head
from lab_utils.train.checkpoint import save as save_ckpt

from experiments.labs.attention_zoom import _resolve_decoder
from experiments.labs.box_policy_zoom import policy_input_dim, build_policy_input
from experiments.labs.box_heatmap_lab import collect_splices, seed_everything
from experiments.labs.zoom_cluster_lab import (
    zoom_head_train_item, evaluate_zoom_head, _TRAIN_SOURCES, _EVAL_SOURCES,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train the learned zoom head (projection + per-cluster value).')
    p.add_argument('--init_checkpoint', required=True, help='trained detector checkpoint (frozen).')
    p.add_argument('--run_dir', required=True)

    g = p.add_argument_group('dataset roots')
    for attr in ('sagid_root', 'casia_root', 'imd2020_root'):
        g.add_argument(f'--{attr}', default=None)

    g = p.add_argument_group('decode / zoom')
    g.add_argument('--decoder', default='kmeans', choices=['kmeans', 'hdbscan'])
    g.add_argument('--min_crop_frac', type=float, default=0.25)
    g.add_argument('--no_attn_channel', action='store_true')
    g.add_argument('--no_patch_logit_channel', action='store_true')

    g = p.add_argument_group('projection head (z → z\')')
    g.add_argument('--proj_dim', type=int, default=32, help='projected dim (kept above the too-shallow 8-16).')
    g.add_argument('--proj_hidden', type=int, default=128)
    g.add_argument('--proj_depth', type=int, default=2, help='hidden layers (>=2 ⇒ a real MLP, not linear).')
    g.add_argument('--proj_dropout', type=float, default=0.0)
    g.add_argument('--proj_margin', type=float, default=0.2)

    g = p.add_argument_group('value head')
    g.add_argument('--width', type=int, default=128)
    g.add_argument('--depth', type=int, default=2)
    g.add_argument('--n_heads', type=int, default=4)
    g.add_argument('--value_dropout', type=float, default=0.1)

    g = p.add_argument_group('clustering (HDBSCAN on z\' + CC split)')
    g.add_argument('--min_cluster_size', type=int, default=8)
    g.add_argument('--min_samples', type=int, default=None)
    g.add_argument('--cluster_dilate', type=int, default=1)
    g.add_argument('--cluster_min_patches', type=int, default=4)
    g.add_argument('--cluster_max_regions', type=int, default=6)
    g.add_argument('--cluster_pad_frac', type=float, default=0.06)
    g.add_argument('--cluster_min_box_size', type=int, default=6)
    g.add_argument('--cluster_min_pad_frac', type=float, default=0.04)
    g.add_argument('--square_cap', type=float, default=1.4)
    g.add_argument('--overlap_kill_frac', type=float, default=0.75,
                   help='drop a smaller box when > this fraction sits inside a larger one.')

    g = p.add_argument_group('gate / reward')
    g.add_argument('--delta', type=float, default=0.10, help='operating gate: zoom region iff predicted advantage > δ.')
    g.add_argument('--reward', default='f1', choices=['f1', 'iou'])
    g.add_argument('--patch_frac', type=float, default=0.25, help='GT→patch coverage for instance labels.')
    g.add_argument('--lambda_value', type=float, default=1.0)
    g.add_argument('--lambda_proj', type=float, default=1.0)
    g.add_argument('--huber_beta', type=float, default=0.1)

    g = p.add_argument_group('training')
    g.add_argument('--num_epochs', type=int, default=20)
    g.add_argument('--train_per_epoch', type=int, default=1000)
    g.add_argument('--lr', type=float, default=3e-4)
    g.add_argument('--weight_decay', type=float, default=1e-4)
    g.add_argument('--grad_accum', type=int, default=8)
    g.add_argument('--max_grad_norm', type=float, default=5.0)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--log_every', type=int, default=50)

    g = p.add_argument_group('eval')
    g.add_argument('--eval_per_source', type=int, default=150)

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'])
    return p


def main() -> None:
    args = _build_parser().parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    log_line(f'[zh] loading frozen detector: {args.init_checkpoint}')
    model, cfg, res = load_eval_model(args.init_checkpoint, device=device, strict=False)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    if getattr(model, 'contrastive_proj', None) is None:
        raise RuntimeError('train_zoom_head: detector has no contrastive head — embeddings required.')

    decode_fn, decoder_name = _resolve_decoder(args.decoder)
    use_attn = not args.no_attn_channel
    use_patch_logit = not args.no_patch_logit_channel

    train_by_source = collect_splices(args, _TRAIN_SOURCES, res, split='train')
    train_splices = [it for items in train_by_source.values() for it in items]
    if not train_splices:
        raise RuntimeError('train_zoom_head: no train splices — check --casia_root / --sagid_root.')

    eval_full = collect_splices(args, _EVAL_SOURCES, res, split='val')
    eval_by_source = {
        src: deterministic_subsample(items, args.eval_per_source, seed=f'zh_eval:{src}')
        for src, items in eval_full.items()
    }
    log_line(f'[zh] train splices={len(train_splices)}  '
             f'eval={[(s, len(v)) for s, v in eval_by_source.items()]}')

    # Probe one item to size the heads.
    probe_t = load_image_tensor(train_splices[0], res, device=device)
    probe_info = model_info(model, probe_t, device=device, amp=use_amp, amp_dtype=args.amp_dtype)
    if probe_info.embeddings is None:
        raise RuntimeError('train_zoom_head: detector returned no embeddings on the probe item.')
    emb_dim = int(probe_info.embeddings.shape[1])
    value_in_dim = policy_input_dim(probe_info, use_attn=use_attn, use_patch_logit=use_patch_logit)
    n_patches = int(probe_info.embeddings.shape[0])
    log_line(f'[zh] emb_dim={emb_dim} value_in_dim={value_in_dim} n_patches={n_patches} '
             f'(use_attn={use_attn} use_patch_logit={use_patch_logit})')

    zoomhead = build_zoom_head(
        emb_dim, value_in_dim, device=device,
        proj_dim=args.proj_dim, proj_hidden=args.proj_hidden, proj_depth=args.proj_depth,
        proj_dropout=args.proj_dropout, value_width=args.width, value_depth=args.depth,
        value_heads=args.n_heads, value_dropout=args.value_dropout,
        max_positions=max(1024, n_patches + 8),
    )
    n_params = sum(p.numel() for p in zoomhead.parameters() if p.requires_grad)
    log_line(f'[zh] zoom head trainable params: {n_params:,}')
    optimizer = torch.optim.AdamW(zoomhead.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    cluster_kwargs = dict(
        min_cluster_size=args.min_cluster_size, min_samples=args.min_samples,
        dilate=args.cluster_dilate, min_patches=args.cluster_min_patches,
        max_regions=args.cluster_max_regions, pad_frac=args.cluster_pad_frac,
        min_box_size=args.cluster_min_box_size, min_pad_frac=args.cluster_min_pad_frac,
        square_cap=args.square_cap, overlap_kill_frac=args.overlap_kill_frac,
    )

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'run_config.json').write_text(json.dumps(vars(args), indent=2, default=str))

    flat_cache: Dict[str, float] = {}
    attn_cache: Dict[str, float] = {}
    train_kwargs = dict(
        decode_fn=decode_fn, decoder_name=decoder_name, use_attn=use_attn,
        use_patch_logit=use_patch_logit, patch_frac=args.patch_frac,
        cluster_kwargs=cluster_kwargs, min_crop_frac=args.min_crop_frac,
        lambda_value=args.lambda_value, lambda_proj=args.lambda_proj,
        proj_margin=args.proj_margin, reward=args.reward, huber_beta=args.huber_beta,
        delta=args.delta,
        use_amp=use_amp, amp_dtype=args.amp_dtype,
    )

    best_metric = -1.0
    for epoch in range(args.num_epochs):
        rng = random.Random(args.seed + epoch)
        order = list(train_splices)
        rng.shuffle(order)
        epoch_items = order[:args.train_per_epoch]

        zoomhead.train()
        optimizer.zero_grad(set_to_none=True)
        L, VL, PL, NR, MA, PF = [], [], [], [], [], []     # loss / value / proj / regions / adv / pos-frac
        n_used, n_in_accum = 0, 0
        for i, item in enumerate(epoch_items):
            try:
                out = zoom_head_train_item(model, zoomhead, item, res, device=device, **train_kwargs)
            except Exception as exc:
                log_line(f'[zh] WARN: train item {item.item_id} failed: {exc}')
                out = None
            if out is None:
                continue
            loss, st = out
            (loss / args.grad_accum).backward()
            n_in_accum += 1
            n_used += 1
            L.append(st.loss); VL.append(st.value_loss); PL.append(st.proj_loss)
            NR.append(st.n_regions); MA.append(st.mean_adv); PF.append(st.pos_frac)

            if n_in_accum == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(zoomhead.parameters(), args.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); n_in_accum = 0

            if (i + 1) % args.log_every == 0:
                k = args.log_every
                log_line(f'[zh] epoch={epoch} {i + 1}/{len(epoch_items)} '
                         f'loss~{np.mean(L[-k:]):.4f} (val~{np.mean(VL[-k:]):.4f} '
                         f'proj~{np.mean(PL[-k:]):.4f}) regions~{np.mean(NR[-k:]):.2f} '
                         f'adv~{np.mean(MA[-k:]):+.4f} pos~{np.mean(PF[-k:]):.2f}')

        if n_in_accum > 0:
            torch.nn.utils.clip_grad_norm_(zoomhead.parameters(), args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)

        log_line(f'[zh] epoch={epoch} done  n_used={n_used}  '
                 f'mean_loss={np.mean(L) if L else float("nan"):.4f} '
                 f'(value={np.mean(VL) if VL else float("nan"):.4f} '
                 f'proj={np.mean(PL) if PL else float("nan"):.4f})  '
                 f'regions/img={np.mean(NR) if NR else 0:.2f}  '
                 f'mean_adv={np.mean(MA) if MA else 0:+.4f}  pos_frac={np.mean(PF) if PF else 0:.2f}')

        med = evaluate_zoom_head(
            model, zoomhead, eval_by_source, res, device=device, decode_fn=decode_fn,
            decoder_name=decoder_name, delta=args.delta, cluster_kwargs=cluster_kwargs,
            flat_cache=flat_cache, attn_cache=attn_cache, use_attn=use_attn,
            use_patch_logit=use_patch_logit, min_crop_frac=args.min_crop_frac,
            reward=args.reward, use_amp=use_amp, amp_dtype=args.amp_dtype, epoch=epoch,
        )

        state = {
            'epoch': epoch, 'zoom_head': zoomhead.state_dict(),
            'emb_dim': emb_dim, 'value_in_dim': value_in_dim,
            'best_metric': max(med, best_metric), 'cfg': vars(args),
            'meta': {'recipe': 'zoom_head', 'decoder': decoder_name,
                     'init_checkpoint': args.init_checkpoint,
                     'use_attn': use_attn, 'use_patch_logit': use_patch_logit},
        }
        save_ckpt(state, str(run_dir / f'epoch_{epoch:04d}.pt'), is_main=True)
        if med >= best_metric:
            best_metric = med
            save_ckpt(state, str(run_dir / 'best.pt'), is_main=True)
            log_line(f'[zh] best zoom head saved  median_policy_f1={best_metric:.4f}')

    log_line(f'[zh] done  best median policy F1 = {best_metric:.4f}')


if __name__ == '__main__':
    main()
