"""experiments.scripts.train_zoom_box — train the dense per-patch zoom-box head.

ISOLATED recipe (frozen detector): load a trained detector, FREEZE it, and train a
ZoomBoxHead — a self-attention encoder + per-patch (box, confidence) heads — as an
offline contextual bandit.  See docs/zoom_box_spec.md.

    Phase 0 (warm-start, supervised):  regress boxes toward GT-component frac boxes and
        confidence toward the inside/outside label.  Seeds the basin.
    Phase 1 (bandit, AWR):  per high-prior patch, jitter the box into K candidates, score
        each candidate's frozen zoom-ADVANTAGE over baseline=max(flat, attn), advantage-
        weight-regress the box toward the winners, regress confidence toward realized
        advantage.  σ (exploration) anneals over the bandit phase.

Train on casia + sagid splices; eval on imd2020 + sagid (policy vs flat/attn/baseline,
confidence↔advantage calibration, δ-sweep).

    python -m experiments.scripts.train_zoom_box \
        --init_checkpoint /runs/r032/best.pt --run_dir /runs/zoom_box/r032 \
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
from lab_utils.model.zoom_box_head import build_zoom_box_head
from lab_utils.train.checkpoint import save as save_ckpt

from experiments.labs.attention_zoom import _resolve_decoder
from experiments.labs.box_policy_zoom import policy_input_dim
from experiments.labs.box_heatmap_lab import collect_splices, seed_everything
from experiments.labs.zoom_box_lab import (
    zoom_box_train_item, evaluate_zoom_box, _TRAIN_SOURCES, _EVAL_SOURCES,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Train the dense per-patch zoom-box head (bandit / AWR).')
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

    g = p.add_argument_group('head architecture')
    g.add_argument('--width', type=int, default=128)
    g.add_argument('--depth', type=int, default=2)
    g.add_argument('--n_heads', type=int, default=4)
    g.add_argument('--dropout', type=float, default=0.0)
    g.add_argument('--dist_bias', type=float, default=-1.0, help='pre-softplus box bias ⇒ cold-start box size.')
    g.add_argument('--conf_bias', type=float, default=0.0)
    g.add_argument('--min_box_half', type=float, default=0.04, help='per-side box distance floor (frac).')
    g.add_argument('--max_box_half', type=float, default=0.35, help='per-side box distance cap (frac); bounds the zoom box so it cannot inflate to the no-op whole-frame.')
    g.add_argument('--shared_trunk', action='store_true', help='share one encoder for box+conf (legacy; risky).')

    g = p.add_argument_group('bandit / AWR')
    g.add_argument('--n_propose', type=int, default=6, help='patches proposing boxes per image.')
    g.add_argument('--n_candidates', type=int, default=5, help='jittered candidates per proposing patch.')
    g.add_argument('--n_background', type=int, default=4, help='random background patches for conf calibration.')
    g.add_argument('--sigma', type=float, default=0.6, help='exploration noise (pre-softplus) at bandit start.')
    g.add_argument('--sigma_final', type=float, default=0.15, help='annealed exploration floor.')
    g.add_argument('--awr_temp', type=float, default=0.05, help='advantage-weighting softmax temperature.')
    g.add_argument('--reward', default='f1', choices=['f1', 'iou'])
    g.add_argument('--huber_beta', type=float, default=0.1)
    g.add_argument('--lambda_box', type=float, default=1.0)
    g.add_argument('--lambda_conf', type=float, default=1.0)

    g = p.add_argument_group('warm-start (supervised)')
    g.add_argument('--warmstart_epochs', type=int, default=2)
    g.add_argument('--patch_frac', type=float, default=0.25, help='GT→patch coverage for warm-start boxes.')
    g.add_argument('--pad_min_patches', type=int, default=1, help='context padding on warm-start GT boxes.')
    g.add_argument('--conf_warm', type=float, default=0.2, help='small ±target for conf warm-start (avoids BCE spike).')

    g = p.add_argument_group('gate / decode (eval)')
    g.add_argument('--delta', type=float, default=0.0, help='operating gate: zoom box iff conf > δ.')
    g.add_argument('--iou_thresh', type=float, default=0.5, help='NMS overlap threshold.')
    g.add_argument('--max_boxes', type=int, default=4)
    g.add_argument('--baseline_warmup_epochs', type=int, default=3,
                   help='bandit epochs to train against baseline=flat (drop attn) before max(flat,attn).')

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


def _sigma_for_epoch(args, epoch: int) -> float:
    """Linear anneal from --sigma to --sigma_final over the bandit phase (floored)."""
    n_bandit = max(1, args.num_epochs - args.warmstart_epochs)
    e = epoch - args.warmstart_epochs
    if e <= 0:
        return args.sigma
    frac = min(1.0, e / max(1, n_bandit - 1))
    return float(args.sigma + frac * (args.sigma_final - args.sigma))


def main() -> None:
    args = _build_parser().parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    log_line(f'[zb] loading frozen detector: {args.init_checkpoint}')
    model, cfg, res = load_eval_model(args.init_checkpoint, device=device, strict=False)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    if getattr(model, 'contrastive_proj', None) is None:
        raise RuntimeError('train_zoom_box: detector has no contrastive head — embeddings required.')

    decode_fn, decoder_name = _resolve_decoder(args.decoder)
    use_attn = not args.no_attn_channel
    use_patch_logit = not args.no_patch_logit_channel

    train_by_source = collect_splices(args, _TRAIN_SOURCES, res, split='train')
    train_splices = [it for items in train_by_source.values() for it in items]
    if not train_splices:
        raise RuntimeError('train_zoom_box: no train splices — check --casia_root / --sagid_root.')

    eval_full = collect_splices(args, _EVAL_SOURCES, res, split='val')
    eval_by_source = {
        src: deterministic_subsample(items, args.eval_per_source, seed=f'zb_eval:{src}')
        for src, items in eval_full.items()
    }
    log_line(f'[zb] train splices={len(train_splices)}  '
             f'eval={[(s, len(v)) for s, v in eval_by_source.items()]}')

    # probe one item to size the head
    probe_t = load_image_tensor(train_splices[0], res, device=device)
    probe_info = model_info(model, probe_t, device=device, amp=use_amp, amp_dtype=args.amp_dtype)
    if probe_info.embeddings is None:
        raise RuntimeError('train_zoom_box: detector returned no embeddings on the probe item.')
    in_dim = policy_input_dim(probe_info, use_attn=use_attn, use_patch_logit=use_patch_logit)
    n_patches = int(probe_info.embeddings.shape[0])
    log_line(f'[zb] in_dim={in_dim} n_patches={n_patches} grid={probe_info.grid_hw} '
             f'(use_attn={use_attn} use_patch_logit={use_patch_logit})')

    head = build_zoom_box_head(
        in_dim, device=device, width=args.width, depth=args.depth, n_heads=args.n_heads,
        dropout=args.dropout, dist_bias=args.dist_bias, conf_bias=args.conf_bias,
        min_box_half=args.min_box_half, max_box_half=args.max_box_half,
        shared_trunk=args.shared_trunk, max_positions=max(2048, n_patches + 8),
    )
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    log_line(f'[zb] zoom-box head trainable params: {n_params:,}')
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'run_config.json').write_text(json.dumps(vars(args), indent=2, default=str))

    flat_cache: Dict[str, float] = {}
    attn_cache: Dict[str, float] = {}
    common_kwargs = dict(
        decode_fn=decode_fn, decoder_name=decoder_name, flat_cache=flat_cache, attn_cache=attn_cache,
        use_attn=use_attn, use_patch_logit=use_patch_logit, patch_frac=args.patch_frac,
        pad_min_patches=args.pad_min_patches, n_propose=args.n_propose, n_candidates=args.n_candidates,
        n_background=args.n_background, awr_temp=args.awr_temp, reward=args.reward,
        min_crop_frac=args.min_crop_frac, huber_beta=args.huber_beta, lambda_box=args.lambda_box,
        lambda_conf=args.lambda_conf, conf_warm=args.conf_warm, use_amp=use_amp, amp_dtype=args.amp_dtype,
    )

    best_metric = -1e9
    for epoch in range(args.num_epochs):
        phase = 'warmstart' if epoch < args.warmstart_epochs else 'bandit'
        sigma = _sigma_for_epoch(args, epoch)
        # baseline curriculum: first --baseline_warmup_epochs bandit epochs train against
        # flat (no-zoom) so zoom-favorable images give positive advantage to climb; then
        # switch to the DEPLOYABLE incumbent attn (NOT the oracle max(flat,attn)).
        train_ref = 'flat' if epoch < (args.warmstart_epochs + args.baseline_warmup_epochs) else 'attn'
        rng = random.Random(args.seed + epoch)
        order = list(train_splices)
        rng.shuffle(order)
        epoch_items = order[:args.train_per_epoch]
        log_line(f'[zb] epoch={epoch} phase={phase} sigma={sigma:.3f} '
                 f'baseline={train_ref} items={len(epoch_items)}')

        head.train()
        optimizer.zero_grad(set_to_none=True)
        L, BL, CL, NR, MA, BA, PF, AR = [], [], [], [], [], [], [], []
        n_used, n_in_accum = 0, 0
        for i, item in enumerate(epoch_items):
            try:
                out = zoom_box_train_item(
                    model, head, item, res, phase=phase, device=device, sigma=sigma,
                    ref=train_ref, rng=rng, **common_kwargs,
                )
            except Exception as exc:
                log_line(f'[zb] WARN: train item {item.item_id} failed: {exc}')
                out = None
            if out is None:
                continue
            loss, st = out
            (loss / args.grad_accum).backward()
            n_in_accum += 1
            n_used += 1
            L.append(st.loss); BL.append(st.box_loss); CL.append(st.conf_loss)
            NR.append(st.n_propose); MA.append(st.mean_adv); BA.append(st.best_adv)
            PF.append(st.pos_frac); AR.append(st.mean_box_area)

            if n_in_accum == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
                optimizer.step(); optimizer.zero_grad(set_to_none=True); n_in_accum = 0

            if (i + 1) % args.log_every == 0:
                k = args.log_every
                base = (f'[zb] epoch={epoch} {phase[:2]} {i + 1}/{len(epoch_items)} '
                        f'loss~{np.mean(L[-k:]):.4f} (box~{np.mean(BL[-k:]):.4f} '
                        f'conf~{np.mean(CL[-k:]):.4f})')
                if phase == 'warmstart':
                    log_line(f'{base} inside~{np.mean(NR[-k:]):.1f} area~{np.mean(AR[-k:]):.3f}')
                else:
                    log_line(f'{base} adv~{np.mean(MA[-k:]):+.4f} best~{np.mean(BA[-k:]):+.4f} '
                             f'pos~{np.mean(PF[-k:]):.2f} area~{np.mean(AR[-k:]):.3f}')

        if n_in_accum > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)

        done = (f'[zb] epoch={epoch} done phase={phase}  n_used={n_used}  '
                f'mean_loss={np.mean(L) if L else float("nan"):.4f} '
                f'(box={np.mean(BL) if BL else float("nan"):.4f} '
                f'conf={np.mean(CL) if CL else float("nan"):.4f})  '
                f'box_area={np.mean(AR) if AR else 0:.3f}')
        if phase == 'bandit':
            done += (f'  adv={np.mean(MA) if MA else 0:+.4f}  best={np.mean(BA) if BA else 0:+.4f}'
                     f'  pos_frac={np.mean(PF) if PF else 0:.2f}')
        log_line(done)

        metric = evaluate_zoom_box(
            model, head, eval_by_source, res, device=device, decode_fn=decode_fn,
            decoder_name=decoder_name, delta=args.delta, iou_thresh=args.iou_thresh,
            max_boxes=args.max_boxes, flat_cache=flat_cache, attn_cache=attn_cache,
            use_attn=use_attn, use_patch_logit=use_patch_logit, min_crop_frac=args.min_crop_frac,
            reward=args.reward, use_amp=use_amp, amp_dtype=args.amp_dtype, epoch=epoch,
        )

        state = {
            'epoch': epoch, 'zoom_box_head': head.state_dict(), 'in_dim': in_dim,
            'best_metric': max(metric, best_metric), 'cfg': vars(args),
            'meta': {'recipe': 'zoom_box', 'decoder': decoder_name,
                     'init_checkpoint': args.init_checkpoint, 'phase': phase,
                     'use_attn': use_attn, 'use_patch_logit': use_patch_logit},
        }
        save_ckpt(state, str(run_dir / f'epoch_{epoch:04d}.pt'), is_main=True)
        # only select the best from the bandit phase (warm-start metric is not comparable)
        if phase == 'bandit' and metric >= best_metric:
            best_metric = metric
            save_ckpt(state, str(run_dir / 'best.pt'), is_main=True)
            log_line(f'[zb] best zoom-box head saved  captured_advantage={best_metric:+.4f}')

    log_line(f'[zb] done  best captured advantage = {best_metric:+.4f}')


if __name__ == '__main__':
    main()
