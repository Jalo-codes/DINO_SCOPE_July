"""experiments.scripts.train_single_box — supervised single-box heatmap MVP.

ISOLATED recipe (like train_box_policy): the DINOv3 detector is FROZEN; only the
small :class:`BoxHeatmap` head trains, by BCE+Dice, toward the raw GT splice mask
(the clean label; ALL geometry is handled at read-off, never baked in).  This is
the simplest test of "can the head predict where to zoom from frozen features" —
no RL, no sampling, no union credit.

    detector (frozen) ──model_info──▶ [z|attn|patch_logit]
        ──BoxHeatmap──▶ per-patch logit (heatmap)
        ──BCE+Dice vs GT splice mask──▶ loss            (GT via metric, I3)
    eval: threshold heatmap → read-off bbox → run_bbox_zoom → realized F1.

Default experiment: train on SAGID + CASIA splices (400/epoch), evaluate each
epoch on 50 splices from each of SAGID / CASIA / IMD vs the flat-decode baseline,
and dump a few heatmap/box visualizations per epoch.

Usage:
    python -m experiments.scripts.train_single_box \\
        --init_checkpoint /runs/base/best.pt \\
        --sagid_root /data/sagid --casia_root /data/casia --imd2020_root /data/imd \\
        --run_dir /runs/singlebox01
"""

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from lab_utils.data.sampling import deterministic_subsample
from lab_utils.eval.fetch import model_info
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.logging.text import log_line
from lab_utils.model.box_heatmap import build_box_heatmap
from lab_utils.train.checkpoint import save as save_ckpt

from experiments.labs.attention_zoom import _resolve_decoder
from experiments.labs.box_policy_zoom import policy_input_dim
from experiments.labs.box_heatmap_lab import (
    _EVAL_SOURCES,
    _SOURCE_ROOT,
    box_heatmap_train_item,
    collect_splices,
    evaluate,
    seed_everything,
)


_TRAIN_SOURCES = ('sagid', 'casia')


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='train_single_box',
        description='Supervised single-box heatmap head on a frozen detector.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--init_checkpoint', required=True,
                   help='Frozen detector checkpoint (needs the contrastive head).')
    p.add_argument('--run_dir', required=True)

    g = p.add_argument_group('dataset roots')
    for attr in sorted(set(_SOURCE_ROOT.values())):
        g.add_argument(f'--{attr}', default=None)

    g = p.add_argument_group('decode / zoom')
    g.add_argument('--decoder', default='kmeans', choices=['kmeans', 'hdbscan'],
                   help='Frozen decoder used inside the zoom crop (= shipped decoder).')
    g.add_argument('--min_crop_frac', type=float, default=0.25)
    g.add_argument('--no_attn_channel',        action='store_true',
                   help='Drop the attention input channel (ablation — NOT recommended).')
    g.add_argument('--no_patch_logit_channel', action='store_true',
                   help='Drop the patch-logit input channel.')

    g = p.add_argument_group('head architecture')
    g.add_argument('--width',  type=int, default=128)
    g.add_argument('--depth',  type=int, default=2)
    g.add_argument('--n_heads', type=int, default=4)
    g.add_argument('--bias_init', type=float, default=-2.0,
                   help='Head bias; negative ⇒ heatmap starts mostly-OFF (sparse target).')
    g.add_argument('--dropout', type=float, default=0.1,
                   help='Encoder dropout — regularizes the head (helps the val plateau).')

    g = p.add_argument_group('target (GT grid)')
    g.add_argument('--patch_frac', type=float, default=0.25,
                   help='A patch counts as GT when ≥ this fraction of its pixels are GT. '
                        'The target is the raw GT splice mask; all geometry is read-off.')

    g = p.add_argument_group('training')
    g.add_argument('--num_epochs',      type=int,   default=20)
    g.add_argument('--train_per_epoch', type=int,   default=400)
    g.add_argument('--lr',              type=float, default=3e-4)
    g.add_argument('--weight_decay',    type=float, default=1e-4)
    g.add_argument('--grad_accum',      type=int,   default=8)
    g.add_argument('--pos_weight',      type=float, default=8.0,
                   help='BCE up-weight on box (1) patches vs background (0).')
    g.add_argument('--loss', default='bce_dice', choices=['dice', 'bce_dice'],
                   help='Per-image loss. Both are SIZE-INVARIANT (Dice makes each image '
                        'pull the gradient equally regardless of splice size, fixing the '
                        'large-splice bias). "bce_dice" adds BCE\'s stable per-pixel term.')
    g.add_argument('--max_grad_norm',   type=float, default=5.0)
    g.add_argument('--seed',            type=int,   default=42)
    g.add_argument('--log_every',       type=int,   default=50)

    g = p.add_argument_group('read-off (eval)')
    g.add_argument('--thresh',      type=float, default=0.5,
                   help='Heatmap threshold for reading off the box.')
    g.add_argument('--min_patches', type=int,   default=2,
                   help='Minimum ON patches to emit a box (else: no zoom).')
    g.add_argument('--dilate', type=int, default=1,
                   help='proximity grouping radius in patches (ON cells within this '
                        'distance join one box). Smaller ⇒ more, separate boxes.')
    g.add_argument('--max_regions', type=int, default=3,
                   help='Hard cap on boxes per image at read-off (use them well).')
    g.add_argument('--readoff_pad_frac', type=float, default=0.05,
                   help='Proportional read-off padding (shrinks with box size, so large '
                        'boxes get ~none).  Lower ⇒ tighter crops across the board.')
    g.add_argument('--readoff_min_pad_frac', type=float, default=0.0,
                   help='Floor on the per-side read-off padding fraction (0 = off; the '
                        'min-box-size floor handles over-magnification now).')
    g.add_argument('--readoff_min_box_size', type=int, default=6,
                   help='Min read-off box size in patches/side (0 = off). The floor that '
                        'stops a tiny splice being over-magnified into a sliver; small '
                        'splices otherwise get only one patch of padding.')
    g.add_argument('--square_cap', type=float, default=1.4,
                   help='Max aspect ratio at read-off (partial squaring). 1.0 = fully '
                        'square (over-expands thin boxes); 1.4 matches the attention path.')
    g.add_argument('--overlap_kill_frac', type=float, default=0.30,
                   help='Drop a smaller box when > this fraction of it lies inside a '
                        'larger box (kills redundant nested/overlapping windows). '
                        '0 = off.')
    g.add_argument('--large_area_frac', type=float, default=0.6,
                   help='If the RAW ON set covers ≥ this fraction of the frame, defer to '
                        'flat (large splice → don\'t zoom), decided pre-pad.')
    g.add_argument('--no_gate_logit', action='store_true',
                   help='Disable the MIL logit gate (by default a zoom crop that does '
                        'not out-score the full frame defers to the flat decode).')
    g.add_argument('--gate_margin', type=float, default=0.0,
                   help='Slack on the MIL logit gate: a crop is kept if its logit ≥ '
                        'full_logit - gate_margin.')

    g = p.add_argument_group('eval + viz')
    g.add_argument('--with_hdbscan', action='store_true',
                   help='Add an HDBSCAN-partition-zoom reference column to eval '
                        '(box from the HDBSCAN decode components; needs the HDBSCAN '
                        'backend; static ⇒ cached).')
    g.add_argument('--eval_per_source', type=int, default=150,
                   help='Eval splices PER source (capped by availability). Larger ⇒ '
                        'less noisy val metrics.')
    g.add_argument('--viz_per_source', type=int, default=0,
                   help='Visualizations saved PER eval source. 0 (default) = ALL eval '
                        'images for that source.')

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'])
    return p


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    log_line(f'[sb] loading frozen detector: {args.init_checkpoint}')
    model, cfg, res = load_eval_model(args.init_checkpoint, device=device, strict=False)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    if getattr(model, 'contrastive_proj', None) is None:
        raise RuntimeError('train_single_box: detector has no contrastive head — embeddings required.')

    decode_fn, decoder_name = _resolve_decoder(args.decoder)

    train_by_source = collect_splices(args, _TRAIN_SOURCES, res, split='train')
    train_splices = [it for items in train_by_source.values() for it in items]
    if not train_splices:
        raise RuntimeError('train_single_box: no train splices — check --sagid_root / --casia_root.')

    eval_full = collect_splices(args, _EVAL_SOURCES, res, split='val')
    eval_by_source = {
        src: deterministic_subsample(items, args.eval_per_source, seed=f'sb_eval:{src}')
        for src, items in eval_full.items()
    }
    log_line(f'[sb] train splices={len(train_splices)}  '
             f'eval={[(s, len(v)) for s, v in eval_by_source.items()]}')

    use_attn = not args.no_attn_channel
    use_patch_logit = not args.no_patch_logit_channel
    probe_t = load_image_tensor(train_splices[0], res, device=device)
    probe_info = model_info(model, probe_t, device=device, amp=use_amp, amp_dtype=args.amp_dtype)
    in_dim = policy_input_dim(probe_info, use_attn=use_attn, use_patch_logit=use_patch_logit)
    log_line(f'[sb] head in_dim={in_dim} (use_attn={use_attn} use_patch_logit={use_patch_logit})')

    head = build_box_heatmap(in_dim, device=device, width=args.width, depth=args.depth,
                             n_heads=args.n_heads, bias_init=args.bias_init, dropout=args.dropout)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'run_config.json').write_text(json.dumps(vars(args), indent=2, default=str))

    if args.with_hdbscan:
        from lab_utils.eval.decode.hdbscan import hdbscan_available
        if not hdbscan_available():
            log_line('[sb] WARN: --with_hdbscan set but no HDBSCAN backend '
                     '(need sklearn>=1.3 or the hdbscan package); disabling.')
            args.with_hdbscan = False

    flat_cache: Dict[str, float] = {}
    attn_cache: Dict[str, float] = {}        # attention-zoom F1 per item (static reference)
    hdb_cache: Dict[str, float] = {}         # hdbscan-partition-zoom F1 per item (static)
    train_kwargs = dict(
        decoder_name=decoder_name, use_attn=use_attn, use_patch_logit=use_patch_logit,
        pos_weight=args.pos_weight, loss_mode=args.loss, patch_frac=args.patch_frac,
        use_amp=use_amp, amp_dtype=args.amp_dtype,
    )
    single_kwargs = dict(
        use_attn=use_attn, use_patch_logit=use_patch_logit, thresh=args.thresh,
        min_patches=args.min_patches, dilate=args.dilate,
        max_regions=args.max_regions,
        readoff_pad_frac=args.readoff_pad_frac,
        readoff_min_pad_frac=args.readoff_min_pad_frac,
        readoff_min_box_size=args.readoff_min_box_size, square_cap=args.square_cap,
        overlap_kill_frac=args.overlap_kill_frac, large_area_frac=args.large_area_frac,
        gate_logit=not args.no_gate_logit, gate_margin=args.gate_margin,
        min_crop_frac=args.min_crop_frac,
    )

    best_metric = -1.0
    for epoch in range(args.num_epochs):
        rng = random.Random(args.seed + epoch)
        order = list(train_splices)
        rng.shuffle(order)
        epoch_items = order[:args.train_per_epoch]

        head.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss, run_pos, kinds = [], [], {'box': 0, 'large': 0, 'no_gt': 0}
        n_used, n_in_accum = 0, 0
        for i, item in enumerate(epoch_items):
            try:
                out = box_heatmap_train_item(model, head, item, res, device=device, **train_kwargs)
            except Exception as exc:
                log_line(f'[sb] WARN: train item {item.item_id} failed: {exc}')
                out = None
            if out is None:
                continue
            loss, stats = out
            (loss / args.grad_accum).backward()
            n_in_accum += 1
            n_used += 1
            run_loss.append(stats.loss); run_pos.append(stats.pos)
            kinds[stats.kind] = kinds.get(stats.kind, 0) + 1

            if n_in_accum == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_in_accum = 0

            if (i + 1) % args.log_every == 0:
                log_line(f'[sb] epoch={epoch} {i + 1}/{len(epoch_items)} '
                         f'loss~{np.mean(run_loss[-args.log_every:]):.4f} '
                         f'pos~{np.mean(run_pos[-args.log_every:]):.1f}')

        if n_in_accum > 0:
            torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        log_line(f'[sb] epoch={epoch} done  n_used={n_used}  '
                 f'mean_loss={np.mean(run_loss) if run_loss else float("nan"):.4f}  '
                 f'targets(box/large/no_gt)={kinds["box"]}/{kinds["large"]}/{kinds["no_gt"]}')

        med = evaluate(
            model, head, eval_by_source, res, device=device, decode_fn=decode_fn,
            decoder_name=decoder_name, use_amp=use_amp, amp_dtype=args.amp_dtype,
            flat_cache=flat_cache, attn_cache=attn_cache, hdb_cache=hdb_cache,
            with_hdbscan=args.with_hdbscan, viz_per_source=args.viz_per_source,
            viz_dir=run_dir / 'viz' / f'epoch_{epoch:04d}', epoch=epoch, single_kwargs=single_kwargs,
            patch_frac=args.patch_frac,
            max_regions=args.max_regions, readoff_pad_frac=args.readoff_pad_frac,
        )

        state = {
            'epoch': epoch, 'head': head.state_dict(), 'in_dim': in_dim,
            'best_metric': max(med, best_metric), 'cfg': vars(args),
            'meta': {'recipe': 'single_box', 'decoder': decoder_name,
                     'init_checkpoint': args.init_checkpoint,
                     'use_attn': use_attn, 'use_patch_logit': use_patch_logit},
        }
        save_ckpt(state, str(run_dir / f'epoch_{epoch:04d}.pt'), is_main=True)
        if med >= best_metric:
            best_metric = med
            save_ckpt(state, str(run_dir / 'best.pt'), is_main=True)
            log_line(f'[sb] best head saved  median_splice_f1={best_metric:.4f}')

    log_line(f'[sb] done  best median splice F1 = {best_metric:.4f}')


if __name__ == '__main__':
    main()
