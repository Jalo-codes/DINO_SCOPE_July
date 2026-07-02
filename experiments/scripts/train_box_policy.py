"""experiments.scripts.train_box_policy — RL finetune of the learned box policy.

ISOLATED recipe (DESIGN_GUIDE I7 exception, like train_tgif): the DINOv3 detector
is FROZEN; only the small :class:`BoxPolicy` head trains, by REINFORCE, against the
realized zoom-localization F1.  Nothing here touches the standard supervised
``train.py`` flow.

    detector (frozen) ──model_info──▶ [z|attn|patch_logit]
        ──BoxPolicy.act──▶ grid-locked boxes ──run_bbox_zoom (frozen)──▶ union
        ──metric (GT, train-time only)──▶ F1 = reward
    advantage = reward − attention_zoom_F1(baseline)   # baseline static ⇒ cached
    loss = −advantage·log π(action) − β·entropy

Default experiment (per request): train on SAGID + CASIA splices (400/epoch),
evaluate each epoch on 50 splices from each of SAGID / CASIA / IMD, and dump 5
box visualizations per epoch.

All model access goes through ``model_info`` / ``run_bbox_zoom`` (I2); GT is read
only inside ``metric`` (I3), at train time, to form the reward.

Usage:
    python -m experiments.scripts.train_box_policy \\
        --init_checkpoint /runs/base/best.pt \\
        --sagid_root /data/sagid --casia_root /data/casia --imd2020_root /data/imd \\
        --run_dir /runs/boxpolicy01 \\
        --num_epochs 20 --train_per_epoch 400 --eval_per_source 50 --viz_n 5
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
from typing import Dict, List, Optional

import numpy as np
import torch

from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution
from lab_utils.data.datasets.registry import REGISTRY
from lab_utils.data.sampling import deterministic_subsample
from lab_utils.eval.fetch import model_info
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.zoom import mask_to_bbox
from lab_utils.logging.text import log_line
from lab_utils.model.box_policy import build_box_policy
from lab_utils.train.checkpoint import save as save_ckpt

from experiments.labs.attention_zoom import _resolve_decoder
from experiments.labs.box_policy_zoom import (
    attention_baseline_f1,
    box_policy_single,
    policy_input_dim,
    policy_train_item,
)


_TRAIN_SOURCES = ('sagid', 'casia')
_EVAL_SOURCES = ('sagid', 'casia', 'imd2020')
_SOURCE_ROOT = {
    'sagid': 'sagid_root', 'casia': 'casia_root', 'imd2020': 'imd2020_root',
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_percentile(val: str):
    try:
        return float(val)
    except ValueError:
        return val


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='train_box_policy',
        description='REINFORCE-train the grid-locked box policy on a frozen detector.',
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
                   help='Frozen decoder used inside each zoom crop (= the shipped decoder).')
    g.add_argument('--attn_percentile', type=_parse_percentile, default='otsu',
                   help='Threshold for the candidate / attention hot-set.')
    g.add_argument('--max_boxes',  type=int,   default=8, help='Hard cap on boxes per image.')
    g.add_argument('--cand_cap',   type=int,   default=64, help='Max candidate patches sampled.')
    g.add_argument('--min_crop_frac', type=float, default=0.25)
    g.add_argument('--no_attn_channel',        action='store_true',
                   help='Drop the attention input channel (z-only ablation).')
    g.add_argument('--no_patch_logit_channel', action='store_true',
                   help='Drop the patch-logit input channel.')

    g = p.add_argument_group('policy architecture')
    g.add_argument('--width',  type=int, default=128)
    g.add_argument('--depth',  type=int, default=2)
    g.add_argument('--n_heads', type=int, default=4)
    g.add_argument('--size_init',      type=float, default=0.30)
    g.add_argument('--keep_bias_init', type=float, default=-3.0,
                   help='Negative ⇒ sparse start.  σ(bias)·cand_cap ≈ expected '
                        'boxes; keep well under --max_boxes so the cap rarely fires.')
    g.add_argument('--size_min_frac',  type=float, default=0.18,
                   help='Min box extent per side (frac of frame).  Floors zoom '
                        'magnification: too small ⇒ crop is mostly interpolation. '
                        '~0.18 ≈ 5 patches at 448/16.')

    g = p.add_argument_group('RL training')
    g.add_argument('--num_epochs',     type=int,   default=20)
    g.add_argument('--train_per_epoch', type=int,  default=400,
                   help='Splices sampled per epoch (compute cap).')
    g.add_argument('--lr',             type=float, default=3e-4)
    g.add_argument('--weight_decay',   type=float, default=1e-4)
    g.add_argument('--grad_accum',     type=int,   default=8)
    g.add_argument('--baseline',       default='flat', choices=['flat', 'attn_zoom'],
                   help='REINFORCE baseline (centers advantage; same optimum either '
                        'way). "flat" = no-zoom decode F1 (cheap, well-centered); '
                        '"attn_zoom" = attention-zoom F1 (higher bar, 2 fwd/item).')
    g.add_argument('--credit',         default='per_box', choices=['per_box', 'global'],
                   help='Credit assignment. "per_box" = leave-one-out difference '
                        'reward per box (fixes uniform sizes + keep=0 inflation); '
                        '"global" = vanilla one-advantage REINFORCE.')
    g.add_argument('--entropy_beta',   type=float, default=0.01,
                   help='Exploration bonus.  Pushes keep-probs toward 0.5 (⇒ more '
                        'boxes); too low ⇒ collapses to no-zoom before learning.')
    g.add_argument('--box_cost',       type=float, default=0.005,
                   help='Per-proposal F1 cost (λ) on the PRE-cap sampled box count '
                        '(beyond the first).  MUST be well below the available zoom '
                        'gain (attn_zoom−flat, ~0.03 on easy sources) or the policy '
                        'collapses to no-zoom.  Count lever: raise if boxes inflate.')
    g.add_argument('--max_grad_norm',  type=float, default=5.0)
    g.add_argument('--seed',           type=int,   default=42)
    g.add_argument('--log_every',      type=int,   default=50)

    g = p.add_argument_group('eval + viz')
    g.add_argument('--eval_per_source', type=int, default=50,
                   help='Splices scored per eval source each epoch.')
    g.add_argument('--viz_n', type=int, default=5, help='Box visualizations per epoch.')

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'])
    return p


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── data collection ──────────────────────────────────────────────────────────────

def _collect_splices(args, sources, res, *, split: str) -> Dict[str, List[Item]]:
    """{source: [splice Items]} from the requested split ('train' | 'val')."""
    out: Dict[str, List[Item]] = {}
    for source in sources:
        root_str = getattr(args, _SOURCE_ROOT[source], None)
        if not root_str:
            continue
        root = Path(root_str)
        if not root.exists():
            log_line(f'[bp] WARN: root not found for {source}: {root}')
            continue
        train_ds, val_ds = REGISTRY[source](root, res=res)
        ds = train_ds if split == 'train' else val_ds
        splices = [it for it in ds.items if not it.is_real]
        out[source] = splices
        log_line(f'[bp] {source} ({split}): {len(splices)} splices')
    return out


# ── per-epoch eval (+ viz) ───────────────────────────────────────────────────────

@torch.no_grad()
def _flat_f1(model, item, res, *, device, decode_fn, decoder_name, use_amp, amp_dtype) -> float:
    img_t = load_image_tensor(item, res, device=device)
    info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    rec = eval_metric(decode_fn(info), info, item, decoder=decoder_name)
    return float(rec.f1)


@torch.no_grad()
def evaluate(
    model, policy, eval_by_source, res, *,
    device, decode_fn, decoder_name, use_amp, amp_dtype,
    baseline_cache, flat_cache, viz_n, viz_dir, epoch, single_kwargs,
) -> float:
    """Median splice F1 of the policy across all eval sources (drives best.pt).

    Also reports the attention-zoom baseline and flat-decode medians (both static
    on a frozen detector ⇒ computed once and cached) for lift context, and saves
    up to ``viz_n`` box visualizations.
    """
    from experiments.labs.viz import plot_box_policy_result

    policy.eval()
    overall: List[float] = []
    n_viz = 0
    if viz_dir is not None:
        viz_dir.mkdir(parents=True, exist_ok=True)

    for source, items in eval_by_source.items():
        pol, base, flat = [], [], []
        for item in items:
            rec, debug = box_policy_single(
                model, policy, item, res, device=device, decode_fn=decode_fn,
                decoder_name=decoder_name, use_amp=use_amp, amp_dtype=amp_dtype,
                return_debug=True, **single_kwargs,
            )
            pol.append(float(rec.f1))

            b = baseline_cache.get(item.item_id)
            if b is None:
                b = attention_baseline_f1(model, item, res, device=device,
                                          decode_fn=decode_fn, use_amp=use_amp, amp_dtype=amp_dtype)
                baseline_cache[item.item_id] = b
            base.append(b)

            f = flat_cache.get(item.item_id)
            if f is None:
                f = _flat_f1(model, item, res, device=device, decode_fn=decode_fn,
                             decoder_name=decoder_name, use_amp=use_amp, amp_dtype=amp_dtype)
                flat_cache[item.item_id] = f
            flat.append(f)

            if n_viz < viz_n and viz_dir is not None:
                gt_box = None
                if rec.gt_mask is not None and rec.gt_mask.any():
                    gt_box = mask_to_bbox(rec.gt_mask.astype(bool))
                fig = plot_box_policy_result(
                    debug['img_pil'], debug['boxes'], debug['keep_prob'], debug['grid_hw'],
                    candidates=debug['candidates'], attn=debug['attn1'],
                    union_mask=debug['mask_zoom'], gt_mask=rec.gt_mask, gt_box=gt_box,
                    title=f'{source} {item.item_id}  policy_f1={rec.f1:.3f} '
                          f'attn_zoom={base[-1]:.3f} flat={flat[-1]:.3f}  n_boxes={len(debug["boxes"])}',
                )
                fig.savefig(viz_dir / f'{n_viz:02d}_{source}_{item.item_id}.png',
                            dpi=120, bbox_inches='tight')
                import matplotlib.pyplot as plt
                plt.close(fig)
                n_viz += 1

        def _med(x):
            return float(np.median(x)) if x else float('nan')
        log_line(f'[bp-eval] {source:>8} (n={len(pol)}): '
                 f'policy_med_f1={_med(pol):.4f}  attn_zoom={_med(base):.4f}  flat={_med(flat):.4f}')
        overall.extend(pol)

    med = float(np.median(overall)) if overall else float('nan')
    log_line(f'[bp-eval] epoch={epoch} OVERALL policy median splice F1 = {med:.4f} (n={len(overall)})')
    return med


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()
    _seed_everything(args.seed)

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    # ── Frozen detector ────────────────────────────────────────────────────────
    log_line(f'[bp] loading frozen detector: {args.init_checkpoint}')
    model, cfg, res = load_eval_model(args.init_checkpoint, device=device, strict=False)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    if getattr(model, 'contrastive_proj', None) is None:
        raise RuntimeError('train_box_policy: detector has no contrastive head — embeddings required.')

    decode_fn, decoder_name = _resolve_decoder(args.decoder)

    # ── Data ───────────────────────────────────────────────────────────────────
    train_by_source = _collect_splices(args, _TRAIN_SOURCES, res, split='train')
    train_splices = [it for items in train_by_source.values() for it in items]
    if not train_splices:
        raise RuntimeError('train_box_policy: no train splices found — check --sagid_root / --casia_root.')

    eval_full = _collect_splices(args, _EVAL_SOURCES, res, split='val')
    eval_by_source = {
        src: deterministic_subsample(items, args.eval_per_source, seed=f'bp_eval:{src}')
        for src, items in eval_full.items()
    }
    log_line(f'[bp] train splices={len(train_splices)}  '
             f'eval={[(s, len(v)) for s, v in eval_by_source.items()]}')

    # ── Policy (size in_dim from a real signal) ────────────────────────────────
    use_attn = not args.no_attn_channel
    use_patch_logit = not args.no_patch_logit_channel
    probe_t = load_image_tensor(train_splices[0], res, device=device)
    probe_info = model_info(model, probe_t, device=device, amp=use_amp, amp_dtype=args.amp_dtype)
    in_dim = policy_input_dim(probe_info, use_attn=use_attn, use_patch_logit=use_patch_logit)
    log_line(f'[bp] policy in_dim={in_dim} (use_attn={use_attn} use_patch_logit={use_patch_logit})')

    policy = build_box_policy(
        in_dim, device=device, width=args.width, depth=args.depth, n_heads=args.n_heads,
        size_init=args.size_init, keep_bias_init=args.keep_bias_init,
        size_min_frac=args.size_min_frac,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'run_config.json').write_text(json.dumps(vars(args), indent=2, default=str))

    baseline_cache: Dict[str, float] = {}   # attention-zoom F1 per item (static)
    flat_cache: Dict[str, float] = {}       # flat-decode F1 per item (static)

    single_kwargs = dict(
        use_attn=use_attn, use_patch_logit=use_patch_logit, attn_percentile=args.attn_percentile,
        cand_cap=args.cand_cap, max_boxes=args.max_boxes,
        min_crop_frac=args.min_crop_frac,
    )
    train_kwargs = dict(
        decode_fn=decode_fn, decoder_name=decoder_name,
        baselines=baseline_cache, baseline_mode=args.baseline, flat_cache=flat_cache,
        use_attn=use_attn, use_patch_logit=use_patch_logit, attn_percentile=args.attn_percentile,
        cand_cap=args.cand_cap, max_boxes=args.max_boxes, entropy_beta=args.entropy_beta,
        box_cost=args.box_cost, credit_mode=args.credit, min_crop_frac=args.min_crop_frac,
        use_amp=use_amp, amp_dtype=args.amp_dtype,
    )

    best_metric = -1.0
    for epoch in range(args.num_epochs):
        # ── epoch sample: shuffle (seeded by epoch) then cap ───────────────────
        rng = random.Random(args.seed + epoch)
        order = list(train_splices)
        rng.shuffle(order)
        epoch_items = order[:args.train_per_epoch]

        policy.train()
        optimizer.zero_grad(set_to_none=True)
        run_R, run_adv, run_boxes, run_sampled = [], [], [], []
        n_used, n_in_accum = 0, 0
        for i, item in enumerate(epoch_items):
            try:
                out = policy_train_item(model, policy, item, res, device=device, **train_kwargs)
            except Exception as exc:
                log_line(f'[bp] WARN: train item {item.item_id} failed: {exc}')
                out = None
            if out is None:
                continue
            loss, stats = out
            (loss / args.grad_accum).backward()
            n_in_accum += 1
            n_used += 1
            run_R.append(stats.reward); run_adv.append(stats.advantage)
            run_boxes.append(stats.n_boxes); run_sampled.append(stats.n_sampled)

            if n_in_accum == args.grad_accum:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_in_accum = 0

            if (i + 1) % args.log_every == 0:
                log_line(f'[bp] epoch={epoch} {i + 1}/{len(epoch_items)} '
                         f'reward~{np.mean(run_R[-args.log_every:]):.3f} '
                         f'adv~{np.mean(run_adv[-args.log_every:]):+.3f} '
                         f'boxes~{np.mean(run_boxes[-args.log_every:]):.2f} '
                         f'sampled~{np.mean(run_sampled[-args.log_every:]):.2f}')

        if n_in_accum > 0:   # flush trailing partial accumulation
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        log_line(f'[bp] epoch={epoch} done  n_used={n_used}  '
                 f'mean_reward={np.mean(run_R) if run_R else float("nan"):.4f}  '
                 f'mean_adv={np.mean(run_adv) if run_adv else float("nan"):+.4f}  '
                 f'mean_boxes={np.mean(run_boxes) if run_boxes else float("nan"):.2f}  '
                 f'mean_sampled={np.mean(run_sampled) if run_sampled else float("nan"):.2f}')

        # ── eval + viz ─────────────────────────────────────────────────────────
        med = evaluate(
            model, policy, eval_by_source, res, device=device, decode_fn=decode_fn,
            decoder_name=decoder_name, use_amp=use_amp, amp_dtype=args.amp_dtype,
            baseline_cache=baseline_cache, flat_cache=flat_cache,
            viz_n=args.viz_n, viz_dir=run_dir / 'viz' / f'epoch_{epoch:04d}',
            epoch=epoch, single_kwargs=single_kwargs,
        )

        # ── checkpoint ───────────────────────────────────────────────────────────
        state = {
            'epoch': epoch,
            'policy': policy.state_dict(),
            'in_dim': in_dim,
            'best_metric': max(med, best_metric),
            'cfg': vars(args),
            'meta': {'recipe': 'box_policy', 'decoder': decoder_name,
                     'init_checkpoint': args.init_checkpoint,
                     'use_attn': use_attn, 'use_patch_logit': use_patch_logit},
        }
        save_ckpt(state, str(run_dir / f'epoch_{epoch:04d}.pt'), is_main=True)
        if med >= best_metric:
            best_metric = med
            save_ckpt(state, str(run_dir / 'best.pt'), is_main=True)
            log_line(f'[bp] best policy saved  median_splice_f1={best_metric:.4f}')

    log_line(f'[bp] done  best median splice F1 = {best_metric:.4f}')


if __name__ == '__main__':
    main()
