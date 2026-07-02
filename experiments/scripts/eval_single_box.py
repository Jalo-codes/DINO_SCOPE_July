"""experiments.scripts.eval_single_box — read-off sweep over a TRAINED head.

The head is frozen here; NOTHING trains.  This is the "cheap as shit test": every
read-off knob (threshold, padding, box count, squaring) is an EVAL-time decision,
so a trained ``best.pt`` can be re-scored across a grid of them WITHOUT retraining.

    best.pt  ──BoxHeatmap (frozen)──▶ heatmap   (re-thresholded per sweep point)
        ──read-off (thresh / pad / max_regions / …)──▶ boxes ──zoom──▶ F1

Because flat / attention-zoom / hdbscan references depend ONLY on the frozen
detector (not on the head read-off), they are computed ONCE and shared across all
sweep points — only the policy column is recomputed per combo.

Sweepable axes accept comma lists; their cartesian product is evaluated:
    --thresh 0.4,0.5,0.6   --max_regions 3,5   --readoff_pad_frac 0.04,0.08

Usage:
    python -m experiments.scripts.eval_single_box \\
        --head_checkpoint /runs/singlebox09/best.pt \\
        --init_checkpoint /runs/base/epoch_004.pt \\
        --sagid_root /data/sagid --casia_root /data/casia --imd2020_root /data/imd \\
        --thresh 0.4,0.5,0.6
"""

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import torch

from lab_utils.data.sampling import deterministic_subsample
from lab_utils.eval.load_model import load_eval_model
from lab_utils.logging.text import log_line
from lab_utils.model.box_heatmap import build_box_heatmap
from lab_utils.train.checkpoint import load as load_ckpt

from experiments.labs.attention_zoom import _resolve_decoder
from experiments.labs.box_heatmap_lab import (
    _EVAL_SOURCES,
    _SOURCE_ROOT,
    collect_splices,
    evaluate,
    seed_everything,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _floats(s):
    return [float(x) for x in str(s).split(',') if x != '']


def _ints(s):
    return [int(x) for x in str(s).split(',') if x != '']


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='eval_single_box',
        description='Read-off sweep over a trained single-box heatmap head (no training).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--head_checkpoint', required=True,
                   help='Trained BoxHeatmap checkpoint (e.g. best.pt).')
    p.add_argument('--init_checkpoint', default=None,
                   help='Frozen detector checkpoint. Default: the one recorded in the '
                        'head checkpoint meta (override when paths differ across machines).')

    g = p.add_argument_group('dataset roots')
    for attr in sorted(set(_SOURCE_ROOT.values())):
        g.add_argument(f'--{attr}', default=None)

    g = p.add_argument_group('decode')
    g.add_argument('--decoder', default=None, choices=['kmeans', 'hdbscan'],
                   help='Frozen decoder. Default: the decoder the head trained against.')
    g.add_argument('--min_crop_frac', type=float, default=None)

    # Sweep axes — comma lists; the cartesian product is evaluated.  Default None
    # ⇒ use the single value the head trained with (from the checkpoint cfg).
    g = p.add_argument_group('read-off sweep axes (comma lists)')
    g.add_argument('--thresh', type=_floats, default=None,
                   help='Heatmap thresholds. Higher ⇒ tighter/smaller ON set.')
    g.add_argument('--max_regions', type=_ints, default=None,
                   help='Box caps per image. Higher ⇒ more boxes allowed.')
    g.add_argument('--readoff_pad_frac', type=_floats, default=None,
                   help='Read-off padding fractions. Lower ⇒ tighter crops.')
    g.add_argument('--readoff_min_box_size', type=_ints, default=None,
                   help='Min read-off box size (patches/side). Lower ⇒ tighter on tiny splices.')
    g.add_argument('--square_cap', type=_floats, default=None,
                   help='Max aspect ratio at read-off (1.0 = fully square).')

    # Fixed read-off knobs — single value; default None ⇒ inherit from checkpoint cfg.
    g = p.add_argument_group('read-off (fixed)')
    g.add_argument('--min_patches', type=int, default=None)
    g.add_argument('--dilate', type=int, default=None)
    g.add_argument('--readoff_min_pad_frac', type=float, default=None)
    g.add_argument('--overlap_kill_frac', type=float, default=None)
    g.add_argument('--large_area_frac', type=float, default=None)
    g.add_argument('--gate_margin', type=float, default=None)
    g.add_argument('--no_gate_logit', action='store_true',
                   help='Disable the MIL logit gate (default: inherit the trained setting).')
    g.add_argument('--patch_frac', type=float, default=None,
                   help='GT-grid patch threshold (used by viz/green target only).')

    g = p.add_argument_group('eval + viz')
    g.add_argument('--eval_per_source', type=int, default=150)
    g.add_argument('--with_hdbscan', action='store_true')
    g.add_argument('--viz_per_source', type=int, default=0,
                   help='Viz saved PER source, written for the BEST sweep point only '
                        '(0 = none). Sweeps stay viz-free to keep them cheap.')
    g.add_argument('--run_dir', default=None,
                   help='Where to write viz (required if --viz_per_source > 0).')

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default=None, choices=['float16', 'bfloat16'])
    g.add_argument('--seed', type=int, default=42)
    return p


# ── head reconstruction ──────────────────────────────────────────────────────────

def _load_head(path: str, device: torch.device):
    """Rebuild the BoxHeatmap from a training checkpoint and load its weights."""
    state = load_ckpt(path, map_location=str(device))
    cfg = state.get('cfg', {})
    meta = state.get('meta', {})
    in_dim = state['in_dim']
    head = build_box_heatmap(
        in_dim, device=device,
        width=cfg.get('width', 128), depth=cfg.get('depth', 2),
        n_heads=cfg.get('n_heads', 4), bias_init=cfg.get('bias_init', -2.0),
        dropout=cfg.get('dropout', 0.0),
    )
    head.load_state_dict(state['head'])
    head.eval()
    log_line(f'[sb-eval] head in_dim={in_dim} '
             f'(use_attn={meta.get("use_attn")} use_patch_logit={meta.get("use_patch_logit")})')
    return head, cfg, meta


# ── main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    use_amp = (not args.no_amp) and device.type == 'cuda'

    head, cfg, meta = _load_head(args.head_checkpoint, device)
    use_attn = bool(meta.get('use_attn', True))
    use_patch_logit = bool(meta.get('use_patch_logit', True))
    amp_dtype = args.amp_dtype or cfg.get('amp_dtype', 'float16')

    # Inherit any unspecified knob from the trained config so a bare invocation
    # reproduces the run; CLI flags override per-axis.
    def fixed(name, default):
        v = getattr(args, name)
        return v if v is not None else cfg.get(name, default)

    def axis(name, default):
        v = getattr(args, name)
        return v if v is not None else [cfg.get(name, default)]

    decoder = args.decoder or meta.get('decoder') or cfg.get('decoder', 'kmeans')
    decode_fn, decoder_name = _resolve_decoder(decoder)

    init_ckpt = args.init_checkpoint or meta.get('init_checkpoint')
    if not init_ckpt:
        raise RuntimeError('eval_single_box: no detector checkpoint — pass --init_checkpoint '
                           '(none recorded in the head checkpoint meta).')
    log_line(f'[sb-eval] loading frozen detector: {init_ckpt}')
    model, _cfg, res = load_eval_model(init_ckpt, device=device, strict=False)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    if args.with_hdbscan:
        from lab_utils.eval.decode.hdbscan import hdbscan_available
        if not hdbscan_available():
            log_line('[sb-eval] WARN: --with_hdbscan set but no HDBSCAN backend; disabling.')
            args.with_hdbscan = False

    eval_full = collect_splices(args, _EVAL_SOURCES, res, split='val')
    eval_by_source = {
        src: deterministic_subsample(items, args.eval_per_source, seed=f'sb_eval:{src}')
        for src, items in eval_full.items()
    }
    if not eval_by_source:
        raise RuntimeError('eval_single_box: no eval splices — check the dataset roots.')
    log_line(f'[sb-eval] eval={[(s, len(v)) for s, v in eval_by_source.items()]}')

    patch_frac = fixed('patch_frac', 0.25)
    # Fixed read-off knobs shared by every sweep point.
    base_single = dict(
        use_attn=use_attn, use_patch_logit=use_patch_logit,
        min_patches=fixed('min_patches', 2), dilate=fixed('dilate', 1),
        readoff_min_pad_frac=fixed('readoff_min_pad_frac', 0.0),
        overlap_kill_frac=fixed('overlap_kill_frac', 0.30),
        large_area_frac=fixed('large_area_frac', 0.6),
        # gate off if explicitly requested, else inherit the trained setting.
        gate_logit=False if args.no_gate_logit else (not cfg.get('no_gate_logit', False)),
        gate_margin=fixed('gate_margin', 0.0),
        min_crop_frac=fixed('min_crop_frac', 0.25),
    )

    # Sweep grid (cartesian product over the comma-list axes).
    thresh_v = axis('thresh', 0.5)
    maxreg_v = axis('max_regions', 3)
    pad_v = axis('readoff_pad_frac', 0.05)
    minbox_v = axis('readoff_min_box_size', 6)
    sqcap_v = axis('square_cap', 1.4)
    grid = list(itertools.product(thresh_v, maxreg_v, pad_v, minbox_v, sqcap_v))
    log_line(f'[sb-eval] sweeping {len(grid)} read-off combo(s) over a frozen head')

    # Shared reference caches — flat / attn / hdbscan depend ONLY on the frozen
    # detector, so compute them once and reuse across every sweep point.
    flat_cache: Dict[str, float] = {}
    attn_cache: Dict[str, float] = {}
    hdb_cache: Dict[str, float] = {}

    results: List = []
    for i, (thr, maxreg, pad, minbox, sqcap) in enumerate(grid):
        single_kwargs = dict(
            base_single, thresh=thr, max_regions=maxreg, readoff_pad_frac=pad,
            readoff_min_box_size=minbox, square_cap=sqcap,
        )
        log_line(f'[sb-eval] === combo {i + 1}/{len(grid)}: thresh={thr} max_regions={maxreg} '
                 f'pad={pad} min_box={minbox} square_cap={sqcap} ===')
        med = evaluate(
            model, head, eval_by_source, res, device=device, decode_fn=decode_fn,
            decoder_name=decoder_name, use_amp=use_amp, amp_dtype=amp_dtype,
            flat_cache=flat_cache, attn_cache=attn_cache, hdb_cache=hdb_cache,
            with_hdbscan=args.with_hdbscan, viz_per_source=0, viz_dir=None, epoch=i,
            single_kwargs=single_kwargs, patch_frac=patch_frac, max_regions=maxreg,
            readoff_pad_frac=pad,
        )
        results.append((med, dict(thresh=thr, max_regions=maxreg, readoff_pad_frac=pad,
                                  readoff_min_box_size=minbox, square_cap=sqcap)))

    results.sort(key=lambda r: r[0], reverse=True)
    log_line('[sb-eval] ── sweep summary (overall median policy F1, best first) ──')
    for med, combo in results:
        log_line(f'[sb-eval]   med={med:.4f}  {combo}')

    # Optional: viz the best combo only (kept out of the sweep loop to stay cheap).
    if args.viz_per_source > 0:
        if not args.run_dir:
            raise RuntimeError('eval_single_box: --viz_per_source > 0 needs --run_dir.')
        best_med, best = results[0]
        log_line(f'[sb-eval] writing viz for best combo {best} (med={best_med:.4f})')
        single_kwargs = dict(
            base_single, thresh=best['thresh'], max_regions=best['max_regions'],
            readoff_pad_frac=best['readoff_pad_frac'],
            readoff_min_box_size=best['readoff_min_box_size'], square_cap=best['square_cap'],
        )
        evaluate(
            model, head, eval_by_source, res, device=device, decode_fn=decode_fn,
            decoder_name=decoder_name, use_amp=use_amp, amp_dtype=amp_dtype,
            flat_cache=flat_cache, attn_cache=attn_cache, hdb_cache=hdb_cache,
            with_hdbscan=args.with_hdbscan, viz_per_source=args.viz_per_source,
            viz_dir=Path(args.run_dir) / 'viz_best', epoch=0,
            single_kwargs=single_kwargs, patch_frac=patch_frac,
            max_regions=best['max_regions'], readoff_pad_frac=best['readoff_pad_frac'],
        )

    log_line('[sb-eval] done')


if __name__ == '__main__':
    main()
