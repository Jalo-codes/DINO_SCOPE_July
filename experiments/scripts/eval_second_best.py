"""experiments.scripts.eval_second_best — multi-bbox / second-best zoom eval.

Runs the second-best (MIL-hide) and spatial multi-box zoom architectures over
one or more datasets and A/Bs them against the single-box baseline.  Thin CLI
over experiments.labs.multi_zoom_bench; all model/decode/metric logic lives in
lab_utils + labs (no cross-script imports, C-script).

Usage:
    python -m experiments.scripts.eval_second_best \\
        --checkpoint /runs/exp01/best.pt \\
        --casia_root /data/casia \\
        --decoder kmeans \\
        --mode all \\
        --max_items 50
"""

import argparse

import torch

from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.val_sources import add_source_root_args, collect_val_items
from lab_utils.logging.text import log_line
from lab_utils.train.distributed import unwrap_model

from experiments.labs.multi_zoom_bench import ALL_MODES, multi_zoom_bench, run_zoom_viz


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='eval_second_best',
        description='Multi-bbox / second-best zoom eval, A/B vs single-box.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint file')
    p.add_argument('--decoder', default='kmeans',
                   choices=['kmeans', 'threshold', 'hdbscan'],
                   help='Decoder used inside every zoom crop')
    p.add_argument('--mode', default='all',
                   choices=['single', 'multi', 'second_best', 'all'],
                   help='Which zoom mode(s) to run (all = the full A/B). '
                        'single = one attention bbox; multi = efficient box cover '
                        'over the attention hot set (gated); second_best = MIL '
                        'pool-peel (paused, kept working).')

    g = p.add_argument_group('dataset roots (at least one required)')
    add_source_root_args(g)

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'])

    g = p.add_argument_group('eval control')
    g.add_argument('--max_items', type=int, default=None,
                   help='Limit items evaluated per source (smoke test mode)')
    g.add_argument('--sources', nargs='*', default=None,
                   help='Restrict to these source names (default: all configured)')

    g = p.add_argument_group('single-mode crop window (vanilla mask finding)')
    g.add_argument('--attn_min_pad_frac', type=float, default=0.06,
                   help='Floor on per-side crop padding fraction so the margin '
                        'does not collapse to ~0 on medium/large boxes. 0 = legacy.')

    g = p.add_argument_group('visualisation')
    g.add_argument('--out_dir', default=None,
                   help='Save per-item zoom figures here (PNG). Enables viz.')
    g.add_argument('--viz_n', type=int, default=0,
                   help='Render the first N non-real items. >0 switches to a '
                        'single-mode viz run instead of the multi-mode A/B.')
    g.add_argument('--show', action='store_true',
                   help='Display figures inline in an IPython/Colab kernel or a '
                        'graphics terminal (iTerm2/kitty/chafa). Under a plain '
                        '"!python" subprocess in Colab there is no display channel '
                        '— use --out_dir, or call run_zoom_viz from a notebook cell.')
    return p


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()
    device = torch.device(
        args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    )
    use_amp = (not args.no_amp) and (device.type == 'cuda')

    log_line(f'[eval] loading checkpoint: {args.checkpoint}')
    model, _cfg, res = load_eval_model(args.checkpoint, device=device, strict=False)
    bare_model = unwrap_model(model)

    all_items = collect_val_items(args, res)
    if not all_items:
        raise RuntimeError(
            'eval_second_best.py: no dataset roots configured or found. '
            'Pass at least one of --imd2020_root, --casia_root, etc.'
        )

    # Crop-window tuning for the vanilla 'single' finder (ignored by other modes).
    single_zoom_kwargs = {'attn_min_pad_frac': args.attn_min_pad_frac}

    viz = (args.viz_n > 0) or args.show or (args.out_dir is not None)
    if viz:
        # Viz runs one mode (with its numbers).  'all' isn't a single mode, so
        # default the viz to the active multi-window finder.
        viz_mode = 'multi' if args.mode == 'all' else args.mode
        if args.mode == 'all':
            log_line("[eval] viz requested with --mode all → visualising 'multi' "
                     '(run without viz flags for the full A/B table)')
        run_zoom_viz(
            bare_model, all_items, res,
            mode=viz_mode, device=device, decoder=args.decoder,
            out_dir=args.out_dir, viz_n=max(args.viz_n, 1) if args.show else args.viz_n,
            show=args.show, use_amp=use_amp, amp_dtype=args.amp_dtype,
            single_zoom_kwargs=single_zoom_kwargs,
        )
    else:
        modes = ALL_MODES if args.mode == 'all' else (args.mode,)
        multi_zoom_bench(
            bare_model, all_items, res,
            device=device, decoder=args.decoder, modes=modes,
            use_amp=use_amp, amp_dtype=args.amp_dtype,
            single_zoom_kwargs=single_zoom_kwargs,
        )


if __name__ == '__main__':
    main()
