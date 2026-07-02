"""experiments.scripts.eval_laundering — entry point for super-resolution laundering evaluations.

Preserves all default evaluation behaviors (otsu threshold, decoders, zoom, device)
by loading the general parser from `lab_utils.eval.numbers` and running the evaluation.

Usage:
    python -m experiments.scripts.eval_laundering \
        --checkpoint /runs/tgif_fr/best.pt \
        --tgif2_root /media/ssd/DINO_SCOPE_DATA/tgif2_flux \
        --launder_mode bicubic_x2 \
        --decoder kmeans
"""

import argparse
import sys
import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from lab_utils.eval.numbers import build_parser, run


def main() -> None:
    wp = argparse.ArgumentParser(
        prog='eval_laundering',
        description='Evaluate a DINO_SCOPE_final checkpoint under upscaling-downsampling laundering attacks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    wp.add_argument('--checkpoint', nargs='+', required=True,
                    help='One or more checkpoints to evaluate (first is reference).')
    wp.add_argument('--label', nargs='+', default=None,
                    help='Optional labels for the checkpoints.')
    wp.add_argument('--tgif2_root', required=True,
                    help='Path to the tgif2_flux dataset directory.')
    wp.add_argument('--launder_mode', required=True,
                    choices=['none', 'bicubic_x2', 'bicubic_x4', 'real_esrgan_x2', 'real_esrgan_x4'],
                    help='The laundering mode to evaluate.')
    wp.add_argument('--prelaundered_root', default=None,
                    help='Root path to pre-processed Real-ESRGAN/SR images.')
    wp.add_argument('--decoders', nargs='+', default=['kmeans', 'hdbscan'],
                    help='Decoders to run.')
    wp.add_argument('--no_zoom', action='store_true',
                    help='Disable attention-guided zooming.')
    wp.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    wp.add_argument('--max_items', type=int, default=None,
                    help='Limit number of items processed per cell (smoke test).')
    wp.add_argument('--out_json', '--summary_out', dest='out_json', default=None,
                    help='Optional output path for results JSON. '
                         '(--summary_out is an alias so the sweep orchestrator, which '
                         'auto-injects --summary_out for eval modules, can drive this.)')

    a = wp.parse_args()

    # Reconstruct arguments to match the general numbers evaluator parser
    argv = [
        '--checkpoint', *a.checkpoint,
        '--tgif2_root', a.tgif2_root,
        '--launder_mode', a.launder_mode,
        '--decoders', *a.decoders,
        '--device', a.device,
    ]
    if a.label:
        argv += ['--label', *a.label]
    if a.prelaundered_root:
        argv += ['--prelaundered_root', a.prelaundered_root]
    if a.no_zoom:
        argv += ['--no-zoom']
    if a.out_json:
        argv += ['--out_json', a.out_json]
    if a.max_items is not None:
        argv += ['--max_items', str(a.max_items)]

    # Parse using the canonical eval_numbers parser and run the evaluation
    run(build_parser().parse_args(argv))


if __name__ == '__main__':
    main()
