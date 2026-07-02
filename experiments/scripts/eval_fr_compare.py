"""experiments.scripts.eval_fr_compare — preset: retrain vs. baseline on FR + IMD.

A thin entry point over ``experiments.scripts.eval_numbers`` that pins the
recurring comparison: the hidden TGIF2 **FR** held-out split (eval_per_cell=500,
seed='tgif_fr_half', types={fr}) plus IMD2020 val, both decoders, flat + zoom,
scored on TWO checkpoints over the identical images.

This is just a preset — it builds the general eval's args and calls into it, so
there is no duplicated eval logic.  For any other dataset / partition / split,
use experiments.scripts.eval_numbers directly.

Usage:
    python -m experiments.scripts.eval_fr_compare \\
        --retrain  /runs/tgif_fr/best.pt \\
        --baseline /runs/base/epoch_004.pt \\
        --imd2020_root /data/IMD2020 \\
        --tgif2_root   /data/flux_originals \\
        --out_json /runs/tgif_fr/fr_compare.json
"""

import argparse

from lab_utils.eval.numbers import build_parser, run


def main() -> None:
    wp = argparse.ArgumentParser(
        prog='eval_fr_compare',
        description='Compare a finetuned checkpoint against its starting checkpoint '
                    'on the hidden TGIF2-FR holdout + IMD.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    wp.add_argument('--retrain',  required=True, help='Finetuned checkpoint (reference).')
    wp.add_argument('--baseline', required=True, help='Starting checkpoint to compare against.')
    wp.add_argument('--imd2020_root', required=True)
    wp.add_argument('--tgif2_root',   required=True)
    wp.add_argument('--out_json', default=None)
    wp.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    wp.add_argument('--max_items', type=int, default=None, help='Cap per source (smoke test).')
    # Exposed so the split can track a non-default training run; defaults match FR.
    wp.add_argument('--tgif_eval_per_cell', type=int, default=500)
    wp.add_argument('--tgif_split_seed', default='tgif_fr_half')
    a = wp.parse_args()

    argv = [
        '--checkpoint', a.retrain, a.baseline,
        '--label', 'retrain', 'baseline',
        '--imd2020_root', a.imd2020_root,
        '--tgif2_root', a.tgif2_root,
        '--tgif_types', 'fr',
        '--tgif_eval_per_cell', str(a.tgif_eval_per_cell),
        '--tgif_split_seed', a.tgif_split_seed,
        '--device', a.device,
    ]
    if a.out_json:
        argv += ['--out_json', a.out_json]
    if a.max_items is not None:
        argv += ['--max_items', str(a.max_items)]

    # Reuse the general parser so every default (decoders, zoom, otsu thresh,
    # dinov3 arch override, …) is filled exactly as eval_numbers defines them.
    run(build_parser().parse_args(argv))


if __name__ == '__main__':
    main()
