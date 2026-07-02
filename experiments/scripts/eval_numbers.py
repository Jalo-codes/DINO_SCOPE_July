"""experiments.scripts.eval_numbers — CLI for the general numerical eval.

Thin entry point; all logic lives in ``lab_utils.eval.numbers`` (so script entry
points can share it without importing each other).  Configure any subset of the
registry sources via --<source>_root; TGIF takes split kwargs to reproduce a
training run's hidden held-out split.  See that module's docstring for details.

Usage:
    python -m experiments.scripts.eval_numbers \\
        --checkpoint /runs/tgif_fr/best.pt /runs/base/epoch_004.pt \\
        --label retrain baseline \\
        --imd2020_root /data/IMD2020 \\
        --tgif2_root   /data/flux_originals \\
        --tgif_types fr \\
        --out_json /runs/tgif_fr/eval_numbers.json
"""

from lab_utils.eval.numbers import build_parser, run


def main() -> None:
    run(build_parser().parse_args())


if __name__ == '__main__':
    main()
