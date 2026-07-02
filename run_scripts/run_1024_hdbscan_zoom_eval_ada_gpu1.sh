#!/usr/bin/env bash
# hdbscan ZOOM eval — res_1024 epoch_0004, imd + indomain + tgif. GPU 1.
# Run this in screen/tmux pane 1. Pair with gpu0 script (nozoom).
# Usage:
#   export PY=/home/studentresearch2/dino_venv/bin/python
#   bash run_scripts/run_1024_hdbscan_zoom_eval_ada_gpu1.sh
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/studentresearch2/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/home/studentresearch2/runs/ablation_eval}"
OUT="$RUNS/res_1024_hdbscan_zoom_e4"
mkdir -p "$OUT"
cd "$REPO"

echo "[eval] res_1024 hdbscan zoom -> GPU 1  run_root=$OUT"
CUDA_VISIBLE_DEVICES="1" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_r16_eval_1024_hdbscan_zoom_ada_gpu1.json \
  --run_root "$OUT" --cwd "$REPO" "$@"
echo "[eval] done. results in $OUT"
