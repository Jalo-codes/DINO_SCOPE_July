#!/usr/bin/env bash
# hdbscan NO-ZOOM eval — res_1024 epoch_0004, imd + indomain + tgif. GPU 0.
# Run this in screen/tmux pane 0. Pair with gpu1 script (zoom).
# Usage:
#   export PY=/home/studentresearch2/dino_venv/bin/python
#   bash run_scripts/run_1024_hdbscan_nozoom_eval_ada_gpu0.sh
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/studentresearch2/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/home/studentresearch2/runs/ablation_eval}"
OUT="$RUNS/res_1024_hdbscan_nozoom_e4"
mkdir -p "$OUT"
cd "$REPO"

echo "[eval] res_1024 hdbscan nozoom -> GPU 0  run_root=$OUT"
CUDA_VISIBLE_DEVICES="0" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_r16_eval_1024_hdbscan_nozoom_ada_gpu0.json \
  --run_root "$OUT" --cwd "$REPO" "$@"
echo "[eval] done. results in $OUT"
