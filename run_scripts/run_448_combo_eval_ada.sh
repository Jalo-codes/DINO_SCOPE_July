#!/usr/bin/env bash
# Combo eval of res_448 epoch_0004 on tgif + imd + indomain: no-zoom kmeans, no-zoom hdbscan, zoom hdbscan.
# Usage:
#   export PY=/home/studentresearch2/dino_venv/bin/python
#   GPU=1 bash run_scripts/run_448_combo_eval_ada.sh
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/studentresearch2/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/home/studentresearch2/runs/ablation_eval}"
GPU="${GPU:-1}"
OUT="$RUNS/res_448_combo_e4"
mkdir -p "$OUT"
cd "$REPO"

echo "[eval] res_448 combo (kmeans-nozoom, hdbscan-nozoom, hdbscan-zoom) -> GPU $GPU  run_root=$OUT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_r16_eval_448_combo_ada.json \
  --run_root "$OUT" --cwd "$REPO" "$@"
echo "[eval] done. results in $OUT"
