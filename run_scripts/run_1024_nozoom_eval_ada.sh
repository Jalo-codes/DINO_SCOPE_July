#!/usr/bin/env bash
# Coarse (no-zoom) eval of res_1024 epoch_0004 on tgif + imd.
# Usage:
#   export PY=/home/studentresearch2/dino_venv/bin/python
#   GPU=1 bash run_scripts/run_1024_nozoom_eval_ada.sh
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/studentresearch2/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/home/studentresearch2/runs/ablation_eval}"
GPU="${GPU:-1}"
OUT="$RUNS/res_1024_nozoom_e4"
mkdir -p "$OUT"
cd "$REPO"

echo "[eval] res_1024 coarse (no zoom) -> GPU $GPU  run_root=$OUT"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_r16_eval_1024_nozoom_ada.json \
  --run_root "$OUT" --cwd "$REPO" "$@"
echo "[eval] done. results in $OUT"
