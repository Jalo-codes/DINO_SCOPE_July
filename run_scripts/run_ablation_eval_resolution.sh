#!/usr/bin/env bash
# Full eval of every resolution ablation best.pt — zoom (two-pass), 500/source
# in-domain + 500/cell on tgif2. Run on ONE gpu. Resume-safe.
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=1 bash run_scripts/run_ablation_eval_resolution.sh            # run on GPU 1
#   GPU=1 bash run_scripts/run_ablation_eval_resolution.sh --dry_run  # preview
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/media/ssd/runs/ablation_eval}"
GPU="${GPU:-1}"
mkdir -p "$RUNS"
cd "$REPO"

echo "[ablation_eval] resolution -> GPU $GPU  run_root=$RUNS/res_sweep"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_ablation_eval_resolution.json \
  --run_root "$RUNS/res_sweep" --cwd "$REPO" "$@"
echo "[ablation_eval] done. summary: $RUNS/res_sweep/sweep_summary.csv"
