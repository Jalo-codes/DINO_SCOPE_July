#!/usr/bin/env bash
# Ablation sweep 2 — RESOLUTION sweep (fixed rank 32). Run on ONE gpu.
# Resume-safe (ORCH_DONE markers); rerun to resume.
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=1 bash run_scripts/run_ablation_resolution.sh            # run on GPU 1
#   GPU=1 bash run_scripts/run_ablation_resolution.sh --dry_run  # print commands only
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/media/ssd/runs/ablation}"
GPU="${GPU:-1}"
mkdir -p "$RUNS"
cd "$REPO"

echo "[ablation] resolution sweep -> GPU $GPU  run_root=$RUNS/res_sweep"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_ablation_resolution.json \
  --run_root "$RUNS/res_sweep" --cwd "$REPO" "$@"
echo "[ablation] done. summary: $RUNS/res_sweep/sweep_summary.csv"
