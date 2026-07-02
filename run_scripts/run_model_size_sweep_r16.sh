#!/usr/bin/env bash
# Ablation sweep 3 (r16) — DINOv3 MODEL-SIZE sweep (fixed rank 16 / 448 res). Run on ONE gpu.
# Resume-safe (ORCH_DONE markers); rerun to resume.
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=0 bash run_scripts/run_model_size_sweep_r16.sh            # run on GPU 0
#   GPU=0 bash run_scripts/run_model_size_sweep_r16.sh --dry_run  # print commands only
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/media/ssd/runs/ablation}"
GPU="${GPU:-0}"
mkdir -p "$RUNS"
cd "$REPO"

echo "[ablation] model-size sweep (r16) -> GPU $GPU  run_root=$RUNS/model_size_sweep_r16"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_model_size_r16.json \
  --run_root "$RUNS/model_size_sweep_r16" --cwd "$REPO" "$@"
echo "[ablation] done. summary: $RUNS/model_size_sweep_r16/sweep_summary.csv"
