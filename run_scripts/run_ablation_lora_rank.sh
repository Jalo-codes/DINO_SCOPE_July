#!/usr/bin/env bash
# Ablation sweep 1 — LoRA RANK sweep (fixed 448 res). Run on ONE gpu.
# Resume-safe (ORCH_DONE markers); rerun to resume.
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=0 bash run_scripts/run_ablation_lora_rank.sh            # run on GPU 0
#   GPU=0 bash run_scripts/run_ablation_lora_rank.sh --dry_run  # print commands only
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/media/ssd/runs/ablation}"
GPU="${GPU:-0}"
mkdir -p "$RUNS"
cd "$REPO"

echo "[ablation] LoRA-rank sweep -> GPU $GPU  run_root=$RUNS/lora_rank_sweep"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_ablation_lora_rank.json \
  --run_root "$RUNS/lora_rank_sweep" --cwd "$REPO" "$@"
echo "[ablation] done. summary: $RUNS/lora_rank_sweep/sweep_summary.csv"
