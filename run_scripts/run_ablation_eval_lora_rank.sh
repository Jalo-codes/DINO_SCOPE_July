#!/usr/bin/env bash
# Full eval of every LoRA-rank ablation best.pt — zoom (two-pass), 500/source
# in-domain + 500/cell on tgif2. Run on ONE gpu. Resume-safe.
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=0 bash run_scripts/run_ablation_eval_lora_rank.sh            # run on GPU 0
#   GPU=0 bash run_scripts/run_ablation_eval_lora_rank.sh --dry_run  # preview
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/media/ssd/runs/ablation_eval}"
GPU="${GPU:-0}"
mkdir -p "$RUNS"
cd "$REPO"

echo "[ablation_eval] LoRA-rank -> GPU $GPU  run_root=$RUNS/lora_rank_sweep"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_ablation_eval_lora_rank.json \
  --run_root "$RUNS/lora_rank_sweep" --cwd "$REPO" "$@"
echo "[ablation_eval] done. summary: $RUNS/lora_rank_sweep/sweep_summary.csv"
