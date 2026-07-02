#!/usr/bin/env bash
# Run frozen backbone (no LoRA) ablation sweep on RTX 6000 Ada.
#
# Usage:
#   source run_scripts/env_ada.sh
#   GPU=0 bash run_scripts/run_ablation_frozen_backbone.sh
#
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/studentresearch2/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/home/studentresearch2/runs/ablation}"
GPU="${GPU:-0}"
mkdir -p "$RUNS"
cd "$REPO"

echo "[ablation] Running training sweep: lora_rank=0 -> GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_frozen_backbone.json \
  --run_root "$RUNS/frozen_backbone_sweep" --cwd "$REPO" "$@"

echo "[ablation] Running evaluation sweep: lora_rank=0 -> GPU $GPU"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_eval_frozen_backbone.json \
  --run_root "$RUNS/frozen_backbone_sweep" --cwd "$REPO" "$@"

echo "[ablation] Done. Results summary inside: $RUNS/frozen_backbone_sweep"
