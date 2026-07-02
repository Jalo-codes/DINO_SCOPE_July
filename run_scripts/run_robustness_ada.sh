#!/usr/bin/env bash
# Robustness eval on the Ada rig — 8 JPEG + 8 noise levels, no clean.
# TGIF 300/cell, IMD 1k random. Each GPU runs in its own named screen session.
#
# Usage (from repo root on the Ada box):
#   source run_scripts/env_ada.sh
#   bash run_scripts/run_robustness_ada.sh [--dry_run]
#
# Monitor:
#   screen -r robust_jpeg    # GPU 0 — JPEG half
#   screen -r robust_noise   # GPU 1 — noise half
#   screen -ls               # list both sessions
#
# Outputs land under $RUNS_ROOT/robustness_ada/{jpeg,noise}/{tgif_jpeg_a,...}/
set -euo pipefail

: "${PY:?source run_scripts/env_ada.sh first}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-$HOME/runs/ablation}/robustness_ada"
mkdir -p "$RUNS"

echo "[robust_ada] sending jobs to existing screens gpu0 / gpu1..."
echo "  GPU 0  screen gpu0  →  $RUNS/jpeg"
echo "  GPU 1  screen gpu1  →  $RUNS/noise"

screen -S gpu0 -X stuff "cd '$REPO' && CUDA_VISIBLE_DEVICES=0 '$PY' -m experiments.scripts.orchestrate --queue sweeps/sweep_robustness_ada_e4_gpu0.json --run_root '$RUNS/jpeg' --cwd '$REPO' $*\n"

screen -S gpu1 -X stuff "cd '$REPO' && CUDA_VISIBLE_DEVICES=1 '$PY' -m experiments.scripts.orchestrate --queue sweeps/sweep_robustness_ada_e4_gpu1.json --run_root '$RUNS/noise' --cwd '$REPO' $*\n"

echo ""
echo "Jobs sent to existing screens. Attach with:"
echo "  screen -r gpu0"
echo "  screen -r gpu1"
echo ""
echo "Detach with Ctrl+A D"
