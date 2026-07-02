#!/usr/bin/env bash
# OPTIMAL headline run on Colab (L4): ViT-H+/16 @ 688, LoRA rank 16 / alpha 32,
# all layers, trained to best.pt. This is the deliverable model that combines the
# rank-16 peak with the 688 resolution top — NOT an ablation cell (no epoch-4 cap).
#
# 688 trains at batch 1 / grad_accum 8 (eff. batch 8), the same memory profile the
# resolution sweep used for 688. amp fp16 is pinned in the queue (L4 is bf16-capable
# but we keep the fp16 regime constant). Drive-rooted so it survives disconnect and
# auto-resumes from the latest epoch.
#
# Usage (L4 Colab runtime, repo cwd):
#   PY=python bash run_scripts/run_optimal_colab.sh
#   PY=python bash run_scripts/run_optimal_colab.sh --dry_run   # preview the command
set -euo pipefail
cd "$(dirname "$0")/.."                          # repo root
REPO="$(pwd)"
PY=${PY:-python}

RUNS="${RUNS_ROOT:-/content/drive/MyDrive/DINO_SCOPE_RUNS/optimal}"
mkdir -p "$RUNS"

echo "[optimal] ViT-H+/16 @ 688 r16  ->  run_root=$RUNS  (Drive — persists + auto-resumes)"
"$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_optimal_688_r16.json \
  --run_root "$RUNS" --cwd "$REPO" "$@"
echo "[optimal] done. best.pt -> $RUNS/optimal_h16plus_688_r16/best.pt"
