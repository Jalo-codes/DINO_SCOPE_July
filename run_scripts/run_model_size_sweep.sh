#!/usr/bin/env bash
# Ablation sweep 3 — DINOv3 MODEL-SIZE sweep (fixed rank 32 / 448 res). Run on ONE gpu.
# Resume-safe (ORCH_DONE markers); rerun to resume.
#
# Trains the standard config (== rank-sweep r032) on each smaller DINOv3 ViT:
#   vits16 (21M) · vits16plus (29M) · vitb16 (86M) · vitl16 (300M)
# ViT-H+/16 (840M) is NOT here — it is the existing r032 cell, the baseline these
# overlay against. All are patch-16, so 448 stays clean (28×28=784 patches) and
# the head feat_dim auto-detects from the backbone; LoRA hits q/k/v/o_proj +
# up_proj/down_proj in every variant (gate_proj on the SwiGLU S+ stays unadapted,
# matching the H+ baseline). Smaller backbones use less memory than H+, so the
# 11 GB 2080 Ti has ample headroom at batch 8 — kept at 8 for comparability.
#
# DINOv3 weights are GATED on HF; this uses the box's existing accepted license +
# HF token (already used to pull H+), which covers the whole collection.
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=0 bash run_scripts/run_model_size_sweep.sh            # run on GPU 0
#   GPU=0 bash run_scripts/run_model_size_sweep.sh --dry_run  # print commands only
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/media/ssd/runs/ablation}"
GPU="${GPU:-0}"
mkdir -p "$RUNS"
cd "$REPO"

echo "[ablation] model-size sweep -> GPU $GPU  run_root=$RUNS/model_size_sweep"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_model_size.json \
  --run_root "$RUNS/model_size_sweep" --cwd "$REPO" "$@"
echo "[ablation] done. summary: $RUNS/model_size_sweep/sweep_summary.csv"
