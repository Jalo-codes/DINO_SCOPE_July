#!/usr/bin/env bash
# New-box environment for the 2× RTX 6000 Ada rig (binghamton-rbtg, user studentresearch2).
# Source before launching sweeps:   source run_scripts/env_ada.sh
#
# Hardware: 2× NVIDIA RTX 6000 Ada (48 GB each, sm_89, bf16-native). No sudo on this account.
# Python:   uv venv at ~/dino_venv (py3.12, torch 2.6.0+cu124, transformers 5.12.1).
#           uv venvs have no pip — use `uv pip install --python "$PY" ...`.
# Data:     active set on the Gen4 NVMe under $HOME (copied from fri-team-4's /media/ssd).
#
# Per-card sweeps (max throughput): pin ONE independent cell/queue per GPU, e.g.
#   GPU=0 bash run_scripts/run_ablation_resolution.sh
#   GPU=1 bash run_scripts/run_model_size_sweep.sh
# Reserve DDP/FSDP for a single latency-critical or oversized model (the 7B), NOT for sweeps.
#
# amp regime: the existing ablation cells pin fp16 for L4/T4 comparability — DO NOT change
# those (it would break apples-to-apples with prior runs). New Ada-only runs (7B, 700+ res)
# should use bf16 (amp_dtype=bf16), which these cards do natively.

export PY="$HOME/dino_venv/bin/python"
export DATA_ROOT="$HOME/DINO_SCOPE_DATA"          # active dataset root (Gen4 NVMe)
export RUNS_ROOT="$HOME/runs/ablation"            # checkpoints on fast NVMe; archive finished runs to /media/HD1
mkdir -p "$RUNS_ROOT"

echo "[env_ada] PY=$PY"
echo "[env_ada] DATA_ROOT=$DATA_ROOT  RUNS_ROOT=$RUNS_ROOT"
"$PY" -c "import torch; print('[env_ada] torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| gpus', torch.cuda.device_count())" 2>/dev/null \
  || echo "[env_ada] WARN: torch import failed — check the venv"
