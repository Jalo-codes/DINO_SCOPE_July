#!/usr/bin/env bash
# Extended eval of the DINOv3 model-size sweep — epoch_0004.pt, zoom (two-pass),
# 1k tgif2/cell + 1k imd2020 per cell. Run on ONE gpu. Resume-safe (ORCH_DONE).
#
# The per-epoch val during training was only flux1dev @100/cell (enough to drive
# early-stop); this is the real cross-size comparison on the same harness as the
# rank/resolution e4 evals. Overlay against the H+/16 baseline (rank-sweep r032).
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=0 bash run_scripts/run_ablation_eval_model_size.sh            # run on GPU 0
#   GPU=0 bash run_scripts/run_ablation_eval_model_size.sh --dry_run  # preview
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${RUNS_ROOT:-/media/ssd/runs/ablation_eval}"
GPU="${GPU:-0}"
mkdir -p "$RUNS"
cd "$REPO"

echo "[ablation_eval] model-size -> GPU $GPU  run_root=$RUNS/model_size_sweep"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_ablation_eval_model_size_e4.json \
  --run_root "$RUNS/model_size_sweep" --cwd "$REPO" "$@"

echo "[ablation_eval] done. Roll up the per-cell tables with:"
echo "  $PY -m experiments.scripts.rollup_ablation_eval \\"
echo "      --run_root $RUNS/model_size_sweep --only_suffix _tgif --metric mean \\"
echo "      --out_prefix $RUNS/model_size_sweep/rollup_tgif"
echo "  $PY -m experiments.scripts.rollup_ablation_eval \\"
echo "      --run_root $RUNS/model_size_sweep --only_suffix _imd --metric mean \\"
echo "      --out_prefix $RUNS/model_size_sweep/rollup_imd"
