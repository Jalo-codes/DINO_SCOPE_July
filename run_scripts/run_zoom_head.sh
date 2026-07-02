#!/usr/bin/env bash
# LEARNED ZOOM HEAD (projection + per-cluster value) on the 2080 Ti.
# Implements docs/zoom_head_spec.md: freeze a trained detector, train a light
# projection (z→z' for clean HDBSCAN clusters) + a per-patch value head regressed
# to per-cluster zoom-ADVANTAGE (F1 improvement over the no-zoom baseline). Gate
# at inference: zoom a region iff predicted advantage > δ.
#
#   train on : casia + sagid splices, 1000 items/epoch
#   eval on  : 150 imd2020 + 150 sagid (policy-F1 vs flat/attn refs,
#              pred↔realized advantage calibration, δ-sweep)
#   backbone : FROZEN ⇒ cheap. Warm-starts from the r032 (448/rank-32) detector.
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=1 bash run_scripts/run_zoom_head.sh
#   INIT=/media/ssd/runs/.../best.pt GPU=1 bash run_scripts/run_zoom_head.sh
set -euo pipefail
: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
cd "$(dirname "$0")/.."

INIT="${INIT:-/media/ssd/runs/ablation/lora_rank_sweep/r032/best.pt}"
RUN_DIR="${RUN_DIR:-/media/ssd/runs/zoom_head/r032_boxhead}"
GPU="${GPU:-1}"
DATA=/media/ssd/DINO_SCOPE_DATA

if [ ! -f "$INIT" ]; then echo "[zoom_head] init checkpoint not found: $INIT" >&2; exit 1; fi
echo "[zoom_head] frozen detector=$INIT  GPU=$GPU  run_dir=$RUN_DIR"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.train_zoom_head \
  --init_checkpoint "$INIT" \
  --run_dir "$RUN_DIR" \
  --casia_root "$DATA/casia" \
  --sagid_root "$DATA/SAGI_D" \
  --imd2020_root "$DATA/IMD2020" \
  --train_per_epoch 1000 \
  --eval_per_source 150 \
  --device cuda --amp_dtype float16 \
  "$@"

echo "[zoom_head] done. best head -> $RUN_DIR/best.pt"
echo "[zoom_head] watch per-epoch: [zh-eval] policy vs flat/attn, calibration corr, δ-sweep."
