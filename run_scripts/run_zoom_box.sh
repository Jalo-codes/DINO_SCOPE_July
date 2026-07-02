#!/usr/bin/env bash
# DENSE ZOOM-BOX HEAD (per-patch box + confidence, contextual bandit) on the 2080 Ti / L4.
# Implements docs/zoom_box_spec.md: freeze a trained detector, train a ZoomBoxHead — a
# self-attention encoder + per-patch (FCOS box, confidence) heads. Phase 0 warm-starts the
# box head supervised toward GT-component boxes; phase 1 is an offline contextual bandit
# (AWR): jitter each patch's box → score frozen zoom-ADVANTAGE over baseline=max(flat,attn)
# → advantage-weight-regress toward the winners; regress confidence toward realized
# advantage. Decode: gate conf > δ → NMS by confidence (least overlap) → zoom union; else
# fall back to the better of flat / attention-zoom.
#
#   train on : casia + sagid splices, 1000 items/epoch
#   eval on  : 150 imd2020 + 150 sagid (policy vs flat/attn/baseline, conf↔advantage
#              calibration, δ-sweep, captured-advantage selection metric)
#   backbone : FROZEN. Warm-starts from the r032 (448/rank-32) detector by default.
#
# NOTE: needs an HDBSCAN backend ONLY if --decoder hdbscan (kmeans is the default and
# has no extra dep).
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=1 bash run_scripts/run_zoom_box.sh
#   INIT=/media/ssd/runs/.../best.pt GPU=1 bash run_scripts/run_zoom_box.sh
set -euo pipefail
: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
cd "$(dirname "$0")/.."

INIT="${INIT:-/media/ssd/runs/ablation/lora_rank_sweep/r032/best.pt}"
RUN_DIR="${RUN_DIR:-/media/ssd/runs/zoom_box/r032_fcos}"
GPU="${GPU:-1}"
DATA=/media/ssd/DINO_SCOPE_DATA

if [ ! -f "$INIT" ]; then echo "[zoom_box] init checkpoint not found: $INIT" >&2; exit 1; fi
echo "[zoom_box] frozen detector=$INIT  GPU=$GPU  run_dir=$RUN_DIR"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.train_zoom_box \
  --init_checkpoint "$INIT" \
  --run_dir "$RUN_DIR" \
  --casia_root "$DATA/casia" \
  --sagid_root "$DATA/SAGI_D" \
  --imd2020_root "$DATA/IMD2020" \
  --train_per_epoch 1000 \
  --eval_per_source 150 \
  --warmstart_epochs 2 \
  --device cuda --amp_dtype float16 \
  "$@"

echo "[zoom_box] done. best head -> $RUN_DIR/best.pt"
echo "[zoom_box] watch per-epoch: [zb-eval] policy vs attn/baseline, calibration corr, δ-sweep."
