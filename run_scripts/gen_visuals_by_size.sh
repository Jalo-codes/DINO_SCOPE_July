#!/usr/bin/env bash
# gen_visuals_by_size.sh — Save visual panels for IMD2020 + TGIF2 partitioned by
# splice-area bucket (tiny / small / medium / large).
# Runs from the r032 (448, rank-32) checkpoint on the 2080 Ti.
#
# Pre-buckets every item via a cheap PIL mask read, so inference only runs
# on items that will actually be saved (~4 buckets × 2 sources × MAX_PER_BUCKET).
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=1 bash run_scripts/gen_visuals_by_size.sh
#
# Overrides:
#   CKPT, GPU, OUT, MAX_PER_BUCKET, VIZ_MAX_PANEL_H
set -euo pipefail
: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
cd "$(dirname "$0")/.."   # repo root

CKPT="${CKPT:-/media/ssd/runs/ablation/lora_rank_sweep/r032/best.pt}"
GPU="${GPU:-1}"
OUT="${OUT:-results/visuals/size_buckets}"
MAX_PER_BUCKET="${MAX_PER_BUCKET:-75}"   # panels saved per (source × bucket) cell
export VIZ_MAX_PANEL_H="${VIZ_MAX_PANEL_H:-384}"   # cap panel height → smaller PNGs

DATA=/media/ssd/DINO_SCOPE_DATA
ROOTS="--imd2020_root $DATA/IMD2020 \
  --tgif2_root $DATA/content/flux_originals"

if [ ! -f "$CKPT" ]; then
  echo "[size_viz] checkpoint not found: $CKPT" >&2
  exit 1
fi

echo "[size_viz] ckpt=$CKPT  GPU=$GPU  max_per_bucket=$MAX_PER_BUCKET  panel_h<=$VIZ_MAX_PANEL_H  out=$OUT"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.gen_size_bucket_visuals \
  --checkpoint "$CKPT" \
  --out_dir "$OUT" \
  --sources imd2020 tgif2 \
  --decoder kmeans --zoom \
  --device cuda --amp_dtype float16 \
  --max_per_bucket "$MAX_PER_BUCKET" \
  --tgif_eval_per_cell 60 \
  --imd_val_split 1.0 \
  $ROOTS

echo
echo "[size_viz] done -> $OUT"
echo "  layout: $OUT/{imd2020,tgif2}/{tiny,small,medium,large}/*.png"
du -sh "$OUT" 2>/dev/null || true
echo
echo "[size_viz] commit + push:"
echo "  git add results/visuals/size_buckets && git commit -m 'visuals: size-bucket panels imd+tgif 448 r032' && git push"
