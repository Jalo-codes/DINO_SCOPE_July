#!/usr/bin/env bash
# Full-fakes eval (whole-image AI generations, no splice boundary) against all
# 6 BCE-emergence cells at epoch 5. Uses the OTHER gpu by default (GPU=1) so
# it can run alongside whatever's already on GPU 0 (training / probe eval).
set -eu

PY="${PY:-$HOME/dino_venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/media/ssd/runs/bce_emergence}"
FULL_FAKES="${FULL_FAKES:-/media/ssd/DINO_SCOPE_DATA/full_fakes}"
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "[full_fakes] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 1. Per-cell eval @ epoch 5. bce_* -> threshold decoder (patch-BCE head);
#    cont_* -> kmeans decoder (contrastive embedding clustering). --viz_n
#    renders a handful of predictions per cell for a quick sanity look.
declare -A DECODER=(
  [bce_inpaint_s0]=threshold  [bce_splice_s0]=threshold  [bce_both_s0]=threshold
  [cont_inpaint_s0]=kmeans    [cont_splice_s0]=kmeans    [cont_both_s0]=kmeans
)
for cell in bce_inpaint_s0 bce_splice_s0 bce_both_s0 cont_inpaint_s0 cont_splice_s0 cont_both_s0; do
  ckpt="$RUN_ROOT/$cell/epoch_0005.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[skip] $cell: no epoch_0005.pt yet"
    continue
  fi
  echo "[full_fakes-eval] $cell (decoder=${DECODER[$cell]})"
  "$PY" -m experiments.scripts.eval \
    --checkpoint "$ckpt" \
    --decoder "${DECODER[$cell]}" \
    --sources full_fakes \
    --full_fakes_root "$FULL_FAKES" \
    --out_dir "$RUN_ROOT/$cell/full_fakes_eval" \
    --summary_out "$RUN_ROOT/$cell/full_fakes_eval/summary.json" \
    --viz_n 6
done

# 2. Report: per-generator AUROC + localization distribution (predicted-
#    positive fraction), across all 6 cells.
"$PY" -m experiments.labs.full_fakes_report \
  --records bce_inpaint_s0="$RUN_ROOT/bce_inpaint_s0/full_fakes_eval/threshold_records.csv" \
  --records bce_splice_s0="$RUN_ROOT/bce_splice_s0/full_fakes_eval/threshold_records.csv" \
  --records bce_both_s0="$RUN_ROOT/bce_both_s0/full_fakes_eval/threshold_records.csv" \
  --records cont_inpaint_s0="$RUN_ROOT/cont_inpaint_s0/full_fakes_eval/kmeans_records.csv" \
  --records cont_splice_s0="$RUN_ROOT/cont_splice_s0/full_fakes_eval/kmeans_records.csv" \
  --records cont_both_s0="$RUN_ROOT/cont_both_s0/full_fakes_eval/kmeans_records.csv" \
  --out_csv "$RUN_ROOT/full_fakes_report.csv"
