#!/usr/bin/env bash
# Region-probe eval (ai_interior/ai_boundary/sp_interior/sp_boundary/fr_bg/real_crop)
# against all 6 BCE-emergence cells at epoch 5 (min_epochs floor -- every cell
# is guaranteed to have this checkpoint regardless of when it early-stopped).
set -eu

PY="${PY:-$HOME/dino_venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/media/ssd/runs/bce_emergence}"
SAGID="/media/ssd/DINO_SCOPE_DATA/SAGI_D"
CASIA="/media/ssd/DINO_SCOPE_DATA/casia"
PROBE_SOURCES="ai_interior ai_boundary sp_interior sp_boundary fr_bg real_crop"
PROBE_ROOTS=(
  --ai_interior_root "$SAGID" --ai_boundary_root "$SAGID"
  --real_crop_root   "$SAGID" --fr_bg_root       "$SAGID"
  --sp_interior_root "$CASIA" --sp_boundary_root "$CASIA"
)

# 0. Manifest + render eyeball (data-only, no checkpoint -- run once).
"$PY" -m experiments.labs.probe_manifest \
  "${PROBE_ROOTS[@]}" \
  --out_csv "$RUN_ROOT/probe_manifest.csv" \
  --render_dir "$RUN_ROOT/probe_renders" --render_n 12

# 1. Per-cell eval @ epoch 5. bce_* -> threshold decoder (patch-BCE head);
#    cont_* -> kmeans decoder (contrastive embedding clustering).
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
  echo "[probe-eval] $cell (decoder=${DECODER[$cell]})"
  "$PY" -m experiments.scripts.eval \
    --checkpoint "$ckpt" \
    --decoder "${DECODER[$cell]}" \
    --sources $PROBE_SOURCES \
    "${PROBE_ROOTS[@]}" \
    --out_dir "$RUN_ROOT/$cell/probe_eval"
done

# 2. Contrasts report across all 6 cells.
"$PY" -m experiments.labs.probe_contrasts \
  --manifest "$RUN_ROOT/probe_manifest.csv" \
  --records bce_inpaint_s0="$RUN_ROOT/bce_inpaint_s0/probe_eval/threshold_records.csv" \
  --records bce_splice_s0="$RUN_ROOT/bce_splice_s0/probe_eval/threshold_records.csv" \
  --records bce_both_s0="$RUN_ROOT/bce_both_s0/probe_eval/threshold_records.csv" \
  --records cont_inpaint_s0="$RUN_ROOT/cont_inpaint_s0/probe_eval/kmeans_records.csv" \
  --records cont_splice_s0="$RUN_ROOT/cont_splice_s0/probe_eval/kmeans_records.csv" \
  --records cont_both_s0="$RUN_ROOT/cont_both_s0/probe_eval/kmeans_records.csv" \
  --out_csv "$RUN_ROOT/probe_contrasts.csv"
