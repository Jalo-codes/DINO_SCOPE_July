#!/usr/bin/env bash
# Region-probe eval (ai_interior/ai_boundary/sp_interior/sp_boundary/fr_bg/real_crop)
# against all 6 BCE-emergence cells at epoch 5 (min_epochs floor -- every cell
# is guaranteed to have this checkpoint regardless of when it early-stopped).
set -eu

PY="${PY:-$HOME/dino_venv/bin/python}"
RUN_ROOT="${RUN_ROOT:-/media/ssd/runs/bce_emergence}"
SAGID="/media/ssd/DINO_SCOPE_DATA/SAGI_D"
IMD="/media/ssd/DINO_SCOPE_DATA/IMD2020"
TGIF2="/media/ssd/DINO_SCOPE_DATA/content/flux_originals"
PROBE_SOURCES="ai_interior ai_boundary sp_interior sp_boundary fr_bg real_crop ai_interior_tgif ai_boundary_tgif real_crop_tgif"
# ai_*/real_crop -> sagid; sp_* -> imd2020 (~171 val fakes vs casia's ~28 --
# more shots at clearing the interior floor); fr_bg -> tgif2 restricted to
# 'fr' manipulations (registry default), a held-out OOD fr pool distinct
# from sagid's own frs. *_tgif -> a SECOND parent pool for ai_interior/
# ai_boundary/real_crop: tgif2's 'sp' manipulations (paste-back AI edits --
# 341 coco_ids x up to 3 models, a much bigger haystack than sagid's 169 val
# fakes for clearing the interior floor). Items still carry Item.source ==
# the base condition name, so these merge into the same ai_interior/
# ai_boundary/real_crop pool automatically in eval.py's records CSV.
PROBE_ROOTS=(
  --ai_interior_root "$SAGID" --ai_boundary_root "$SAGID"
  --real_crop_root   "$SAGID" --fr_bg_root       "$TGIF2"
  --sp_interior_root "$IMD"   --sp_boundary_root "$IMD"
  --ai_interior_tgif_root "$TGIF2" --ai_boundary_tgif_root "$TGIF2"
  --real_crop_tgif_root   "$TGIF2"
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
    --out_dir "$RUN_ROOT/$cell/probe_eval" \
    --summary_out "$RUN_ROOT/$cell/probe_eval/summary.json"
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
