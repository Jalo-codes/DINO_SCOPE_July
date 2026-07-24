#!/usr/bin/env bash
# run_bce_calib_sweep_box.sh — adaptive-vs-fixed calibration campaign on the
# six-condition bce_emergence checkpoints (the three bce_* have a patch head;
# cont_* do not and are out of scope — no logit axis).
#
# Per condition it produces, in ONE neat place per checkpoint:
#   results/bce_emergence/<cond>/probe_calib/
#       threshold_records.csv     thr@0.5 (fixed production)
#       kmeans_logit_records.csv  otsu (adaptive per-image cut on the logits)
#       kmeans_feats_records.csv  whole 1280-d vector (reference)
#       sweep/sweep_records.csv   fixed-threshold curve -> oracle global-t
#       otsu_vs_threshold.{csv,txt}  the collapsed decision table (the deliverable)
#       log_bench_cached.txt / log_bench_feats.txt / log_sweep.txt  (traceability;
#           each carries the [ckpt] identity line: exact checkpoint + arch + epoch)
#
# Decode polarity is LOGIT-based (BCE head defines fakeness) — see commit e994a2d.
# 2080 box is Turing -> fp16 only (never bf16). Full region-probe pools (no cap),
# matching the canonical probe_eval2, so numbers are apples-to-apples with it.
#
# Usage (from repo root):
#   ./run_scripts/run_bce_calib_sweep_box.sh                 # all three bce_*
#   ./run_scripts/run_bce_calib_sweep_box.sh bce_splice_s0   # one condition
#   PY=~/dino_venv/bin/python ./run_scripts/run_bce_calib_sweep_box.sh
set -euo pipefail

PY=${PY:-python}
RUNS_ROOT=${RUNS_ROOT:-/media/ssd/runs/bce_emergence}
DATA=${DATA:-/media/ssd/DINO_SCOPE_DATA}
RESULTS=${RESULTS:-results/bce_emergence}
EPOCH=${EPOCH:-epoch_0005}
AMP=${AMP:-float16}

CONDS=("$@")
if [ ${#CONDS[@]} -eq 0 ]; then
  CONDS=(bce_both_s0 bce_inpaint_s0 bce_splice_s0)
fi

# Full region-probe root set — identical flags to the canonical probe_eval2 run.
ROOTS=(
  --ai_interior_root      "$DATA/SAGI_D"
  --ai_boundary_root      "$DATA/SAGI_D"
  --real_crop_root        "$DATA/SAGI_D"
  --sp_interior_root      "$DATA/IMD2020"
  --sp_boundary_root      "$DATA/IMD2020"
  --fr_bg_matched_root    "$DATA/content/flux_originals"
  --ai_interior_tgif_root "$DATA/content/flux_originals"
  --ai_boundary_tgif_root "$DATA/content/flux_originals"
  --real_crop_tgif_root   "$DATA/content/flux_originals"
)

for COND in "${CONDS[@]}"; do
  CKPT="$RUNS_ROOT/$COND/$EPOCH.pt"
  CACHE="$RUNS_ROOT/$COND/probe_calib_cache"
  OUT="$RESULTS/$COND/probe_calib"
  echo "=================================================================="
  echo "[calib] $COND  ckpt=$CKPT"
  echo "=================================================================="
  if [ ! -f "$CKPT" ]; then
    echo "[calib] MISSING checkpoint: $CKPT — skipping $COND" >&2
    continue
  fi
  mkdir -p "$OUT/sweep"

  # 1) build fresh cache + decode threshold & kmeans_logit from it (0 extra fwd)
  $PY -m experiments.scripts.eval \
    --checkpoint "$CKPT" --decoder threshold kmeans_logit --bench \
    --amp_dtype "$AMP" --cache_dir "$CACHE" --overwrite_cache \
    "${ROOTS[@]}" --out_dir "$OUT" 2>&1 | tee "$OUT/log_bench_cached.txt"

  # 2) kmeans_feats — needs raw patch_feats, so a fresh forward (no cache)
  $PY -m experiments.scripts.eval \
    --checkpoint "$CKPT" --decoder kmeans_feats \
    --amp_dtype "$AMP" \
    "${ROOTS[@]}" --out_dir "$OUT" 2>&1 | tee "$OUT/log_bench_feats.txt"

  # 3) fixed-threshold sweep over the same cache -> oracle global-t
  $PY -m experiments.scripts.eval_threshold_sweep \
    --cache_dir "$CACHE" --out_dir "$OUT/sweep" \
    "${ROOTS[@]}" 2>&1 | tee "$OUT/log_sweep.txt"

  # 4) collapse to the decision table
  $PY -m analysis.otsu_vs_threshold \
    --eval_dir "$OUT" --sweep_dir "$OUT/sweep" | tee "$OUT/otsu_vs_threshold.txt"

  echo "[calib] $COND done -> $OUT/otsu_vs_threshold.csv"
done

echo "[calib] all conditions complete."
