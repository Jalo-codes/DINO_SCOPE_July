#!/usr/bin/env bash
# run_bce_emergence_noise.sh — noise-reliance probe sweep (JPEG ladder).
#
# WHY: corrupting high-frequency content partially isolates WHAT signal each
# objective learned. If bce_* and cont_* detection/localization degrade
# differently down the JPEG ladder, the objectives are relying on different
# signals (high-freq generator fingerprint vs relational/contextual evidence).
# Corruption is applied at MODEL INPUT (post-resize) so every crop gets
# identical model-space frequency destruction — the isolation instrument,
# not the laundering threat model (that is --corrupt_at native).
#
# Usage mirrors run_bce_emergence_rerun.sh:
#   ./run_bce_emergence_noise.sh                  # all six conditions
#   ./run_bce_emergence_noise.sh bce_both_s0 ...  # subset (two-GPU split via
#                                                 # CUDA_VISIBLE_DEVICES)
# Override the ladder: LEVELS="clean jpeg_50 noise_0.10" ./run_bce_emergence_noise.sh
set -euo pipefail

PY=${PY:-$HOME/dino_venv/bin/python}
DATA=/media/ssd/DINO_SCOPE_DATA
RUNS=/media/ssd/runs/bce_emergence
OUT=results/bce_emergence
LEVELS=${LEVELS:-"clean jpeg_90 jpeg_70 jpeg_50 jpeg_30"}
# Fixed epoch-5 study checkpoint — see run_bce_emergence_rerun.sh. NOT best.pt.
CKPT_FILE=${CKPT_FILE:-epoch_0005.pt}

"$PY" -c 'import torch' 2>/dev/null || {
  echo "ERROR: $PY has no torch — set PY=/path/to/venv/python" >&2; exit 1; }

SAGID=$DATA/SAGI_D
IMD2020=$DATA/IMD2020
TGIF2=$DATA/content/flux_originals

ROOTS=(
  --ai_interior_root "$SAGID" --ai_boundary_root "$SAGID" --real_crop_root "$SAGID"
  --sp_interior_root "$IMD2020" --sp_boundary_root "$IMD2020"
  --fr_bg_matched_root "$TGIF2"
  --ai_interior_tgif_root "$TGIF2" --ai_boundary_tgif_root "$TGIF2" --real_crop_tgif_root "$TGIF2"
)

ALL=(bce_both_s0 bce_inpaint_s0 bce_splice_s0 cont_both_s0 cont_inpaint_s0 cont_splice_s0)
CONDS=("${@:-${ALL[@]}}")

for cond in "${CONDS[@]}"; do
  ckpt="$RUNS/$cond/$CKPT_FILE"
  [[ -f $ckpt ]] || { echo "ERROR: missing checkpoint $ckpt" >&2; exit 1; }
  outdir="$OUT/$cond/noise_probe"
  mkdir -p "$outdir"

  if [[ $cond == bce_* ]]; then decoder=threshold; else decoder=kmeans; fi

  echo "=== $cond (decoder=$decoder, levels: $LEVELS) ==="
  # shellcheck disable=SC2086  # LEVELS is intentionally word-split
  "$PY" -m experiments.scripts.eval_robustness \
      --checkpoint "$ckpt" \
      --decoder "$decoder" \
      --amp_dtype float16 \
      --conditions $LEVELS \
      --corrupt_at model_input \
      "${ROOTS[@]}" \
      --out_dir "$outdir" \
      --summary_out "$outdir/robustness_summary.json"
done
echo "DONE: ${CONDS[*]}"
