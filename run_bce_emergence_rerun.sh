#!/usr/bin/env bash
# run_bce_emergence_rerun.sh — fr_bg_matched probe re-eval + BCE threshold sweep.
#
# Box: binghamton-rbtg (2x RTX 2080 Ti, Turing -> fp16, never bf16).
# Paths below are the box's confirmed layout (repo CLAUDE.md / ANALYSIS_NOTES
# "Planned rerun"). eval.py's cache skips already-computed items, so re-running
# after a crash never repeats finished GPU work.
#
# Usage:
#   ./run_bce_emergence_rerun.sh                 # all six conditions + manifest
#   ./run_bce_emergence_rerun.sh bce_both_s0 ... # subset (no manifest step)
#
# Two-GPU split (two screens, roughly halves wall clock):
#   screen -L -S sweep0
#   CUDA_VISIBLE_DEVICES=0 ./run_bce_emergence_rerun.sh bce_both_s0 bce_inpaint_s0 bce_splice_s0
#   screen -L -S sweep1
#   CUDA_VISIBLE_DEVICES=1 ./run_bce_emergence_rerun.sh cont_both_s0 cont_inpaint_s0 cont_splice_s0
#   # then run the manifest once after both finish:
#   ./run_bce_emergence_rerun.sh manifest
set -euo pipefail

PY=${PY:-$HOME/dino_venv/bin/python}
DATA=/media/ssd/DINO_SCOPE_DATA
RUNS=/media/ssd/runs/bce_emergence
OUT=results/bce_emergence
# The study checkpoint is the FIXED epoch-5 snapshot for every condition —
# the same file all official evals used (equal training budget across
# conditions; best.pt lands on epochs 0-9 and confounds objective with
# training length). Do NOT switch to best.pt.
CKPT_FILE=${CKPT_FILE:-epoch_0005.pt}
# Cache dir is keyed to the checkpoint file: build_cache reuses existing npz
# blindly, so a cache built from different weights would silently poison the
# sweep.
CACHE_NAME=probe_cache_${CKPT_FILE%.pt}

# Fail fast if PY is not the torch venv (bare `python` on this box is a
# torchless conda-base 3.7 — see 2080-box notes).
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

run_manifest() {
  "$PY" -m experiments.labs.probe_manifest \
      "${ROOTS[@]}" \
      --image_size 448 --patch_size 16 \
      --out_csv "$OUT/probe_manifest2.csv"
}

if [[ "${1:-}" == manifest ]]; then
  run_manifest; exit 0
fi

CONDS=("${@:-${ALL[@]}}")

for cond in "${CONDS[@]}"; do
  ckpt="$RUNS/$cond/$CKPT_FILE"
  [[ -f $ckpt ]] || { echo "ERROR: missing checkpoint $ckpt" >&2; exit 1; }
  outdir="$OUT/$cond/probe_eval2"
  mkdir -p "$outdir"

  if [[ $cond == bce_* ]]; then
    decoder=threshold
    cache_args=(--cache_dir "$RUNS/$cond/$CACHE_NAME")
  else
    decoder=kmeans
    cache_args=()
  fi

  echo "=== $cond (decoder=$decoder) ==="
  "$PY" -m experiments.scripts.eval \
      --checkpoint "$ckpt" \
      --decoder "$decoder" \
      --amp_dtype float16 \
      "${cache_args[@]}" \
      "${ROOTS[@]}" \
      --out_dir "$outdir"

  if [[ $cond == bce_* ]]; then
    swout="$OUT/$cond/threshold_sweep"
    mkdir -p "$swout"
    "$PY" -m experiments.scripts.eval_threshold_sweep \
        --cache_dir "$RUNS/$cond/$CACHE_NAME" \
        --out_dir "$swout" \
        "${ROOTS[@]}"
  fi
done

# Manifest only on a full (no-arg) run — for split runs, invoke `manifest`
# once after both halves finish so concurrent writers never collide.
if [[ $# -eq 0 ]]; then
  run_manifest
fi
echo "DONE: ${CONDS[*]}"
