#!/usr/bin/env bash
# Qualitative VISUALS across datasets from the r032 (448, rank-32) LoRA-sweep model.
# Run on the 2080 Ti. Renders the zoom two-pass debug panel per manipulated image
# (original · 1st-pass attention · flat mask · GT mask+box · zoom crop · attn-on-crop
# · zoom mask), titled with F1 — so each figure shows the failure mode AND what zoom
# did to it. Output lands in results/visuals/ (committed via the gitignore exception)
# so it can be pushed to GitHub and reviewed before any zoom-tuning work.
#
# The box has NO matplotlib — plot_hdbscan_result falls back to a PIL compositor
# (same 7 panels) and eval.py saves the returned PIL image directly. No mpl needed.
#
# INFERENCE IS CAPPED to exactly the sampled set (not the whole dataset):
#   TGIF      — TGIF_PER_CELL (=100) items PER CELL × 12 cells = ~1200, all visualized.
#   in-domain — DOM_N (=100) items per source × 4 sources = ~400, all visualized.
# VIZ_MAX_PANEL_H caps panel height so the ~1600 committed PNGs stay small.
#
# Usage (on the box, repo cwd):
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=1 bash run_scripts/gen_visuals.sh
#   TGIF_PER_CELL=100 DOM_N=100 GPU=1 bash run_scripts/gen_visuals.sh
set -euo pipefail
: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
cd "$(dirname "$0")/.."                          # repo root

CKPT="${CKPT:-/media/ssd/runs/ablation/lora_rank_sweep/r032/best.pt}"
GPU="${GPU:-1}"
TGIF_PER_CELL="${TGIF_PER_CELL:-100}"            # items inferred + visualized per TGIF cell
DOM_N="${DOM_N:-100}"                            # items inferred + visualized per in-domain source
OUT="${OUT:-results/visuals/r032_448}"
export VIZ_MAX_PANEL_H="${VIZ_MAX_PANEL_H:-384}" # cap panel height → small PNGs

DATA=/media/ssd/DINO_SCOPE_DATA
ROOTS="--imd2020_root $DATA/IMD2020 --casia_root $DATA/casia \
  --sagid_root $DATA/SAGI_D \
  --coco_inpaint_root $DATA/INPAINT_COCO/content/inpaint_coco/images \
  --tgif2_root $DATA/content/flux_originals"

if [ ! -f "$CKPT" ]; then echo "[visuals] checkpoint not found: $CKPT" >&2; exit 1; fi
echo "[visuals] model=$CKPT  GPU=$GPU  tgif=$TGIF_PER_CELL/cell  in-domain=$DOM_N/src  panel_h<=$VIZ_MAX_PANEL_H  -> $OUT"

# TGIF: one call, capped per CELL. tgif_eval_per_cell bounds inference to N/cell
# (~12 cells → ~1200 items); viz_n is set above that so every inferred item is saved.
echo "=== [visuals] tgif2 ($TGIF_PER_CELL/cell) ==="
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.eval \
  --checkpoint "$CKPT" --sources tgif2 --decoder kmeans --zoom \
  --device cuda --amp_dtype float16 \
  --tgif_eval_per_cell "$TGIF_PER_CELL" \
  --viz_n $((TGIF_PER_CELL * 24)) --out_dir "$OUT/tgif2" \
  $ROOTS

# In-domain: one call per source, capped at DOM_N items (inference) and DOM_N figures.
for SRC in imd2020 casia coco_inpaint sagid; do
  echo "=== [visuals] $SRC ($DOM_N) ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.eval \
    --checkpoint "$CKPT" --sources "$SRC" --decoder kmeans --zoom \
    --device cuda --amp_dtype float16 \
    --max_items "$DOM_N" \
    --viz_n "$DOM_N" --out_dir "$OUT/$SRC" \
    $ROOTS
done

echo
echo "[visuals] done -> $OUT/<source>/kmeans_viz/*.png"
echo "[visuals] NOTE: ~1600 PNGs. Check total size before committing to git history:"
echo "  du -sh $OUT"
echo "[visuals] commit + push (on the box):"
echo "  git add results/visuals && git commit -m 'visuals: r032 zoom panels across datasets' && git push"
