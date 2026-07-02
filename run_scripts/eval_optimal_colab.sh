#!/usr/bin/env bash
# Standalone evaluation and visualization script for the optimal checkpoint on Colab (L4).
#
# Generates:
#   - 100 visualizations per TGIF2 subcategory type (12 cells * 100 = 1200 images)
#   - Evaluates ONLY these 1200 visualised TGIF2 items.
#
# Visualizations show: full image, zoom crop, attention maps, bounding boxes, etc.
# landing in /content/drive/MyDrive/DINO_SCOPE_RUNS/optimal/visuals/tgif2/
#
# Usage (Colab repo cwd):
#   PY=python bash run_scripts/eval_optimal_colab.sh

set -euo pipefail
cd "$(dirname "$0")/.."                          # repo root
REPO="$(pwd)"
PY=${PY:-python}

CKPT="${CKPT:-/content/drive/MyDrive/DINO_SCOPE_RUNS/optimal/optimal_h16plus_688_r16/best.pt}"
RUNS_DIR="${RUNS_ROOT:-/content/drive/MyDrive/DINO_SCOPE_RUNS/optimal}"

if [ ! -f "$CKPT" ]; then
  echo "[eval] checkpoint not found: $CKPT" >&2
  exit 1
fi

echo "[eval] Using checkpoint: $CKPT"
echo "[eval] Visuals output root: $RUNS_DIR/visuals/tgif2"

# 1. TGIF2: Evaluate exactly 100 items per cell (12 subcategories * 100 = 1200 total) and save all 1200 visuals
echo "=== Evaluating and generating visuals for TGIF2 (100/cell -> 1200 visuals) ==="
OUT_TGIF="$RUNS_DIR/visuals/tgif2"
mkdir -p "$OUT_TGIF"
"$PY" -m experiments.scripts.eval \
  --checkpoint "$CKPT" \
  --sources tgif2 \
  --decoder kmeans \
  --zoom \
  --device cuda \
  --amp_dtype float16 \
  --tgif2_root "/content/dataset_root/content/flux_originals" \
  --tgif_eval_per_cell 100 \
  --viz_n 1200 \
  --out_dir "$OUT_TGIF"
