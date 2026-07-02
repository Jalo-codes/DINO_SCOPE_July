#!/usr/bin/env bash
# Wall-clock inference benchmark across the resolution-sweep checkpoints.
# Times the forward pass over a FIXED set of TGIF images (25/cell ≈ 300 imgs)
# for every res_*/best.pt, so latency-vs-resolution is apples-to-apples.
# Run on ONE gpu. Writes a CSV + PNG graph. Resume-safe (CSV rewritten per cell).
#
# Usage:
#   export PY=/home/fri-team-4/dino_venv/bin/python
#   GPU=1 bash run_scripts/run_bench_resolution.sh                 # box defaults
#   GPU=1 bash run_scripts/run_bench_resolution.sh --zoom          # 2-pass latency
#   GPU=1 PER_CELL=25 bash run_scripts/run_bench_resolution.sh
set -euo pipefail

: "${PY:?set PY to the dino_venv python, e.g. /home/fri-team-4/dino_venv/bin/python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GPU="${GPU:-0}"
RUN_ROOT="${RUN_ROOT:-/media/ssd/runs/ablation/res_sweep}"
TGIF_ROOT="${TGIF_ROOT:-/media/ssd/DINO_SCOPE_DATA/content/flux_originals}"
PER_CELL="${PER_CELL:-25}"
OUT_CSV="${OUT_CSV:-$REPO/results/bench_resolution.csv}"
cd "$REPO"

echo "[bench] GPU=$GPU run_root=$RUN_ROOT per_cell=$PER_CELL -> $OUT_CSV"
CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m experiments.scripts.bench_resolution \
  --run_root "$RUN_ROOT" \
  --tgif2_root "$TGIF_ROOT" \
  --tgif_eval_per_cell "$PER_CELL" \
  --warmup 10 \
  --out_csv "$OUT_CSV" \
  --out_json "${OUT_CSV%.csv}.json" \
  --plot "$@"
echo "[bench] done. csv=$OUT_CSV  graph=${OUT_CSV%.csv}.png"
