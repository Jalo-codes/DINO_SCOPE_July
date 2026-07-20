#!/usr/bin/env bash
# Deep/shallow-LoRA location-sweep EVAL on Colab (L4), epoch_0004.pt, zoom two-pass.
#
# Mirrors the box rank-eval harness but roots the eval run on DRIVE. This matters:
# the per-cell `subgroup=… f1` numbers are written ONLY to orchestrator.log (they
# are never serialized into eval_summary.json), so if run_root sat on /content a
# runtime disconnect would lose the per-cell breakdown even though the overall
# json survived. Drive-rooted ⇒ logs + summaries + ORCH_DONE markers all persist,
# and a re-run after a disconnect skips finished cells.
#
# Eval is a single-item loop (no DataLoader / num_workers / batching), so the only
# throughput lever is amp fp16 — already pinned in the queue's base_args. Cost is
# tgif_eval_per_cell=1000 × 12 cells × 2 forwards (zoom) per checkpoint.
#
# Usage (on the L4 Colab runtime, repo cwd):
#   PY=python bash run_scripts/run_deep_lora_eval_colab.sh
#   PY=python bash run_scripts/run_deep_lora_eval_colab.sh --dry_run   # preview
set -euo pipefail
cd "$(dirname "$0")/.."                         # repo root
REPO="$(pwd)"
PY=${PY:-python}

# Drive-rooted eval output so per-cell logs survive a runtime disconnect.
RUNS="${RUNS_ROOT:-/content/drive/MyDrive/DINO_SCOPE_RUNS/deep_lora_eval}"
mkdir -p "$RUNS"

echo "[deep_lora_eval] queue=sweeps/sweep_ablation_eval_deep_lora_colab.json"
echo "[deep_lora_eval] run_root=$RUNS  (Drive — persists across disconnect)"
"$PY" -m experiments.scripts.orchestrate \
  --queue sweeps/sweep_ablation_eval_deep_lora_colab.json \
  --run_root "$RUNS" --cwd "$REPO" "$@"

echo "[deep_lora_eval] done. Roll up the per-cell table with:"
echo "  $PY -m analysis.rollup_ablation_eval \\"
echo "      --run_root $RUNS --only_suffix _tgif --metric mean \\"
echo "      --out_prefix $RUNS/rollup_tgif"
echo "  $PY -m analysis.rollup_ablation_eval \\"
echo "      --run_root $RUNS --only_suffix _imd --metric mean \\"
echo "      --out_prefix $RUNS/rollup_imd"
