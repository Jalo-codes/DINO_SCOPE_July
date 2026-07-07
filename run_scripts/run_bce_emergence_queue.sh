#!/usr/bin/env bash
# BCE-emergence sweep — dynamic 2-GPU queue on the 2080 Ti box.
#
# Each worker (one per GPU) atomically CLAIMS the next unclaimed cell via
# mkdir (atomic on POSIX) and runs it through orchestrate --only <cell>, so a
# fast cell frees its GPU immediately instead of waiting on a chained list.
# Resume-safe twice over: orchestrate skips cells whose ORCH_DONE.json says
# exit 0, and stale claims of finished cells are harmless. To retry a FAILED
# cell: rm -r "$RUN_ROOT/<cell>/.claim" and rerun.
#
# Usage:
#   bash run_scripts/run_bce_emergence_queue.sh              # both GPUs
#   GPUS="0" bash run_scripts/run_bce_emergence_queue.sh     # single GPU
set -u

PY="${PY:-$HOME/dino_venv/bin/python}"
QUEUE="${QUEUE:-sweeps/sweep_bce_emergence.json}"
RUN_ROOT="${RUN_ROOT:-/media/ssd/runs/bce_emergence}"
GPUS="${GPUS:-0 1}"

mkdir -p "$RUN_ROOT"

# Cell names from the queue JSON (stdlib only — box venv has no jq).
CELLS=$("$PY" - "$QUEUE" <<'EOF'
import json, sys
print('\n'.join(e['name'] for e in json.load(open(sys.argv[1]))['runs']))
EOF
)

worker() {
  local gpu="$1"
  for cell in $CELLS; do
    local claim="$RUN_ROOT/$cell/.claim"
    mkdir -p "$RUN_ROOT/$cell"
    if mkdir "$claim" 2>/dev/null; then
      echo "[queue] GPU$gpu -> $cell"
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.scripts.orchestrate \
        --queue "$QUEUE" --run_root "$RUN_ROOT" --only "$cell" \
        2>&1 | tee "$RUN_ROOT/$cell/queue_gpu$gpu.log"
      echo "[queue] GPU$gpu finished $cell"
    fi
  done
  echo "[queue] GPU$gpu: no unclaimed cells left"
}

pids=()
for gpu in $GPUS; do
  worker "$gpu" &
  pids+=($!)
done
wait "${pids[@]}"
echo "[queue] all workers done — summary at $RUN_ROOT"
