#!/bin/bash
# preprocess_realesrgan.sh — Index-driven Real-ESRGAN launder pass (CUDA / PyTorch).
#
# Thin wrapper: installs the pip `realesrgan` stack, then runs the actual launder
# in run_scripts/preprocess_realesrgan.py.  Runs on the GPU via PyTorch/CUDA —
# no Vulkan, no ICD.  See the .py header for the full behavior.
#
# Usage:
#   bash run_scripts/preprocess_realesrgan.sh [INPUT_DIR] [OUTPUT_DIR] [SCALE_2_OR_4] [MAX_IMAGES]
#
# Example:
#   bash run_scripts/preprocess_realesrgan.sh \
#       /content/dataset_root/content/flux_originals \
#       /content/dataset_root/content/flux_originals_esrgan_x2 \
#       2 10000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$SCRIPT_DIR/preprocess_realesrgan.py"

if [ "$#" -lt 2 ]; then
    echo "Usage: bash run_scripts/preprocess_realesrgan.sh [INPUT_DIR] [OUTPUT_DIR] [SCALE_2_OR_4] [MAX_IMAGES]"
    exit 1
fi

# Install the realesrgan stack if absent.  basicsr requires only torch>=1.7
# (no upper bound), so this will NOT downgrade Colab's torch/CUDA build.
if ! python3 -c "import realesrgan, basicsr" >/dev/null 2>&1; then
    echo "[preprocess] Installing realesrgan + basicsr..."
    pip install -q realesrgan
fi

# Reduce allocator fragmentation so the per-batch VRAM sawtooth has safer
# headroom (lets the cached blocks grow/shrink without re-reserving).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python3 "$PY" "$@"
