#!/usr/bin/env bash
# Deep-LoRA layer-LOCATION sweep on Colab (L4).
#
# L4 is Ada (CC 8.9, 24 GB) and bf16-capable, so resolve_hardware would auto-pick
# bf16 — we PIN --amp_dtype fp16 to (a) match the box all-layers fp16 overlay and
# (b) cleanly resume the fp16 partials trained on the earlier T4. Batch/res stay
# at 8/448 for the same comparability reason (the 24 GB headroom is unused on
# purpose). Only throughput knobs (num_workers) are raised for the L4.
#
#   deep-half  @ {8,16,32}  → adapt only blocks [N/2, N)  (--lora_block_start HALF)
#   shallow-half @ {8,16}   → adapt only blocks [0, N/2)  (--lora_block_end   HALF)
#   alpha = 2 x rank throughout.
#
# Schedule-matched to the all-layers coarse rank sweep (num_epochs=10) and
# truncated at epoch 4 via --max_train_epochs, so each cell's epoch_0004.pt
# overlays the box all-layers e4 evals (r008 .423 / r016 .499 / r032 .409).
#
# Resume-safe: checkpoints live on Drive; re-running after a runtime disconnect
# auto-resumes each cell (a finished cell exits immediately at epoch 4).
set -e
cd "$(dirname "$0")/.."                         # repo root
PY=${PY:-python}
NW=${NW:-8}                                      # L4 runtimes have more vCPUs than T4
MODEL=facebook/dinov3-vith16plus-pretrain-lvd1689m

# ── auto-compute the deep/shallow block split (no model weights loaded) ───────
read N HALF < <($PY - <<EOF
from transformers import AutoConfig
c = AutoConfig.from_pretrained("$MODEL")
N = getattr(c, "num_hidden_layers", None) or getattr(c, "n_layers", None) or getattr(c, "depth", None)
N = int(N); print(N, N // 2)
EOF
)
echo "[deep_lora] backbone N=$N blocks  →  deep-half=[$HALF,$N)   shallow-half=[0,$HALF)"

# ── shared config (Colab roots, T4: fp16 default, num_workers 2) ──────────────
ROOTS="--imd2020_root /content/IMD2020 --casia_root /content/casia \
  --sagid_root /content/sagi_d_partial \
  --coco_inpaint_root /content/inpaint_coco/images \
  --tgif2_root /content/dataset_root/content/flux_originals"

COMMON="$ROOTS --casia_train --imd_val_only \
  --image_size 448 --batch_size 8 --grad_accum 1 --device cuda \
  --contrastive_dim 64 --pool_hidden 256 \
  --lambda_image_bce 1.0 --lambda_contrastive 2.0 \
  --paste_frac 0.5 --noise_prob 0.8 --jpeg_prob 0.55 \
  --train_samples 3000 --num_workers $NW --amp_dtype fp16 \
  --num_epochs 10 --max_train_epochs 5 --min_epochs 5 --warmup_epochs 1.0 \
  --early_stop_patience 2 --early_stop_reduce mean \
  --val_zoom --tgif_val_models flux1dev --val_per_cell 100"

RUNS=/content/drive/MyDrive/DINO_SCOPE_RUNS/deep_lora

run_cell () {  # $1=name $2=rank $3=alpha $4..=location flag(s)
  local name=$1 rank=$2 alpha=$3; shift 3
  echo "=== [deep_lora] $name  rank=$rank alpha=$alpha  $* ==="
  $PY -m experiments.scripts.train \
    --checkpoint_root "$RUNS/$name" \
    --lora_rank "$rank" --lora_alpha "$alpha" "$@" $COMMON
}

run_cell dh_r008  8  16 --lora_block_start "$HALF"
run_cell dh_r016 16  32 --lora_block_start "$HALF"
run_cell dh_r032 32  64 --lora_block_start "$HALF"
run_cell sh_r008  8  16 --lora_block_end   "$HALF"
run_cell sh_r016 16  32 --lora_block_end   "$HALF"

echo "[deep_lora] training done → $RUNS"
