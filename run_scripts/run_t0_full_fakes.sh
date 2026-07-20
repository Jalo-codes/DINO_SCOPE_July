#!/usr/bin/env bash
# T0 — how (non-)trivial is whole-image fake detection, with the heads AS THEY STAND?
#
# Image head only by default (HEADS=image); HEADS=image+patch adds patch-BCE.
# Never contrastive. Trains on the OpenFake TRAIN split (full_fakes layout) and
# scores against a separate eval root. Motivated by a measured failure, not
# curiosity: the FullySynthesized recall 0.15 crater.
#
# Two things this is designed to keep honest:
#   1. Leakage is GATED, not assumed — the run refuses to start until the train
#      and eval roots are proven disjoint by md5 of raw bytes.
#   2. The headline number is meaningless without the robustness ladder. OpenFake
#      preserves original bytes on purpose (container/compression is real eval
#      signal), which means a model CAN lean on it. If detection collapses under
#      re-encoding, that is what it was doing. Run step 4.
#
# Precision is autodetected from compute capability (see DTYPE below); override
# with DTYPE=fp16|bf16 if you need to pin it.
set -euo pipefail

FF_TRAIN="${FF_TRAIN:-/content/openfake_train_ff}"
FF_VAL="${FF_VAL:-/content/openfake_ff}"          # existing test-split download
RUN_DIR="${RUN_DIR:-/content/drive/MyDrive/DINO_SCOPE_RUNS/t0_full_fakes}"
EPOCHS="${EPOCHS:-6}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-3000}"
BATCH="${BATCH:-8}"
VAL_PER_POOL="${VAL_PER_POOL:-25}"     # per generator pool
VAL_REALS="${VAL_REALS:-100}"          # reals in the per-epoch val
HEADS="${HEADS:-image}"                # 'image' | 'image+patch'
# bf16 needs Ampere+ (L4/Ada = ok). A T4 or 2080 Ti is Turing -> fp16 ONLY, and
# bf16 there fails at runtime rather than falling back. Autodetect unless told.
if [[ -z "${DTYPE:-}" ]]; then
    DTYPE=$(python - <<'PY' 2>/dev/null || echo fp16
import torch
print('bf16' if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else 'fp16')
PY
)
fi
ROBUST_DTYPE=$([[ "$DTYPE" == "bf16" ]] && echo bfloat16 || echo float16)
echo "[cfg] dtype=$DTYPE (robustness: $ROBUST_DTYPE)  heads=$HEADS"

# HEADS=image is the default after the first T0 run: with --patch_bce, OOD image
# AUC fell 0.9693 -> 0.9217 across one epoch while BOTH losses dropped. On
# whole-image fakes the patch target is the image target copied 784x
# (all-positive for fakes, all-negative for reals), so it adds no information
# but dominates the gradient into the shared backbone — the image head emits one
# attention-pooled scalar, the patch head emits 784 dense signals. The cheapest
# way to satisfy 784 per-patch labels is local texture, i.e. generator-specific
# artifacts, which is precisely what fails to transfer to held-out generators.
case "$HEADS" in
  image)
    # --contrastive_dim 0 is NOT optional. It defaults to 64, so the projector is
    # BUILT even at --lambda_contrastive 0, and --val_decoder auto then resolves
    # to kmeans over an UNTRAINED random projection — plausible-looking garbage
    # masks. Zero it and auto correctly resolves to 'none'.
    HEAD_FLAGS=(--lambda_image_bce 1.0 --lambda_contrastive 0.0
                --lambda_patch_bce 0.0 --contrastive_dim 0)
    VAL_DECODER=none
    ROBUST_DECODER=none
    ;;
  image+patch)
    # patch_pos_weight 1.0, not the default 10.0: under --balance_real_fake the
    # patch classes are already ~50/50, so a 10x positive weight just teaches the
    # head to fire everywhere (75% of patches lit on REAL images in run 1).
    HEAD_FLAGS=(--patch_bce --lambda_image_bce 1.0 --lambda_patch_bce 1.0
                --lambda_contrastive 0.0 --patch_pos_weight 1.0)
    VAL_DECODER=threshold
    ROBUST_DECODER=threshold
    ;;
  *) echo "HEADS must be 'image' or 'image+patch', got '$HEADS'" >&2; exit 2 ;;
esac

echo "=== [1/4] leakage gate ==="
# Exits 1 on any shared md5. set -e turns that into a hard stop, by design:
# a contaminated eval set makes every downstream number unreadable.
python -m analysis.audit_openfake_split_overlap \
    --train_root "$FF_TRAIN" --eval_root "$FF_VAL"

echo
echo "=== [2/4] data diet ==="
for d in "$FF_TRAIN"/*/; do
    printf '%6d  %s\n' "$(ls -1 "$d" 2>/dev/null | wc -l)" "$(basename "$d")"
done | sort -rn
echo "train total: $(find "$FF_TRAIN" -type f ! -name manifest.csv | wc -l) images"

echo
echo "=== [3/4] train (HEADS=$HEADS) ==="
python -m experiments.scripts.train \
    --full_fakes_root     "$FF_TRAIN" \
    --full_fakes_val_root "$FF_VAL" \
    --checkpoint_root     "$RUN_DIR" \
    "${HEAD_FLAGS[@]}" \
    --balance_real_fake \
    --full_fakes_val_per_pool "$VAL_PER_POOL" \
    --full_fakes_val_reals    "$VAL_REALS" \
    --aug_severity medium \
    --image_size 448 --lora_rank 16 --lora_alpha 32 \
    --batch_size "$BATCH" --num_epochs "$EPOCHS" --train_samples "$TRAIN_SAMPLES" \
    --seed 42 \
    --base_dtype "$DTYPE" --amp_dtype "$DTYPE" \
    --num_workers 2 \
    --val_decoder "$VAL_DECODER" \
    --no-val_zoom

echo
echo "=== [4/4] robustness ladder — the number that decides if step 3 meant anything ==="
CKPT="$(ls -1 "$RUN_DIR"/epoch_*.pt 2>/dev/null | sort | tail -1)"
if [[ -z "$CKPT" ]]; then
    echo "no checkpoint found under $RUN_DIR — skipping"; exit 0
fi
echo "checkpoint: $CKPT"
# NOTE eval_robustness spells precision 'float16'/'bfloat16' while train.py wants
# 'fp16'/'bf16' — same thing, different vocabularies, and it REJECTS 'bf16'.
# Hence ROBUST_DTYPE. --decoder likewise defaults to kmeans; there is no
# contrastive head here, so it is set explicitly.
python -m experiments.scripts.eval_robustness \
    --checkpoint "$CKPT" \
    --full_fakes_root "$FF_VAL" \
    --decoder "$ROBUST_DECODER" \
    --amp_dtype "$ROBUST_DTYPE" \
    --corrupt_at native \
    --out_dir "$RUN_DIR/robustness"

cat <<'NOTES'

── how to read this ────────────────────────────────────────────────────────────
image AUROC          the headline. Compare to the FullySynth 0.15 recall crater.
AUROC under JPEG     if it falls off a cliff, the model was reading container
                     statistics, not generated content. That reframes a good
                     clean number as an artifact.
patch-BCE on THIS    near-meaningless alone: sentinel masks label every patch of
  data               a fake positive, so the head can just copy the image
                     decision. The real patch test is the transfer eval below.

NEXT, and it is the cheap one worth doing: run this checkpoint's PATCH head over
SPLICED images (CASIA / IMD / TGIF). If patch-BCE — trained only on whole-image
labels — lights up just the spliced region, localization falls out of image-level
supervision with zero new machinery. If it lights the whole frame, AI-ness needs
global context, which kills the 'lit-local vs lit-global' row of the signature
table. Either answer is decisive and neither needs another training run.
NOTES
