#!/usr/bin/env bash
# Experiment 1 — vanilla vs equal-budget patch BCE, splice-only (docs/equal_budget_bce_spec.md Part C).
#
# Question: does per-image budget balancing
# (lab_utils.model.losses.bce.equal_budget_patch_bce_loss) fix the small-splice
# punishment gap under today's mean-reduced patch BCE (a k-fake-patch splice is
# punished k/N as hard as a whole N-patch fake for being TOTALLY missed) —
# WITHOUT sprinkling false alarms on clean reals?
#
# Data is tgif2 'sp' ONLY (no full fakes — single-class images are a no-op for
# this loss and would dilute the A/B; no 'fr' — never average sp/fr, CLAUDE.md
# rule 5). Both arms are otherwise identical: image head + patch-BCE on,
# contrastive OFF (--contrastive_dim 0 is NOT optional — it defaults to 64 and
# silently builds an untrained kmeans decoder otherwise, same trap documented
# in run_t0_full_fakes.sh).
#
# Comparison is THRESHOLD-FREE (patch AUROC), never at a fixed decode
# threshold: per-image balancing changes what a patch's sigmoid means (rarity-
# suppressed posterior -> appearance likelihood-ratio), so any fixed-t f1/iou
# comparison between arms measures the calibration shift, not localization
# quality (CLAUDE.md rule 1, self-inflicted on purpose here).
#
# Steps: [0] pre-run kp-budget gate (exits 1 and STOPS the run if k_min would
# mute the small-splice stratum) -> [1] train arm A (global, today's behavior)
# -> [2] train arm B (per_image, equal-budget) -> [3] collect patch scores for
# BOTH checkpoints over the SAME items -> [4] paired-bootstrap comparison
# (the pre-registered decision gate is printed at the end).
#
# Precision is autodetected from compute capability (Turing = fp16 only,
# Ampere+/L4 = bf16); override with DTYPE=fp16|bf16.
set -euo pipefail

TGIF_ROOT="${TGIF_ROOT:?set TGIF_ROOT (tgif2 dataset root)}"
IMD_ROOT="${IMD_ROOT:-}"          # optional OOD eval-only addition (imd2020 is never trained on)
RUN_DIR="${RUN_DIR:-/content/drive/MyDrive/DINO_SCOPE_RUNS/exp1_equal_budget}"
EPOCHS="${EPOCHS:-6}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-2000}"
BATCH="${BATCH:-8}"
K_MIN="${K_MIN:-4}"
BAND_LOW="${BAND_LOW:-0.2}"
BAND_HIGH="${BAND_HIGH:-0.8}"
N_BOOTSTRAP="${N_BOOTSTRAP:-1000}"

# bf16 needs Ampere+ (L4/Ada = ok). A T4 or 2080 Ti is Turing -> fp16 ONLY, and
# bf16 there fails at runtime rather than falling back. Autodetect unless told.
# (Same block as run_scripts/run_t0_full_fakes.sh — keep these in sync.)
if [[ -z "${DTYPE:-}" ]]; then
    DTYPE=$(python - <<'PY' 2>/dev/null || echo fp16
import torch
print('bf16' if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else 'fp16')
PY
)
fi
# analysis.compare_patch_auroc (like eval_robustness/eval_numbers) spells
# precision 'float16'/'bfloat16' where train.py wants 'fp16'/'bf16' — same
# thing, different vocabularies.
EVAL_DTYPE=$([[ "$DTYPE" == "bf16" ]] && echo bfloat16 || echo float16)
echo "[cfg] dtype=$DTYPE (eval: $EVAL_DTYPE)  k_min=$K_MIN  band=($BAND_LOW, $BAND_HIGH)"

echo
echo "=== [0/4] pre-run gate: kp budget audit (docs/equal_budget_bce_spec.md C0) ==="
# Exits 1 if k_min would mute the small-splice stratum this experiment exists
# to measure. set -e turns that into a hard stop, by design — do not lower
# this gate's bar to make a run start; fix k_min/band or the corpus instead.
python -m analysis.audit_patch_budget \
    --tgif2_root "$TGIF_ROOT" --tgif_types sp \
    --band "$BAND_LOW" "$BAND_HIGH" --k_min "$K_MIN"

echo
echo "=== [1/4] train — arm A: patch_balance=global (today's behavior, baseline) ==="
python -m experiments.scripts.train \
    --tgif2_root "$TGIF_ROOT" --tgif_types sp --seed 42 \
    --patch_bce --lambda_image_bce 1.0 --lambda_patch_bce 1.0 \
    --lambda_contrastive 0.0 --contrastive_dim 0 \
    --patch_pos_weight 1.0 --patch_balance global \
    --val_decoder threshold \
    --image_size 448 --lora_rank 16 --lora_alpha 32 --batch_size "$BATCH" \
    --aug_severity medium --num_epochs "$EPOCHS" --train_samples "$TRAIN_SAMPLES" \
    --base_dtype "$DTYPE" --amp_dtype "$DTYPE" \
    --checkpoint_root "$RUN_DIR/arm_global"

echo
echo "=== [2/4] train — arm B: patch_balance=per_image (equal-budget redesign) ==="
python -m experiments.scripts.train \
    --tgif2_root "$TGIF_ROOT" --tgif_types sp --seed 42 \
    --patch_bce --lambda_image_bce 1.0 --lambda_patch_bce 1.0 \
    --lambda_contrastive 0.0 --contrastive_dim 0 \
    --patch_balance per_image --patch_band "$BAND_LOW" "$BAND_HIGH" --patch_k_min "$K_MIN" \
    --val_decoder threshold \
    --image_size 448 --lora_rank 16 --lora_alpha 32 --batch_size "$BATCH" \
    --aug_severity medium --num_epochs "$EPOCHS" --train_samples "$TRAIN_SAMPLES" \
    --base_dtype "$DTYPE" --amp_dtype "$DTYPE" \
    --checkpoint_root "$RUN_DIR/arm_perimage"

echo
echo "=== [3/4] collect patch scores — BOTH checkpoints over the SAME items ==="
CKPT_A="$(ls -1 "$RUN_DIR"/arm_global/epoch_*.pt 2>/dev/null | sort | tail -1)"
CKPT_B="$(ls -1 "$RUN_DIR"/arm_perimage/epoch_*.pt 2>/dev/null | sort | tail -1)"
if [[ -z "$CKPT_A" || -z "$CKPT_B" ]]; then
    echo "missing a checkpoint under $RUN_DIR — skipping eval"; exit 0
fi
echo "arm_global:   $CKPT_A"
echo "arm_perimage: $CKPT_B"

# Scoring the identical item set for both labels is what makes the paired
# bootstrap in step 4 exact — do not vary --tgif_types / --imd2020_root
# between arms.
if [[ -n "$IMD_ROOT" ]]; then
    python -m analysis.compare_patch_auroc collect \
        --checkpoint "$CKPT_A" "$CKPT_B" --label global per_image \
        --tgif2_root "$TGIF_ROOT" --tgif_types sp \
        --imd2020_root "$IMD_ROOT" --imd_val_split 1.0 \
        --band "$BAND_LOW" "$BAND_HIGH" --amp_dtype "$EVAL_DTYPE" \
        --out_csv "$RUN_DIR/exp1_patch_scores.csv"
else
    python -m analysis.compare_patch_auroc collect \
        --checkpoint "$CKPT_A" "$CKPT_B" --label global per_image \
        --tgif2_root "$TGIF_ROOT" --tgif_types sp \
        --band "$BAND_LOW" "$BAND_HIGH" --amp_dtype "$EVAL_DTYPE" \
        --out_csv "$RUN_DIR/exp1_patch_scores.csv"
fi

echo
echo "=== [4/4] paired-bootstrap comparison (decision gate C4) ==="
python -m analysis.compare_patch_auroc analyze \
    --from_csv "$RUN_DIR/exp1_patch_scores.csv" \
    --n_bootstrap "$N_BOOTSTRAP" --seed 0 \
    --summary_out "$RUN_DIR/exp1_compare_summary.json"

cat <<'NOTES'

── decision gate (PRE-REGISTERED — docs/equal_budget_bce_spec.md C4) ──────────
PRIMARY   bucket=small AUROC delta (per_image - global), printed above. CI
          excluding 0 in favor of per_image -> mechanism confirmed; proceed to
          the {loss} x {full fakes present} 2x2 experiment.
GUARD 1   real_bg_p99 delta (the sprinkle canary, printed above) must not be
          worse for per_image beyond CI noise. Violated -> the symmetric
          false-alarm pool failed in PRACTICE, not just in theory — stop and
          analyze before proceeding, this is a headline finding either way.
GUARD 2   image-level detection AUC (see each arm's own training/val logs)
          should sit within noise between arms — the image head must not pay
          for the patch-loss fix.
KILL      PRIMARY delta's CI does not exclude 0 in per_image's favor -> the
          redesign is dead. Do NOT tune k_min / band / lambda to resurrect it
          — report the null result and stop.
NOTES
