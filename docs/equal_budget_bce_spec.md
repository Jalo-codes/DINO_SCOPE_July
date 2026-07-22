# Spec — Equal-Budget Patch BCE + Patch-AUROC Readout + Experiment 1 A/B

Implementation spec for a builder agent. Everything needed is in this file; when
this file and the code disagree, STOP and flag it rather than improvising.
Read CLAUDE.md first — rules 1, 2, 3, 5, 6 all bind here.

## 0. Framing (do not skip — it decides several design calls below)

The patch-BCE head is a **detector**: each patch's sigmoid answers "does this
patch look AI-generated" — an absolute, per-patch appearance judgment. The
contrastive head answers a different question — "which patches differ from the
rest of THIS image" — relative and self-calibrating. **BCE is not a replacement
for contrastive and must never be framed or evaluated as one.** They will
eventually co-train; this spec touches only the BCE side.

Future context that shapes today's decisions (out of scope to build, in scope
to not preclude): whole-image AE-passthrough images (a REAL photo passed
through an autoencoder) will be added as **hard NEGATIVES** — label real,
everywhere. Purpose: force "AI-ness" to be independent of high-frequency /
resampling artifacts. The loss below handles this for free (a single-class
negative image is identical to a clean real from the loss's point of view).
Do not add any assumption that label-0 patches are untouched pixels.

The problem being fixed: under today's mean-reduced patch BCE, missing ALL 7
fake patches of a small splice costs 7/784 ≈ 1/112 of missing all 784 patches
of a full fake — the model's punishment is proportional to splice area, so it
rationally ignores small splices. The fix gives every image the same total
punishment budget for total blindness, redistributed within the image. No
image is up- or down-weighted relative to any other; the sampler is untouched.

## 1. Deliverables map

| # | path | action |
|---|---|---|
| D1 | `lab_utils/model/losses/bce.py` | ADD `equal_budget_patch_bce_loss()` (new function — do NOT modify `selective_patch_bce_loss`) |
| D2 | `lab_utils/train/loop.py` | ADD tensor band twin `_mask_to_patch_labels_soft_t()`; wire loss selection |
| D3 | `experiments/scripts/train.py` | ADD flags `--patch_balance`, `--patch_k_min`, `--patch_band` |
| D4 | `lab_utils/eval/patch_scores.py` | NEW module — patch-AUROC readout |
| D5 | `lab_utils/eval/numbers.py` | ADD `--patch_auroc` integration |
| D6 | `analysis/audit_patch_budget.py` | NEW — k⁺ distribution audit (pre-run gate) |
| D7 | `analysis/compare_patch_auroc.py` | NEW — paired-bootstrap A/B comparison |
| D8 | `tests/test_equal_budget_bce.py`, `tests/test_patch_scores.py` | NEW tests |
| D9 | `run_scripts/run_exp1_equal_budget_ab.sh` | NEW — both arms + eval, one command |

---

## Part A — the loss

### A1. Math (normative)

Per active image `i`, over its `N` patches (784 at 448px/16):

- `y_ij ∈ {0,1}` — patch label (1 = fake)
- `pw_ij ∈ [0,1]` — per-patch supervision weight from the band (all-ones when
  band disabled)
- `ℓ_ij` — plain per-patch BCE-with-logits, `pos_weight` NOT applied
  (per-image balancing replaces it wholesale)

```
kp_i = Σ_j pw_ij · y_ij                 (weighted fake count)
kn_i = Σ_j pw_ij · (1 − y_ij)           (weighted not-fake count)

wpos_i = 1 / max(kp_i, k_min)
wneg_i = 1 / max(kn_i, k_min)

L_i = Σ_j  pw_ij · [ y_ij · wpos_i + (1 − y_ij) · wneg_i ] · ℓ_ij     ← SUM over j, never mean

loss = Σ_i s_i · L_i / max(Σ_i s_i, 1)      over active images
       where s_i = sample_weights_i · 1[kp_i + kn_i > 0]
```

The per-image SUM is load-bearing: it is what concentrates the fixed budget
instead of re-diluting it. `kp/kn/wpos/wneg` are label-derived constants —
compute under `torch.no_grad()` in fp32, then cast to the logits dtype.

Properties the tests must pin (see A5):

1. **Flat budget**: for any image with `kp ≥ k_min`, missing every fake patch
   with per-patch wrongness `c` costs exactly `c` — independent of `kp`.
2. **Single-class identity**: on an all-fake or all-real image (band off),
   `L_i` equals today's `selective_patch_bce_loss` at `pos_weight=1`
   *exactly* (both reduce to the plain patch mean). The scheme only changes
   mixed images.
3. **Clamp**: `kp < k_min` ⇒ budget shrinks linearly to `kp / k_min`
   (deliberate mistrust of tiny-splice labels).
4. **Symmetric FP pricing**: false-alarm cost per background patch is
   `1/max(kn, k_min)` — scarce background (mostly-fake images) is expensive,
   abundant background (clean reals) costs `1/N`, same as today.

### A2. New function (D1) — `lab_utils/model/losses/bce.py`

Add (do not touch existing functions):

```python
def equal_budget_patch_bce_loss(
    logits: torch.Tensor,            # (B, N) per-patch logits
    labels: torch.Tensor,            # (B, N) {0,1}
    active_mask: torch.Tensor,       # (B,) bool
    k_min: float = 4.0,
    sample_weights: torch.Tensor = None,   # (B,)
    patch_weights: torch.Tensor = None,    # (B, N) in [0,1]; None = all-ones
) -> tuple[torch.Tensor, dict]:
```

Body: implement A1 literally. Zero-active or all-`s_i`-zero batches return
`logits.sum() * 0.0` (grad-safe zero — copy the existing idiom at
`bce.py:30`). Returned diag dict:

```python
{
  'realized_P': float,        # mean of kp_i * wpos_i over images with kp_i > 0
  'realized_Q': float,        # mean of kn_i * wneg_i over images with kn_i > 0
  'max_patch_w': float,       # max over batch of (wpos_i, wneg_i) — weight-blowup canary
  'n_no_supervision': int,    # images zeroed by 1[kp+kn > 0]
  'patch_pos_frac': float,    # as in selective_patch_bce_loss, over pw>0 patches
  'pred_pos_frac': float,     # ditto (logit >= 0)
}
```

`realized_P`/`realized_Q` are the budget-symmetry canaries (P≈Q≈1 when
healthy above the clamp); `max_patch_w` caps at `1/k_min` by construction —
if it ever exceeds that, the clamp is broken.

### A3. Band wiring (D2) — `lab_utils/train/loop.py`

The band SEMANTICS already exist:
`lab_utils/data/resolution.py::mask_to_patch_labels_soft` (line ~267). It is
the **source of truth**. It takes PIL masks; the train loop has batched GPU
tensors, so add a tensor twin `_mask_to_patch_labels_soft_t(mask_t, patch_size,
low, high) -> (labels (B,N) long, weights (B,N) float)` next to the existing
`_mask_to_patch_labels` implementing the identical piecewise rule:

```
density = avg_pool2d(mask_t, patch_size).flatten(1)      # (B, N)
label   = (density >= low)
weight  = 1.0  where density == 0.0        (confident background)
        = 0.0  where 0 < density < low     (IGNORE — boundary noise)
        = ramp (density−low)/(high−low)  where low <= density < high
        = 1.0  where density >= high       (confident fake)
```

Note the asymmetry is intentional and inherited: ONLY exact-zero density is
confident background — any mask-touched patch is ramped or ignored (splice
masks are antialiased; this is the right paranoia). Do not "fix" it.

**Parity test is mandatory** (A5, test 5): random masks through the PIL
function and the tensor twin must produce identical labels and weights
(exact for labels, atol 1e-6 for weights). This is what makes it reuse
rather than rebuild.

Call-site changes in `run_train_epoch` (around line 149 and 190):

```python
patch_labels = _mask_to_patch_labels(mask, cfg.patch_size)   # UNCHANGED — contrastive input

# ── Patch-BCE head ──
if patch_active and patch_logit is not None:
    active_patch = ~(is_splice & is_single)
    band = getattr(cfg, 'patch_band', None)
    if band:
        bce_labels, bce_pw = _mask_to_patch_labels_soft_t(
            mask, cfg.patch_size, float(band[0]), float(band[1]))
    else:
        bce_labels, bce_pw = patch_labels, None
    if getattr(cfg, 'patch_balance', 'global') == 'per_image':
        patch_loss, patch_diag = equal_budget_patch_bce_loss(
            patch_logit, bce_labels, active_mask=active_patch,
            k_min=float(getattr(cfg, 'patch_k_min', 4.0)),
            patch_weights=bce_pw)
    else:
        patch_loss, patch_diag = selective_patch_bce_loss(
            patch_logit, bce_labels, active_mask=active_patch,
            pos_weight=cfg.patch_pos_weight, patch_weights=bce_pw)
```

HARD CONSTRAINTS:
- `patch_labels` fed to the CONTRASTIVE loss stays the original
  `_mask_to_patch_labels` output. Band labels feed the BCE loss ONLY. The
  contrastive head's inputs must be bit-identical before/after this change.
- When `patch_balance == 'per_image'` and `cfg.patch_pos_weight != 1.0`, log
  ONE warning at epoch start: pos_weight is ignored under per_image.
- Log `realized_P / realized_Q / max_patch_w / n_no_supervision` once per
  epoch (accumulate means across steps, emit in the existing epoch summary
  log line).
- Use `getattr(cfg, ..., default)` for every new cfg field so old configs /
  resumed checkpoints load unchanged.

### A4. Flags (D3) — `experiments/scripts/train.py`

```
--patch_balance {global,per_image}   default 'global'   (global = bit-exact current behavior)
--patch_k_min FLOAT                  default 4.0
--patch_band LOW HIGH                nargs=2, type=float, default None (= no band, current binarize)
```

Defaults MUST reproduce current behavior exactly — a run with none of these
flags is byte-identical to today. Validate `0 < low < high <= 1` at parse
time (mirror the check in `mask_to_patch_labels_soft`).

### A5. Tests (D8) — `tests/test_equal_budget_bce.py`

Torch-dependent tests follow the repo's existing skip idiom (see
`tests/test_full_fakes.py`). Build small synthetic batches (e.g. N=16
patches) with hand-set logits. Required cases:

1. **Single-class identity**: all-fake image and all-real image, band off —
   `equal_budget_patch_bce_loss(..., k_min≤N)` equals
   `selective_patch_bce_loss(..., pos_weight=1.0)` to 1e-6.
2. **Flat budget**: image A with kp=7 of 784(use N=100, kp=7), image B all
   fake — identical confidently-wrong logits on every fake patch ⇒ equal
   `L_i` to 1e-5.
3. **Clamp**: kp=2, k_min=4 ⇒ positive-side `L_i` is exactly half the kp=4
   case's (same per-patch wrongness).
4. **FP pricing**: one false-alarm patch on an image with kn=10 costs 5× the
   same false alarm on an image with kn=50 (ratio 1/10 : 1/50).
5. **Band parity**: random (B,1,S,S) masks (include exact-0, sub-low,
   in-ramp, above-high densities) → `_mask_to_patch_labels_soft_t` ==
   `mask_to_patch_labels_soft` per item (convert tensor→PIL 'L' for the
   reference call).
6. **Degenerate**: image with all patches in the ignore band ⇒ contributes
   0, no NaN, `n_no_supervision` counts it; empty active mask ⇒ grad-safe 0.
7. **Gradients**: `loss.backward()` runs; all grads finite.

---

## Part B — patch-AUROC readout (experiment 0)

### B1. Why it exists

The eval stack only scores committed, thresholded masks
(`lab_utils/eval/metric.py::metric`). But per-image balancing changes output
calibration (outputs become appearance-likelihood-ratios, not
rarity-suppressed posteriors), so ANY fixed-threshold comparison between the
two loss arms measures the calibration shift, not localization quality —
CLAUDE.md rule 1's confound, self-inflicted. The A/B therefore REQUIRES a
threshold-free metric: per-patch AUROC on raw sigmoid scores. This readout is
decoder-free and permanently useful beyond experiment 1.

### B2. Module (D4) — `lab_utils/eval/patch_scores.py`

No sklearn (only a lazy HDBSCAN import exists in-repo; do not add a hard
dep). Numpy implementations:

```python
def weighted_auroc(scores, labels, weights=None) -> float
    # Mann-Whitney / rank formulation with sample weights:
    # sort by score; cumulative weighted TPR/FPR; trapezoid. Handle ties by
    # averaging ranks. Return nan if either class has zero weight.

def collect_patch_scores(model, items, res, *, device, use_amp, amp_dtype,
                         band=(0.2, 0.8), log_tag) -> dict
```

`collect_patch_scores` runs its own forward loop (one forward per item, same
pattern as `numbers.py::_flat_records`, including the every-10% progress
log — stdout is piped on Colab, tqdm is dead, see eval_robustness history):

- Skip items with `meta['gt_mask_reliable'] is False` and items with
  `meta['crop_window']` (geometry not 1:1 with the input frame). Log skip counts.
- Fake items: load GT mask PIL, resize to `(res.image_size, res.image_size)`
  with NEAREST, run `mask_to_patch_labels_soft(mask, res, low=band[0],
  high=band[1])` — the SAME band function, PIL path, no twin needed here.
  Collect `(sigmoid(patch_logits), label, weight)` for patches with weight > 0.
  Stratum: `fake` (label 1) / `splice_bg` (label 0).
- Real items: all N patches are `(score, 0, 1.0)`, stratum `real_bg`.
- Every patch also records its item's `area_to_bucket(item.mask_area(res))`
  (grep for `area_to_bucket` / `BUCKET_LABELS` and import from where they
  live; reals get bucket `'real'`), plus `item_id`.

Returned dict (this shape goes verbatim into the JSON):

```python
{
  'n_items': int, 'n_skipped_unreliable': int, 'n_skipped_cropwin': int,
  'auroc_pooled':      float,   # fake vs (splice_bg + real_bg)
  'auroc_vs_splice_bg': float,  # fake vs splice_bg only
  'auroc_vs_real_bg':  float,   # fake vs real_bg only
  'auroc_by_bucket': {bucket: float},  # that bucket's fake patches vs ALL background
  'score_quantiles': {stratum: {'p50':…, 'p90':…, 'p99':…}},  # sprinkle watch
  'per_image': [ {'item_id':…, 'bucket':…, 'n_fake':…, 'n_bg':…,
                  'scores_fake_mean':…, 'scores_bg_mean':…}, … ],
}
```

`score_quantiles['real_bg']['p99']` is the sprinkle canary: if per_image
training pushes it up materially vs global, the symmetric FP pool failed —
that is a headline result, not a footnote.

### B3. Integration (D5) — `lab_utils/eval/numbers.py`

- Flag: `--patch_auroc` (`BooleanOptionalAction`, default False) and
  `--patch_gt_band LOW HIGH` (nargs=2, float, default `0.2 0.8`).
- In `run()`, per checkpoint per source, when the flag is set AND the model
  has a patch head (`ModelInfo.patch_logits` non-None — probe the first item;
  if None, log and skip), call `collect_patch_scores` and store under
  `out['sources'][src]['patch_auroc']`. Log a compact block: pooled, the two
  vs-strata AUROCs, per-bucket line each, and the three real_bg quantiles.
- This is a SEPARATE forward loop from `_flat_records` — do not entangle
  them. Cost is one extra forward per item; exp-1 eval sets are small.

### B4. Tests (D8) — `tests/test_patch_scores.py`

1. `weighted_auroc`: perfect separation → 1.0; anti-separation → 0.0; random
   interleave → ≈0.5; ties averaged; weights respected (a weight-2 patch ==
   two weight-1 duplicates); one-class → nan.
2. `collect_patch_scores` on a fake model/info stub (monkeypatch the forward
   fetch): synthetic items with known masks → correct strata counts, bucket
   assignment, skip logic for `gt_mask_reliable=False` and `crop_window`.

---

## Part C — Experiment 1: vanilla vs equal-budget, splice-only

### C0. Pre-run audit (D6) — `analysis/audit_patch_budget.py`

Model-free. Builds a source via the registry (reuse
`lab_utils/eval/val_sources.py` arg plumbing), computes per-item banded
`kp` (PIL band path, same as B2), prints: percentiles of kp overall and per
area bucket, fraction of fake items with kp=0 (banded out entirely — these
train as pure background, count them!), and fraction with kp < k_min.
Run BEFORE training:

```bash
python -m analysis.audit_patch_budget --tgif2_root <ROOT> --tgif_types sp \
    --band 0.2 0.8 --k_min 4
```

GATE: if >20% of small-bucket items have kp < k_min, k_min is muting the
exact stratum the experiment measures — STOP and report; do not train.

### C1. Design

- **Data: tgif2 `sp` ONLY.** No full fakes (single-class images are a no-op
  for the loss — they'd dilute the A/B). No `fr` (rule: never average sp and
  fr). Train side = the established hidden split (`--tgif_types sp`); eval =
  the held-out side plus imd2020 (`--imd_val_split 1.0`, never trained, OOD).
- **Arms identical except the loss flags.** Both: image head ON, patch-BCE
  ON, contrastive OFF. `--contrastive_dim 0` is MANDATORY (defaults to 64
  and silently builds an untrained kmeans decoder otherwise — known trap).
  `--patch_pos_weight 1.0` in the global arm (the theory's baseline is the
  plain mean; the default 10.0 would handicap it with a known
  fire-everywhere miscalibration).
- Same seed (42), same epochs/batch/lora as the established recipe
  (448px, lora_rank 16, alpha 32, batch 8, aug medium).

```bash
COMMON="--tgif2_root $TGIF --tgif_types sp --seed 42 \
  --patch_bce --lambda_image_bce 1.0 --lambda_patch_bce 1.0 \
  --lambda_contrastive 0.0 --contrastive_dim 0 \
  --image_size 448 --lora_rank 16 --lora_alpha 32 --batch_size 8 \
  --aug_severity medium --val_decoder threshold"

# Arm A — vanilla
python -m experiments.scripts.train $COMMON --patch_pos_weight 1.0 \
    --patch_balance global --checkpoint_root $RUNS/exp1_arm_global

# Arm B — equal budget
python -m experiments.scripts.train $COMMON \
    --patch_balance per_image --patch_band 0.2 0.8 --patch_k_min 4 \
    --checkpoint_root $RUNS/exp1_arm_perimage
```

(D9 wraps this + C0 + C2 into one script, with the DTYPE autodetect block
copied from `run_scripts/run_t0_full_fakes.sh` — Turing boxes are fp16-only.)

### C2. Eval — one invocation, both checkpoints, identical items

`eval_numbers` is multi-checkpoint: items are built once and scored by each
checkpoint over the identical images. Use that — it makes the bootstrap
pairing exact by construction:

```bash
python -m experiments.scripts.eval_numbers \
    --checkpoint $RUNS/exp1_arm_global/best.pt $RUNS/exp1_arm_perimage/best.pt \
    --label global per_image \
    --tgif2_root $TGIF --tgif_types sp \
    --imd2020_root $IMD --imd_val_split 1.0 \
    --decoders threshold --patch_auroc \
    --amp_dtype float16 \
    --out_json $RUNS/exp1_patch_auroc.json
```

Threshold-decode f1 may be *looked at* but is NOT a comparison metric between
arms (calibration confound, B1). The comparison lives entirely in the
patch_auroc block.

### C3. Comparison (D7) — `analysis/compare_patch_auroc.py`

Input: the out_json (or two). Paired bootstrap: resample IMAGES with
replacement (patches within an image are correlated — never bootstrap
patches), same resampled item_id multiset applied to both arms, 1000 reps,
report mean Δ and 95% CI for: pooled AUROC, each bucket's AUROC, and
`real_bg` p99. Print absolute values for both arms first, Δ as additional
info (Jake's reporting conventions: absolute numbers, means, comparisons as
extra).

### C4. Decision gate (pre-registered — write the answer down before running)

- **PRIMARY**: smallest-area-bucket patch AUROC, Δ(per_image − global). CI
  excluding 0 in favor of per_image ⇒ mechanism confirmed, proceed to the
  2×2 full-fakes experiment.
- **GUARD 1 (sprinkle)**: `real_bg` p99 score for per_image not worse than
  global beyond CI noise. Violated ⇒ the symmetric FP pool failed in
  practice — headline finding, stop and analyze before proceeding.
- **GUARD 2 (detection)**: image_auc within noise between arms (the image
  head must not pay for the patch fix).
- **KILL**: primary Δ ≤ 0 ⇒ the redesign is dead. Do not tune
  k_min/band/λ to resurrect it; report and stop.

---

## Do-NOT list (each item has burned this repo before)

1. Do NOT modify `_mask_to_patch_labels`, `selective_patch_bce_loss`, or the
   contrastive loss's inputs. New function + new call path only.
2. Do NOT let any new flag's default change current behavior — flagless runs
   must be byte-identical to today.
3. Do NOT compare the two arms at any fixed decode threshold (rule 1).
4. Do NOT report full-fakes localization anywhere (rule 2) — n/a here, but
   the eval prints it if you let it.
5. Do NOT mix `fr` into experiment 1 or average sp/fr anywhere (rule 5 +
   SAGI-D provenance memory).
6. Do NOT run per_image against pseudo-mask sources (a spurious 1-patch blob
   inherits an entire image budget).
7. Do NOT touch the sampler / `--balance_real_fake`.
8. Do NOT use tqdm-only progress in any new loop — stdout is piped on Colab;
   use the `log_line` every-10% idiom.
9. Scripts do not import each other (C-script invariant) — shared logic goes
   in `lab_utils/`, thin entries in `experiments/scripts/`.

## Verification checklist (run all before calling it done)

```bash
python3 -m py_compile lab_utils/model/losses/bce.py lab_utils/train/loop.py \
    lab_utils/eval/patch_scores.py lab_utils/eval/numbers.py \
    analysis/audit_patch_budget.py analysis/compare_patch_auroc.py
python3 -m pytest tests/test_equal_budget_bce.py tests/test_patch_scores.py -q
python3 -m pytest -q          # full suite — zero regressions
python -m experiments.scripts.train --help          # flags present, defaults right
python -m experiments.scripts.eval_numbers --help   # --patch_auroc present
```

Plus one semantic smoke: a 2-epoch tiny run (`--train_samples 64
--num_epochs 2`) per arm on any splice source — confirm the per_image arm
logs `realized_P/realized_Q` near 1.0 and `max_patch_w ≤ 1/k_min`, and the
global arm's loss curve matches a pre-change run at the same seed.
