# Component-isolation sweep — head interference under a shared backbone

**Status:** spec, not yet run. Written 2026-07-19.
**Box:** 2080 Ti ×2 (Turing → **fp16, never bf16**), dynamic 2-GPU claim queue.
**Prereq reading:** `CLAUDE.md` methodology rules 1–7, `ANALYSIS_NOTES_bce_emergence.md`.

## 1. The question

Everything in the class-targeting plan (5-way readout over per-head signatures) rides on
one unverified assumption: **that a single DINO backbone can carry several largely
independent tasks at once without them degrading each other.** Adding a patch-BCE
AI-ness head, a noise head, and full fakes to the diet all at once and measuring only
the end result cannot distinguish "the design works" from "two additions cancelled."

So each add-on is verified **independently against a shared baseline** before any are
combined, and the combination cells measure the *interaction* rather than assuming it
is zero.

Each cell answers exactly one question:

| cell | question |
|---|---|
| `*_base` | what does this objective do alone, at medium aug? (the anchor) |
| `*_ff` | does putting whole-image fakes in the diet cost localization on local edits? |
| `*_nz` | does a noise head steal capacity from / get entangled with the semantic heads? |
| `*_ffnz` | are the two effects additive, or do they interact? |
| `dual_*` | do patch-BCE and contrastive interfere when trained on one backbone? |

## 2. Design — 3 × 4 factorial (12 cells) + 1 gate run

| | base | +full-fakes | +noise head | +FF +noise |
|---|---|---|---|---|
| **BCE** | `bce_base_s0` | `bce_ff_s0` | `bce_nz_s0` | `bce_ffnz_s0` |
| **Cont** | `cont_base_s0` | `cont_ff_s0` | `cont_nz_s0` | `cont_ffnz_s0` |
| **Dual** | `dual_base_s0` | `dual_ff_s0` | `dual_nz_s0` | `dual_ffnz_s0` |

Full factorial, not a star design: every main effect *and* every two-way interaction is
separately estimable. Dropping `dual_base` or `*_nz` would confound the interaction
cells — if `dual_ff` underperforms you could not tell whether the dual-head coupling or
full fakes caused it.

**`dual_*` cells run WITHOUT an orthogonality penalty.** Their job is to measure raw
BCE↔contrastive interference. Baking in the remedy first would make it impossible to
say whether it was ever needed. If interference shows, orthogonality goes in as a
labelled remedy arm afterward — and that ordering is itself the paper-ready ablation.

**Pico gate (`cont_pico_s0`)** — separate, not part of the factorial. Folds
`pico_pseudo` into `cont_base_s0`'s diet and regression-checks IMD/TGIF-FR. Run on the
`cont` config because production configs are contrastive, so that is the regression that
matters. If clean → fold pico into everything downstream; if it degrades, that is a
mask-quality finding and gates the "little guy" mask-repair mini-work.

## 3. Fixed recipe (all 12 cells + gate)

Mirrors `sweeps/sweep_bce_emergence.json` `base_args` so the two studies stay
comparable, with **one deliberate change**:

```
image_size 448, patch_size 16, lora_rank 16, lora_alpha 32
batch_size 8, grad_accum 1, contrastive_dim 64, pool_hidden 256
lambda_image_bce 1.0, paste_frac 0.4, fr_bg_negative_prob 0.12
seed 42, train_samples 3000, num_epochs 10, min_epochs 5
warmup_epochs 1.0, early_stop_patience 2, val_zoom true
casia_train true, imd_val_only true          # IMD is NEVER trained on — hard rule
aug_severity medium                          # <-- THE deliberate change
```

`--aug_severity medium` = prob 0.35, JPEG q55–80, gaussian σ 0.04–0.12, resize
0.70–0.92, poisson peak 32–96. This is Jake's "slightly increased, very not extreme" —
`heavy` (q35–65, σ up to 0.30) is the next tier up and is NOT used here.

**Fresh medium-aug baselines are mandatory, not redundant.** The six `bce_emergence`
cells were trained at `light` (prob 0.0 — the heavy harness never fires), so they cannot
anchor a medium-aug comparison. Bonus: `*_base_s0` (medium) vs `bce_emergence` (light)
is a free, already-paid-for ablation of the aug bump itself — which the challenge
robustness findings say we want measured anyway (rank-robust ≠ threshold-robust).

Train mix is held **fixed** at the `both` recipe across all 12 cells —
`sagid=0.25, coco_inpaint=0.25, casia=0.5`. The bce_emergence study already varied the
mix; varying it here too would make this 36 cells to answer a question about heads.

## 4. Per-cell flags

Objective axis:

| objective | flags |
|---|---|
| `bce` | `--patch_bce --lambda_patch_bce 1.0 --lambda_contrastive 0.0` |
| `cont` | `--lambda_contrastive 2.0` (no `--patch_bce`) |
| `dual` | `--patch_bce --lambda_patch_bce 1.0 --lambda_contrastive 2.0` |

Add-on axis:

| add-on | flags |
|---|---|
| `+ff` | `--full_fakes_root <root>` and splice_mix → `sagid=0.21 coco_inpaint=0.21 casia=0.43 full_fakes=0.15` |
| `+nz` | `--noise_head --lambda_noise <w>` (**to be built — §5 B3**) |
| gate | `--pico_pseudo_root <root>`, splice_mix → `sagid=0.21 coco_inpaint=0.21 casia=0.43 pico_pseudo=0.15` |

FF weight 0.15 = base three scaled by 0.85. Rationale: enough signal for the image head
and patch-BCE to learn "whole frame is fake" without letting a source that carries **no
localization signal** (rule 2) dominate the diet.

## 5. Data routing rules (the part that is easy to get silently wrong)

**R1 — Full fakes feed the image head and patch-BCE, never the contrastive head.**
An all-fake crop has no real patches to contrast against; contrastive on it is
degenerate (rules 2/3). Note it would NOT crash today — `selective_symmetric_contrastive_loss`
takes `is_single` and downweights single-class items to `single_class_weight=0.05` — so
without an explicit gate the effect is a quiet 5%-weight contribution of pure noise.
Gate it properly.

**R2 — Patch-BCE on full fakes gets the all-fake target for free.**
`full_fakes.build` attaches a shared all-white sentinel mask (`_synthetic_full_mask`),
so every patch is labelled fake with no new target construction. Start with **plain
BCE**; the worst-violator-dominated loss is a *second* experiment, not the baseline —
do not stack two novelties in an isolation test.

**R3 — Splices are NEGATIVES for the noise head, not positives.**
The head must answer "was this content *generated*", not "was this edited". Spliced real
content carries no generator fingerprint. This is what preserves the class-2-vs-3
boundary in the signature table.

**R4 — Never average sp and fr localization; never compare raw F1/IoU across the
bce↔cont boundary** (rule 1: decoder is confounded with objective). Cross-objective
comparisons use AUROC, or same-decoder re-eval.

## 6. Build items

| id | scope | size | blocks |
|---|---|---|---|
| **B1** | `--full_fakes_root` flag + `source_map` entry in `experiments/scripts/train.py` (roots group ~L70, map ~L311) | **2 lines** | T2 |
| **B2** | Gate FF out of contrastive: thread a per-item sentinel flag into the batch and AND it into `active_cont` at `lab_utils/train/loop.py:174` | small, follows the existing `active_mask` precedent | T2 |
| **B3** | **Noise head** — separate small residual encoder (Bayar/SRM or NoisePrint++-style), NOT a DINO subspace; image-level label; late fusion into the image logit; v1 emits a scalar, family-vector comes later | **the real build; needs its own spec** | T3, T4 |
| **B4** | Pico — none needed, `--pico_pseudo_root` already exists | 0 | — |

Verified non-issues:
- `full_fakes` is already in `lab_utils/data/datasets/registry.py` — no builder needed.
- `train.py` passes **no** `verify_policy`, so `full_fakes.build`'s relaxation of
  `max_mask_area` to 1.0 applies and the 100%-coverage sentinels survive. (Passing an
  explicit policy would silently drop every FF item — do not add one.)
- `_mask_kind()` in train.py's data-diet summary already recognises `sentinel`, so the
  pre-flight table will show FF correctly.

## 7. Eval protocol

Per cell, against the **matching medium-aug baseline** (never against bce_emergence):

1. `probe_eval2`-style clean detection/localization with the `fr_bg_matched` null
   (rule 4 — declare the null; interior AUROC vs matched `real_crop`, tgif2-restricted
   per rule 5).
2. `noise_probe` JPEG ladder (clean/90/70/50/30) — the compression-consistency tripwire
   (see class-targeting plan: if boundary-crop localization F1 degrades much faster than
   image AUROC, the paired-consistency loss earns its cost).
3. `full_fakes_eval` — **image-level AUC only** (rule 2).
4. Per-class signature harness (OpenFake→1, CASIA/IMD→2, SAGI-D/TGIF→3, pico→4,
   reals→0) — the diagnostic pass that makes these cells interpretable as class evidence.

Report **means**, absolute numbers, with bootstrap CIs vs chance (rules 6 and the
reporting conventions). Condition labels `BCE·X` / `Cont·X` / `Dual·X`.

## 8. Scheduling (2080 Ti ×2)

Reuse the existing runner unchanged — it is generic over the queue file:

```bash
QUEUE=sweeps/sweep_component_isolation.json \
RUN_ROOT=/media/ssd/runs/component_isolation \
bash run_scripts/run_bce_emergence_queue.sh
```

Atomic `mkdir` claim per cell, resume-safe (`ORCH_DONE.json`); retry a failed cell with
`rm -r "$RUN_ROOT/<cell>/.claim"`.

Tiers are ordered so **every tier is interpretable on its own** — a slipped night never
produces an unreadable partial result:

| tier | cells | needs | note |
|---|---|---|---|
| **T1** | `bce_base`, `cont_base`, `dual_base`, `cont_pico` (gate) | **nothing — runnable today** | 4 runs; establishes every anchor |
| **T2** | `bce_ff`, `cont_ff`, `dual_ff` | B1 + B2 | 3 runs |
| **T3** | `bce_nz`, `cont_nz`, `dual_nz` | B3 | 3 runs |
| **T4** | `bce_ffnz`, `cont_ffnz`, `dual_ffnz` | B1+B2+B3 | 3 runs; interactions only readable after T2+T3 |

Wall-clock per cell: **measure it off the first T1 cell** rather than guessing — 3000
samples × ≤10 epochs at 448/r16 on Turing fp16, and the box's throughput is not recorded
anywhere trustworthy.

## 9. Decision rules (write these down BEFORE reading results)

- **Dual-head interference:** `dual_base` vs `bce_base` (on patch-BCE metrics) and vs
  `cont_base` (on contrastive metrics). Degradation beyond the bootstrap CI on **either**
  side ⇒ interference is real ⇒ orthogonality-penalty remedy arm is justified. No
  degradation ⇒ the heads coexist and the class-targeting design's core assumption holds.
- **FF cost:** `*_ff` vs `*_base` on **local-edit** localization. FF is supposed to buy
  full-fake image AUC; if it costs local localization beyond CI, the diet weight (0.15)
  is the first knob, not the design.
- **Noise entanglement:** `*_nz` vs `*_base` on semantic-head metrics. The noise branch
  is a *secondary* signal — if it degrades the semantic heads at all, late fusion is not
  isolating it and the head needs to be more detached.
- **Interaction:** `*_ffnz` vs additive prediction from `*_ff` and `*_nz`. A large
  negative residual means the two add-ons compete for capacity — that is the finding
  that would force a bigger backbone or staged training.
- **Pico gate:** IMD/TGIF-FR regression within CI ⇒ fold pico in everywhere.

## 10. Known gotchas

- **fp16 only on this box** (Turing). Never bf16.
- Do not pass `--verify_policy` on FF cells (§6).
- `--splice_mix` is **ignored under DDP** (train.py:502) — keep these single-GPU-per-cell,
  which the claim queue already does.
- Cell names carry the `_s0` suffix = seed 42; replicates are re-queues with a new seed
  suffix, matching bce_emergence convention.
- IMD stays `imd_val_only` in every cell. It is never trained on, anywhere, in either study.
