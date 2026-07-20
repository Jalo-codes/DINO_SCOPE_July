# Phase 0 — noise-subspace exploration (generators-as-cameras)

**Status:** spec, not yet run. Written 2026-07-19.
**Platform:** Colab T4 (Turing → **fp16 only, never bf16**; ~2 CPUs → `--num_workers 2`;
checkpoints to Drive).
**Prereq reading:** `CLAUDE.md` rules 1–7; `docs/component_isolation_sweep.md` (this is the
B3 prerequisite).

## 1. What this phase is for

Before the noise head can enter the component-isolation sweep (cells `*_nz`, `*_ffnz`), we
need to know three things that no amount of design settles:

1. **Is there a generator fingerprint in LoRA-adapted DINO patch tokens at all?**
2. **Does the embedding encode fingerprint, or is it laundering semantics?**
3. **Does a fingerprint trained with zero mask supervision localize?**

All three are answerable with **image-level labels only**. That is the whole reason this
phase comes first: it is unblocked by the mask-quality work, needs no new dataset, and
fails cheaply.

**Framing (Jake, 2026-07-19):** the goal is not a standalone Noiseprint. It is a *bounded*
noise channel — enough signal to be useful, low enough capacity that it cannot poison the
semantic heads downstream. Low capacity is a feature, not a compromise.

## 2. The core idea

Noiseprint learns a camera-model fingerprint from a Siamese net where **positive pairs are
patches from the same camera model AND the same spatial position**, using a distance-based
logistic (DBL) loss, with zero forgery supervision. Localization at test time is *blind*:
per-image anomaly detection over the residual.

We port it by treating **each generator as a camera model**.

This is nearly free to set up: `lab_utils/data/datasets/full_fakes.py` already discovers
`root/real/` + `root/<generator>/` and writes the folder name to `meta['generator']`
([full_fakes.py:113](../lab_utils/data/datasets/full_fakes.py)). That *is* the camera-model
label, already parsed. `real/` becomes one more class.

### Why it may map well
Decoder artifacts are periodic with a fixed phase (VAE/diffusion upsampling grid), the
direct analogue of the JPEG 8×8 grid and CFA sampling pattern that make Noiseprint's
position-matching necessary. (Position-matching itself is dropped in this port —
rescale-as-degradation scrambles phase by design, see P2/P3 — so what is hunted is the
scale-robust remainder of these artifacts, not their phase.)

### Why it may not — predict this up front
A camera model is a **fixed** pipeline. A generator is not: sampler, step count, CFG scale,
scheduler, seed and prompt all vary *within* one folder. Within-class variance will be far
higher than Noiseprint ever faced. §7 R0 measures this before any training.

## 3. Hard preconditions (the ways this experiment lies to you)

**P1 — Container-format leakage. Audit before training, no exceptions.**
If `sdxl-juggernaut/` is all PNG and `real/` is all JPEG q90, the task is trivial and the
result is meaningless. Before anything: tabulate per generator folder the file extension
mix, native resolution distribution, and JPEG quantization table where present. If they
differ systematically, **re-encode every image through one identical pipeline** first.
This is the single most likely way to get a beautiful, worthless number.

**P2 — Geometry is a degradation, not a constant: random crop + random rescale on EVERY
image.** (Jake, 2026-07-19 — supersedes the earlier crop-only rule.) Scale is itself a
real confound: generators correlate with native resolution, so a no-rescale pipeline lets
the head cheat on scale statistics instead of generator physics — the P1 logic extended
to geometry. Randomizing crop + rescale independently per image makes scale uninformative
in expectation (same argument as P4) and matches deployment, where challenge images
arrive pre-resized by unknown hands. Cost, on record: the grid-phase-locked component of
the fingerprint is deliberately traded away for the scale-robust component. Jake's bet:
harder to find, not destroyed. The rescale ladder (§8 A1) prices that bet.

**P3 — Same-image pairs are MASKED out of the loss.** With position-matching dead
(rescale scrambles phase — §5), groups collapse to generator-only, and that quietly makes
tokens from the SAME image positives for each other — a pairing content encoding
satisfies for free. Positives must be cross-image: same generator, DIFFERENT image.
Same-image pairs are excluded from both the positive set and the softmax denominator —
not labelled negative (they genuinely share the fingerprint; pushing them apart fights
the signal), simply absent. This masking REPLACES position-matching as the anti-semantic
mechanism, and it is the single line of the loss most worth unit-testing.

**P4 — Degradation randomized independently per image, and only after base normalization.**
Per Jake: plop varied noise on every leg rather than matching one. Correct, and stronger
than matching — it makes degradation uninformative in expectation and covers degradations
we did not think to match. Two conditions: draws are **independent per image** (one shared
draw re-introduces the confound), and P1's base normalization happens **first** (random
degradation on top of a systematic gap dilutes it, it does not erase it).
Run `deg_strength=0` first to establish the clean ceiling, then ladder up.

**P5 — NO POOLING, ANYWHERE.** The loss operates on per-patch residuals. Pooling to an
image vector kills localization dead and you would not discover it until decode time.
Noiseprint's training images are homogeneous too (one camera each, never a mixed image) and
localization still works — because the loss pushes *patches* from different models apart,
so at test time a spliced region's patches land elsewhere in the space. Homogeneous
training data is fine. Pooling is not.

## 4. Architecture

```
NoiseHead(nn.Module):
    proj: Linear(feat_dim -> d_noise)      # mirrors contrastive_proj exactly
    forward(patch_feats):                  # (B, N, feat_dim)
        r = proj(patch_feats)              # (B, N, d_noise)
        return F.normalize(r, p=2, dim=-1) # L2-norm PER PATCH
```

- **Backbone: LoRA-adapted DINOv3.** Jake is certain (2026-07-19) the noise signal only
  surfaces under LoRA fine-tuning, not frozen. Independently corroborated: GAPL
  ([arXiv 2512.12982](https://arxiv.org/abs/2512.12982), CVPR 2026) diagnoses a
  *frozen-encoder bottleneck* as the cause of detector degradation under generator
  diversity, and fixes it with LoRA. The frozen arm is therefore **dropped**.
- Phase 0 trains **noise head only** — no image head, no patch-BCE, no contrastive. Full
  isolation. Interference is measured later, in the sweep's `*_nz` cells.
- `d_noise = 64` (matches `contrastive_dim` default).
- `image_size 448, patch_size 16` → **N = 784 tokens** per crop, on a 28×28 grid.

**The reuse win:** this output is shape-identical to `forward()['contrastive']` —
`(B, N, d)`, L2-normalized per patch ([multi_head_detector.py:215](../lab_utils/model/multi_head_detector.py)).
So `decode_kmeans`, `decode_hdbscan`, and the existing eval/record machinery consume noise
embeddings **with no changes at all**. Localization evaluation is nearly free.

## 5. Batch construction (adapted from Noiseprint's group structure)

Noiseprint: minibatch = 200 patches as **50 groups of 4**, each group internally homogeneous
(same model, same position), heterogeneous across groups; lifted to O(N²) pairs. Position
is dropped in this port (P2/P3), so the group key is **generator alone** and geometry is
fully independent per image:

```
per batch:
  draw 4 generator classes  (real/ counts as one class)
  draw 4 images per class                                 -> 16 image forwards
  per image, INDEPENDENTLY: random crop -> random rescale -> random degradation (P4)
  draw P = 12 token positions per image (free to differ across images)
  element set = 16 x 12 = 192 tokens
  group key   = generator
  loss mask   = all same-image pairs excluded (P3)
```

Per anchor token: **36 positives** (tokens from the 3 other same-generator images),
**11 masked** (same image), **144 negatives** (other generators). 4 images per class is
kept deliberately: each anchor draws positives from 3 distinct partner images, matching
Noiseprint's 3-positive-partner structure and preventing any single partner image's
content from dominating the positive signal. (8 classes × 2 images is the
higher-class-diversity alternative at the same forward cost — a knob, not a redesign.)
16 forwards per step fits a T4 at 448/fp16.

## 6. Loss — distance-based logistic, not triplet

From the paper (verified against the source PDF, §III):

```
d_ij  = || r_i - r_j ||^2                                  # squared Euclidean between residuals
p_i(j) = exp(-d_ij) / sum_{k != i, img(k) != img(i)} exp(-d_ik)
L_i    = -log sum_{j : gen(j) = gen(i), img(j) != img(i)} p_i(j)
L      = mean_i L_i
```

Same-image tokens appear **nowhere** — neither as positives nor in the denominator (P3).

Chosen over triplet deliberately: it lifts one batch into O(N²) pairs instead of O(N),
which removes triplet mining entirely and matters on a T4. Their note on why negatives
carry weight is worth keeping in view — negatives *teach the network to discard
information common to all models*, which is precisely the semantic-suppression mechanism.

## 7. Readouts (pre-registered — decide these before looking)

**R0 — Feasibility precheck, BEFORE training.** On raw LoRA-DINO tokens, compute mean
within-generator vs between-generator distance. Ratio ≈ 1.0 is expected pre-training; this
is the baseline the trained numbers must beat, and it also surfaces P1 leakage early (a
ratio already far from 1 on an untrained model means format leakage, not fingerprint).

**R1a — Leg vs image separation (Run 1 only). Uninformative ≠ suppressed.**
Corresponding triplets make content *uninformative* — it cannot help discriminate the
legs — but nothing **removes** it. An embedding that is 90% semantics plus one
discriminative noise direction satisfies the loss perfectly. So measure, in the same
batch, whether the embedding separates by **leg** (original / benign / AE — wanted) or by
**image** (content — not wanted):

    S_leg  = between-leg  distance, content held fixed
    S_img  = between-image distance, leg held fixed

`S_img >> S_leg` ⇒ content dominates the subspace even though it earned no gradient.
Discriminative honesty is intact; **isolation is not**, and isolation is what the
downstream decoders need (see R2 — k-means over a semantics-dominated embedding clusters
object boundaries, not noise boundaries, and returns plausible-looking wrong masks).

**R1 — Fingerprint vs semantics. The decisive one, and label-free.**

| statistic | definition |
|---|---|
| `R_gen` | mean distance, same generator, **different** image |
| `R_img` | mean distance, **same image**, different position |
| `R_rand`| mean distance, random token pairs |

- Want `R_gen << R_rand` — a fingerprint exists.
- Want `R_img` **not** much below `R_gen` — if same-image tokens cluster far tighter than
  same-generator tokens, the head is encoding image content, not generator physics.
- `R_img << R_gen` ⇒ **semantic leakage; stop and fix before proceeding.**

No semantic labels needed — image identity is the proxy, which is why this is cheap.
Bonus of P3: the loss never touches same-image pairs in either direction, so `R_img` is a
fully held-out instrument — if it collapses, content leaked in through the cross-image
positives, and there is no way to blame the loss's own bookkeeping.

**R2 — Blind localization.** Run the trained head on held-out **spliced** images
(CASIA / IMD / TGIF), take `(N, d)` token embeddings, decode with the **existing**
`decode_kmeans`, score against GT masks.
Report **boundary-crop** F1/IoU (rule 3 — interiors are degenerate) and image AUROC with a
declared null (rule 4). Compare against that checkpoint's own contrastive decode.
**Do not compare raw F1 across the bce↔cont boundary** (rule 1).
*This is the headline result: localization from a model that never saw a mask.*

**OpenFake's split structure maps onto these readouts for free** (dataset card, verified
2026-07-19): `core/train` ≈ 2.31M rows carrying all in-train generators; `core/validation`
≈ 59K = held-out *images* from those same generators; `core/test` ≈ 91.4K = held-out
**OOD generators** (gpt-image-1.5, gpt-image-2.0, nano-banana-pro …). 100+ generators
exist overall. So: **train** learns fingerprints, **validation** is R1 (same generators,
unseen images), **test** answers whether family structure survives to *unseen* generators —
which is the actual class-3-vs-4 question. This also explains why a test-split pull yields
only ~20 subfolders: that split is a deliberately small OOD set, not the generator census.
Generators-as-cameras therefore requires the **train** split.

**R3 — Family structure.** Hierarchical clustering + confusion matrix over per-generator
centroids. Does it group by *family* (diffusion together, GAN together) or by individual
generator? This is the direct evidence for whether the class-3-vs-4 family vector is
viable. GAPL reports prototypes organizing by artifact family rather than instance, so
family-level structure is the expected outcome.

**Kill criteria, written in advance:**
- `R_gen / R_rand` ≈ 1 after training ⇒ no fingerprint survives into LoRA-DINO tokens. Stop;
  the Bayar-conv encoder arm becomes the fallback, not a refinement.
- `R_img << R_gen` ⇒ semantic leakage. Fix (harder negatives / stronger degradation) before
  any localization claim.
- R2 image AUROC ≈ 0.5 ⇒ fingerprint exists but does not transfer to localization; the head
  is then a detection-only signal and the localization ambition is dropped.

## 8. Ablations

| id | question | arms |
|---|---|---|
| **A1** | How much fingerprint survives per unit rescale? (prices P2's bet) | rescale ladder: none / mild 0.9–1.1 / medium 0.7–1.3 / strong 0.5–1.6 |
| **A2** | Which blocks carry noise? | `lora_block_start/end` early vs late vs all |

The old position/grid-phase ablations died with position-matching — rescale scrambles
phase, so there is no phase structure left to ablate. A1's `none` arm does double duty:
it is the clean ceiling AND the post-mortem on the phase-locked bet — a large none→mild
drop means the phase-locked artifact was real and P2 knowingly spent it; a flat curve
means it was never load-bearing and rescale is free robustness. Either way, information.

## 9. Deliverables

1. `R0/R1/R2/R3` table + the A1–A2 grid.
2. Go/no-go on the noise head entering the sweep as B3, with the encoder decision
   (LoRA-DINO linear head vs. from-scratch Bayar-conv) settled by evidence.
3. If R2 lands: a `decode_noise_em` sketch — Noiseprint's actual localizer is co-occurrence
   features + EM fitting pristine-vs-manipulated *per image*, which is **self-calibrating**,
   the same property that makes k-means beat a fixed threshold under rule 1. Would be a
   genuinely third decoder, independent of both current ones.

## 10. Scope decision, 2026-07-19 (Jake)

**Runs 2 and 3 are scrapped for now** — non-corresponding triplets and
generators-as-cameras both. Not disproven, deferred: there is real signal to chase in
both, and §2/§5 above are kept as the record of how to build them.

**Run 1 (corresponding triplets) is the program.** Its decisive property is that content
is controlled *by construction* rather than in expectation — anchor, positive and
negative are the same image, so semantics cannot discriminate, full stop. Everything
elaborate in this spec (same-image masking, the R1 ratios, degradation randomization)
exists only to approximate in Runs 2/3 what Run 1 gets for free. Dropping them removes
the tax.

The cost is stated in R1a: Run 1 forbids semantic cheating but does not force semantic
*suppression*. Isolation must therefore be bought explicitly, not inherited from the
sampling design — an open question at time of writing, not a settled part of this spec.

## 11. Phase 0.5 — Type A/B triplets (deferred, same harness)

Once the generators-as-cameras harness exists, the real-vs-AE triplets reuse it directly:

- **Type A (same image):** anchor = real patch, positive = degraded real, negative = AE
  repass. High precision, low generalization. Content is fixed, so it cannot force semantic
  invariance on its own.
- **Type B (cross image):** anchor = AE(x₁), positive = AE(x₂) same generator, negative =
  real(x₁). **Stronger than Noiseprint's construction for semantic suppression** — the
  negative shares content with the anchor while the positive does not, so any
  content-encoding direction *simultaneously* increases `d(anchor, pos)` and decreases
  `d(anchor, neg)`. Semantics is actively penalized, not merely left unlearned.

Start 50/50; R1's ratios say which to upweight. P4 (independent per-leg degradation) is
what keeps Type A honest — otherwise it separates on "was this JPEG'd" and never looks at
the AE pass.

**The AE corpus is cheaper than it looks, and it buys the strongest control in the
program.** Take the existing real pool and push every image through N off-the-shelf
autoencoders (SD1.5 KL-f8, SDXL VAE, Flux/SD3 AE, TAESD, one VQ model) — encode–decode
only, no diffusion. Embarrassingly parallel background Colab job; output drops straight
into the full_fakes layout (`root/<ae_name>/` + `root/real/`). Because every class shares
the SAME source images, semantic leakage is impossible **by construction** — this is the
content-controlled version of generators-as-cameras, the same corpus is Type A/B fuel,
and it is the industrial form of the class-targeting plan's fr/change-nothing pair.
Ordering (Jake): generators-as-cameras first, on what is already on disk.

## 11. Known gotchas

- **fp16 only** on T4 (Turing). Never bf16.
- Colab roots: `--casia_root /content/casia`, `--imd2020_root /content/IMD2020`,
  `--sagid_root /content/sagi_d_partial`, checkpoints → `/content/drive/MyDrive/DINO_SCOPE_RUNS/<run>`.
  `full_fakes` root is **not** in the existing Colab root table — needs uploading/mounting.
- `full_fakes.build` returns `(empty_train_ds, val_ds)` — it is **eval-only** today. Phase 0
  trains on it, so it needs a train-split path (or index the folders directly in the
  notebook, which is the lower-risk choice for an exploration phase).
- Do not pass an explicit `verify_policy` to `full_fakes.build` — it silently drops every
  fake item (the sentinel mask is 100% coverage and `DEFAULT_POLICY` caps at 99%).
- Sentinel masks are irrelevant to Phase 0 (no mask supervision) but still gate `is_real`.

## Sources

- Noiseprint — [arXiv 1808.08396](https://arxiv.org/abs/1808.08396) ·
  [GRIP-UNINA](https://grip-unina.github.io/noiseprint/). Position-matching, DBL loss, and
  the 50×4 group structure verified against the paper text.
- Splicebuster (Noiseprint's blind localizer: co-occurrence features + EM).
- GAPL — [arXiv 2512.12982](https://arxiv.org/abs/2512.12982), CVPR 2026. Frozen-encoder
  bottleneck, LoRA fix, family-level prototype structure.
</content>
</invoke>
