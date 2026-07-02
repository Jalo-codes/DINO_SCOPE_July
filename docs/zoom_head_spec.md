# Learned Zoom Head — Full Model Specification

**Status:** design locked 2026-06-24. Supersedes the original `train_single_box`
GT-padded-box recipe. Implementation entry points: `lab_utils/model/box_heatmap.py`
(architecture), `experiments/labs/box_heatmap_lab.py::box_heatmap_train_item`
(target/reward), `lab_utils/eval/multibox.py` (decode).

---

## 0. One line

A learned, **frozen-backbone** module that decides *where to crop ("zoom")* to
maximize downstream localization F1 — by gating HDBSCAN regions of a learned
low-dim contrastive projection with a **per-cluster advantage predictor**.

It replaces the current heuristic ("crop where attention is hot") with a head
trained on the realized downstream reward.

---

## 1. Objective

Maximize **realized localization-F1 improvement over the no-zoom baseline**. The
no-zoom flat decode is a *safety floor*: skipping a zoom keeps flat-F1; a wrong
zoom can fall *below* it (magnifying a clean crop hallucinates a confident splice).
The whole decision structure follows from that asymmetry.

**Defensibility:** `evaluate()` caches static per-item reference F1 for flat /
attention-zoom / hdbscan-zoom on the frozen detector, so every epoch prints the
head's zoom-F1 **head-to-head vs the attention-zoom heuristic**. Learned > heuristic
is the publishable claim.

---

## 2. Design principles (the hard-won constraints)

1. **DINO features are smooth & region-coherent** — great at "this region is
   coherent," terrible at "this patch vs its neighbor." So **every decision is
   per-REGION, never sharp per-patch.** No CenterNet peaks, no per-patch box
   regression, no global magnitude threshold.
2. **It's an offline contextual bandit, not RL.** Frozen backbone ⇒ the reward
   `F1(zoom→region) − F1(no-zoom)` is a cheap, deterministic, queryable function.
   So: **search the reward and distill the winners** (advantage-weighted
   regression / expert iteration), never policy gradient.
3. **The unit of decision is a CLUSTER.** This fixes credit assignment (per-cluster
   advantage) and the mixed-messages problem (per-region supervision) at once.
4. **Reward = advantage** (improvement over the no-zoom baseline). Baseline
   subtraction cancels per-item difficulty — the key variance reducer.
5. **Separate the predictor from the operating point.** The head *predicts*
   advantage (learned, improvable). The gate threshold δ encodes the *cost
   appetite* (fear of FP-zoom vs missed gain) — a tuned scalar you own, NOT a
   gradient target.
6. **MIL is a feature, never a hard filter.** Zoom is attempted broadly (incl.
   image-MIL-"clean" images) for recall on hard misses; FP-zoom is handled by the
   per-cluster gate + fallback, not by skipping images.

---

## 3. Architecture

```
                      ┌──────────────── FROZEN (reused) ────────────────┐
  image ─► DINOv3+LoRA ─► patch feats ─► contrastive head ─► z   (64-d) │
                                      └─► MIL pool head ───► attention, image_logit
                                      └─► (patch head) ───► patch_logit (optional)
                      └─────────────────────────────────────────────────┘
                                              │
                      ┌──────────────── LEARNED (light) ────────────────┐
                z ─►  Projection P : 64 → ~8–16d, L2-norm  ─► z'         │
   [z' , attn , patch_logit , per-crop MIL feat] ─► ZoomValue head V ─► per-patch scalar
                      └─────────────────────────────────────────────────┘
```

- **Frozen:** backbone + LoRA + contrastive head (`z`) + MIL pool head + patch head.
- **Learned (the only new params), both light:**
  - **Projection `P`**: `z (64) → z' (~8–16d)`, L2-normalized. Linear or 1-layer
    MLP. Purpose: a low-dim, **zoom-coherent** space where HDBSCAN is clean
    (density estimation degrades in 64-d → raw-`z` clusters are noisy).
  - **Zoom-value head `V`**: per-patch scalar (≈ the existing `BoxHeatmap` trunk).
    Inputs: `z'` (and/or `z`) + MIL attention + patch_logit + per-crop MIL feature.

---

## 4. Clustering / region extraction

1. **HDBSCAN on `z'`** — variable cluster count (the "non-fixed number" we want),
   density-based, noise label `-1` = "not a region." `min_cluster_size` is a
   *relative* split knob (reward-tunable), not a magnitude threshold.
2. **Spatial coherence:** embedding clusters are *semantic*, not spatial — two
   separate-but-similar splices land in one cluster → one huge low-magnification
   box. Fix: **connected-components within each embedding cluster** before boxing
   (keeps the semantic partition, enforces spatial contiguity at box time).
   *(Alt considered: cluster on `z' ⊕ λ·(x,y)`. CC-within-cluster preferred.)*
3. Output: a set of **spatial candidate regions** per image.

**Why collapse-proof:** clusters are disjoint ⇒ one bbox per region ⇒ "9 identical
boxes" is inexpressible. The head never emits boxes or points.

---

## 5. The zoom decision (per-cluster gate)

For each candidate region:
- `V` produces per-patch scalars; **aggregate over the region** (mean/max) → a
  predicted advantage `â(region)`. Per-region aggregate ⇒ supervision is
  per-region, never sharp per-patch 1/0.
- **Gate (positive framing):** zoom the region iff `â(region) > δ`, with `δ > 0`.
  Default is *don't-zoom-this-region*; act only on positive predicted gain.
- **Applies to EVERY image**, including image-MIL-"clean" ones. A subtle
  manipulation the image-MIL missed can still form a region with positive `â` and
  get zoomed — this is the recall mechanism for hard cases.
- **Multibox:** zoom all gated regions; OR-union the placed-back masks.
- **Fallback (safety floor):** no region clears δ → defer to the flat (no-zoom)
  decode.

**δ is a val-tuned operating point, not learned.** Sweep it to a target FP-zoom
rate; slide along the aggressiveness curve without retraining. Threshold "failure"
= re-sweep δ, never a gradient. (Letting the RL learn δ end-to-end bakes in one
cost assumption and couples calibration to reward noise.)

---

## 6. Reward & training target

- **Reward(region) = soft-IoU(zoom-decode restricted to region) −
  soft-IoU(no-zoom baseline)** = the **advantage**. Soft-IoU (not hard-F1) to keep
  the signal smooth and de-noised. Average over decoders (kmeans+hdbscan) if noisy.
- **Multibox credit:** **leave-one-out** marginal advantage on the union —
  `A(region_i | S) = F1(∪S) − F1(∪ S\{i})`. Cheap (frozen backbone). A region that
  adds nothing to the union gets ~0 advantage → never reinforced (kills redundant
  boxes from the reward side too).
- **Target for `V`:** the region aggregate regresses toward `â = advantage` (or a
  rank objective over regions); optionally **advantage-weighted** (AWR:
  `weight = exp(A/τ)`). No point targets, no thresholds in the loss.
- **Baseline** = flat/no-zoom decode, cached per item (static on a frozen detector).
- **The reward function already exists** as the `flat_cache`/`attn_cache` scoring in
  `evaluate()` — repurpose it to score *candidate regions* during training.

**Candidate generation is free:** the candidates **are** the HDBSCAN regions. No
attention-peak proposals, no DAgger bootstrap — cluster → score each region's
realized advantage → distill.

---

## 7. Projection (`P`) training

- **Objective:** a metric/contrastive loss toward **GT instance grouping** — pull
  within-instance patches together, push cross-instance & clean apart. Well-defined
  from GT instance masks (unlike "optimal box").
- **Light:** linear or 1-layer + L2-norm; backbone + main heads stay frozen.
- **Dim reduction does double duty:** cleaner HDBSCAN + a zoom-coherent space.
- **Schedule:** TBD empirically — train `P` first (cluster quality), or jointly
  with `V`. Start with `P` first to get stable clusters before reward distillation.

---

## 8. Inference pipeline (end to end)

```
1. backbone → z, attention, image_logit  (frozen)
2. P → z'
3. HDBSCAN(z') → connected-components → candidate regions
4. V → per-patch scalar → per-region aggregate â
5. gate: keep regions with â > δ
6. (optional) per-crop MIL corroboration on kept regions
7. zoom each kept region (2nd pass) → decode → OR-union (with/over flat)
8. polarity: attention-selected cluster (existing; pool head)
   fallback: no region clears δ → flat decode (safety floor)
```

---

## 9. MIL signals — features, not filters

- **Image-level MIL logit:** a *feature* into `â` (global context), and a possible
  soft prior — but **never a skip**. Zoom is attempted regardless so hard misses
  are recoverable.
- **Per-crop MIL logit** (`gate_boxes_by_logit`: does the crop look *more*
  manipulated isolated than in the full frame?): an **input feature** to `V`
  (clean crops won't score higher ⇒ correlates with negative advantage) and a cheap
  self-supervised **secondary corroboration** at accept time. Used in tandem.

---

## 10. Collapse modes & defenses

| mode | why it can't win |
|---|---|
| 9 identical / overlapping boxes | disjoint clusters — inexpressible; redundant region → leave-one-out advantage ≈ 0 |
| one giant box over everything | low magnification → ~0 advantage; CC-within-cluster + `min_cluster_size` |
| mixed messages to adjacent patches | decision is per-REGION (aggregate), never per-patch 1/0 |
| collapse to "never zoom" | warm-start; positive advantage on zoomable regions pulls it to act |
| FP-zoom (clean region) | per-cluster advantage gate + δ + flat fallback + per-crop MIL — **not** an image skip |
| brittle threshold | HDBSCAN (relative partition) replaces the magnitude threshold; box = bbox-of-members (threshold-free extent) |

---

## 11. What is frozen / learned / tuned

- **Frozen:** DINOv3 backbone, LoRA, contrastive head, MIL pool head, patch head.
- **Learned (small):** projection `P`, zoom-value head `V`.
- **Tuned — owned by us, not the RL:** δ (operating point), HDBSCAN
  `min_cluster_size` / spatial-split params, box pad, AWR temperature τ.

---

## 12. Evaluation

- **Built-in head-to-head:** per-source zoom-F1 vs cached flat / attention-zoom /
  hdbscan-zoom references (`evaluate()`).
- **Sources:** `sagid`, `casia`, `imd2020` — **add `tgif2`** (TGIF is where zoom
  matters most; see the resolution/zoom ablation).
- **Warm-start detector:** start on **r032** (448/rank-32, available, frozen,
  cheap). **Headline number:** retrain the head on the **optimal 688/r16** model
  once it finishes on the L4.

---

## 13. Open / empirical decisions (settle while prototyping)

- Projection depth & output dim (start linear, ~8–16d).
- `z` vs `z'` as input to `V`.
- Staged (`P` then `V`) vs joint training.
- Regress-advantage vs rank objective for `V`.
- δ operating point (val sweep to target FP-zoom rate).
- Per-crop MIL as feature, accept-gate, or both.
- HDBSCAN `min_cluster_size`, CC-split parameters.
- AWR temperature τ (or hard argmax distill).
