# bce_emergence — analysis notes & findings

Companion to `CLAUDE.md`. All numbers below verified from the eval records CSVs
(`{condition}/full_fakes_eval/` and `{condition}/probe_eval/`) and `probe_manifest.csv`.
Measures are **mean** unless stated. `rank_auc(pos,neg)` = Mann-Whitney
`(sum(rank_pos) − n1(n1+1)/2)/(n1·n2)`.

## 1. Full-fakes benchmark (whole-image), mean

| condition | image AUC | F1 (mean) | IoU (mean) | reals acc |
|---|---|---|---|---|
| BCE·both | 0.875 | 0.344 | 0.285 | 0.990 |
| BCE·inpaint | 0.944 | 0.510 | 0.440 | 0.992 |
| BCE·splice | 0.437 | 0.040 | 0.027 | 0.973 |
| Cont·both | 0.855 | 0.565 | 0.412 | 0.551 |
| Cont·inpaint | 0.917 | 0.576 | 0.426 | 0.549 |
| Cont·splice | 0.493 | 0.570 | 0.417 | 0.599 |

- **F1/IoU here are meaningless** (k-means k=2 byproduct). Only image AUC is real.
- Splice-trained models are near-chance at detecting *whole* fakes (0.44 / 0.49) — they
  learned a boundary/region cue, not a whole-image cue.
- Contrastive reals-acc ~0.55 vs BCE ~0.99: the k-means decoder has no "off" state and
  flags ~40% of patches even on genuine reals. BCE thresholding sits near-zero false-flag.

## 2. Probe benchmark — localization (BOUNDARY crops only; interiors are degenerate)

Boundary F1/IoU, equal-weight mean of ai_boundary + sp_boundary type-means:

| metric | BCE·both | BCE·inpaint | BCE·splice | Cont·both | Cont·inpaint | Cont·splice |
|---|---|---|---|---|---|---|
| F1 | 0.745 | 0.542 | 0.689 | 0.842 | 0.798 | 0.706 |
| IoU | 0.667 | 0.485 | 0.613 | 0.790 | 0.742 | 0.642 |

Per-type boundary F1:

| type | BCE·both | BCE·inpaint | BCE·splice | Cont·both | Cont·inpaint | Cont·splice |
|---|---|---|---|---|---|---|
| ai_boundary | 0.802 | 0.779 | 0.625 | 0.871 | 0.884 | 0.623 |
| sp_boundary | 0.687 | 0.306 | 0.754 | 0.813 | 0.712 | 0.790 |

- **Contrastive leads BCE at every regime on boundary localization** (subject to the
  decoder confound — see caveat). Largest gap at inpaint (F1 +0.256, IoU +0.257).
- **BCE·inpaint collapses on sp_boundary (F1 0.306)** — trained only on inpaint, it cannot
  localize splice boundaries. Contrastive holds up cross-manipulation.
- CAVEAT: bce_* use threshold decoder, cont_* use k-means; a same-decoder re-eval is
  needed to fully attribute this to the objective vs the decoder. Still UNRUN.

## 3. Probe benchmark — image-level detection AUROC (matched null)

Honest per-type detection AUROC against the best-available null (interiors vs matched
`real_crop`; boundaries vs pooled reals — boundary AUROCs are reference-robust, move
<0.06 across nulls). Bootstrap 4000×; NO cell's 95% CI touches 0.5 under these nulls.

| type | BCE·both | BCE·inpaint | BCE·splice | Cont·both | Cont·inpaint | Cont·splice |
|---|---|---|---|---|---|---|
| ai_boundary | 0.988 | 0.985 | 0.945 | 0.982 | 0.980 | 0.939 |
| ai_interior | 0.861 | 0.948 | 0.776 | 0.880 | 0.915 | 0.749 |
| sp_boundary | 0.985 | 0.947 | 0.995 | 0.984 | 0.893 | 0.992 |
| sp_interior | 0.789 | 0.874 | 0.847 | 0.831 | 0.813 | 0.878 |

## 4. Reals-null sensitivity (why the null matters)

- fr_bg drifts fake-ward vs real_crop: separability AUC(fr_bg>real_crop) =
  0.60 / 0.62 / 0.69 / 0.74 / 0.81 / 0.53 (BCE·both…Cont·splice).
- Interior AUROC swings up to 0.243 depending on null (real_crop vs fr_bg vs pooled);
  boundary AUROC moves <0.06. → interiors MUST be reported vs matched real_crop.
- **Honest ai_interior AUROC (tgif2-only, n=300, vs matched tgif real_crop)** is HIGHER
  than the official pooled number — the feared reals shift was depressing it, not
  inflating: BCE·both 0.84 / BCE·inpaint 0.94 / BCE·splice 0.76 / Cont·both 0.86 /
  Cont·inpaint 0.90 / Cont·splice 0.72.

## 5. Splice-interior is near-real (validated user hypothesis)

- Image scores for sp_interior are bimodal: ~60–82% fall below 0.2 (called real), a tail
  above 0.8. High AUROC comes from a hairline rank shift, NOT a large magnitude gap.
- Low-bin (<0.2) medians are ~1.3–2.8× reals' medians but both tiny (~0.01–0.03 vs
  ~0.001–0.01). Even the "missed" bulk sits a hairline above reals. Salience / low-level
  statistics is a plausible-but-unconfirmed mechanism. Salience concern applies broadly.

## 6. fr_bg size-match test — CORRECTED FINDING (this is the one I got wrong first)

Question: is fr_bg's fake-ward drift a WINDOW-SIZE artifact? (fr_bg median 126px vs
ai_interior 96px, 1.31× larger; real_crop == ai_interior exactly, TV dist 0.)

Method: reweight fr_bg negatives to ai_interior's FULL size distribution (density-ratio
histogram weights on native side px; verified TV distance to target = 0.000, effective
N≈131/300), full two-sided bootstrap 4000× resampling BOTH ai positives and matched
fr_bg negatives. tgif2↔tgif2 on both sides (n=300 each).

**Result — size matters critically for the two splice conditions:**

| condition | raw fr_bg AUROC | size-matched [95% CI] | verdict |
|---|---|---|---|
| BCE·both | 0.80 | 0.80 [0.76, 0.84] | signal holds |
| BCE·inpaint | 0.88 | 0.88 [0.85, 0.91] | signal holds |
| **BCE·splice** | 0.62 | **0.53 [0.48, 0.58]** | **collapses to RANDOM (CI incl. 0.5)** |
| Cont·both | 0.71 | 0.70 [0.66, 0.74] | signal holds |
| Cont·inpaint | 0.76 | 0.75 [0.71, 0.79] | signal holds |
| **Cont·splice** | 0.66 | **0.59 [0.55, 0.64]** | pulled toward chance |

- Within fr_bg, score DECREASES with window size (Spearman rho −0.06 to −0.44, negative in
  every condition) → size correction can only DEPRESS the raw AUROC, never inflate it.
  That is exactly why the splice numbers drop and none rise.
- **Lesson (why my first pass was wrong):** I read "4 of 6 barely move, mostly
  insignificant" as "not a size artifact." That missed the point. On the two splice
  conditions size correction moves BCE·splice from *weak-but-present* (0.62) to
  *statistically random* (0.53, CI straddles 0.5). A point estimate that barely moves on
  the majority can still flip a minority from signal to nothing — you MUST report the CI
  vs chance, and match the full distribution (not a single target px).
- This is NOT contradicted by §3/§4: the matched `real_crop` null controls size + location
  + provenance TOGETHER, and against it splice still separates (BCE·splice 0.76). The
  size-matched fr_bg result isolates the size variable alone and shows the splice models'
  fr_bg-referenced signal is size-driven. For splice-trained conditions, do NOT report
  interior detection as a clean result without stating the null.

## Open threads
- Same-decoder re-eval (contrastive+threshold AND bce+k-means) to break objective↔decoder
  confound — UNRUN, the clean test. PARTIAL substitute now implemented: the threshold
  sweep below bounds the calibration half of the confound without retraining.
- Salience mechanism for interior/splice signal — unconfirmed; chase per-pair score margin
  vs window contrast / edit magnitude within matched pairs.
- cont_both_s0 was marked status=skipped in sweep_summary.csv but has complete eval
  outputs; included in all analysis, flagged for double-check.

## Planned rerun (2080 box) — fr_bg_matched + threshold sweep [2026-07-09]

Code landed (this repo, unrun as of writing):

1. **`fr_bg` condition RETIRED, replaced by `fr_bg_matched`** (registry key, source
   name, and `--fr_bg_matched_root` flag). Same tgif2-'fr' outside-mask windows, but
   (h, w) drawn iid from the re-derived tgif2-'sp' ai_interior window pool — the §6
   size mismatch is now fixed BY CONSTRUCTION (fr_bg drew sides from [floor, 1.6·floor]
   while interior sides scale with the mask's inscribed rectangle; two different
   generating processes). Full N, no post-hoc reweighting. All existing
   `results/bce_emergence` records/manifest rows still carry the OLD fr_bg.
2. **`build_cache` crop-window bug FIXED** (lab_utils/eval/cache.py): it opened
   item.image directly, silently caching FULL frames for probe items. Any prior
   probe eval that used `--cache_dir` is suspect (the recorded runs did not; they
   ran the live path — verified eval.py flow).
3. **`experiments/scripts/eval_threshold_sweep.py`** — model-free F1/IoU-vs-threshold
   sweep over a ModelInfo cache, per probe condition, with the ORACLE-BEST row labeled
   as an upper envelope (eval_oracle.py convention: best-t leaks GT, report next to
   production t=0.5, never as headline). Purpose: separate "BCE features don't rank
   fake patches" from "fixed 0.5 threshold is miscalibrated OOD" — if contrastive's
   fixed k-means F1 ≥ BCE's best-t envelope, the boundary-localization gap is FEATURES;
   if the envelope closes it, it was CALIBRATION (itself a production argument for
   contrastive: k-means needs no threshold).

Run plan (2080 box — Turing: use fp16, NOT bf16; eval.py default --amp_dtype float16
is already correct):

    # per condition (all six), fresh out_dir; bce_* also get --cache_dir for the sweep
    $PY -m experiments.scripts.eval \
        --checkpoint <run_dir>/best.pt \
        --decoder threshold            # kmeans for cont_* \
        --cache_dir <run_dir>/probe_cache   # bce_* only (sweep input) \
        --ai_interior_root $SAGID --ai_boundary_root $SAGID --real_crop_root $SAGID \
        --sp_interior_root $IMD2020 --sp_boundary_root $IMD2020 \
        --fr_bg_matched_root $TGIF2 \
        --ai_interior_tgif_root $TGIF2 --ai_boundary_tgif_root $TGIF2 \
        --real_crop_tgif_root $TGIF2 \
        --out_dir results/bce_emergence/<cond>/probe_eval2

    # sweep (CPU, after the cache exists), bce_* only
    $PY -m experiments.scripts.eval_threshold_sweep \
        --cache_dir <run_dir>/probe_cache \
        --out_dir results/bce_emergence/<cond>/threshold_sweep \
        <same dataset-root flags>

    # regenerate the manifest once (any box with the data)
    $PY -m experiments.labs.probe_manifest <same roots> \
        --out_csv results/bce_emergence/probe_manifest2.csv

Shortcut if 2080 time is tight: cont_* only need `--sources fr_bg_matched` (their
other condition rows are unchanged — window RNG untouched); analysis then joins new
fr_bg_matched rows onto the existing kmeans_records.csv and drops old fr_bg rows.
Full rerun preferred for provenance if time allows.

Analysis after the run: (a) interior AUROC vs fr_bg_matched with bootstrap CIs — the
§6 size-matched table, now by construction (expect BCE·splice ≈ chance again; verify
realized size TV distance in the new manifest); (b) threshold-sweep envelope vs
contrastive fixed k-means per boundary type — the calibration-vs-features read;
(c) paired BCE↔Cont detection AUROC-difference bootstrap (same item_ids across
conditions) for the small consistent BCE image-detection edge.

## Noise-reliance probe sweep (JPEG ladder) [planned 2026-07-11]

Rationale (Jake): corrupting the high-frequency band partially ISOLATES the signal a
model is using. If bce_* and cont_* degrade differently down a JPEG ladder, the
objectives learned fundamentally different signals — the direct test of the
"absolute AI-ness (high-freq fingerprint) vs relational/contextual evidence" split
suggested by the clean results (BCE leads image detection where AI texture exists;
contrastive leads boundary localization and cross-type generalization).

Predictions the ladder discriminates:
- If BCE·inpaint's ai_interior edge (0.94) is a high-freq generator fingerprint, it
  should collapse toward Cont's level (or below) by jpeg_50.
- Boundary detection/localization (seam evidence is partly structural) should be the
  most compression-robust stratum for BOTH objectives; if contrastive's boundary F1
  advantage GROWS as quality drops, its signal is the lower-frequency one.
- sp_* (real-content splices, no generator fingerprint available at all) should be
  the flattest curves — a sanity anchor.

Mechanics: `run_bce_emergence_noise.sh` → eval_robustness.py over the probe
conditions, `--conditions clean jpeg_90 jpeg_70 jpeg_50 jpeg_30`,
`--corrupt_at model_input` (corruption AFTER the 448 resize → identical model-space
frequency destruction for every crop; `native` would confound the ladder with each
crop's upsample factor). Per-item records CSVs land in
`results/bce_emergence/<cond>/noise_probe/{decoder}_{level}_records.csv` — same
format as probe_eval records, so all stratified/matched-null AUROC analyses apply
per level. The re-run `clean` level should reproduce probe_eval2 (consistency
check). eval_robustness fixes that made this possible: probe crop windows now
respected (was full-frame), per-item CSVs + durable eval.log added, pixel-res
masks stripped from accumulated records (the old host-OOM cause).

## Figure inventory (artifacts in Claude Science project proj_6a53bb0928d9)
fig1 fullfakes aggregate · fig2 probe by-type (F1/IoU/AUROC/imgscore/predpos) · fig3
generator AUROC · fig4 comparisons · fig5 sp_interior distribution · fig6 bce-vs-cont
boundary localization · fig7 mean-vs-AUROC divergence · fig8 reals-reference sensitivity ·
fig9 honest interior AUROC · fig10 fr_bg size-match (corrected).
