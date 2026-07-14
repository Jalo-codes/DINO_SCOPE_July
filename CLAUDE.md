# CLAUDE.md — DINO_SCOPE_July

Working notes for any Claude (Code or Science) picking up this repo. Keep this file
truthful and current; correct it in place when a claim is superseded.

## Repo orientation

- `results/bce_emergence/` — the six-condition BCE-vs-contrastive emergence sweep (the
  current focus). Conditions: `bce_both_s0`, `bce_inpaint_s0`, `bce_splice_s0`,
  `cont_both_s0`, `cont_inpaint_s0`, `cont_splice_s0`.
  - `both` / `inpaint` / `splice` = which manipulation family the model was trained on.
  - `bce_*` vs `cont_*` = localization objective (per-patch BCE head vs contrastive head).
- Per condition, **which subdirectory is canonical (as of 2026-07-14, all six
  conditions, epoch_0005, validated)** — go-to list, use these for new analysis:
  | dir | contents |
  |---|---|
  | `probe_eval2/` | clean-condition detection/localization, `fr_bg_matched` null |
  | `noise_probe/` | JPEG ladder (clean/90/70/50/30); `clean` level ≈ `probe_eval2` (validated, mean\|Δscore\|<2e-5) |
  | `threshold_sweep/` (bce_* only) | oracle BCE-threshold envelope, clean level |
  | `noise_threshold_sweep/<level>/` (bce_* only) | oracle envelope, per JPEG level |
  | `full_fakes_eval/` | whole-image AUC only (rule 2 — ignore its F1/IoU) |

  Each records CSV is decoder-specific: `threshold_*.csv` for `bce_*`, `kmeans_*.csv`
  for `cont_*`.

  **Superseded — do not use for new analysis:**
  | path | why |
  |---|---|
  | `<cond>/probe_eval/` | pre-checkpoint-fix AND pre-`fr_bg_matched` (uses retired `fr_bg`, see rule 4). Kept only as the validity-check baseline against `probe_eval2` — do not cite its numbers directly. |
  | `probe_manifest.csv` | old `fr_bg` geometry. Use `probe_manifest2.csv` (current join table, `fr_bg_matched` geometry) instead. |
  | `probe_contrasts.csv` / `.log` | generated before the checkpoint fix and before `fr_bg_matched` existed (the original Claude-Science report pass). Numbers in `REPORT_bce_emergence.md` sourced from this are stale — cross-check against `ANALYSIS_NOTES_bce_emergence.md` before citing. Not regenerated yet. |
  | `probe_renders/` | illustrative crops only; some filenames still say `fr_bg` (retired name). Qualitative reference, not a data source. |

- Eval code: `lab_utils/eval/aggregate.py`, `lab_utils/data/datasets/region_probes.py`,
  `experiments/labs/probe_contrasts.py`, `experiments/scripts/eval_robustness.py` (JPEG
  ladder), `experiments/scripts/eval_threshold_sweep.py` (oracle sweep, model-free over a
  frozen cache).
- `ANALYSIS_NOTES_bce_emergence.md` — detailed findings + corrected procedures (read it
  before re-deriving anything; it's the current source of truth, ahead of
  `REPORT_bce_emergence.md`).

## Hard-won methodology rules (do not relearn the hard way)

1. **Decoder is confounded with objective.** `bce_*` decode masks by thresholding;
   `cont_*` decode by spherical k-means (k=2). NEVER compare raw localization F1/IoU
   across the bce↔cont boundary as if it measured representation quality — the
   contrastive interior F1 sits right at the mechanical k-means floor (2r/(1+r)≈0.57 at
   recall 0.4). Same-decoder re-eval is the clean fix and is still UNRUN. The
   calibration HALF of the confound is now boundable without retraining:
   `experiments/scripts/eval_threshold_sweep.py` sweeps the BCE threshold over a frozen
   eval cache (built with `eval.py --cache_dir`). Best-t is an ORACLE upper envelope —
   label it as such next to the production t=0.5 number, never headline it. k-means
   self-calibrates per crop; a fixed threshold cannot — that asymmetry is exactly what
   the sweep measures.

2. **Full-fakes localization is meaningless.** It is a byproduct of spherical k-means
   k=2 on whole fakes. Ignore full-fakes F1/IoU entirely; only image-level AUC there is
   real.

3. **Localization signal lives in BOUNDARY crops, not interiors.** Interiors are all-fake
   (contrastive precision pins at exactly 1.000 for 100% of crops — degenerate). Boundary
   crops have real+fake patches (precision 0.73–0.90, only 1–14% at precision=1), so
   boundary F1/IoU is a genuine localization measure.

4. **AUROC needs a declared null, and the null choice changes the answer.** Probe reals
   come in two strata that are NOT interchangeable:
   - `real_crop` = SAME interior window re-derived on the PRISTINE ORIGINAL with the same
     deterministic RNG as its paired ai_interior fake → matched geometry, edit is the only
     difference. This is the honest interior null.
   - `fr_bg` = window OUTSIDE the mask on the MODIFIED image, the train-time negative
     distribution. Drifts fake-ward; pooling it INTO the null depresses interior AUROC.
     RETIRED 2026-07-09: its sides came from a different generating process
     ([floor, 1.6·floor]) than interiors (mask-inscribed) — the size artifact. Replaced
     by `fr_bg_matched` (sizes drawn from the re-derived tgif2-sp ai_interior pool →
     matched by construction, full N; flag `--fr_bg_matched_root`). Rerun on the 2080
     box (Turing → fp16, never bf16) is DONE for all six conditions — see
     `probe_eval2/` and `noise_probe/` (canonical, above), not `probe_eval/` (old rows).
   Report interior detection AUROC against the matched `real_crop`, not pooled reals.

5. **Provenance is mixed — restrict before comparing.** ai_interior and real_crop are each
   300 tgif2 + 50 sagid; fr_bg is pure tgif2; sp_* is imd2020. For an apples-to-apples
   interior test, restrict all sides to the tgif2 subset (n=300).

6. **When correcting for a nuisance variable, match the whole DISTRIBUTION, not a point,
   and always report a bootstrap CI vs chance.** A point-estimate AUROC that "barely
   moves" can still be the difference between weak-signal and statistically random once
   you look at the CI — this bit us on the splice conditions (see notes).

7. **mean image_score is NOT comparable across the bce↔cont boundary** (contrastive reals
   baseline sits 2–3× higher). Use AUROC for cross-condition detection comparisons; read
   mean only next to that condition's own reals mean.

## Reporting conventions the user (Jake) wants

- Absolute numbers, not difference measures; comparisons shown as additional info.
- All measures reported as **mean** (not median).
- Condition labels: `BCE·X` / `Cont·X` (objective-instead-of framing), never `+Cont`.
- Figures color-code: bce_both `#08519c`, bce_inpaint `#3182bd`, bce_splice `#9ecae1`,
  cont_both `#a63603`, cont_inpaint `#e6550d`, cont_splice `#fdae6b`.
