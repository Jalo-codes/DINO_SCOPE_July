# Zoom-box head (v2) — dense per-patch box + confidence, trained as a contextual bandit

Supersedes the cluster/projection design in `docs/zoom_head_spec.md`. That design had a
**disjoint supervision** problem: the projection head was trained on GT-instance grouping
while the value head was trained on reward, the regions came entirely from the projection
(so the value head could only select, never reshape), and post-hoc padding/squaring meant
the reward was measured on a box the model never chose. This design removes all of that:
one head, one objective (reward), and the box is the model's **direct output**, so the
reward signal matches the prediction as tightly as possible.

## 1. The problem — a continuous-action contextual bandit

One-step MDP (no transitions, no discount). Per **patch**, the action is a box; the reward
is the localization-F1 improvement from zooming it. There is no sequential structure — each
zoom's reward is evaluated independently. It is a bandit; it cannot be otherwise.

## 2. Context / state

Frozen detector forward on image `I` → `G×G` patch grid, `N=G²` patches. Per patch `p`:
the feature `f_p = [z_p ‖ attn_p ‖ ℓ_p]` (`build_policy_input`), and its fractional centre
`(x_p, y_p)`. A **self-attention encoder** `E` over `{f_p}` produces contextualized tokens
`h_p`. The attention is essential: a patch must see the whole blob to know the splice's
extent — that's what makes box *regression* possible where a per-patch scalar could not.

## 3. Action — FCOS distance-to-sides box

Patch `p` emits four non-negative distances `(dt,dl,db,dr) = softplus(box_head(h_p))` and a
fractional box anchored at its centre:

    box(p) = [ y_p − dt, x_p − dl, y_p + db, x_p + dr ]  ∩ [0,1]²

Neighbouring patches over one splice centre see similar context ⇒ regress nearly the same
absolute box (**consensus**, not contradiction — the discrimination moves to decode-time
NMS, never sharp per-patch targets). No padding, no squaring: the box is the output.

## 4. Reward — advantage over the DEPLOYABLE baseline = attention-zoom

With `m(·)` = `eval_metric(mask).f1`:

    A(b; I) = m(zoom→b) − m_attn          (deploy);   − m_flat   (early curriculum)

- **baseline = attention-zoom**, the incumbent heuristic — a FIXED, GT-free action. The gate
  then means "zoom only where the learned box beats attn", and the system is an honest
  cascade over attn (fall back to attn when the gate is off).
- **DO NOT use `max(flat, attn)` as the baseline or the fallback.** Picking the per-image max
  needs the GT F1 of each ⇒ it is an **oracle**, not deployable; using it as the fallback
  makes `policy ≡ max(flat,attn)` and `Δ vs attn ≥ 0` a tautology (a real leak we hit and
  removed). `max(flat,attn)` is reported ONLY as a clearly-labeled oracle *ceiling*.
- Choice of baseline is **irrelevant to the box head** (AWR weights are softmax over a
  constant-shifted advantage ⇒ shift-invariant); it only sets the **confidence/gate** scale.
- **Curriculum:** first few bandit epochs use `baseline = flat` (positive advantage on
  zoom-favorable images ⇒ cold-start signal), then switch to `attn`.
- Frozen backbone ⇒ `A` is deterministic and cheap. Trivial / failed box ⇒ `A = flat − base`.
- For a kept set `B` (NMS ⇒ disjoint): `A(B) ≈ Σ A(b)`, so per-box reward is a valid target.

## 5. Heads

- **Box** `softplus(box_head(h_p)) ∈ ℝ⁴₊` → frac box (above).
- **Confidence** `conf(h_p) ∈ ℝ` = predicted advantage of `box(p)`. Drives NMS ranking and
  the δ-gate; regressed to realized advantage so "`conf > δ`" means "predicted advantage > δ"
  (consistent with the δ-sweep tooling).

## 6. Decode (inference)

1. Encode → `{box(p), conf_p}`.
2. **Gate:** candidates `C = { box(p) : conf_p > δ }`.
3. **NMS:** sort `C` by `conf` desc; keep `b` if `IoU(b, kept) < η`; drop trivial boxes;
   cap at `max_boxes` → kept set `B`.
4. `B = ∅` → **fall back to max(flat, attn)** (safety floor).
5. Else → zoom-union over `B` → final mask.

`δ` is a val-tuned operating point (not learned), chosen from the δ-sweep.

## 7. Training — AWR / search-and-distill

**Phase 0 — warm-start (supervised, `--warmstart_epochs`).** For patches inside a GT
component: regress box → that component's (lightly padded) frac box; confidence → 1.
Elsewhere confidence → 0. Seeds the box head into a sensible basin so the bandit has signal.

**Phase 1 — bandit (AWR).** Per proposing patch `p` (top `n_propose` by manipulation prior,
+ `n_background` random patches for negative calibration):

1. Candidates: `b⁰=μ_p` (greedy) and `bⁱ = softplus(raw_p + σ·εᵢ)`, `i=1..K` (exploration).
2. Score each: `Aⁱ = A(box(bⁱ))` (frozen, deduped to a 0.02-frac grid).
3. Weights `wⁱ = softmax(Aⁱ/τ)`.
4. **Box loss** `Σᵢ wⁱ · smooth_l1(μ_p, sg[bⁱ])` — pull the mean toward high-advantage samples.
5. **Confidence loss** `smooth_l1(conf_p, sg[A⁰])` — calibrate to the deployed box's advantage.

Background patches contribute only the confidence term (their greedy box's advantage, ~≤0),
never a box-pull.

**Exploration (both required):** `σ` anneals `--sigma → --sigma_final` with a floor (keeps
candidate search alive); `n_background` random patches feed the confidence head the negative
examples it would otherwise never see.

**Cost:** `|P|·(1+K)` deduped frozen crop-forwards/image (defaults ≈ 6·6 + 4 ≈ 40, ~−40% after
dedup). Warm-start is forward-free of zooms.

## 8. Eval & selection

Per source (never pooled): policy vs flat / attn / **baseline=max(flat,attn)**, headline
`Δ vs attn` and `Δ vs baseline`; `conf↔realized-advantage` calibration corr; a δ-sweep of
mean captured advantage. **Selection metric = mean captured advantage (policy − baseline)**
over the eval set — the quantity the bandit maximizes. `best.pt` is chosen from the **bandit
phase only** (warm-start metric is not comparable).

## 9. Frozen / learned / tuned

| | what |
|---|---|
| frozen | detector backbone + all heads (z, attn, patch-logit, MIL pool) |
| learned | encoder + box head + confidence head (the only new params) |
| tuned (val, not learned) | δ (gate), η (NMS IoU), max_boxes, σ-schedule, awr_temp |

## 10. Code

- `lab_utils/model/zoom_box_head.py` — `ZoomBoxHead`, `patch_centers`, `boxes_from_distances`.
- `experiments/labs/zoom_box_lab.py` — reward/baseline, AWR train step, warm-start, NMS decode, eval.
- `experiments/scripts/train_zoom_box.py` — harness (warm-start → bandit, σ-anneal).
- `run_scripts/run_zoom_box.sh` — launcher (warm-starts from r032; kmeans decoder = no extra dep).
