# DINO_SCOPE — Rebuild Gameplan

> Execution order for the rebuild specified in [`DESIGN_GUIDE.md`](DESIGN_GUIDE.md).
> The guide says *what* to build and the invariants (I1–I9); this says *in what
> order* and *how each step is verified before the next starts*. Build bottom-up,
> one layer per commit, each behind a green gate. Nothing proceeds on an
> unverified layer.

---

## Cross-cutting contracts (thread through every phase)

These two are not phases — they are constraints every relevant step must honor,
called out here so they don't get rediscovered late.

### C1 — Checkpoint compatibility (old `.pt` files must keep loading)
Existing checkpoints are a single dict:
`{'model': state_dict, 'optimizer', 'scheduler', 'scaler', 'cfg': dict, 'epoch'}`
loaded via `model.load_state_dict(ckpt['model'])`. The weight keys are determined
entirely by the `nn.Module` attribute tree in `lab_utils/model/*`.

- **Rule:** the rebuilt `model/{image_bce_detector,multi_head_detector}.py` keep
  the **same module attribute names and nesting** (backbone, heads, projections).
  `model/` is KEEP (not rebuilt) precisely to protect this — do not "clean up"
  layer names or restructure submodules.
- **Rule:** `eval/fetch.py:model_info` reads the same `forward()` output keys the
  current model emits (`patch_logit`, `pool_attention`, `contrastive`,
  `image_logit`). `fetch` is new code wrapping an unchanged forward.
- **Gate (set up in Phase 0, enforced from Phase 5 on):** a golden pre-rebuild
  checkpoint loads into the rebuilt model with `load_state_dict(strict=True)` and
  reproduces a cached forward within tolerance. See `test_checkpoint_compat`.

### C2 — Run-settings printout is derived, never hand-written
The current printout is f-strings off `args.*` emitted *before* config overrides
apply — it can disagree with what actually runs, and new knobs go unprinted.
Kill the drift at the structural level:

- There is **one resolved config object** (`RunConfig`, a frozen dataclass) built
  once in a single `resolve_config(args) -> RunConfig` step. All CLI args, file
  defaults, and overrides collapse into it **before anything runs**.
- That same object — and only that object — is passed to model build, data build,
  the train loop, and eval. Nothing downstream reads raw `args`.
- The settings printout is **auto-rendered from `RunConfig`'s fields**
  (`log_run_config(cfg)` iterates `dataclasses.fields`), not composed by hand. A
  new field prints automatically; a printed value is by construction the value
  that runs.
- `RunConfig` is serialized into the checkpoint `cfg` slot and written to the run
  dir, so **printed settings == saved settings == resumed settings.**
- **Gate:** `test_run_config_roundtrip` — every field shown by `log_run_config`
  is a field actually consumed; `resolve_config → serialize → reload` is identity.

### C3 — Hardware portability (dual 2080 Ti · Colab · CPU dev box)
The code runs on three shapes of machine and must resolve each cleanly without
per-script special-casing:

| Target | GPUs | AMP dtype | Distributed |
|---|---|---|---|
| Dual 2080 Ti (Turing, CC 7.5) | 2 | **fp16** (no bf16 on Turing) | DDP / nccl, world=2 |
| Colab (A100/L4/T4, CC 8.0+ usually) | 1 | bf16 where CC≥8, else fp16 | none (world=1) |
| Dev box / CI (this machine) | 0 | off | none |

- **Consolidate hardware management into one resolver:**
  `lab_utils/train/hardware.py` — device selection (cuda/mps/cpu + availability),
  capability probe, and the AMP/distributed wiring that is currently inline in the
  god script (`train_multi_head.py:1167-1204`). It delegates to the **existing**
  `train/amp.py:resolve_amp` (keep as-is — it already does CC≥8→bf16 / else fp16)
  and `train/distributed.py:setup` (keep — DDP/nccl + work-sharding). The resolver
  returns a `HardwareInfo` (device, dtype, world_size, gpu name, cc).
- **Ties into C2:** `HardwareInfo` is recorded on `RunConfig` and printed by
  `log_run_config`, so a run's log + checkpoint state *which* hardware/precision it
  actually used (a 2080-Ti-fp16-DDP run is distinguishable from a Colab-A100-bf16
  run in the saved settings).
- **Torch-free import boundary (this is why the suite is red locally).** Today
  `lab_utils/__init__.py` eagerly imports torch-bound modules, so *nothing* in
  `lab_utils` imports without torch. In the rebuild, the top-level package and the
  torch-free layers (`item`, `verify`, `buckets`, `record`, `aggregate`,
  `run_config`) must import with **no torch**; only `fetch`, `model/*`, `train/*`,
  and decoders that genuinely need tensors pull torch (lazily). This makes the
  torch-free test tier runnable anywhere.
- **Two test tiers** (the suite is split, not all-or-nothing):
  - **torch-free tier** — item/verify/augment-mask-logic/buckets/aggregate/
    run_config/cache-load. Runs on the CPU dev box and Colab-CPU. *This is the tier
    that gates local work.*
  - **gpu/torch tier** — fetch, checkpoint-compat (C1), decode-on-real-signal,
    train smoke. Marked `needs_gpu`; runs on the 2080 Ti box or Colab-GPU.
- **Gate:** `test_torch_free_import` — importing `lab_utils` and each torch-free
  module succeeds in an environment with torch uninstalled (or mocked absent).

---

## Phase 0 — Freeze the past, stand up the skeleton
Goal: capture everything needed to prove nothing was lost, then move legacy aside.

Phase 0 spans two machines (C3). Steps are tagged **[gpu]** (2080 Ti box / Colab)
or **[any]** (runnable on the CPU dev box). The **[gpu]** steps must complete and
their artifacts be committed before the irreversible restructure, so the
"nothing lost" gate is real.

1. **[gpu] Confirm the live baseline is green** on the 2080 Ti box (full suite,
   torch present). This is the actual baseline — it cannot be verified on the CPU
   dev box, where torch is absent and the whole suite fails to collect. Record the
   pass/fail list; known-pitch tests (`test_window_geometry`,
   `test_sliding_window_geometry`) may already be slated for removal.
2. **[gpu] Capture golden fixtures from the live tree** (they must come from real
   hardware):
   - one small trained checkpoint → `tests/fixtures/golden.pt`;
   - its forward output on 2–3 fixed images → `tests/fixtures/golden_modelinfo.npz`
     (the §2.1 `ModelInfo` fields). This is the C1 oracle and the cache-fixture
     for the torch-free decode/metric tier (§2.6, C3). Commit these before step 4.
3. **[any] Stand up the test-tier split** (C3): a `needs_gpu` marker (already in
   `pyproject.toml`) on the torch tier, so the torch-free tier collects and runs
   on the dev box. Verify the torch-free tier is green locally.
4. **[any] `git mv`** the entire current tree into `legacy/` in one commit, no
   edits. The legacy suite still points at `legacy/`; re-run it **[gpu]** — it
   stays green (proves the move lost nothing).
5. **[any] Create the empty target skeleton** (`lab_utils/{data,eval,...}`,
   `experiments/`, `tests/fixtures/`) with `__init__`s. No logic yet.

**Gate (split by tier, C3):** torch-free tier green on the dev box; full legacy
suite green on the 2080 Ti box from its new `legacy/` location; golden fixtures
committed; skeleton imports cleanly with **no torch**.

---

## Phase 1 — Data layer (bottom of the stack)
Order within the phase follows the dependency chain.

1. `data/item.py` — `Item` (the triplet). **Decide the `Item`/`ImageTriplet`
   naming now** (one class, declare the alias if both names are used) so every
   later annotation is consistent.
   - **Verify:** `test_item` — `is_real`, `mask_area(res)`, `load(res)` on fixtures.
2. `data/verify.py` — `verify_all()` drop-and-log (§3.3).
   - **Verify:** `test_verify` — corrupt / all-white / empty-mask / out-of-area
     triplets are dropped with a logged reason; good ones pass.
3. `data/augment/*` — port `light` (keep primitive ops; **retire the
   `apply_light_augmentations` compound** — its flip-last order violates the I6
   two-stage rule), then `corruptions`, `degradation`, `composite`, `blob`, and
   `crop.py` (keeps the `oracle_` name, train-only tripwire).
   - **Verify:** `test_augment` — each op is label-correct (geometric ops move the
     mask, appearance ops pass it through); two-stage ordering puts geometric
     before appearance.
4. `data/dataset.py` — the one general `Dataset` (holds `list[Item]`, owns the
   two-stage pipeline, uniform tensor output §3.5).
   - **Verify:** `test_dataset` — `__getitem__` shape contract; output carries
     **no bucket/area_tier field** (I5); eval mode runs zero train-only ops.
5. `data/datasets/*.py` + `registry.py` — port each `build()` from `indexer.py`
   (read each in full first).
   - **Verify:** `test_uniform_item` — every registered `build()` returns a
     `Dataset` of `Item`s that pass the verifier on a fixture root.

**Gate:** all data tests green; `data/` imports nothing from `eval/` or scripts.

---

## Phase 2 — Eval core (the three-stage contract)
This is the heart (§2). Build strictly in contract order.

1. `eval/fetch.py` — `model_info()` + `ModelInfo`. The **sole** model entry (I2).
   Wraps the unchanged forward (C1).
   - **Verify:** `test_fetch_is_sole_model_entry`; `test_checkpoint_compat`
     (C1 gate goes live here — golden.pt loads strict, `model_info` reproduces
     `golden_modelinfo.npz` within tolerance); `model_info` has **no GT param**.
2. `eval/record.py` — `EvalRecord` dataclass.
3. `eval/decode/{threshold,kmeans,graph,hdbscan}.py` — plain, pure, silent
   functions (§2.2). Polarity by attention mass, never GT. HDBSCAN hard-dep (drop
   the soft guard).
   - **Verify:** `test_decode_is_gt_free` (signatures take only `ModelInfo` +
     numeric params; module imports nothing from `data/`); `test_decode_is_silent`
     (no `print`/logging); `test_graph_decode` correctness on the cached fixture.
4. `eval/metric.py` — `metric(patch_mask, info, triplet) -> EvalRecord`. The
   **only** GT touch; projects patches→pixels, scores, derives `bucket` from
   `triplet.mask_area`, computes `image_score` (sigmoid(image_logit), fallback
   pooled).
   - **Verify:** `test_metric` — f1/iou/precision/recall on known masks;
     `test_no_oracle_outside_train_crop` (scoped to `eval/`+`labs/`: only `metric`
     reads `triplet.mask`; no `oracle` token anywhere under `eval/`/`labs/`).

**Gate:** a cached `ModelInfo` → decode → metric pipeline runs end-to-end with
**no model and no GT outside `metric`**.

---

## Phase 3 — Aggregation, buckets, cache
Pure functions over `List[EvalRecord]` (§2.5) — no model, no new GT.

1. `eval/buckets.py` — area→bucket cutoffs (reuse current `area_tiers`).
2. `eval/cache.py` — `build_cache`/`load_cache` over `ModelInfo` bundles. The
   golden fixture from Phase 0 *is* a cache entry.
3. `eval/aggregate.py` — `summarize/by_bucket/by_source/by_decoder`, median-led
   reporting (reals pooled separately, full percentiles) per
   [[feedback_eval_display]].
4. `eval/robustness.py` — record-based.
   - **Verify:** `test_buckets`, `test_aggregate` (golden record set → stable
     report), cache roundtrip identity.

**Gate:** full eval readout producible from a cache fixture alone.

---

## Phase 4 — Hardware resolver + RunConfig + derived settings printer (C2, C3)
Do this *before* the train/eval scripts so they consume the resolved objects, not
raw args.

1. `train/hardware.py` (C3) — `resolve_hardware(args) -> HardwareInfo` (device,
   dtype, world_size, gpu name, cc), consolidating the inline device/AMP/dist
   wiring from the god script. Delegates to the **kept** `train/amp.py:resolve_amp`
   and `train/distributed.py:setup`. Must degrade to CPU cleanly (no CUDA → device
   cpu, amp off, world 1) so the dev box and Colab-CPU work.
   - **Verify:** `test_hardware_cpu_fallback` (torch-free-ish: CPU path resolves
     without CUDA); on the 2080 Ti box, an `[gpu]` check that CC 7.5 → fp16 (not
     bf16) and world_size=2 under torchrun.
2. `experiments/configs/run_config.py` — `RunConfig` (frozen) + `resolve_config`
   (curated CLI + file defaults + overrides + `HardwareInfo` → one object, all
   resolution here).
3. `logging/run_config.py` — `log_run_config(cfg)` auto-rendered from fields
   (including the resolved hardware/precision, C3); and `to_dict`/`from_dict` for
   the checkpoint `cfg` slot and run-dir dump.
   - **Verify:** `test_run_config_roundtrip` (C2 gate) — printed fields ⊆ consumed
     fields; serialize→reload is identity; overrides + resolved hardware are
     reflected in the printout.

**Gate:** the only place settings are rendered is `log_run_config`; grep shows no
hand-built `[cfg] ...` f-strings in scripts; the printed line names the resolved
device/precision/world_size.

---

## Phase 5 — Train script (slim)
1. `train/loop.py` — epoch/step extracted from the god script.
2. `experiments/scripts/train.py` — parse curated flags → `resolve_config` →
   `log_run_config` → build model/data/loop → call `lab_utils.eval.*` for
   per-epoch validation (same fetch→decode→metric→aggregate, no private copy).
   **Zero eval/metric bodies defined here** (§5). Checkpoint save writes
   `RunConfig` into `cfg` (C2) and preserves the existing dict shape (C1).
   - **Verify:** `test_no_cross_script_imports`; a 1-step smoke train resumes from
     `golden.pt` (C1) and round-trips its own checkpoint.

**Gate:** train.py under one screen of CLI; resumes old checkpoints; settings line
matches the run.

---

## Phase 6 — Eval script + labs
1. `experiments/scripts/eval.py` — triplets → fetch → decode → metric → aggregate;
   replaces `eval_checkpoint.py` (imports **only** `lab_utils`, zero trainer
   coupling).
2. `experiments/labs/attention_zoom.py` — THE multi-pass strategy, GT-free
   (`attention_zoom_bbox`), consolidates the scattered zoom paths. **Resolve the
   zoom→metric geometry seam** here: define whether `metric` receives the
   placed-back patch mask + full-image `info`, or a pixel-mask entry path.
3. `experiments/labs/{decoder_bench,viz}.py`.
   - **Verify:** `test_zoom_is_gt_free`; `eval.py` reproduces the Phase-3 readout
     on the golden checkpoint (live path == cache path).

**Gate:** `eval.py` on `golden.pt` matches the cached-fixture readout.

---

## Phase 7 — Cleanup & docs
1. Wire the new invariant tests into CI; delete the pitched/smoke tests (§7).
2. Rewrite `experiments/README.md`, fix `lab_utils/README.md`, fill root
   `README.md`. Correct all "v1/v2/v3" docstrings.
3. Confirm `legacy/` is imported by nothing (`test_no_legacy_imports`); archive or
   delete on your word.

**Final gate:** every §7 invariant test green; nothing imports `legacy/`; old
checkpoints load; the settings printout provably equals the run.

---

## One-line dependency order
```
fixtures+legacy-move
  → item → verify → augment → dataset → datasets/*
    → fetch(+C1) → record → decode/* → metric
      → buckets → cache → aggregate → robustness
        → hardware(C3) → RunConfig+printer(C2,C3)
          → loop → train.py
            → eval.py → attention_zoom → labs/*
              → tests/docs/cleanup
```
Each arrow is a green gate. Parallelizable by subagent within a phase; the arrows
across phases are hard ordering.
