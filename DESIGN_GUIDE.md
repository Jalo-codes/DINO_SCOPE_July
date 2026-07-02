# DINO_SCOPE — Design Guide & Rebuild Specification

> **Status:** authoritative spec. No code has been moved yet. This document
> governs the rebuild: everything that gets built, copied, changed, or pitched
> must comply with the invariants in §1 and the contracts in §2–§3. Subagents
> executing any section build *to this doc*, not to the legacy code's habits.
>
> This is the single source of truth for the rebuild. Why the current tree is
> being torn down — inverted deps, dead shadow modules, god-script, doc drift —
> is recapped where relevant below; build from the invariants in §1.

---

## 0. The shape we are building toward

```
shared utilities (lab_utils)  ──►  ONE train script  +  N eval scripts/labs
```

Three layers, one direction of dependency. Utilities never import scripts.
Scripts never import each other. Nothing reaches across a module boundary into a
`_`-private name. The current tree violates all three; the rebuild restores them.

Terminology note: "swin" in the legacy tree meant *sliding-window inference*
(not the Swin Transformer). **Sliding-window is being removed entirely** (§2.3) in
favor of attention-zoom — no `swin`/sliding-window concept survives the rebuild.

---

## 1. Non-negotiable invariants

These are hard rules. A change that breaks one is wrong even if it "works."

### I1 — No oracle in eval. Ever.
"Oracle" = any use of ground truth to *produce or select* a prediction. Banned
everywhere in the eval path, unconditionally. Concretely, the following current
behaviors are **deleted, not ported**:
- `decode_oracle_labels` (partition.py) — picking cluster polarity by GT.
- `_oracle_polarity`, `_oracle_pixel` (localization.py) — "better of the two
  labelings by F1/IoU vs GT." This is the core offense.
- `report_oracle_tax` and every "oracle tax" report — the concept dies with the
  oracle metric.
- GT-centered eval crops / GT-assisted early-stop metrics.

**Disambiguation (do not over-apply the ban):** `oracle_mask_crop` /
`oracle_fallback` in the *dataset* are a **training augmentation** — a GT-centered
crop used while training on tiny splices. GT at *train* time is allowed, **at eval
NEVER.** **Keep the `oracle_` name on purpose** (do *not* rename it): the word
"oracle" is a deliberate tripwire. It is allowed to appear in exactly one place —
the train-only crop in `data/augment/crop.py` — and nowhere else. The §7 test
enforces precisely that: any `oracle` token under `eval/` or `labs/` is a bug.

The model must **commit to a foreground polarity using only its own signal**
(attention / patch-logit), then eval scores that commitment. If the commitment is
a coin-flip, the score reflects that — that is the point.

### I2 — The model is touched only through the fetch call.
Eval code never calls `model(...)` directly, never reads `out['contrastive']`
ad hoc, never re-implements an autocast forward. There is exactly one function
that turns an image into model signal: **`model_info(...)`** (§2.1). Every
decoder, every lab, every script goes through it. This is what makes eval
uniform and the model swappable.

### I3 — Eval is a fixed three-stage pipeline producing one packaged record.
```
ModelInfo  = fetch(model, image)            # raw signal, GT-free, uniform
PredMask   = decode(ModelInfo, decoder)     # committed mask, GT-free
EvalRecord = metric(PredMask, triplet)      # GT enters HERE and only here
```
`fetch → decode → metric` is **specific and uniform for all downstream uses.**
Every higher-level eval (bucket breakdown, robustness sweep, decoder bench,
attention-zoom study) is *aggregation over `List[EvalRecord]`* — never a new
forward, never a new GT touch. The `EvalRecord` packages GT mask, predicted mask,
attention, image score, per-image metrics, and derived bucket together (§2.4) so
downstream code slices instead of recomputing.

### I4 — One uniform sample format across every dataset.
Every dataset — IMD2020, CASIA, TGIF2/FLUX, inpaint-triplet, AnyEdit, indoor-real
— is reduced to a list of **`ImageTriplet`** (§3.1): `(authentic, manipulated,
mask)`. Reals are the degenerate case (`authentic == manipulated`, empty mask).
Downstream code never branches on "which dataset"; it operates on triplets.

### I5 — Dataset owns the mask and its area. Buckets are eval-derived.
Mask area (fraction of pixels manipulated) is computed and owned by the
dataset/triplet. **Size "buckets" (tiny/small/medium/large) are an eval reporting
concept only** — derived on demand from the triplet's mask area inside the eval
aggregation layer. Buckets must not appear as a training signal, a sampler input,
a dataset field, or a model input. (Current code leaks size into training/sampling
— that is removed.)

### I6 — Augmentation mechanics are shared utils; the pipeline is dataset-owned.
The augmentation **mechanics** (primitive ops — light full-image, corruptions,
multi-region degradation, splice compositing, blob masks, mask-centered crop) are
reusable functions taking **explicit numeric parameters** (no op reads a config).
The augmentation **pipeline** — which ops run, in what order, and the train/eval
gating — is owned by the `Dataset` (its build config), because order depends on
sample kind and on train-vs-eval. (See §3.6 for where the mechanics physically
live.) **Ordering rule that dissolves the crop ambiguity:** ops split by what they
touch, and run in two stages —
1. **Geometric stage** (operates on image *and* mask jointly, at native
   resolution): mask-centered (`oracle`) crop → flip → resize-to-`res`.
2. **Appearance stage** (operates on the image only; mask passes through): jpeg,
   gaussian/poisson noise, blur, resize-jitter, degradation/corruptions.

The mask-centered crop is therefore unambiguous: it is the *first* geometric op,
on the coupled (image, mask) before anything changes appearance. Synthetic
compositing happens before stage 1 (it constructs the manipulated image).

### I7 — Minimal CLI. One train script.
One training entrypoint. A small, curated flag set (the genuinely varied knobs:
data roots, run dir, epochs, lr, model heads, the few regime switches). The
~300-line `_build_parser` and its "9 trillion flags" are not ported wholesale —
each flag must justify its existence or die. Eval scripts get the few flags they
need from shared builders in one place.

### I8 — Legacy isolation; minimal, fully-read copying.
Everything current moves under `legacy/` untouched (§5). The new tree is built by
hand. Code is **reimplemented, not bulk-moved.** When a legacy function is good
enough to carry over, the file it comes from is **read in full first** and the
copy is verified to comply with I1–I7 before it lands. "Copied" never means
"imported from legacy" — `legacy/` is a graveyard for reference, not a dependency.

### I9 — Tests are curated and partitioned.
One clear `tests/` tree mirroring the package. Keep the tests that pin real
contracts (geometry, metrics, sampling determinism, decode correctness,
oracle-free guarantees). Pitch the fragmented/temporary/smoke-only tests. Add the
new invariant tests (§7): "no eval module imports GT into decode," "no script
imports another script," "fetch is the only model entry."

---

## 2. The eval contract (detailed)

All of this lives under `lab_utils/eval/`. This is the heart of the rebuild.

### 2.1 Stage 1 — the fetch (`model_info`)
The single, uniform model interface. Raw signal only; no GT, no thresholds, no
decode.

```python
@dataclass(frozen=True)
class ModelInfo:
    patch_logits: np.ndarray | None     # (N,)   dense per-patch splice logits  (patch-BCE head)
    attention:    np.ndarray | None     # (N,)   per-patch pool attention
    embeddings:   np.ndarray | None     # (N, D) L2-normalized contrastive embeddings
    image_logit:  float | None          # scalar image-level BCE logit
    grid_hw:      tuple[int, int]        # (n_side, n_side) patch grid
    res:          Resolution             # geometry for patch→pixel projection

def model_info(model, image_tensor, *, device, amp=True) -> ModelInfo: ...
```
- Maps directly onto today's `multi_head_detector.forward` output
  (`patch_logit`, `pool_attention`, `contrastive`, `image_logit`). It is the
  *only* place an autocast forward is written.
- Heads that are disabled yield `None` fields; decoders declare which fields they
  require and error cleanly if absent.
- **No GT parameter exists on this function.** That is structurally enforced
  (test in §7).

### 2.2 Stage 2 — decode (info → committed mask)
**No `Decoder` class, no Protocol, no registry — that is bloat.** A decode is a
**plain, siloed function per use case**: it takes a `ModelInfo` and returns the
committed foreground signal. Hard rules:
- **Pure.** Input is `ModelInfo` + explicit numeric params. **No GT parameter.**
- **Silent.** No printing, no logging, no plotting — decode *only* outputs the
  decoded signal.
- **Single-pass.** A decode reads `ModelInfo`; it does not re-enter the model.
  (Multi-pass strategies that re-fetch — attention-zoom, sliding-window — are
  *eval strategies* in `labs/`, not decodes. See §2.3.)

```python
# eval/decode/<usecase>.py — one file per use case, each a plain function:
def decode_threshold(info: ModelInfo, *, t: float) -> np.ndarray: ...      # (n_side,n_side) 0/1
def decode_kmeans(info: ModelInfo, *, n_init: int = 4) -> np.ndarray: ...  # polarity from attention
def decode_graph(info: ModelInfo, *, spec: GraphSpec) -> np.ndarray: ...   # calibrated band + components
def decode_hdbscan(info: ModelInfo, *, ...) -> np.ndarray: ...             # MAIN decode set (not a lab)
```
Return is the committed **patch mask** (the decoded signal). Pixel projection,
image score, and packaging happen in `metric` (§2.4), which owns the geometry and
`ModelInfo.image_logit` — decode stays minimal. A decode that produces extra
signal (graph component count, abstain flag) may return `(mask, aux: dict)`, but
`aux` is data only — never a printed side effect.

Main decode set: **`threshold`, `kmeans`, `graph` (+ spatial variant), `hdbscan`.**
HDBSCAN is promoted from the graph lab into this first-class set (hard dep, §9.5).

**Polarity rule (critical):** wherever a 2-cluster decode must choose which side
is foreground, it chooses by **attention / patch-logit mass** (`polarity_attn`),
never GT. `decode_oracle_labels` does not exist in the new tree.

### 2.3 Attention-zoom is THE multi-pass eval strategy (in `labs/`), oracle-free
**Sliding-window ("swin") is removed entirely — attention-zoom replaces it.** Do
not port `eval/sliding_window.py`, `eval/window_geometry.py`, or `viz_swin.py`;
they go to `legacy/` and stay there.

Attention-zoom is a **multi-pass strategy**: it re-enters the model, so it is not
a §2.2 decode. It lives in `experiments/labs/attention_zoom.py` and composes
`fetch → decode` more than once. Its one inviolable rule: **the crop bbox is
chosen from attention only** (`attention_zoom_bbox`), never from GT.
- Stage A: `model_info` on the full image → attention → `attention_zoom_bbox`
  picks a crop. (No GT — locked in, this is the already-fixed behavior.)
- Stage B: `model_info` on the crop → a §2.2 decode → `place_crop_in_full_frame`
  projects the committed mask back to full-frame coords. `multi_zoom_bboxes`
  supported for multi-region.
- The old GT-assisted early-stop / GT-chosen crop variants are **deleted.**
- Consolidate the scattered zoom logic (`eval/zoom.py`, trainer
  `_attention_zoom_second_pass`, `graph_lab/viz_zoom.py`, the zoom paths in
  `eval/localization.py`) into **one** `experiments/labs/attention_zoom.py`.
- Output is still `EvalRecord`s (§2.4) — the strategy just produces the committed
  full-frame mask that `metric` then scores.

### 2.4 Stage 3 — the metric (the only GT touch) → `EvalRecord`
```python
@dataclass(frozen=True)
class EvalRecord:
    # provenance
    item_id: str
    is_real: bool
    source: str                 # 'imd2020' | 'casia' | 'tgif2' | ...
    decoder: str
    # the packaged signal (everything downstream needs, no recompute)
    gt_mask:    np.ndarray       # (H, W)  — present ONLY inside the record
    pred_mask:  np.ndarray       # (H, W)  committed
    attention:  np.ndarray | None
    image_score: float
    # scores
    f1: float; iou: float; precision: float; recall: float; accuracy: float
    # derived reporting dims (computed here, from dataset-owned mask area)
    mask_area: float             # fraction of pixels manipulated (0 for reals)
    bucket: str                  # 'tiny'|'small'|'medium'|'large' — DERIVED, eval-only

def metric(patch_mask: np.ndarray, info: ModelInfo, triplet: ImageTriplet) -> EvalRecord: ...
```
- `metric` takes the decoded patch mask (§2.2), the `ModelInfo` (for attention,
  `image_logit`→`image_score`, and patch→pixel geometry), and the `triplet`. It
  projects to pixels, scores, and packages the record.
- `metric` is the **only** function that reads `triplet.mask`. Everything above it
  is GT-free; everything below it consumes records.
- `bucket` is computed here from `triplet.mask_area` (I5). It is a label on the
  record, nothing else.
- There is no `PredMask` container — decode returns the bare signal, `metric`
  assembles the record. Keeps the surface minimal (I7).

### 2.5 Aggregation layer (everything else is this)
Pure functions `List[EvalRecord] -> report`. No model, no GT beyond what the
records already carry. This is where the eval scripts and labs actually live.
- `summarize(records)` — overall + per-bucket. **Reporting style is fixed by
  preference: median-led with mean alongside, full percentiles, reals pooled and
  reported separately from splices, legible aligned output.**
- `by_bucket(records)`, `by_source(records)`, `by_decoder(records)`.
- `robustness(records_under_augs)` — sweep over augmentation presets (the augs
  come from §3.6, applied before fetch).
- `decoder_bench(records_by_decoder)` — compare the §2.2 main decodes over the
  same fetch. (Replaces `benchmark_decoders.py` and the graph_lab sweep.)

### 2.6 The cache (a util) and the labs
The only thing the rebuild keeps from `graph_lab`'s machinery is **the cache**,
and it becomes a plain utility: `eval/cache.py` with `build_cache(...)` /
`load_cache(...)` that freeze and reload `ModelInfo` bundles (the §2.1 contract).
- `build_cache` = one GPU pass → `ModelInfo`s on disk. `load_cache` = instant,
  model-free. Because the cache *is* the §2.1 contract, a cached signal and a live
  signal are identical — no bespoke per-lab forward exists.
- **Tests may load a cached fixture** to exercise decode/metric/aggregate without
  a model.
- Everything else the old `graph_lab` did (sweeps, viz, zoom studies) is just a
  `labs/` script that loads a cache (or fetches), runs §2.2 decodes / §2.3
  strategies, and aggregates records. The labs hold the *experimentation*; the
  reusable mechanics (fetch, decode, metric, aggregate, cache) are in `lab_utils`.

---

## 3. The data layer (detailed)

All under `lab_utils/data/`. Goal: every dataset → `List[ImageTriplet]` in one
format, verified, with augmentation as a separate composable step.

### 3.1 `Item` — the triplet member class
A small, standalone class (this is the "image triplet"); a `Dataset` holds a list
of these.
```python
@dataclass
class Item:                    # the ImageTriplet
    authentic: Path | None     # original/pristine image; None ⇒ unknown-source real
    manipulated: Path          # the image actually fed to the model
    mask: Path | None          # GT manipulation mask; None/empty ⇒ real
    source: str                # 'imd2020' | 'casia' | 'tgif2' | 'inpaint' | ...
    item_id: str               # stable, deterministic id (for seeds, sort, dedupe)
    meta: dict                 # dataset-specific extras (category, model, mask_type, ...)

    @property
    def is_real(self) -> bool: ...          # mask is None/empty
    def mask_area(self, res) -> float: ...   # dataset-OWNED area; feeds eval buckets (I5)
    def load(self, res): ...                 # → (img tensor, mask tensor) at target res
```
- For reals: `authentic == manipulated`, `mask` empty. Uniform arity (I4).
- `item_id` is the single source of determinism for subsampling/seeds/sorting
  (replaces the scattered `_stable_item_sort_key` / per-script seed hacks).

### 3.2 One general `Dataset` class; each dataset is an instance
There is **no per-dataset subclass and no `Indexer`/`Pairer` class hierarchy** —
that was over-engineered. There is **one** general `Dataset` (a
`torch.utils.data.Dataset`) that holds `list[Item]` and owns the load → augment →
tensorize path. Each real dataset is an **instance** of it, produced by a builder
that lives in **its own file**.
```python
class Dataset(torch.utils.data.Dataset):
    items: list[Item]
    def __init__(self, items, *, res, augment=None): ...
    def __getitem__(self, i) -> dict: ...    # uniform tensor format (§3.5)
    def subsample(self, n, *, seed) -> "Dataset": ...
    def filter(self, pred) -> "Dataset": ...

# data/datasets/imd2020.py  (one file per dataset — the indexing+pairing logic):
def build(root: Path, *, res, augment=None, verify_policy=...) -> Dataset:
    raw   = _discover(root)            # walk the dataset's layout
    items = _pair(raw)                 # match authentic ↔ manipulated ↔ mask → Item
    items, rejected = verify_all(items, res, policy=verify_policy)   # §3.3, drop-and-log
    return Dataset(items, res=res, augment=augment)
```
Each dataset file ports its discovery+pairing from today's `indexer.py`
(read in full first). A registry maps `source -> build` so a script says
`datasets.build('imd2020', root, res=res)` and nothing else.
| Dataset file | From (legacy) | Pairing logic to preserve |
|---|---|---|
| `datasets/imd2020.py` | `index_imd2020` | directory-structure pairing |
| `datasets/casia.py` | `index_casia_exported`, `_parse_casia_base_ids` | filename base-id matching |
| `datasets/inpaint.py` | `index_inpaint_triplet`, `_inpaint_clean_name` | clean-name matching, real_path triplet |
| `datasets/anyedit.py` | `index_anyedit` | as-is |
| `datasets/bfree.py` | `index_bfree`, `_resolve_bfree_root` | folder-resolution pairing |
| `datasets/indoor.py` | `index_indoor_dataset`, manifest/recursive | reals (no mask) |
| `datasets/tgif2.py` | `experiments/tgif2_flux.build_tgif2_items` | coco_id → manipulations; OOD probe |

### 3.3 `Verifier`
A required gate between pairing and use. Rejects bad triplets with a typed
`DataError` (or drops-with-log, configurable):
- mask file exists for any non-real triplet;
- image loads and is not corrupt (PIL open + decode);
- image is not all-white / all-black / single-color (variance/threshold check);
- mask is non-empty and within `[min_area, max_area]` for splices;
- shapes are consistent / loadable at target `Resolution`.
```python
def verify(triplet: ImageTriplet, res, *, policy) -> VerifyResult: ...
def verify_all(triplets, res, *, policy) -> tuple[list[ImageTriplet], list[Rejection]]: ...
```
Verification runs once at index time; the report (counts, reasons) is logged.

### 3.4 `Dataset.__getitem__` behavior
(Same general `Dataset` from §3.2 — this pins its per-item behavior.) Replaces the
1454-line `LabDataset`'s kind-dispatch sprawl.
- `__getitem__` loads the `Item`, applies the configured augmentation pipeline
  (§3.6), returns the model-ready tensor dict + a lightweight ref back to the
  item (for eval-record provenance — **not** the GT, which the dataset still owns
  but eval only reads through `metric`).
- No `area_tier` / bucket fields in the output (I5). No oracle-anything in the
  output. The mask-centered training crop (§3.6) is the only GT-using step and is
  train-only.

### 3.5 Uniform sample (tensor) format
`__getitem__` → exactly:
```python
{
  'img':   FloatTensor (3, S, S),         # S = res.image_size
  'mask':  FloatTensor (1, S, S) or zeros,# GT mask at input res (train supervision; eval ignores via contract)
  'meta':  {'item_id', 'source', 'is_real', ...},   # NO bucket, NO area_tier
}
```
One shape contract, asserted (keep today's shape-contract assertion — it is good).

### 3.6 Augmentations — mechanics + the dataset-owned pipeline
**Mechanics** (primitive ops, explicit params, I6). Port and consolidate:
| Module | From | Stage | Purpose |
|---|---|---|---|
| `light.py` | `augment/light.py` | appearance (+flip is geometric) | KEEP as-is; full-image label-preserving augs |
| `corruptions.py` | `augment/corruptions.py` | appearance | corruption families |
| `degradation.py` | `augment/degradation.py` | appearance | multi-region degradation |
| `composite.py` | `data/paste.py` + dataset paste logic | pre-stage | synthetic splice construction |
| `blob.py` | `data/blob.py` | pre-stage | synthetic ellipse masks |
| `crop.py` | `oracle_mask_crop`/`oracle_fallback` — **keep `oracle_` name** | geometric | **train-only** GT-centered crop; the one sanctioned "oracle" (tripwire, I1) |

**Pipeline** (ordering + gating) is **owned by the `Dataset`** (I6), applied in
`__getitem__` in the two-stage order: `composite → [geometric: oracle-crop → flip
→ resize] → [appearance: jpeg/noise/blur/degradation] → tensorize`. Eval datasets
run no train-only ops (no oracle crop, deterministic). Named presets (which numbers
= which regime) stay in `experiments/configs/augment.py`, passed in as explicit
values.

Mechanics live in shared `lab_utils/data/augment/` (one impl per op, DRY, §9.7);
the `Dataset` imports and sequences them. Co-locating aug under `data/datasets/`
was considered and rejected (would duplicate primitive ops).

---

## 4. Target directory tree

```
lab_utils/
├── errors.py
├── paths.py
├── data/
│   ├── item.py             # NEW  Item (the ImageTriplet) — the member class
│   ├── dataset.py          # REBUILT  ONE general Dataset class (holds list[Item])
│   ├── datasets/           # NEW  one file per dataset → build() returns a Dataset instance
│   │   ├── registry.py     #      source -> build()
│   │   ├── imd2020.py  casia.py  inpaint.py  anyedit.py  bfree.py  indoor.py  tgif2.py
│   ├── verify.py           # NEW  verify_all() — drop-and-log (§3.3, §9.2)
│   ├── loaders.py          # keep (cleaned)
│   ├── resolution.py       # keep (strip oracle refs)
│   ├── sampling.py         # keep  (is_real/is_splice/deterministic_subsample) — item-based
│   └── augment/
│       ├── light.py        # KEEP as-is (full-image label-preserving augs incl. Poisson shortcut-killer)
│       ├── corruptions.py degradation.py composite.py blob.py
│       └── crop.py         # oracle_mask_crop — TRAIN-ONLY, keeps the oracle name (I1 tripwire)
├── model/
│   ├── image_bce_detector.py multi_head_detector.py
│   └── losses/{bce.py, contrastive.py}
├── eval/
│   ├── fetch.py            # NEW  model_info(), ModelInfo — the SOLE model entry
│   ├── decode/            # NEW  plain siloed decode functions (no class, no print, GT-free)
│   │   ├── threshold.py kmeans.py graph.py hdbscan.py
│   ├── metric.py          # NEW  metric() → EvalRecord; f1/iou/auroc (absorb metrics.py); ONLY GT touch
│   ├── record.py          # NEW  EvalRecord
│   ├── aggregate.py       # NEW  summarize/by_bucket/by_source/by_decoder (median-led reporting)
│   ├── buckets.py         # NEW  area→bucket thresholds (eval-only, I5)
│   ├── cache.py           # NEW  build_cache()/load_cache() of ModelInfo bundles (util; tests load)
│   └── robustness.py      # keep (cleaned, record-based)
│   #  no decoders/ class tree · no partition.py monolith (decode math lives in decode/*)
│   #  no window_geometry.py · no sliding_window.py — sliding-window REMOVED (attention-zoom replaces it)
├── logging/{text.py, csv_logger.py, run_dir.py}
├── train/{loop.py(NEW), amp.py, checkpoint.py, distributed.py}
└── viz/composite.py

experiments/                  # renamed from contrastive_inpainting_v1 (ratified §9.1)
├── configs/{base.py, augment.py}
├── scripts/
│   ├── train.py              # THE train script (slim, minimal CLI)      ← from train_multi_head.py
│   ├── eval.py               # the uniform eval readout (records → report)
│   └── orchestrate.py        # keep (torch-free job runner)
├── labs/                     # experimentation + THE multi-pass eval strategy (was graph_lab + viz_*)
│   ├── attention_zoom.py     # the only multi-pass strategy (GT-free) ← zoom.py + trainer 2nd-pass + viz_zoom + eval_zoom_recovery
│   ├── decoder_bench.py      # ← benchmark_decoders.py + graph_lab/sandbox.py
│   └── viz.py                # ← viz_decode.py
└── cli.py                    # shared argparse builders (data paths / decode / eval flags)

legacy/                       # EVERYTHING current, untouched, reference-only
tests/                        # curated, partitioned (mirrors lab_utils/)
DESIGN_GUIDE.md  README.md
```

> Naming: ratified as `experiments/` (§9.1). Fix all "v1/v2/v3" docstrings during
> the rebuild: the dir self-describes as `contrastive_test_v3`, its README lists
> scripts that don't exist, and `lab_utils/README.md` references `contrastive_test_v2/`
> — correct all of these as each file is ported.

---

## 5. File-by-file rebuild ledger

Legend: **NEW** = written from scratch · **PORT** = reimplement after full read,
must comply with I1–I7 · **KEEP** = move with light cleaning · **PITCH** = delete
(lives only in `legacy/`).

### Data layer
| Target | Disposition | Sources (read in full) |
|---|---|---|
| `data/item.py` | NEW | — (the `Item`/triplet class) |
| `data/dataset.py` | REBUILT(slim) | `lab_utils/data/dataset.py` — ONE general `Dataset`; drop kind-sprawl, area_tier, oracle output |
| `data/datasets/registry.py` | NEW | `source -> build()` |
| `data/datasets/{imd2020,casia,inpaint,anyedit,bfree,indoor}.py` | PORT | `lab_utils/data/indexer.py` (the matching `index_*`/`_parse_*` fns) — each `build()` returns a `Dataset` instance |
| `data/datasets/tgif2.py` | PORT | `experiments/tgif2_flux.py` |
| `data/verify.py` | NEW | logic seeds: dataset.py shape contract; new all-white/black/corrupt checks; drop-and-log |
| `data/augment/{light,corruptions,degradation,composite,blob}.py` | KEEP | `augment/{light,corruptions,degradation}.py`, `data/{paste→composite,blob}.py` |
| `data/augment/crop.py` | PORT(keep oracle name) | `oracle_mask_crop`/`oracle_fallback` in dataset.py — TRAIN-ONLY (I1 tripwire) |
| `data/sampling.py` | KEEP | as-is, retargeted to `Item` |
| `data/resolution.py` | KEEP | strip 7 oracle refs |
| `data/area_tiers.py` | PITCH→`eval/buckets.py` | move concept to eval (I5) |

### Eval layer
| Target | Disposition | Sources |
|---|---|---|
| `eval/fetch.py` (`model_info`,`ModelInfo`) | NEW | shape from `multi_head_detector.forward`; absorb the autocast-forward pattern scattered in trainer/eval_checkpoint |
| `eval/decode/threshold.py` | NEW | `sigmoid(patch_logit)>=t` — plain fn, no print |
| `eval/decode/kmeans.py` | PORT | `partition.spherical_kmeans2` + `polarity_attn` (attention polarity, never GT) |
| `eval/decode/graph.py` | PORT | `partition.graph_components_decode`/`decode_deploy_mask`/`calibrate_graph_decode` (+ spatial) |
| `eval/decode/hdbscan.py` | PORT | `graph_lab/hdbscan_decode.py` — MAIN decode set; **drop the `hdbscan_available` soft guard** (hard dep §9.5) |
| `eval/metric.py` | NEW | absorb `eval/metrics.py` (`f1_iou`,`binary_metrics`,`auroc`); the ONLY GT entry |
| `eval/record.py` | NEW | `EvalRecord` |
| `eval/aggregate.py` | NEW | logic seeds: `localization.summarize_localization`, `report_*` (stripped of oracle) |
| `eval/buckets.py` | NEW | from `data/area_tiers.py` thresholds |
| `eval/cache.py` | PORT | `graph_lab/dump_embeddings.py` — but cache `ModelInfo` bundles, not bespoke arrays |
| `eval/robustness.py` | PORT | record-based |
| `eval/partition.py` | DISSOLVE | decode math → `eval/decode/*`; **DELETE** `decode_oracle_labels` & oracle helpers; no monolith remains |
| `eval/sliding_window.py` (953 ln) | PITCH | sliding-window removed entirely; attention-zoom replaces it |
| `eval/window_geometry.py` | PITCH | only existed to serve sliding-window |
| `eval/localization.py` (2034 ln) | PITCH | replaced by fetch+decode+metric+aggregate; **none of the `_oracle_*` paths survive** (salvage the non-oracle reporting into `aggregate.py` FIRST) |
| `eval/zoom.py` | MOVE→`labs/attention_zoom.py` | the zoom strategy lives in labs (§2.3) |
| `eval/image_bce.py` | FOLD | into fetch (image_logit) + metric; kill the dead/shadow split |
| `eval/gap_utils.py` | PORT-if-used | only if a kept decode needs it |
| `eval/decode_cli.py` | FOLD | into `experiments/cli.py` |

### Scripts / labs / train
| Target | Disposition | Sources |
|---|---|---|
| `experiments/scripts/train.py` | PORT(slim) | `scripts/train_multi_head.py` — keep model/data/opt/loop; **all eval bodies removed** (now call `eval/*`); CLI cut to essentials (I7) |
| `experiments/scripts/eval.py` | NEW | builds triplets → fetch → decode → metric → aggregate; replaces `eval_checkpoint.py` (which imported 9 trainer privates — that coupling is deleted) |
| `experiments/labs/attention_zoom.py` | PORT(consolidate) | `eval/zoom.py` + trainer `_attention_zoom_second_pass` + `graph_lab/viz_zoom.py`/`analyze_zoom.py` + `eval_zoom_recovery.py` — GT-free strategy (§2.3) |
| `experiments/labs/decoder_bench.py` | PORT | `benchmark_decoders.py` + `graph_lab/sandbox.py` |
| `experiments/labs/viz.py` | PORT | `viz_decode.py` |
| ~~`sliding_window` / `window_geometry` / `viz_swin`~~ | PITCH | sliding-window removed entirely |
| `experiments/scripts/orchestrate.py` | KEEP | torch-free runner, already clean |
| `experiments/cli.py` | PORT | `pipeline/cli.py` + `eval/decode_cli.py` |
| `inspect_data.py` | PITCH or fold | into a lab if still useful |
| `experiments/configs/{base,augment}.py` | KEEP | fix "v2/v3" docstrings |
| `experiments/experiments/{imd2020_*,tgif2_*}.py` | FOLD | discovery/pairing logic → `data/datasets/*` (item-builders become each dataset's `build()`) |

(Cache is **not** a lab — it is `lab_utils/eval/cache.py`, §2.6.)

### Train script — the eval bodies that LEAVE it
`_run_image_bce_eval`, `_run_localization_eval`, `_run_patch_bce_loc_eval`,
`_make_bce_eval_callable`, `_attention_zoom_second_pass`, `_prep_tgif_items`,
`_tgif_model_filter`, `_tgif_partition_cells`, `_subsample_items`, `_is_real`,
`_kind_is_splice`, `_splice_balance_weights`, `_run_epoch_viz`, `_outlier_score`
→ all relocate to `lab_utils` (eval/* or data/*). After the rebuild, `train.py`
defines **zero** eval/metric functions.

---

## 6. The train script & CLI

- One file: `experiments/scripts/train.py`. Responsibilities only: parse a small
  flag set, resolve paths, build triplets (verified), build loaders, build model,
  run the epoch loop (`lab_utils/train/loop.py`), and call `lab_utils.eval.*` for
  per-epoch validation (same fetch→decode→metric→aggregate as the standalone eval
  — no private epoch-eval copy).
- CLI audit: take the current `_build_parser` (`train_multi_head.py:702`), list
  every flag, and keep only flags that (a) vary across real runs and (b) have no
  good default. Everything else becomes a config default. Target: a flag set that
  fits on one screen. Eval/decode flags come from `experiments/cli.py` builders
  shared with `eval.py`.

---

## 7. Tests

Partitioned `tests/` mirroring `lab_utils/`. **Keep** (port): metrics
(`test_metrics`), sampling determinism (`test_sampling`), decode correctness
(`test_graph_decode`), area→bucket (`test_area_tiers`→buckets), checkpoint
(`test_find_latest_checkpoint`), run-dir (`test_run_dir`). **Pitch**: import/alias
smoke tests, `test_no_bucket_prediction_imports`, `test_step1_smoke`,
`test_window_geometry` + `test_sliding_window_geometry` (sliding-window removed),
and other temporary/fragmented ones. Tests that need model signal **load a cached
`ModelInfo` fixture** (§2.6) instead of building a model.

**New invariant tests (enforce the principles):**
1. `test_no_oracle_outside_train_crop` — the only `oracle` token in the tree is in
   `data/augment/crop.py`. **No `oracle` anywhere under `eval/` or `labs/`.**
   `metric` is the only function whose signature takes a GT/mask/triplet.
2. `test_fetch_is_sole_model_entry` — no module under `eval/` calls `model(`
   outside `fetch.py` (labs reach the model only via `fetch`).
3. `test_decode_is_gt_free` — every `eval/decode/*` function signature takes only
   `ModelInfo` + numeric params (no GT/mask/triplet), and the module imports
   nothing from `data` (no GT path reachable).
4. `test_decode_is_silent` — `eval/decode/*` modules contain no `print(` and no
   logging calls (decode outputs only the decoded signal).
5. `test_no_cross_script_imports` — no `scripts/`/`labs/` module imports another.
6. `test_uniform_item` — every `datasets/*.build()` returns a `Dataset` of `Item`s
   that pass the verifier on a fixture.

---

## 8. Migration procedure

> The ordered, gated execution plan lives in [`GAMEPLAN.md`](GAMEPLAN.md), which
> also pins three cross-cutting contracts: **checkpoint compatibility** (old `.pt`
> files keep loading), **derived run-settings printing** (the settings readout is
> rendered from the one resolved config that drives the run), and **hardware
> portability** (one resolver for dual 2080 Ti / Colab / CPU; torch-free import
> boundary so the data + aggregation layers run without a GPU).

1. **Ratify this doc** (and the open questions in §9).
2. `git mv` the current tree into `legacy/` in one commit — no edits. Tests still
   point at `legacy/` and stay green (proves nothing was lost).
3. Build the new tree **bottom-up**, one layer per change, each behind its own
   tests: `item → datasets/* → verify → dataset/augment → fetch → decode/* →
   metric/record → aggregate/buckets/cache → train.py → eval.py → labs/*`.
4. Each PORT: open the legacy source, read it **in full**, reimplement to comply,
   diff behavior on a fixture, delete reliance on `legacy/`.
5. When a layer's new tests pass and nothing imports its legacy counterpart, that
   counterpart is dead — leave it in `legacy/` (graveyard) and move on.
6. Final: `legacy/` imported by nothing; delete or archive on your word.

This is naturally parallelizable by subagent — each §5 row with its source(s) and
target is a self-contained unit, gated by the invariant tests in §7.

---

## 9. Ratified decisions

1. **Package name:** `experiments/`. The current `contrastive_inpainting_v1`
   directory (self-described `contrastive_test_v3`) is renamed to `experiments/`
   during the rebuild; all "v1/v2/v3" docstrings are corrected.
2. **Verifier policy:** *drop-and-log at index time* — failed triplets are
   filtered out with a logged reason + aggregate counts. The only hard-error is
   the shape contract at `__getitem__` (§3.5).
3. **Bucket thresholds:** reuse the current `area_tiers` cutoffs, documented in
   `eval/buckets.py` (§5). Recalibrate later only with evidence.
4. **`image_score` source:** canonical is image-BCE `sigmoid(image_logit)`; fall
   back to pooled patch logits when the image-BCE head is absent.
5. **HDBSCAN:** **hard dependency.** Add `hdbscan` to required deps; the
   `hdbscan_available()` soft guard is removed and `eval/decode/hdbscan.py`
   assumes the package is present (§2.2, §5).
6. **Sliding-window: removed.** No `swin`/sliding-window/`window_geometry` in the
   rebuild — attention-zoom is the sole multi-pass strategy (§2.3).
7. **Augmentation pipeline:** two-stage (geometric on image+mask, then appearance
   on image only), ordering owned by the `Dataset` (§3.6, I6). Aug **mechanics**
   live in shared `lab_utils/data/augment/` (one impl per op, DRY); the `Dataset`
   assembles them into its pipeline.
```
