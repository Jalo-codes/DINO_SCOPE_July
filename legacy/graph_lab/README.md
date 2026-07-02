# graph_lab — a compartmentalized bench for the graph-components decode

Twist the knobs on the **calibrated-graph decode** (`graph_components_decode`,
the connected-components-over-similarity-graph replacement for k-means(2)) and
*see* where it helps — without re-running the model every time.

The whole point: **dump once, sweep forever.** One GPU pass freezes real
embeddings to a `.npz`; after that every experiment is pure numpy + PIL on the
cache (no model, no dataset, instant).

```
graph_lab/
  dump_embeddings.py   # run ONCE: model → cache/<run>.npz  (z, attention, GT, thumbnails)
  sandbox.py           # run ANY number of times: cache → labelled PNG composites
  cache/               # the .npz dumps live here (gitignored-friendly)
```

## 1. Dump (once per checkpoint, needs GPU + data)

```bash
python -m graph_lab.dump_embeddings \
    --ckpt /content/drive/MyDrive/DINO_SCOPE_RUNS/<run>/epoch_006.pt \
    --imd2020_root /content/IMD2020 --casia_root /content/casia \
    --casia_train --imd_val_only \
    --tau_pos 0.55 --tau_neg 0.20 \
    --n_items 20 --out graph_lab/cache/e006.npz
```

Caches `z` (L2-normalized contrastive embeddings — the exact decode input),
per-patch BCE attention, the GT mask, and a square thumbnail, plus the run's
`tau_pos/tau_neg` so the sandbox defaults match the training margins.

## 2. Sandbox (instant, numpy + PIL only)

**Single setting** — one composite per image,
`Original | GT | K-means | Graph | Graph+spatial`. Graph panels are coloured per
component (green=accepted, red=rejected, gray=sub-`m_min`) and labelled with the
decode's own reasoning + IoU vs GT. Prints a k-means-vs-graph median-IoU line and
a win count.

```bash
python -m graph_lab.sandbox --cache graph_lab/cache/e006.npz \
    --out graph_lab/out/baseline --s_edge 0.375 --knn 10 --spatial 2
```

**Sweep one knob** — `Original | GT | <a panel per value>` per image, plus a
stdout table of median/mean IoU and abstain-rate at each setting so the knee is
obvious. Sweepable: `s_edge`, `mutual_knn_k`, `r_spatial`, `m_min`, `theta_w`,
`theta_x`, `tau_pos`, `tau_neg`.

```bash
python -m graph_lab.sandbox --cache graph_lab/cache/e006.npz \
    --out graph_lab/out/sweep_sedge --sweep s_edge \
    --sweep_vals 0.30 0.34 0.38 0.42 0.46
```

## HDBSCAN mode (density-based, sandbox-only)

Density clustering with mutual-reachability (resists slow-gradient chaining) and a
noise label (natural abstention). Background = largest stable cluster (not a
fragile largest-connected-component); a non-background cluster is emitted when its
mean sim-to-background (`cross`) ≤ `--hdb_theta_x`. Needs `scikit-learn>=1.3`
(`sklearn.cluster.HDBSCAN`) or the `hdbscan` package — **present in Colab**, and
deliberately kept out of the numpy-only shipped decode.

```bash
# add the HDBSCAN panel next to k-means/graph, shown inline in the notebook
python -m graph_lab.sandbox --cache graph_lab/cache/margin1560.npz \
    --out graph_lab/out/hdb --hdbscan \
    --hdb_min_cluster_size 8 --hdb_theta_x 0.5 --hdb_polarity size --show

# sweep an HDBSCAN knob (min_cluster_size | min_samples | theta_x | spatial_weight)
python -m graph_lab.sandbox --cache graph_lab/cache/margin1560.npz \
    --out graph_lab/out/hdb_mcs --method hdbscan \
    --sweep min_cluster_size --sweep_vals 4 6 8 12 16 --show
```

HDBSCAN knobs: `--hdb_min_cluster_size` (smallest splice to keep), `--hdb_min_samples`
(density smoothing; default = min_cluster_size), `--hdb_theta_x` (accept ceiling on
sim-to-background), `--hdb_spatial_weight` (>0 folds scaled `(row,col)` into the
features so adjacency matters), `--hdb_polarity` (`size` | `attention` — pick
background by largest cluster or lowest BCE attention, for >50% splices).

`--show` renders each composite inline in Colab/Jupyter (via `IPython.display`) in
addition to saving the PNGs. **It only works when run inside the notebook kernel** —
a `!python -m graph_lab.sandbox ... --show` *subprocess* can't reach the frontend
and will only save PNGs. To display inline, call `main` from a cell:

```python
from graph_lab import sandbox
sandbox.main([
    "--cache", "graph_lab/cache/margin1560.npz",
    "--out",   "graph_lab/out/hdb",
    "--s_edge", "0.97", "--hdbscan", "--hdb_theta_x", "0.5", "--show",
])
```

## Attention zoom (coarse→fine), decoder comparison — `viz_zoom.py`

Visualizes the repo's **existing** coarse→fine zoom (the same path as
`collect_coarse_to_fine_samples`), not a reimplementation, run independently for
**K-means / graph / HDBSCAN** so you can compare them under the zoom. Model-bearing
(it re-embeds crops), so it's a separate script from the model-free sandbox.

Per decoder: pass 1 decodes the full frame; the zoom bbox comes from
`_minority_bbox` (single) or attention via `multi_zoom_bboxes` (multi, that
decoder's foreground as hot_mask); pass 2 crops, re-embeds, decodes, and
`_place_fine_in_pixel_frame` pastes it back — kept only when the image head agrees
(`p_zoom >= p_full`, the ratchet).

Each splice is **one multi-row PNG**: a shared context row
(`Original | Attention(full) | GT`), then **one row per decoder** —
`coarse | refined+bbox | zoom crop | zoom heatmap | zoom decode` — so you see the
actual cropped region the model re-embedded and its zoomed attention, not just the
pasted-back mask. The coarse/refined IoU is printed on each decoder's panels. A
per-decoder summary table prints coarse/refined median IoU, how often zoom fired,
and how often it improved.

```bash
python -m graph_lab.viz_zoom \
    --ckpt /content/drive/MyDrive/DINO_SCOPE_RUNS/<run>/best.pt \
    --imd2020_root /content/IMD2020 --imd_val_only \
    --tau_pos 0.60 --tau_neg 0.15 --s_edge 0.97 \
    --hdb_min_cluster_size 8 --hdb_theta_x 0.5 \
    --zoom_mode single --pad_frac 0.25 --n_items 24 \
    --out /content/viz_zoom_margin1560
```
`--methods kmeans,graph,hdbscan` selects the subset to compare. Add `--show` and
call `viz_zoom.main([...])` from a cell to render inline (same in-kernel rule as
the sandbox). `--zoom_mode multi` uses the attention-driven `multi_zoom_bboxes`.

## Knobs (all map to `graph_components_decode` / `DecodeSpec`)

| flag | meaning | default |
|------|---------|---------|
| `--s_edge` | absolute cosine bar for an edge | mid-band `(tau_pos+tau_neg)/2` |
| `--knn` | mutual-kNN k (anti-chaining) | 10 |
| `--spatial` | Chebyshev radius for the `Graph+spatial` panel | 2 (0=skip) |
| `--m_min` | min component size to score | 4 |
| `--theta_w` | component acceptance: cohesion floor | `tau_pos - 0.05` |
| `--theta_x` | component acceptance: sim-to-background ceiling | mid-band |
| `--tau_pos/--tau_neg` | trained margins | from cache |

`None`/unset → the decode's own calibrated default, so leaving a flag off is the
honest baseline. See `GRAPH_DECODE_PLAN.md` for the full formulation.

> Same heavy deps as `scripts/viz_decode.py` (torch only for the dump; PIL+numpy
> for the sandbox). Runs on Colab where the checkpoints live.
