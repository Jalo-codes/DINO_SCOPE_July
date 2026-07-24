# analysis/

Offline analysis & audit scripts: everything here consumes *already-produced*
artifacts (records CSVs, eval caches, run logs, dataset manifests) and emits
tables/reports. Nothing here trains a model or produces canonical eval records —
those live in `experiments/scripts/` (entry points) and `lab_utils/eval/` (logic).

Run from the repo root as modules, e.g. `python -m analysis.probe_contrasts ...`.

| script | reads | emits |
|---|---|---|
| `probe_manifest.py` | probe roots (region_probes datasets) | join-table CSV + renders |
| `probe_contrasts.py` | probe records CSVs + manifest | raw tables, rank-AUC contrasts |
| `full_fakes_report.py` | full-fakes eval records | whole-image AUC report |
| `decoder_bench.py` | frozen eval cache (ModelInfo) | decoder comparison table |
| `otsu_vs_threshold.py` | decoder-bench `*_records.csv` + `sweep_records.csv` | adaptive(otsu)-vs-fixed(thr@0.5/oracle-t) F1 table, per source × size bucket |
| `run_inventory.py` | `run_config.json` under a runs root (Drive/box/local) | one manifest of every run's identity (arch, splice_mix, pico/ff-in-train, checkpoints on disk) |
| `image_auroc_by_size.py` | any decoder records CSV | detection AUROC (fake vs real) per fake mask-size bucket — decoder/zoom-independent |
| `rollup_ablation_eval.py` | orchestrator.log under a run_root | per-cell + headline CSVs |
| `audit_zoom_image_auc.py` | checkpoint (collect) / logits CSV (analyze) | zoom-fusion AUROC audit |
| `coco_leakage_probe.py` | dataset roots / manifests | COCO-provenance leakage audit |
| `audit_coco_overlap.py` | InpaintCOCO metadata stream | OOD-honesty overlap audit |
