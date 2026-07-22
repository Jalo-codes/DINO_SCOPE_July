"""analysis.compare_patch_auroc — paired-bootstrap A/B for patch_balance (D7).

Compares exactly two checkpoints (e.g. --patch_balance global vs per_image,
experiment 1 in docs/equal_budget_bce_spec.md) on threshold-free patch AUROC,
with a 95% CI on the delta from a bootstrap that resamples IMAGES, never
patches — patches within one image are correlated (same generator, same
crop), so resampling patches directly would understate the true variance.

DEVIATION FROM THE WRITTEN SPEC (flagged per docs/equal_budget_bce_spec.md's
own instruction to flag spec/code disagreements): the spec describes this
script as reading eval_numbers.py's --out_json. That JSON only carries
per-item SCALAR summaries (scores_fake_mean / scores_bg_mean), which is
insufficient to recompute AUROC — a nonlinear function of the full per-patch
score distribution — on a resampled item set. Recomputing AUROC on scalar
per-item means would silently answer a different, coarser question ("does
the per-image mean separate the classes") instead of "does the per-PATCH
score separate the classes", which is what the equal-budget redesign is
actually supposed to fix. So this script is two-phase instead (mirroring the
existing analysis.audit_zoom_image_auc precedent): a `collect` phase (GPU,
needs torch + the dataset registry) writes one CSV row per (checkpoint,
item, stratum) with that item's full per-patch score/weight arrays
(semicolon-joined) — enough to exactly reconstruct any resampled AUROC — and
an `analyze` phase (numpy-only, runs anywhere, including a torch-less
laptop) does the bootstrap from that CSV.

Usage (collect, on a GPU box — scores BOTH checkpoints over the SAME item
set, which is what makes the bootstrap pairing exact):
    python -m analysis.compare_patch_auroc collect \\
        --checkpoint /runs/exp1_arm_global/best.pt /runs/exp1_arm_perimage/best.pt \\
        --label global per_image \\
        --tgif2_root /data/tgif2 --tgif_types sp \\
        --imd2020_root /data/imd2020 --imd_val_split 1.0 \\
        --amp_dtype float16 --out_csv /runs/exp1_patch_scores.csv

Usage (analyze, anywhere):
    python -m analysis.compare_patch_auroc analyze \\
        --from_csv /runs/exp1_patch_scores.csv --n_bootstrap 1000
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# The registry (and mask_to_patch_labels_soft) drag in torch — optional here so
# `analyze --from_csv` runs on a torch-less machine (mirrors audit_zoom_image_auc).
try:
    from lab_utils.eval.val_sources import add_source_root_args, collect_val_items_by_source
    _HAVE_DATASETS = True
except ModuleNotFoundError:
    _HAVE_DATASETS = False

from lab_utils.eval.buckets import BUCKET_LABELS
from lab_utils.eval.patch_scores import weighted_auroc
from lab_utils.logging.text import log_line

CSV_FIELDS = ['label', 'item_id', 'bucket', 'stratum', 'scores', 'weights']


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='compare_patch_auroc',
        description='Paired-bootstrap patch-AUROC comparison between two checkpoints.',
    )
    sub = p.add_subparsers(dest='phase', required=True)

    pc = sub.add_parser('collect', help='GPU phase: score both checkpoints, write a CSV.')
    pc.add_argument('--checkpoint', nargs=2, required=True, metavar=('CKPT_A', 'CKPT_B'))
    pc.add_argument('--label', nargs=2, required=True, metavar=('LABEL_A', 'LABEL_B'))
    pc.add_argument('--out_csv', required=True)
    pc.add_argument('--band', type=float, nargs=2, default=(0.2, 0.8), metavar=('LOW', 'HIGH'))
    pc.add_argument('--image_size', type=int, default=448)
    pc.add_argument('--patch_size', type=int, default=16)
    pc.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    pc.add_argument('--no_amp', action='store_true')
    pc.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'])
    pc.add_argument('--tgif_types', nargs='+', default=None, choices=['sp', 'fr'])
    pc.add_argument('--max_items', type=int, default=None)
    if _HAVE_DATASETS:
        g = pc.add_argument_group('dataset roots')
        add_source_root_args(g)

    pa = sub.add_parser('analyze', help='Torch-free phase: bootstrap the delta from a CSV.')
    pa.add_argument('--from_csv', required=True)
    pa.add_argument('--n_bootstrap', type=int, default=1000)
    pa.add_argument('--seed', type=int, default=0)
    pa.add_argument('--summary_out', default=None)

    return p


# ── phase 1: collect ─────────────────────────────────────────────────────────

def _collect_one(model, item, res, *, device, use_amp, amp_dtype, band):
    """Return (bucket, {'fake': (scores, weights), 'splice_bg': (...), 'real_bg': (...)})
    for one item, or None to skip it. Mirrors lab_utils.eval.patch_scores.collect_patch_scores
    item-by-item, but keeps raw per-patch arrays instead of only summary stats."""
    import numpy as _np
    from PIL import Image

    from lab_utils.data.resolution import mask_to_patch_labels_soft
    from lab_utils.eval.buckets import area_to_bucket
    from lab_utils.eval.fetch import model_info
    from lab_utils.eval.preprocess import load_image_tensor

    if item.meta.get('gt_mask_reliable') is False or item.meta.get('crop_window') is not None:
        return None

    img_t = load_image_tensor(item, res, device=device)
    info = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
    if info.patch_logits is None:
        raise RuntimeError('patch_logits is None (patch-BCE head not enabled on this checkpoint)')

    logits = _np.asarray(info.patch_logits, dtype=_np.float64).reshape(-1)
    probs = 1.0 / (1.0 + _np.exp(-logits))
    n_side = info.grid_hw[0]
    n_patches = n_side * n_side

    if item.is_real:
        return 'real', {'real_bg': (probs, _np.ones(n_patches))}

    mask_area = item.mask_area(res)
    bucket = area_to_bucket(mask_area)
    mask_pil = (
        Image.open(item.mask).convert('L')
        .resize((res.image_size, res.image_size), Image.NEAREST)
    )
    labels_t, weights_t = mask_to_patch_labels_soft(mask_pil, res, low=band[0], high=band[1])
    labels = labels_t.numpy().astype(_np.float64).reshape(-1)
    weights = weights_t.numpy().astype(_np.float64).reshape(-1)
    if labels.shape[0] != n_patches:
        raise RuntimeError(f'grid mismatch: mask={labels.shape[0]} model={n_patches}')

    fake_m = (labels > 0.5) & (weights > 0.0)
    bg_m = (labels <= 0.5) & (weights > 0.0)
    strata = {}
    if fake_m.any():
        strata['fake'] = (probs[fake_m], weights[fake_m])
    if bg_m.any():
        strata['splice_bg'] = (probs[bg_m], weights[bg_m])
    return bucket, strata


def collect(args) -> Path:
    if not _HAVE_DATASETS:
        raise SystemExit('collect phase needs torch + the dataset registry '
                         '(this environment could not import them)')
    import torch
    from pathlib import Path as _Path

    from lab_utils.data.resolution import Resolution
    from lab_utils.eval.load_model import load_eval_model
    from lab_utils.train.distributed import unwrap_model

    device = torch.device(args.device)
    use_amp = not args.no_amp and args.device == 'cuda'
    res = Resolution(image_size=args.image_size, patch_size=args.patch_size)

    by_source = collect_val_items_by_source(args, res, log_tag='[patch-cmp]')
    # val_sources.py has no --tgif_types plumbing (that's an eval_numbers-only
    # flag); rebuild tgif2 directly when the caller restricted types, mirroring
    # lab_utils.eval.numbers._build_item_sets's special-case.
    if args.tgif_types and 'tgif2' in by_source:
        from lab_utils.data.datasets.registry import REGISTRY
        tgif_root = _Path(getattr(args, 'tgif2_root'))
        _, val_ds = REGISTRY['tgif2'](
            tgif_root, res=res, build_train_side=False, types=set(args.tgif_types))
        by_source['tgif2'] = val_ds.items
        log_line(f'[patch-cmp] tgif2 restricted to types={sorted(args.tgif_types)}: '
                 f'{len(val_ds.items)} items')

    items = [it for its in by_source.values() for it in its]
    if args.max_items:
        items = items[:args.max_items]
    if not items:
        raise SystemExit('no dataset roots configured/found — nothing to collect')
    log_line(f'[patch-cmp] {len(items)} items total across {len(by_source)} source(s)')

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for ckpt, label in zip(args.checkpoint, args.label):
            log_line(f'[patch-cmp] === loading label={label} ckpt={ckpt} ===')
            model, _cfg, _res = load_eval_model(ckpt, device=args.device, strict=False)
            model = unwrap_model(model)
            model.eval()

            n = len(items)
            every = max(1, n // 20)
            n_written = n_skipped = n_failed = 0
            with torch.no_grad():
                for i, item in enumerate(items):
                    try:
                        result = _collect_one(
                            model, item, res, device=device, use_amp=use_amp,
                            amp_dtype=args.amp_dtype, band=tuple(args.band),
                        )
                    except Exception as exc:
                        log_line(f'[patch-cmp] WARN label={label} failed {item.item_id}: {exc}')
                        n_failed += 1
                        continue
                    if result is None:
                        n_skipped += 1
                        continue
                    bucket, strata = result
                    for stratum, (scores, weights) in strata.items():
                        writer.writerow({
                            'label': label, 'item_id': item.item_id, 'bucket': bucket,
                            'stratum': stratum,
                            'scores': ';'.join(f'{s:.6f}' for s in scores),
                            'weights': ';'.join(f'{w:.6f}' for w in weights),
                        })
                    n_written += 1
                    if (i + 1) % every == 0 or (i + 1) == n:
                        fh.flush()
                        log_line(f'[patch-cmp] label={label} {i + 1}/{n} '
                                 f'(written={n_written} skipped={n_skipped} failed={n_failed})')
            log_line(f'[patch-cmp] label={label}: {n_written} items written, '
                     f'{n_skipped} skipped, {n_failed} failed')
    log_line(f'[patch-cmp] wrote -> {out_csv}')
    return out_csv


# ── phase 2: analyze ─────────────────────────────────────────────────────────

def _parse_arr(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split(';')], dtype=np.float64) if s else np.zeros(0)


def _load_rows(csv_path) -> Dict[str, Dict[str, dict]]:
    """{label: {item_id: {'bucket': str, 'strata': {stratum: (scores, weights)}}}}"""
    data: Dict[str, Dict[str, dict]] = defaultdict(dict)
    with Path(csv_path).open() as fh:
        for row in csv.DictReader(fh):
            entry = data[row['label']].setdefault(row['item_id'], {'bucket': row['bucket'], 'strata': {}})
            entry['strata'][row['stratum']] = (_parse_arr(row['scores']), _parse_arr(row['weights']))
    return data


def _pooled_stats(label_data: Dict[str, dict], id_subset) -> dict:
    """AUROC (pooled + per bucket) and real_bg p99 over a (possibly repeated,
    i.e. bootstrap-resampled) list of item_ids for one label."""
    fake_chunks, sbg_chunks, rbg_chunks = [], [], []
    bucket_fake: Dict[str, list] = defaultdict(list)

    for iid in id_subset:
        entry = label_data.get(iid)
        if entry is None:
            continue
        bucket = entry['bucket']
        if 'fake' in entry['strata']:
            fake_chunks.append(entry['strata']['fake'])
            bucket_fake[bucket].append(entry['strata']['fake'])
        if 'splice_bg' in entry['strata']:
            sbg_chunks.append(entry['strata']['splice_bg'])
        if 'real_bg' in entry['strata']:
            rbg_chunks.append(entry['strata']['real_bg'])

    def _cat(chunks, idx):
        parts = [c[idx] for c in chunks]
        return np.concatenate(parts) if parts else np.zeros(0)

    fake_s, fake_w = _cat(fake_chunks, 0), _cat(fake_chunks, 1)
    sbg_s, sbg_w = _cat(sbg_chunks, 0), _cat(sbg_chunks, 1)
    rbg_s, rbg_w = _cat(rbg_chunks, 0), _cat(rbg_chunks, 1)
    all_bg_s = np.concatenate([sbg_s, rbg_s])
    all_bg_w = np.concatenate([sbg_w, rbg_w])

    def _auc(pos_s, pos_w):
        s = np.concatenate([pos_s, all_bg_s])
        y = np.concatenate([np.ones_like(pos_s), np.zeros_like(all_bg_s)])
        w = np.concatenate([pos_w, all_bg_w])
        return weighted_auroc(s, y, w)

    pooled_auc = _auc(fake_s, fake_w)
    by_bucket = {}
    for b in BUCKET_LABELS:
        bchunks = bucket_fake.get(b, [])
        bs = _cat(bchunks, 0)
        bw = _cat(bchunks, 1)
        by_bucket[b] = _auc(bs, bw) if bs.size else float('nan')
    real_bg_p99 = float(np.quantile(rbg_s, 0.99)) if rbg_s.size else float('nan')
    return {'pooled_auc': pooled_auc, 'by_bucket': by_bucket, 'real_bg_p99': real_bg_p99}


def analyze(csv_path, *, n_bootstrap: int, seed: int, summary_out: Optional[str] = None) -> dict:
    data = _load_rows(csv_path)
    labels = sorted(data.keys())
    if len(labels) != 2:
        raise SystemExit(f'compare_patch_auroc needs exactly 2 labels in the CSV, found {labels}')
    label_a, label_b = labels

    ids_a, ids_b = set(data[label_a]), set(data[label_b])
    common = sorted(ids_a & ids_b)
    if ids_a != ids_b:
        log_line(f'[patch-cmp] WARN: {len(ids_a ^ ids_b)} items present under only one label — '
                 f'using {len(common)} common items for the paired bootstrap')
    if not common:
        raise SystemExit('no items common to both labels — nothing to compare')
    ids_arr = np.array(common, dtype=object)
    n = len(ids_arr)

    point_a = _pooled_stats(data[label_a], common)
    point_b = _pooled_stats(data[label_b], common)

    log_line(f'[patch-cmp] === {label_a} vs {label_b} (n_items={n}) ===')
    log_line(f'[patch-cmp] {label_a:<12} pooled_auc={point_a["pooled_auc"]:.4f} '
             f'real_bg_p99={point_a["real_bg_p99"]:.4f}')
    log_line(f'[patch-cmp] {label_b:<12} pooled_auc={point_b["pooled_auc"]:.4f} '
             f'real_bg_p99={point_b["real_bg_p99"]:.4f}')
    for b in BUCKET_LABELS:
        log_line(f'[patch-cmp]   bucket={b}: {label_a}={point_a["by_bucket"][b]:.4f}  '
                 f'{label_b}={point_b["by_bucket"][b]:.4f}')

    rng = np.random.default_rng(seed)
    boot_pooled = np.empty(n_bootstrap)
    boot_rbgp99 = np.empty(n_bootstrap)
    boot_bucket = {b: np.empty(n_bootstrap) for b in BUCKET_LABELS}
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        resample_ids = ids_arr[idx]
        sa = _pooled_stats(data[label_a], resample_ids)
        sb = _pooled_stats(data[label_b], resample_ids)
        boot_pooled[i] = sb['pooled_auc'] - sa['pooled_auc']
        boot_rbgp99[i] = sb['real_bg_p99'] - sa['real_bg_p99']
        for b in BUCKET_LABELS:
            boot_bucket[b][i] = sb['by_bucket'][b] - sa['by_bucket'][b]
        if (i + 1) % max(1, n_bootstrap // 10) == 0:
            log_line(f'[patch-cmp] bootstrap {i + 1}/{n_bootstrap}')

    def _ci(deltas: np.ndarray) -> Tuple[float, float, float]:
        d = deltas[~np.isnan(deltas)]
        if d.size == 0:
            return float('nan'), float('nan'), float('nan')
        return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    result = {'label_a': label_a, 'label_b': label_b, 'n_items': n,
             'point': {label_a: point_a, label_b: point_b}, 'delta': {}}

    mean, lo, hi = _ci(boot_pooled)
    sig = '*' if (lo > 0 or hi < 0) else ' '
    log_line(f'[patch-cmp] DELTA pooled_auc ({label_b} - {label_a}): '
             f'mean={mean:+.4f} CI95=[{lo:+.4f}, {hi:+.4f}] {sig}')
    result['delta']['pooled_auc'] = {'mean': mean, 'ci95': [lo, hi]}

    for b in BUCKET_LABELS:
        mean, lo, hi = _ci(boot_bucket[b])
        sig = '*' if (lo > 0 or hi < 0) else ' '
        log_line(f'[patch-cmp] DELTA bucket={b} auroc ({label_b} - {label_a}): '
                 f'mean={mean:+.4f} CI95=[{lo:+.4f}, {hi:+.4f}] {sig}')
        result['delta'][f'bucket_{b}_auroc'] = {'mean': mean, 'ci95': [lo, hi]}

    mean, lo, hi = _ci(boot_rbgp99)
    sig = '*' if (lo > 0 or hi < 0) else ' '
    log_line(f'[patch-cmp] DELTA real_bg_p99 ({label_b} - {label_a}) [sprinkle canary — '
             f'worse means MORE false-alarm on clean reals]: '
             f'mean={mean:+.4f} CI95=[{lo:+.4f}, {hi:+.4f}] {sig}')
    result['delta']['real_bg_p99'] = {'mean': mean, 'ci95': [lo, hi]}

    if summary_out:
        import json
        Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_out).write_text(json.dumps(result, indent=2))
        log_line(f'[patch-cmp] summary json -> {summary_out}')

    return result


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.phase == 'collect':
        collect(args)
    else:
        analyze(args.from_csv, n_bootstrap=args.n_bootstrap, seed=args.seed,
               summary_out=args.summary_out)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
