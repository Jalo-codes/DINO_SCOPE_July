"""experiments.scripts.audit_zoom_image_auc — does zoom help image-level AUROC?

Structural fact this audit exists to quantify: the zoom two-pass has NEVER
touched the image score.  attention_zoom_single builds its EvalRecord from the
pass-1 (full-frame) ModelInfo, so image_score == sigmoid(full-frame logit) with
or without zoom; the crop's image_logit is computed and discarded (multi /
second_best use it only to gate masks).  Every "zoomed" image AUROC ever logged
is therefore numerically identical to the flat AUROC.  This script measures the
untapped headroom from actually FUSING the crop logit into the image score.

Two phases, decoupled through a CSV so the GPU pass runs once and fusion-rule
sweeps re-run offline (numpy-only — works on the torch-less Mac venv):

  1. Collect (needs GPU + datasets): per item, pass 1 full-frame logit +
     attention -> DEFAULT_ZOOM bbox (the exact crop geometry val/eval/predict
     share) -> if non-trivial, pass 2 on the crop -> crop logit.  One CSV row
     per item: source, item_id, is_real, full_logit, zoomed, crop_logit,
     bbox_area_frac.

  2. Analyze (--from_csv, no torch): AUROC per source for each fusion rule —
     flat (baseline == production), max(full, crop), crop-when-zoomed,
     mean(full, crop), min(full, crop) — with a bootstrap CI on the delta vs
     flat (methodology rule 6: distribution + CI, never a bare point estimate).
     Also broken out on the zoomed-only subset, where fusion can actually act.

Usage (collect, on a GPU box):
    python -m experiments.scripts.audit_zoom_image_auc \
        --checkpoint /path/epoch_0003.pt \
        --tgif2_root ... --sagid_root ... --imd2020_root ... --casia_root ... \
        --out_csv runs/zoom_audit/gemini_v3_e3.csv

Usage (analyze, anywhere):
    python -m experiments.scripts.audit_zoom_image_auc \
        --from_csv runs/zoom_audit/gemini_v3_e3.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# val_sources drags in the dataset registry, whose Dataset base imports torch —
# optional here so --from_csv analysis runs on a torch-less machine (the flags
# it would add are only meaningful in the collect phase anyway).
try:
    from lab_utils.eval.val_sources import add_source_root_args, collect_val_items_by_source
    _HAVE_DATASETS = True
except ModuleNotFoundError:
    _HAVE_DATASETS = False

from lab_utils.logging.text import install_log, log_line

from experiments.configs.zoom import DEFAULT_ZOOM


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_percentile(val: str):
    try:
        return float(val)
    except ValueError:
        return val


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='audit_zoom_image_auc',
        description='Audit image-level AUROC headroom from fusing zoom-crop logits.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--checkpoint', default=None,
                   help='Path to .pt checkpoint (required unless --from_csv)')
    p.add_argument('--out_csv', default=None,
                   help='Where to write the per-item logit CSV (collect phase)')
    p.add_argument('--from_csv', default=None,
                   help='Skip the forward pass; analyze an existing CSV (no torch needed)')
    p.add_argument('--summary_out', default=None,
                   help='Optional path to write the analysis table as JSON')

    if _HAVE_DATASETS:
        g = p.add_argument_group('dataset roots (collect phase)')
        add_source_root_args(g)

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'],
                   help='Turing (2080) = float16; Ada / L4 = bfloat16.')

    g = p.add_argument_group('eval control')
    g.add_argument('--max_items', type=int, default=None)
    g.add_argument('--sources', nargs='*', default=None)

    # Zoom geometry: same flags + DEFAULT_ZOOM defaults as eval.py, so the
    # audited crop is byte-identical to the one val/eval/predict take.
    g = p.add_argument_group('zoom geometry (defaults = DEFAULT_ZOOM)')
    g.add_argument('--attn_percentile', type=_parse_percentile,
                   default=DEFAULT_ZOOM.attn_percentile)
    g.add_argument('--attn_thresh_mult', type=float, default=DEFAULT_ZOOM.attn_thresh_mult)
    g.add_argument('--attn_pad_frac', type=float, default=DEFAULT_ZOOM.attn_pad_frac)
    g.add_argument('--min_box_size', type=int, default=DEFAULT_ZOOM.min_box_size)
    g.add_argument('--attn_min_pad_frac', type=float, default=DEFAULT_ZOOM.attn_min_pad_frac)
    g.add_argument('--zoom_pad_frac', type=float, default=DEFAULT_ZOOM.pad_side_frac)
    g.add_argument('--zoom_min_area', type=float, default=DEFAULT_ZOOM.min_area_frac)
    g.add_argument('--min_crop_frac', type=float, default=DEFAULT_ZOOM.min_crop_frac)

    g = p.add_argument_group('analysis')
    g.add_argument('--n_bootstrap', type=int, default=2000)
    g.add_argument('--seed', type=int, default=0)
    return p


# ── phase 1: collect ───────────────────────────────────────────────────────────

CSV_FIELDS = ['source', 'item_id', 'is_real', 'subgroup', 'full_logit', 'zoomed',
              'crop_logit', 'bbox_area_frac']


def collect(args) -> Path:
    if not _HAVE_DATASETS:
        raise SystemExit('collect phase needs torch + the dataset registry '
                         '(this environment could not import them)')
    import torch  # lazy: analysis-only runs never need it

    from lab_utils.eval.fetch import model_info
    from lab_utils.eval.load_model import load_eval_model
    from lab_utils.eval.preprocess import load_image_tensor
    from lab_utils.eval.zoom import attention_to_bbox, bbox_is_trivial, crop_to_bbox
    from lab_utils.train.distributed import unwrap_model

    device = torch.device(args.device)
    use_amp = not args.no_amp and args.device == 'cuda'

    model, cfg, res = load_eval_model(args.checkpoint, device=args.device, strict=False)
    model = unwrap_model(model)
    model.eval()

    by_source = collect_val_items_by_source(args, res, log_tag='[zoom]')
    if not by_source:
        raise SystemExit('no dataset roots configured — nothing to collect')

    out_csv = Path(args.out_csv or 'zoom_audit.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    n_done = 0
    with out_csv.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        with torch.no_grad():
            for source, items in by_source.items():
                for item in items:
                    try:
                        row = _collect_one(
                            model, item, res, device=device,
                            use_amp=use_amp, amp_dtype=args.amp_dtype, args=args,
                            attention_to_bbox=attention_to_bbox,
                            bbox_is_trivial=bbox_is_trivial,
                            crop_to_bbox=crop_to_bbox,
                            model_info=model_info,
                            load_image_tensor=load_image_tensor,
                        )
                    except Exception as exc:
                        log_line(f'[zoom] WARN: skipped {item.item_id}: {exc}')
                        continue
                    writer.writerow(row)
                    n_done += 1
                    if n_done % 50 == 0:
                        fh.flush()
                        log_line(f'[zoom] {n_done} items done '
                                 f'(current source={source})')
    log_line(f'[zoom] wrote {n_done} rows -> {out_csv}')
    return out_csv


def _collect_one(model, item, res, *, device, use_amp, amp_dtype, args,
                 attention_to_bbox, bbox_is_trivial, crop_to_bbox,
                 model_info, load_image_tensor) -> dict:
    from PIL import Image as PILImage

    img_pil = PILImage.open(item.image).convert('RGB')
    x = load_image_tensor(img_pil, res, device=device)
    info1 = model_info(model, x, device=device, amp=use_amp, amp_dtype=amp_dtype)

    # subgroup = per-generator cell (full_fakes) or TGIF cell — lets the
    # analysis break AUROC out per generator against the source's shared reals.
    subgroup = item.meta.get('generator') or item.meta.get('tgif_subcat') or ''
    row = {
        'source': item.source,
        'item_id': item.item_id,
        'is_real': int(item.is_real),
        'subgroup': subgroup,
        'full_logit': float(info1.image_logit),
        'zoomed': 0,
        'crop_logit': '',
        'bbox_area_frac': '',
    }
    if info1.attention is None:
        return row

    bbox = attention_to_bbox(
        info1.attention, info1.grid_hw,
        percentile=args.attn_percentile, thresh_mult=args.attn_thresh_mult,
        pad_frac=args.attn_pad_frac, min_box_size=args.min_box_size,
        min_pad_frac=args.attn_min_pad_frac,
        pad_side_frac=args.zoom_pad_frac, min_area_frac=args.zoom_min_area,
    )
    if bbox_is_trivial(bbox, min_crop_frac=args.min_crop_frac):
        return row

    y0, x0, y1, x1 = bbox
    crop_pil = crop_to_bbox(img_pil, bbox)
    crop_tensor = load_image_tensor(crop_pil, res, device=device)
    info2 = model_info(model, crop_tensor, device=device, amp=use_amp, amp_dtype=amp_dtype)

    row['zoomed'] = 1
    row['crop_logit'] = float(info2.image_logit)
    row['bbox_area_frac'] = float((y1 - y0) * (x1 - x0))
    return row


# ── phase 2: analyze ───────────────────────────────────────────────────────────

# Each rule maps (full, crop, zoomed) arrays -> per-item score. crop is NaN
# where zoomed == 0; every rule must fall back to full there.
def _fused(full: np.ndarray, crop: np.ndarray, zoomed: np.ndarray, rule: str) -> np.ndarray:
    c = np.where(zoomed.astype(bool), crop, full)
    if rule == 'flat':
        return full
    if rule == 'crop':
        return c
    if rule == 'max':
        return np.maximum(full, c)
    if rule == 'min':
        return np.minimum(full, c)
    if rule == 'mean':
        return 0.5 * (full + c)
    raise ValueError(f'unknown fusion rule {rule!r}')


FUSION_RULES = ['flat', 'max', 'mean', 'crop', 'min']


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC (Mann-Whitney), tie-aware; NaN if one class missing."""
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _bootstrap_delta(full, crop, zoomed, labels, rule, *, n_boot, rng):
    """95% CI on AUROC(rule) - AUROC(flat), paired resampling over items."""
    n = len(labels)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f, c, z, y = full[idx], crop[idx], zoomed[idx], labels[idx]
        deltas[b] = _auroc(_fused(f, c, z, rule), y) - _auroc(_fused(f, c, z, 'flat'), y)
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return float('nan'), float('nan')
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def analyze(csv_path: Path, *, n_bootstrap: int, seed: int,
            summary_out: Optional[str] = None) -> None:
    rows: List[dict] = []
    with Path(csv_path).open() as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    if not rows:
        raise SystemExit(f'no rows in {csv_path}')

    def _arrays(subset: List[dict]):
        full = np.array([float(r['full_logit']) for r in subset])
        zoomed = np.array([int(r['zoomed']) for r in subset])
        crop = np.array([float(r['crop_logit']) if r['crop_logit'] != '' else np.nan
                         for r in subset])
        labels = np.array([1 - int(r['is_real']) for r in subset])  # 1 = fake
        return full, crop, zoomed, labels

    sources = sorted({r['source'] for r in rows})
    groups: Dict[str, List[dict]] = {s: [r for r in rows if r['source'] == s]
                                     for s in sources}
    if len(sources) > 1:
        groups['POOLED (mixed sources — per-source rows are the primary read)'] = rows

    rng = np.random.default_rng(seed)
    summary = {}
    for name, subset in groups.items():
        full, crop, zoomed, labels = _arrays(subset)
        n = len(subset)
        zoom_rate = float(zoomed.mean())
        zoom_rate_fake = float(zoomed[labels == 1].mean()) if (labels == 1).any() else float('nan')
        zoom_rate_real = float(zoomed[labels == 0].mean()) if (labels == 0).any() else float('nan')

        log_line(f'[zoom] === {name} ===  n={n}  '
                 f'zoom_rate={zoom_rate:.3f} (fake {zoom_rate_fake:.3f} / real {zoom_rate_real:.3f})')
        entry = {'n': n, 'zoom_rate': zoom_rate,
                 'zoom_rate_fake': zoom_rate_fake, 'zoom_rate_real': zoom_rate_real,
                 'rules': {}}

        for rule in FUSION_RULES:
            auc = _auroc(_fused(full, crop, zoomed, rule), labels)
            if rule == 'flat':
                log_line(f'[zoom]   {rule:<6} auroc={auc:.4f}  (baseline == production)')
                entry['rules'][rule] = {'auroc': auc}
                continue
            lo, hi = _bootstrap_delta(full, crop, zoomed, labels, rule,
                                      n_boot=n_bootstrap, rng=rng)
            flat_auc = entry['rules']['flat']['auroc']
            sig = '*' if (lo > 0 or hi < 0) else ' '
            log_line(f'[zoom]   {rule:<6} auroc={auc:.4f}  '
                     f'delta={auc - flat_auc:+.4f} CI95=[{lo:+.4f}, {hi:+.4f}] {sig}')
            entry['rules'][rule] = {'auroc': auc, 'delta': auc - flat_auc,
                                    'delta_ci95': [lo, hi]}

        # Activation scores (mean sigmoid), reals vs fakes, per rule.  Reported
        # per methodology rule 7: a mean is only readable NEXT TO the same
        # condition's reals mean — never compare means across models/decoders,
        # use the AUROCs above for that.
        entry['activation'] = {}
        log_line(f'[zoom]   activation (mean sigmoid):  reals    fakes')
        for rule in ('flat', 'max', 'crop'):
            s = 1.0 / (1.0 + np.exp(-_fused(full, crop, zoomed, rule)))
            m_real = float(s[labels == 0].mean()) if (labels == 0).any() else float('nan')
            m_fake = float(s[labels == 1].mean()) if (labels == 1).any() else float('nan')
            log_line(f'[zoom]     {rule:<6}                     '
                     f'{m_real:.4f}   {m_fake:.4f}')
            entry['activation'][rule] = {'mean_real': m_real, 'mean_fake': m_fake}

        # Per-generator / per-cell breakdown: each subgroup's fakes vs the
        # SOURCE's whole real pool (reals carry no subgroup).  Point estimates
        # only — n per cell is small; the source-level CI above is the read.
        subs = sorted({r['subgroup'] for r in subset if r['subgroup']})
        reals = [r for r in subset if int(r['is_real']) == 1]
        if subs and reals:
            entry['subgroups'] = {}
            for sub in subs:
                cell = [r for r in subset
                        if r['subgroup'] == sub and int(r['is_real']) == 0] + reals
                cf, cc, cz, cy = _arrays(cell)
                flat_auc = _auroc(_fused(cf, cc, cz, 'flat'), cy)
                max_auc = _auroc(_fused(cf, cc, cz, 'max'), cy)
                n_fake = int(cy.sum())
                fake_act = float((1.0 / (1.0 + np.exp(-cf[cy == 1]))).mean())
                log_line(f'[zoom]   -- {sub:<28} n_fake={n_fake:<4} '
                         f'flat={flat_auc:.4f}  max={max_auc:.4f}  '
                         f'act(flat)={fake_act:.4f}')
                entry['subgroups'][sub] = {'n_fake': n_fake, 'flat_auroc': flat_auc,
                                           'max_auroc': max_auc,
                                           'mean_fake_activation_flat': fake_act}

        # Zoomed-only subset: the lens where fusion can actually act.  Small-n
        # and label-imbalanced — read the CI, not the point.
        zmask = zoomed.astype(bool)
        if zmask.sum() >= 10 and len(set(labels[zmask])) == 2:
            zf, zc, zz, zy = full[zmask], crop[zmask], zoomed[zmask], labels[zmask]
            log_line(f'[zoom]   -- zoomed-only subset: n={int(zmask.sum())} '
                     f'({int(zy.sum())} fake / {int((zy == 0).sum())} real)')
            entry['zoomed_only'] = {'n': int(zmask.sum()), 'rules': {}}
            for rule in FUSION_RULES:
                auc = _auroc(_fused(zf, zc, zz, rule), zy)
                log_line(f'[zoom]      {rule:<6} auroc={auc:.4f}')
                entry['zoomed_only']['rules'][rule] = {'auroc': auc}
        summary[name] = entry

    if summary_out:
        Path(summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_out).write_text(json.dumps(summary, indent=2))
        log_line(f'[zoom] summary json -> {summary_out}')


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    if args.from_csv:
        csv_path = Path(args.from_csv).resolve()
        install_log(str(csv_path.with_suffix('.analysis.log')))
        analyze(csv_path, n_bootstrap=args.n_bootstrap,
                seed=args.seed, summary_out=args.summary_out)
        return

    if not args.checkpoint:
        raise SystemExit('--checkpoint is required unless --from_csv is given')
    csv_path = Path(args.out_csv or 'zoom_audit.csv').resolve()
    args.out_csv = str(csv_path)
    install_log(str(csv_path.with_suffix('.log')))
    out_csv = collect(args)
    analyze(out_csv, n_bootstrap=args.n_bootstrap, seed=args.seed,
            summary_out=args.summary_out)


if __name__ == '__main__':
    main()
