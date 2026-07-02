"""experiments.scripts.bench_resolution — wall-clock inference benchmark.

Times the forward pass of one or more checkpoints over a FIXED set of images so
you can plot how inference latency scales with model resolution (token count).
It is the timing-only twin of ``experiments.scripts.eval``: it reuses the exact
same model loader (``load_eval_model``), the exact same image→tensor path
(``load_image_tensor``), and the ONLY forward call site (``model_info``).  It
reads no GT and computes no metrics — just stopwatch numbers.

Because each resolution-sweep checkpoint bakes its own ``image_size`` into the
saved cfg, ``load_eval_model`` recovers it for us; the *resolution is the size
axis*.  TGIF per-cell sampling is seeded, so ``--tgif_eval_per_cell 25`` selects
the same images for every checkpoint → an apples-to-apples curve.

Usage::

    # Auto-discover every res_*/best.pt under a run root and benchmark each:
    python -m experiments.scripts.bench_resolution \\
        --run_root /media/ssd/runs/ablation/res_sweep \\
        --tgif2_root /media/ssd/DINO_SCOPE_DATA/content/flux_originals \\
        --tgif_eval_per_cell 25 \\
        --out_csv results/bench_resolution.csv --plot

    # Or name checkpoints explicitly:
    python -m experiments.scripts.bench_resolution \\
        --checkpoints a/best.pt b/best.pt --tgif2_root ... --out_csv ...

    # Re-draw the graph from an existing CSV (no GPU needed):
    python -m experiments.scripts.bench_resolution \\
        --plot_only --out_csv results/bench_resolution.csv
"""

from __future__ import annotations

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from lab_utils.logging.text import log_line


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='bench_resolution',
        description='Wall-clock forward-pass benchmark across checkpoints.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group('checkpoints (one of --run_root / --checkpoints)')
    g.add_argument('--run_root', default=None,
                   help='Directory whose immediate children each hold a best.pt '
                        '(e.g. .../res_sweep with res_224/best.pt, res_336/best.pt, ...). '
                        'Cells are sorted by their checkpoint image_size.')
    g.add_argument('--ckpt_name', default='best.pt',
                   help='Checkpoint filename to look for inside each run_root child.')
    g.add_argument('--checkpoints', nargs='+', default=None,
                   help='Explicit list of .pt files to benchmark (overrides --run_root).')

    g = p.add_argument_group('dataset roots (TGIF by default)')
    from lab_utils.eval.val_sources import add_source_root_args
    add_source_root_args(g)
    g.add_argument('--sources', nargs='*', default=['tgif2'],
                   help='Source names to draw the fixed image set from.')
    g.add_argument('--max_items', type=int, default=None,
                   help='Hard cap on items per source (after per-cell capping).')

    g = p.add_argument_group('timing control')
    g.add_argument('--warmup', type=int, default=10,
                   help='Untimed forward passes before measurement (kernel autotune).')
    g.add_argument('--repeats', type=int, default=1,
                   help='Number of timed passes over the whole image set; reported '
                        'stats pool all repeats.')
    g.add_argument('--zoom', action='store_true',
                   help='Measure the real two-pass attention-zoom latency end-to-end '
                        'instead of the single backbone forward.')

    g = p.add_argument_group('hardware')
    g.add_argument('--device', default='cuda', choices=['cuda', 'cpu', 'mps'])
    g.add_argument('--no_amp', action='store_true')
    g.add_argument('--amp_dtype', default='float16', choices=['float16', 'bfloat16'])
    g.add_argument('--compile', action='store_true',
                   help='torch.compile each model before timing (compile cost is '
                        'absorbed by warmup).')

    g = p.add_argument_group('output')
    g.add_argument('--out_csv', default='results/bench_resolution.csv',
                   help='Per-checkpoint summary CSV.')
    g.add_argument('--out_json', default=None,
                   help='Optional JSON with full per-image raw timings.')
    g.add_argument('--plot', action='store_true',
                   help='Write a PNG graph next to --out_csv.')
    g.add_argument('--plot_only', action='store_true',
                   help='Skip benchmarking; just (re)draw the graph from --out_csv.')
    return p


# ── checkpoint discovery ──────────────────────────────────────────────────────

def _discover_checkpoints(args) -> List[str]:
    if args.checkpoints:
        return list(args.checkpoints)
    if not args.run_root:
        raise SystemExit('bench_resolution: pass --run_root or --checkpoints')
    root = Path(args.run_root)
    found = sorted(root.glob(f'*/{args.ckpt_name}'))
    if not found:
        raise SystemExit(f'bench_resolution: no */{args.ckpt_name} under {root}')
    return [str(p) for p in found]


# ── timing core ───────────────────────────────────────────────────────────────

def _sync(device) -> None:
    import torch
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _bench_one(checkpoint: str, args, device, use_amp) -> Optional[Dict]:
    """Load one checkpoint, build its image set, and time the forward pass."""
    import torch

    from lab_utils.eval.fetch import model_info
    from lab_utils.eval.load_model import load_eval_model
    from lab_utils.eval.preprocess import load_image_tensor
    from lab_utils.train.distributed import unwrap_model

    from lab_utils.eval.val_sources import collect_val_items_by_source

    log_line(f'[bench] loading checkpoint: {checkpoint}')
    model, cfg, res = load_eval_model(checkpoint, device=device, strict=False)
    if args.compile:
        model = torch.compile(model)
    bare_model = unwrap_model(model)

    # Same image set for every checkpoint (TGIF per-cell sampling is seeded).
    by_source = collect_val_items_by_source(args, res, log_tag='[bench]')
    items = [it for its in by_source.values() for it in its]
    if not items:
        log_line(f'[bench] WARN: no items for {checkpoint}; skipping')
        return None

    # Pre-load tensors to the device so we time compute, not disk/JPEG decode.
    log_line(f'[bench] preloading {len(items)} image tensors @ {res.image_size}px')
    tensors = []
    for it in items:
        try:
            tensors.append(load_image_tensor(it, res, device=device))
        except Exception as exc:
            log_line(f'[bench] WARN: skipped item={it.item_id}: {exc}')
    if not tensors:
        return None

    def _forward(img_t):
        with torch.no_grad():
            model_info(bare_model, img_t, device=device, amp=use_amp, amp_dtype=args.amp_dtype)

    # ── Warmup ────────────────────────────────────────────────────────────────
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    log_line(f'[bench] warmup x{args.warmup}')
    for i in range(args.warmup):
        if args.zoom:
            _zoom_forward(bare_model, items[i % len(items)], res, device, use_amp, args)
        else:
            _forward(tensors[i % len(tensors)])
    _sync(device)

    # ── Timed ─────────────────────────────────────────────────────────────────
    per_image_ms: List[float] = []
    log_line(f'[bench] timing {len(tensors)} imgs x{args.repeats} repeat(s)'
             f"{' [ZOOM 2-pass]' if args.zoom else ''}")
    for _ in range(args.repeats):
        if args.zoom:
            for it in items:
                _sync(device)
                t0 = time.perf_counter()
                _zoom_forward(bare_model, it, res, device, use_amp, args)
                _sync(device)
                per_image_ms.append((time.perf_counter() - t0) * 1e3)
        else:
            for img_t in tensors:
                _sync(device)
                t0 = time.perf_counter()
                with torch.no_grad():
                    model_info(bare_model, img_t, device=device,
                               amp=use_amp, amp_dtype=args.amp_dtype)
                _sync(device)
                per_image_ms.append((time.perf_counter() - t0) * 1e3)

    arr = np.asarray(per_image_ms, dtype=np.float64)
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
               if device.type == 'cuda' else float('nan'))
    dev_name = (torch.cuda.get_device_name(device) if device.type == 'cuda'
                else device.type)

    row = {
        'checkpoint':     checkpoint,
        'name':           Path(checkpoint).parent.name,
        'image_size':     res.image_size,
        'patch_size':     res.patch_size,
        'num_patches':    res.num_patches,
        'n_images':       len(tensors),
        'repeats':        args.repeats,
        'mode':           'zoom2pass' if args.zoom else 'forward',
        'mean_ms':        float(arr.mean()),
        'median_ms':      float(np.median(arr)),
        'p90_ms':         float(np.percentile(arr, 90)),
        'std_ms':         float(arr.std()),
        'throughput_ips': float(1000.0 / arr.mean()) if arr.mean() > 0 else float('nan'),
        'peak_mem_mb':    float(peak_mb),
        'amp_dtype':      'none' if not use_amp else args.amp_dtype,
        'device':         dev_name,
        '_raw_ms':        per_image_ms,
    }
    log_line(f"[bench] {row['name']}: {res.image_size}px / {res.num_patches} tok "
             f"→ {row['mean_ms']:.2f} ms/img (median {row['median_ms']:.2f}, "
             f"{row['throughput_ips']:.1f} img/s, peak {peak_mb:.0f} MB)")

    # Free before the next (larger) checkpoint loads.
    del model, bare_model, tensors
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return row


def _zoom_forward(bare_model, item, res, device, use_amp, args) -> None:
    from experiments.labs.attention_zoom import attention_zoom_single
    attention_zoom_single(
        bare_model, item, res,
        device=device, use_amp=use_amp, amp_dtype=args.amp_dtype,
        decoder='kmeans',
    )


# ── output ────────────────────────────────────────────────────────────────────

_CSV_FIELDS = [
    'name', 'image_size', 'patch_size', 'num_patches', 'n_images', 'repeats',
    'mode', 'mean_ms', 'median_ms', 'p90_ms', 'std_ms', 'throughput_ips',
    'peak_mem_mb', 'amp_dtype', 'device', 'checkpoint',
]


def _write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _CSV_FIELDS})
    log_line(f'[bench] wrote {len(rows)} rows → {path}')


def _read_csv(path: Path) -> List[Dict]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ('image_size', 'num_patches', 'n_images'):
            r[k] = int(float(r[k]))
        for k in ('mean_ms', 'median_ms', 'p90_ms', 'std_ms',
                  'throughput_ips', 'peak_mem_mb'):
            r[k] = float(r[k])
    return rows


def _plot(rows: List[Dict], png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log_line('[bench] WARN: matplotlib not available; skipping plot')
        return
    rows = sorted(rows, key=lambda r: r['image_size'])
    xs   = [r['image_size'] for r in rows]
    mean = [r['mean_ms'] for r in rows]
    p90  = [r['p90_ms'] for r in rows]
    med  = [r['median_ms'] for r in rows]
    err_lo = [m - md for m, md in zip(mean, med)]   # purely cosmetic spread band
    dev  = rows[0]['device'] if rows else ''
    mode = rows[0].get('mode', 'forward') if rows else 'forward'
    n    = rows[0]['n_images'] if rows else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, mean, 'o-', color='#2c6fbb', lw=2, label='mean ms/img')
    ax.plot(xs, p90, 's--', color='#bb4b2c', lw=1.2, ms=4, alpha=0.8, label='p90 ms/img')
    ax.fill_between(xs, med, p90, color='#2c6fbb', alpha=0.08)
    for x, y in zip(xs, mean):
        ax.annotate(f'{y:.0f}', (x, y), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color='#2c6fbb')
    ax.set_xlabel('input resolution (px / side)')
    ax.set_ylabel('wall-clock per image (ms)')
    ttl = 'Two-pass zoom latency' if mode == 'zoom2pass' else 'Single forward-pass latency'
    ax.set_title(f'{ttl} vs resolution\n{dev} · {n} TGIF imgs/model')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(xs)
    ax.legend()
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    log_line(f'[bench] wrote graph → {png_path}')


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys

    # Lightweight replot path: regenerate the graph from an existing CSV without
    # importing torch (handy on a laptop). Recognized before the full parser,
    # whose dataset-root args are torch-bound.
    if '--plot_only' in sys.argv:
        rp = argparse.ArgumentParser(prog='bench_resolution (plot_only)')
        rp.add_argument('--plot_only', action='store_true')
        rp.add_argument('--out_csv', default='results/bench_resolution.csv')
        rargs, _ = rp.parse_known_args()
        out_csv = Path(rargs.out_csv)
        _plot(_read_csv(out_csv), out_csv.with_suffix('.png'))
        return

    args = _build_parser().parse_args()
    out_csv = Path(args.out_csv)

    import torch
    device = torch.device(
        args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu'
    )
    use_amp = (not args.no_amp) and (device.type == 'cuda')
    log_line(f'[bench] device={device} amp={use_amp} dtype={args.amp_dtype}')

    checkpoints = _discover_checkpoints(args)
    log_line(f'[bench] {len(checkpoints)} checkpoint(s) to benchmark')

    rows: List[Dict] = []
    for ckpt in checkpoints:
        try:
            row = _bench_one(ckpt, args, device, use_amp)
            if row is not None:
                rows.append(row)
                _write_csv(rows, out_csv)   # checkpoint-after-each (resume-friendly)
        except Exception as exc:
            log_line(f'[bench] ERROR on {ckpt}: {exc}')

    if not rows:
        raise SystemExit('bench_resolution: no successful benchmarks')

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, 'w') as fh:
            json.dump(rows, fh, indent=2)
        log_line(f'[bench] wrote raw timings → {args.out_json}')

    if args.plot:
        _plot(rows, out_csv.with_suffix('.png'))

    log_line('[bench] done.')


if __name__ == '__main__':
    main()
