"""experiments.labs.probe_manifest — audit the region-probe conditions.

No model, no GT scoring — builds the probe conditions exactly as eval will
(same builders, same deterministic windows) and:

  1. dumps a manifest CSV — one row per probe item with its window geometry,
     upsample factor, and pair_stem.  This CSV is the JOIN TABLE for
     probe_contrasts.py (records CSVs carry item_id; the manifest carries
     everything else), and the audit trail that replaces an on-disk export.
  2. optionally renders N random crops per condition (image + cropped GT mask
     where the condition has one) for eyeballing BEFORE any training run.

Usage (box)::

    $PY -m experiments.labs.probe_manifest \
        --ai_interior_root $SAGID --ai_boundary_root $SAGID --real_crop_root $SAGID \
        --sp_interior_root $CASIA --sp_boundary_root $CASIA \
        --fr_bg_root $SAGID_FR_CLEAN \
        --image_size 448 --patch_size 16 \
        --out_csv runs/probe_manifest.csv --render_dir runs/probe_render --render_n 12
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image

from lab_utils.data.crop_conditions import WINDOW_SPEC, apply_crop_window
from lab_utils.data.resolution import Resolution
from lab_utils.eval.val_sources import add_source_root_args, collect_val_items_by_source
from lab_utils.logging.text import log_line

PROBE_SOURCES = ('ai_interior', 'ai_boundary', 'sp_interior', 'sp_boundary',
                 'fr_bg', 'real_crop',
                 'ai_interior_tgif', 'ai_boundary_tgif', 'real_crop_tgif')

_CSV_COLS = ['item_id', 'source', 'parent_item_id', 'parent_source', 'pair_stem',
             'case_id', 'win_y0', 'win_x0', 'win_y1', 'win_x1',
             'native_w', 'native_h', 'upsample_factor',
             'window_group', 'window_index', 'window_spec', 'has_mask', 'image']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group('data')
    add_source_root_args(g)
    g.add_argument('--sources', nargs='*', default=list(PROBE_SOURCES),
                   help='Conditions to audit (default: all six probe conditions).')
    g.add_argument('--max_items', type=int, default=None)
    p.add_argument('--image_size', type=int, default=448)
    p.add_argument('--patch_size', type=int, default=16)
    p.add_argument('--out_csv',    default='probe_manifest.csv')
    p.add_argument('--render_dir', default=None,
                   help='If set, save render_n random (crop, mask) pairs per condition here.')
    p.add_argument('--render_n',   type=int, default=12)
    p.add_argument('--render_seed', type=int, default=0)
    return p.parse_args()


def _render(items, out_dir: Path, n: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    picks = items if len(items) <= n else rng.sample(items, n)
    for it in picks:
        window = it.meta['crop_window']
        stem = it.item_id[:12]
        crop = apply_crop_window(Image.open(it.image).convert('RGB'), window)
        crop.save(out_dir / f'{stem}_img.png')
        if it.mask is not None:
            m = Image.open(it.mask).convert('L')
            img_size = Image.open(it.image).size
            if m.size != img_size:
                m = m.resize(img_size, Image.NEAREST)
            apply_crop_window(m, window).save(out_dir / f'{stem}_mask.png')


def main():
    args = parse_args()
    res = Resolution(image_size=args.image_size, patch_size=args.patch_size)

    by_source = collect_val_items_by_source(args, res, log_tag='[probe]')
    by_source = {s: v for s, v in by_source.items() if s in set(args.sources)}
    if not by_source:
        log_line('[probe] no probe sources configured — pass --<condition>_root flags')
        return

    out_csv = Path(args.out_csv)
    if out_csv.parent != Path(''):
        out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(_CSV_COLS)
        for source, items in by_source.items():
            for it in items:
                m = it.meta
                y0, x0, y1, x1 = m['crop_window']
                nw, nh = m['window_native_wh']
                w.writerow([
                    # it.source (the condition, e.g. 'ai_interior'), NOT the
                    # registry key -- a second parent pool for the same
                    # condition (e.g. 'ai_interior_tgif') must land under the
                    # same label here so it matches how eval.py's records CSV
                    # groups scores (by EvalRecord.source == Item.source).
                    it.item_id, it.source, m.get('parent_item_id', ''),
                    m.get('parent_source', ''), m.get('pair_stem', ''),
                    m.get('case_id') or '',
                    f'{y0:.6f}', f'{x0:.6f}', f'{y1:.6f}', f'{x1:.6f}',
                    nw, nh, f"{m['upsample_factor']:.4f}",
                    m.get('window_group', ''), m.get('window_index', ''),
                    m.get('window_spec', WINDOW_SPEC.version),
                    int(it.mask is not None), str(it.image),
                ])
                n_rows += 1
    log_line(f'[probe] wrote {n_rows} rows -> {out_csv} (spec={WINDOW_SPEC.version})')

    # Pairing sanity: every real_crop pair_stem should have an ai_interior
    # twin. Union across ALL configured sources (item.source, not registry
    # key) since ai_interior/real_crop may each be fed by more than one
    # parent pool (e.g. sagid + the tgif-sourced variants).
    all_items = [it for items in by_source.values() for it in items]
    ai = {it.meta.get('pair_stem') for it in all_items if it.source == 'ai_interior'}
    rc = {it.meta.get('pair_stem') for it in all_items if it.source == 'real_crop'}
    if ai or rc:
        log_line(f'[probe] pairing: ai_interior={len(ai)} real_crop={len(rc)} '
                 f'matched={len(ai & rc)} (ai-only={len(ai - rc)}, rc-only={len(rc - ai)})')

    if args.render_dir:
        for source, items in by_source.items():
            _render(items, Path(args.render_dir) / source, args.render_n, args.render_seed)
        log_line(f'[probe] rendered up to {args.render_n} crops/condition -> {args.render_dir}')


if __name__ == '__main__':
    main()
