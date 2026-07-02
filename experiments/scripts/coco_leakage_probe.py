"""experiments.scripts.coco_leakage_probe — COCO-provenance leakage audit.

Several sources draw their *background real* images from the same COCO corpus:

    coco_inpaint   (train)        COCO originals, inpainted
    bfree          (train)        COCO_real_512 anchors, SD2.1 inpainted
    tgif2          (eval/finetune) keyed by coco_id
    cocoglide      (eval-only)    COCO backgrounds, GLIDE edits
    opensdi        (eval-only)    SD15 (overlap unverified)

Because every builder splits its *own* cases independently and ``item_id`` is
``md5(source|path)``, the SAME COCO scene can be a training real/background AND
the background of an eval fake. This script quantifies that overlap at the
COCO-image-id level, separately for reals and for fakes (a leaked id taints
both the eval real negative and the untouched background of the eval fake).

It reuses the real registry builders, so the train/val partition it audits is
exactly the one training and eval consume. Image decoding is skipped
(``SKIP_VERIFY``) — only paths are inspected.

Usage (run where the data lives, e.g. Colab / the box with /data):

    python -m experiments.scripts.coco_leakage_probe \
        --coco_inpaint_root /data/coco_inpaint \
        --bfree_root        /data/bfree \
        --cocoglide_root    /data/CocoGlide \
        --opensdi_root      /data/OpenSDI \
        --tgif2_root        /data/tgif2_index.json   # optional

By default coco_inpaint+bfree are treated as TRAIN-contributing and
cocoglide+opensdi as EVAL-only; tgif2 is EVAL-only (matching train_frac=0.0).
Override with --train_sources / --eval_sources if your wiring differs.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from lab_utils.data.datasets.registry import REGISTRY
from lab_utils.data.resolution import Resolution
from lab_utils.data.verify import SKIP_VERIFY

# COCO-derived sources this probe knows how to wire up.
COCO_FAMILY = ('coco_inpaint', 'bfree', 'tgif2', 'cocoglide', 'opensdi')

DEFAULT_TRAIN = ('coco_inpaint', 'bfree')
DEFAULT_EVAL = ('cocoglide', 'opensdi', 'tgif2')

# tgif2 needs an index path rather than a dir root; the registry forwards kwargs.
_BUILD_KWARGS: Dict[str, dict] = {
    'tgif2': {'index_path': None},  # filled from the root arg below
}

# Prefer a 12-digit zero-padded run (canonical COCO), else the longest >=6-digit run.
_RE_12 = re.compile(r'(?<!\d)(\d{12})(?!\d)')
_RE_RUN = re.compile(r'\d{6,}')


def extract_coco_id(path: Optional[Path], min_digits: int) -> Optional[str]:
    """Best-effort COCO image id from a filename. Returns a leading-zero-stripped
    decimal string, or None when no plausible id is present."""
    if path is None:
        return None
    stem = Path(path).name
    m = _RE_12.search(stem)
    if m:
        return str(int(m.group(1)))
    runs = _RE_RUN.findall(stem)
    runs = [r for r in runs if len(r) >= min_digits]
    if not runs:
        return None
    # Longest digit run is the most id-like; ties broken by last occurrence.
    best = max(runs, key=len)
    return str(int(best))


def _id_for_item(item, min_digits: int) -> Optional[str]:
    """Id source: real → the image itself; fake → its background (authentic) when
    known, else the fake filename (which still carries the key for unpaired sets)."""
    src_path = item.image if item.is_real else (item.authentic or item.image)
    return extract_coco_id(src_path, min_digits)


def _collect_ids(items, min_digits: int) -> Tuple[Set[str], int, int]:
    """(unique ids, n_items, n_items_with_id) over an item list, by real/fake."""
    ids: Set[str] = set()
    n_with = 0
    for it in items:
        cid = _id_for_item(it, min_digits)
        if cid is not None:
            ids.add(cid)
            n_with += 1
    return ids, len(items), n_with


def build_source(source: str, root: str, res: Resolution, tgif2_train_frac: float = 0.0):
    """Return (train_items, val_items) via the real registry builder, no decode."""
    kwargs = dict(_BUILD_KWARGS.get(source, {}))
    if source == 'tgif2':
        # tgif2 root arg points at the index json; train_frac>0 to populate train side.
        kwargs['index_path'] = Path(root)
        kwargs['train_frac'] = tgif2_train_frac
        train_ds, val_ds = REGISTRY[source](
            Path(root).parent, res=res, verify_policy=SKIP_VERIFY, **kwargs)
    else:
        train_ds, val_ds = REGISTRY[source](
            Path(root), res=res, verify_policy=SKIP_VERIFY, **kwargs)
    return list(train_ds.items), list(val_ds.items)


def _split_real_fake(items):
    reals = [it for it in items if it.is_real]
    fakes = [it for it in items if not it.is_real]
    return reals, fakes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for s in COCO_FAMILY:
        p.add_argument(f'--{s}_root', default=None)
    p.add_argument('--train_sources', default=','.join(DEFAULT_TRAIN),
                   help='comma-separated sources whose TRAIN split forms the train pool')
    p.add_argument('--eval_sources', default=','.join(DEFAULT_EVAL),
                   help='comma-separated sources whose VAL split is audited for leakage')
    p.add_argument('--image_size', type=int, default=224)
    p.add_argument('--patch_size', type=int, default=14)
    p.add_argument('--min_digits', type=int, default=6,
                   help='shortest digit run accepted as a COCO id when no 12-digit run')
    p.add_argument('--tgif2_train_frac', type=float, default=0.0,
                   help='set >0 if you finetune on tgif2, so its train side is populated')
    args = p.parse_args()

    res = Resolution(image_size=args.image_size, patch_size=args.patch_size)
    train_sources = [s for s in args.train_sources.split(',') if s]
    eval_sources = [s for s in args.eval_sources.split(',') if s]

    # Build every configured source once; keep ids split by side and by real/fake.
    per_source: Dict[str, dict] = {}
    for s in COCO_FAMILY:
        root = getattr(args, f'{s}_root')
        if not root:
            continue
        if not Path(root).exists():
            print(f'[probe] WARN: {s} root not found: {root}')
            continue
        train_items, val_items = build_source(s, root, res, args.tgif2_train_frac)
        tr_real, tr_fake = _split_real_fake(train_items)
        va_real, va_fake = _split_real_fake(val_items)
        rec = {
            'train_real_ids': _collect_ids(tr_real, args.min_digits),
            'train_fake_ids': _collect_ids(tr_fake, args.min_digits),
            'val_real_ids':   _collect_ids(va_real, args.min_digits),
            'val_fake_ids':   _collect_ids(va_fake, args.min_digits),
        }
        per_source[s] = rec

    if not per_source:
        print('[probe] No sources configured. Pass at least two --<source>_root flags.')
        return

    # ── Extraction coverage (sanity-check the id parser found ids) ─────────────
    print('\n=== COCO-id extraction coverage (per source) ===')
    print(f'{"source":<14}{"side":<7}{"items":>8}{"with_id":>9}{"unique":>8}')
    for s, rec in per_source.items():
        for side in ('train_real_ids', 'train_fake_ids', 'val_real_ids', 'val_fake_ids'):
            ids, n, nw = rec[side]
            if n == 0:
                continue
            label = side.replace('_ids', '').replace('train_', 'tr/').replace('val_', 'va/')
            cov = f'{100.0 * nw / n:.0f}%' if n else '-'
            print(f'{s:<14}{label:<7}{n:>8}{nw:>9}{len(ids):>8}  ({cov} parsed)')

    # ── Train pool of COCO ids (reals + fake backgrounds) ──────────────────────
    train_pool: Set[str] = set()
    train_pool_by_src: Dict[str, Set[str]] = {}
    for s in train_sources:
        if s not in per_source:
            continue
        ids = per_source[s]['train_real_ids'][0] | per_source[s]['train_fake_ids'][0]
        train_pool_by_src[s] = ids
        train_pool |= ids
    print(f'\n=== TRAIN COCO-id pool: {len(train_pool)} unique ids '
          f'from {sorted(train_pool_by_src)} ===')

    # ── Leakage: eval ids that appear in the train pool, split real vs fake ────
    print('\n=== EVAL → TRAIN leakage (eval items sitting on a trained COCO id) ===')
    print(f'{"eval_src":<14}{"kind":<6}{"uniq_ids":>9}{"leaked_ids":>11}{"id_leak%":>10}')
    grand = defaultdict(int)
    for s in eval_sources:
        if s not in per_source:
            continue
        for kind, key in (('real', 'val_real_ids'), ('fake', 'val_fake_ids')):
            ids, n_items, _ = per_source[s][key]
            if not ids:
                continue
            leaked = ids & train_pool
            pct = 100.0 * len(leaked) / len(ids) if ids else 0.0
            print(f'{s:<14}{kind:<6}{len(ids):>9}{len(leaked):>11}{pct:>9.1f}%')
            grand[f'{kind}_ids'] += len(ids)
            grand[f'{kind}_leaked'] += len(leaked)

    # ── Cross-source val leakage among the COCO-family (own held-out vs others) ─
    print('\n=== Cross-source VAL leakage (a source\'s own val id seen in ANOTHER '
          'source\'s train) ===')
    for s, rec in per_source.items():
        own_val = rec['val_real_ids'][0] | rec['val_fake_ids'][0]
        if not own_val:
            continue
        others_train: Set[str] = set()
        for o, orec in per_source.items():
            if o == s:
                continue
            others_train |= orec['train_real_ids'][0] | orec['train_fake_ids'][0]
        leaked = own_val & others_train
        pct = 100.0 * len(leaked) / len(own_val) if own_val else 0.0
        flag = '  <-- LEAK' if leaked else ''
        print(f'  {s:<14} val_ids={len(own_val):>6}  in_other_train={len(leaked):>6}'
              f'  ({pct:.1f}%){flag}')

    # ── Headline ───────────────────────────────────────────────────────────────
    print('\n=== HEADLINE ===')
    for kind in ('real', 'fake'):
        ni, nl = grand[f'{kind}_ids'], grand[f'{kind}_leaked']
        if ni:
            print(f'  eval {kind:<4}: {nl}/{ni} COCO ids ({100.0*nl/ni:.1f}%) '
                  f'also appear in training')
    print('  (a leaked fake id means the eval fake\'s untouched background was '
          'seen at train time)')


if __name__ == '__main__':
    main()
