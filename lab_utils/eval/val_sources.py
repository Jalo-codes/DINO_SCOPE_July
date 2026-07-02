"""lab_utils.eval.val_sources — single source of truth for eval datasets + CLI.

Both eval scripts (and any future one) collect validation items from the same
set of registered sources.  Keeping the source→``--<x>_root`` flag mapping and
the collection loop here means adding a dataset is a ONE-line change in
``SOURCE_ROOT_ARGS`` (plus the registry), not an edit in every script.

No model, no GT — just dataset construction via the registry.  Scripts may not
import each other (C-script invariant), so this shared logic lives in lab_utils.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from lab_utils.data.datasets.registry import REGISTRY
from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line

# source name (must be a REGISTRY key) → argparse attribute / flag stem.
# `--<value>` is the CLI flag; getattr(args, <value>) is the configured root.
SOURCE_ROOT_ARGS: Dict[str, str] = {
    'imd2020':      'imd2020_root',
    'casia':        'casia_root',
    'indoor':       'indoor_root',
    'coco_inpaint': 'coco_inpaint_root',
    'sagid':        'sagid_root',
    'bfree':        'bfree_root',
    'anyedit':      'anyedit_root',
    'tgif2':        'tgif2_root',
    'cocoglide':    'cocoglide_root',
    'opensdi':      'opensdi_root',
    'sid_set':      'sid_set_root',
    'pico_banana':  'pico_banana_root',
}


def add_source_root_args(group) -> None:
    """Register one ``--<source>_root`` argument per known source on an argparse
    group, so every script exposes the same dataset flags."""
    for attr in SOURCE_ROOT_ARGS.values():
        group.add_argument(f'--{attr}', default=None)
    group.add_argument('--tgif_eval_per_cell', type=int, default=None,
                       help='Limit/cap validation items to this many per cell in TGIF2')
    group.add_argument('--sagid_val_split', type=float, default=None,
                       help='Override SAGI-D val_split fraction (use 1.0 to eval the '
                            'entire sagid_root, e.g. a purpose-built clean val set)')
    group.add_argument('--imd_val_split', type=float, default=None,
                       help='Override IMD2020 val_split fraction (use 1.0 to eval the '
                            'entire imd2020_root — IMD is never trained on, so the '
                            'train/val distinction is moot)')


def collect_val_items_by_source(
    args,
    res: Resolution,
    *,
    log_tag: str = '[eval]',
) -> Dict[str, List]:
    """Build val items for every configured source → {source: [Item]}.

    Reads ``args.<source>_root`` (skips unset / missing dirs), honours optional
    ``args.sources`` (restrict set) and ``args.max_items`` (per-source cap).
    Sources with no configured root are silently skipped; a configured-but-
    missing dir logs a WARN and is skipped.
    """
    restrict = getattr(args, 'sources', None)
    max_items = getattr(args, 'max_items', None)

    by_source: Dict[str, List] = {}
    for source, attr in SOURCE_ROOT_ARGS.items():
        if restrict and source not in restrict:
            continue
        root_str = getattr(args, attr, None)
        if not root_str:
            continue
        root = Path(root_str)
        if not root.exists():
            log_line(f'{log_tag} WARN: root not found for {source}: {root}')
            continue
        
        kwargs = {}
        if source == 'tgif2':
            if getattr(args, 'tgif_eval_per_cell', None) is not None:
                kwargs['eval_per_cell'] = args.tgif_eval_per_cell
        if source == 'sagid':
            if getattr(args, 'sagid_val_split', None) is not None:
                kwargs['val_split'] = args.sagid_val_split
        if source == 'imd2020':
            if getattr(args, 'imd_val_split', None) is not None:
                kwargs['val_split'] = args.imd_val_split
        _, val_ds = REGISTRY[source](root, res=res, **kwargs)
        items = val_ds.items
        if max_items:
            items = items[:max_items]
        by_source[source] = items
        log_line(f'{log_tag} {source}: {len(items)} val items')
    return by_source


def collect_val_items(args, res: Resolution, *, log_tag: str = '[eval]') -> List:
    """Flat list of val items across all configured sources (order = SOURCE_ROOT_ARGS)."""
    by_source = collect_val_items_by_source(args, res, log_tag=log_tag)
    return [it for items in by_source.values() for it in items]
