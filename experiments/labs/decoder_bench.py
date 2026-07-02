"""experiments.labs.decoder_bench — compare decoder strategies head-to-head.

Runs multiple decoders on the same cached ModelInfo objects and prints a
comparison table (via lab_utils.eval.aggregate.decoder_bench).

Typical usage::

    python -m experiments.labs.decoder_bench \\
        --cache_dir /tmp/eval_cache \\
        --items_pkl /tmp/val_items.pkl \\
        --decoder kmeans threshold hdbscan

Or call directly::

    from experiments.labs.decoder_bench import run_decoder_bench
    table = run_decoder_bench(infos, items, decoders=['kmeans', 'threshold'])
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from lab_utils.data.item import Item
from lab_utils.eval.aggregate import decoder_bench
from lab_utils.eval.cache import iter_cache
from lab_utils.eval.decode.hdbscan import decode_hdbscan
from lab_utils.eval.decode.kmeans import decode_kmeans
from lab_utils.eval.decode.threshold import decode_threshold
from lab_utils.eval.fetch import ModelInfo
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.record import EvalRecord
from lab_utils.logging.text import log_line


_DECODERS = {
    'kmeans':    decode_kmeans,
    'threshold': decode_threshold,
    'hdbscan':   decode_hdbscan,
}


def run_decoder_bench(
    infos: Dict[str, ModelInfo],
    items: List[Item],
    *,
    decoders: Optional[List[str]] = None,
    metric: str = 'f1',
    log_tag: str = '[buckets]',
) -> Dict[str, List[EvalRecord]]:
    """Run all requested decoders over pre-fetched ModelInfo objects.

    Args:
        infos:    {item_id: ModelInfo} — pre-computed forward outputs.
        items:    List of Item objects with matching item_ids.
        decoders: Decoder names to compare (default: all available).
        metric:   Which EvalRecord field to bench.
        log_tag:  Log tag for comparison table output.

    Returns:
        {decoder_name: List[EvalRecord]}
    """
    if decoders is None:
        decoders = list(_DECODERS)

    item_by_id: Dict[str, Item] = {item.item_id: item for item in items}
    records_by_decoder: Dict[str, List[EvalRecord]] = {}

    for dec_name in decoders:
        fn = _DECODERS.get(dec_name)
        if fn is None:
            log_line(f'[buckets] WARN: unknown decoder {dec_name!r}, skipping')
            continue

        records: List[EvalRecord] = []
        for item_id, info in infos.items():
            item = item_by_id.get(item_id)
            if item is None:
                log_line(f'[buckets] WARN: no item for cache id={item_id}')
                continue
            try:
                patch_mask = fn(info)
                rec        = eval_metric(patch_mask, info, item, decoder=dec_name)
                records.append(rec)
            except Exception as exc:
                log_line(f'[buckets] WARN: {dec_name} failed for item={item_id}: {exc}')
        records_by_decoder[dec_name] = records
        log_line(f'{log_tag} {dec_name}: {len(records)} records')

    decoder_bench(records_by_decoder, metric=metric, log_tag=log_tag)
    return records_by_decoder


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog='decoder_bench',
        description='Compare decoder strategies over a pre-built cache.',
    )
    p.add_argument('--cache_dir',  required=True,
                   help='Directory of .npz ModelInfo cache files')
    p.add_argument('--items_pkl',  required=True,
                   help='Pickled List[Item] file (produced by eval.py --save_items)')
    p.add_argument('--decoder', nargs='+', default=list(_DECODERS),
                   choices=list(_DECODERS))
    p.add_argument('--metric', default='f1')
    args = p.parse_args()

    with open(args.items_pkl, 'rb') as fh:
        items: List[Item] = pickle.load(fh)

    log_line(f'[buckets] loading cache from {args.cache_dir}')
    infos: Dict[str, ModelInfo] = {}
    for item_id, info in iter_cache(args.cache_dir):
        infos[item_id] = info
    log_line(f'[buckets] loaded {len(infos)} cached ModelInfo objects')

    run_decoder_bench(infos, items, decoders=args.decoder, metric=args.metric)


if __name__ == '__main__':
    main()
