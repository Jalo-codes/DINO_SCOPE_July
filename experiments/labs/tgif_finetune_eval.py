"""experiments.labs.tgif_finetune_eval — eval readout for the TGIF finetune recipe.

This lab is the per-epoch (and final) eval surface used by
``experiments/scripts/train_tgif.py``.  It is exceptional to the standard eval
flow on purpose: the finetune recipe trains on TGIF, so we want to watch BOTH
the OOD generalization target (IMD) and the in-domain TGIF held-out set, the
latter broken into its (model|type|family) subcategory cells.

Four readouts per surface — ``{kmeans, hdbscan} × {flat, attention-zoom}``:
  * flat  — one ``model_info`` forward → §2.2 decode → ``metric`` (shared forward
            across both decoders, so the flat pass is computed once per item).
  * zoom  — ``attention_zoom_single`` (the GT-free two-pass strategy, §2.3) with
            the named decoder.

Design compliance:
  * GT is still touched ONLY in ``metric`` — the subcategory partition is pure
    aggregation over the GT-free ``EvalRecord.subgroup`` label (set from
    ``Item.meta['tgif_subcat']``).
  * The model is reached only through ``model_info`` / ``attention_zoom_single``
    (which itself fetches via ``model_info``).
  * No oracle, anywhere.

It lives in ``labs/`` (not ``lab_utils``) because it composes the labs-level
attention-zoom strategy with the lab_utils eval pipeline — the same reason
``experiments/scripts/eval.py`` imports ``attention_zoom``.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution
from lab_utils.data.sampling import deterministic_subsample
from lab_utils.eval.aggregate import summarize, summarize_by_subgroup
from lab_utils.eval.decode.kmeans import decode_kmeans
from lab_utils.eval.fetch import model_info
from lab_utils.eval.metric import metric as eval_metric
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.eval.record import EvalRecord
from lab_utils.logging.text import log_line
from lab_utils.train.distributed import unwrap_model

from experiments.labs.attention_zoom import attention_zoom_single

SUBGROUP_KEY = 'tgif_subcat'


def _flat_decoders(names: Sequence[str]) -> Dict[str, callable]:
    """Resolve flat-decode functions by name (hdbscan imported lazily)."""
    out: Dict[str, callable] = {}
    for n in names:
        if n == 'kmeans':
            out[n] = decode_kmeans
        elif n == 'hdbscan':
            from lab_utils.eval.decode.hdbscan import decode_hdbscan
            out[n] = decode_hdbscan
        elif n == 'threshold':
            from lab_utils.eval.decode.threshold import decode_threshold
            out[n] = decode_threshold
        else:
            raise ValueError(f'tgif_finetune_eval: unknown decoder {n!r}')
    return out


def _cap_per_cell(items: List[Item], cap: Optional[int]) -> List[Item]:
    """Per-epoch compute cap: ``cap`` splices per subgroup cell; reals kept full.

    Deterministic via per-cell seeded subsample.  ``cap=None`` → no capping.
    """
    if cap is None:
        return items
    splices = [it for it in items if not it.is_real]
    reals   = [it for it in items if it.is_real]
    by_cell: Dict[str, List[Item]] = defaultdict(list)
    for it in splices:
        by_cell[str(it.meta.get(SUBGROUP_KEY, 'none'))].append(it)
    kept: List[Item] = []
    for cell in sorted(by_cell):
        kept.extend(deterministic_subsample(by_cell[cell], cap, seed=f'ft_eval:{cell}'))
    return kept + reals


@torch.no_grad()
def _flat_records(
    model: torch.nn.Module,
    items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    use_amp: bool,
    amp_dtype: str,
    decoders: Sequence[str],
    log_tag: str,
    phase: str = '',
) -> Dict[str, List[EvalRecord]]:
    """{f'{decoder}_flat': [records]} — one shared forward per item, N decodes."""
    fns = _flat_decoders(decoders)
    out: Dict[str, List[EvalRecord]] = {f'{d}_flat': [] for d in decoders}
    fail: Dict[str, int] = {d: 0 for d in decoders}

    n_total = len(items)
    every = max(1, n_total // 10)
    for i, item in enumerate(items):
        sub = item.meta.get(SUBGROUP_KEY)
        try:
            img_t = load_image_tensor(item, res, device=device)
            info  = model_info(model, img_t, device=device, amp=use_amp, amp_dtype=amp_dtype)
        except Exception as exc:
            log_line(f'{log_tag} WARN: fetch failed item={item.item_id}: {exc}')
            continue
        for d in decoders:
            try:
                mask = fns[d](info)
                rec  = eval_metric(mask, info, item, decoder=f'{d}_flat', subgroup=sub)
                out[f'{d}_flat'].append(rec)
            except Exception:
                fail[d] += 1
        if (i + 1) % every == 0 or (i + 1) == n_total:
            log_line(f'{log_tag} {phase} flat {i + 1}/{n_total}')

    for d, n in fail.items():
        if n:
            log_line(f'{log_tag} WARN: {d}_flat decode failed on {n} item(s)')
    return out


@torch.no_grad()
def _zoom_records(
    model: torch.nn.Module,
    items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    use_amp: bool,
    amp_dtype: str,
    decoders: Sequence[str],
    log_tag: str,
    phase: str = '',
) -> Dict[str, List[EvalRecord]]:
    """{f'{decoder}_zoom': [records]} via attention_zoom_single (two-pass, GT-free)."""
    out: Dict[str, List[EvalRecord]] = {f'{d}_zoom': [] for d in decoders}
    n_total = len(items)
    every = max(1, n_total // 20)
    for i, item in enumerate(items):
        sub = item.meta.get(SUBGROUP_KEY)
        for d in decoders:
            try:
                rec = attention_zoom_single(
                    model, item, res,
                    device=device, use_amp=use_amp, amp_dtype=amp_dtype, decoder=d,
                )
                out[f'{d}_zoom'].append(dataclasses.replace(rec, subgroup=sub))
            except Exception as exc:
                log_line(f'{log_tag} WARN: {d}_zoom failed item={item.item_id}: {exc}')
        if (i + 1) % every == 0 or (i + 1) == n_total:
            log_line(f'{log_tag} {phase} zoom {i + 1}/{n_total}')
    return out


def _median_splice_f1(records: List[EvalRecord]) -> float:
    splices = [r.f1 for r in records if not r.is_real]
    return float(np.median(splices)) if splices else float('nan')


@torch.no_grad()
def run_tgif_finetune_eval(
    model: torch.nn.Module,
    imd_items: List[Item],
    tgif_items: List[Item],
    res: Resolution,
    *,
    device: torch.device,
    use_amp: bool = False,
    amp_dtype: str = 'float16',
    decoders: Sequence[str] = ('kmeans', 'hdbscan'),
    val_per_cell: Optional[int] = None,
    imd_max_items: Optional[int] = None,
    primary_decoder: str = 'kmeans',
    primary_mode: str = 'zoom',
    primary_surface: str = 'imd',
    log_tag: str = '[ft-eval]',
) -> float:
    """Print the 4-readout IMD + TGIF-by-subcategory eval; return the primary metric.

    IMD is the OOD generalization surface (``summarize`` — overall + buckets).
    TGIF held-out is the in-domain surface (``summarize_by_subgroup`` — the 12
    (model|type|family) cells).  Each surface is reported under all four
    ``decoder×mode`` labels.

    The returned scalar is the median splice F1 of IMD under
    ``{primary_decoder}_{primary_mode}`` (default kmeans-zoom) — the train loop
    uses it to drive best.pt / early-stop.

    Args:
        imd_items:     IMD val items (primary OOD metric; subgroup ignored).
        tgif_items:    TGIF held-out items carrying meta['tgif_subcat'].
        decoders:      Decoders to run flat AND zoom (default kmeans + hdbscan).
        val_per_cell:  Cap TGIF splices per subgroup cell this epoch (compute).
        imd_max_items: Cap IMD items this epoch (compute).
    """
    bare = unwrap_model(model)
    bare.eval()

    imd_eval  = imd_items[:imd_max_items] if imd_max_items else list(imd_items)
    tgif_eval = _cap_per_cell(list(tgif_items), val_per_cell)

    log_line(
        f'{log_tag} eval start: IMD n={len(imd_eval)} TGIF n={len(tgif_eval)} '
        f'decoders={list(decoders)} (flat+zoom) val_per_cell={val_per_cell}'
    )

    common = dict(device=device, use_amp=use_amp, amp_dtype=amp_dtype,
                  decoders=decoders, log_tag=log_tag)

    # ── IMD (OOD generalization — overall + bucket summary) ────────────────────
    imd_records = {}
    imd_records.update(_flat_records(bare, imd_eval, res, phase='IMD', **common))
    imd_records.update(_zoom_records(bare, imd_eval, res, phase='IMD', **common))
    log_line(f'{log_tag} ═══ IMD (OOD) ═══')
    for label in sorted(imd_records):
        summarize(imd_records[label], log_tag=log_tag, tag=f'IMD/{label}')

    # ── TGIF held-out (in-domain — per-subcategory breakdown) ──────────────────
    tgif_records = {}
    tgif_records.update(_flat_records(bare, tgif_eval, res, phase='TGIF', **common))
    tgif_records.update(_zoom_records(bare, tgif_eval, res, phase='TGIF', **common))
    log_line(f'{log_tag} ═══ TGIF held-out (by subcategory) ═══')
    for label in sorted(tgif_records):
        summarize_by_subgroup(tgif_records[label], log_tag=log_tag, tag=f'TGIF/{label}')

    # ── Primary metric (drives best.pt / early-stop) ───────────────────────────
    primary_label = f'{primary_decoder}_{primary_mode}'
    src_records = tgif_records if primary_surface == 'tgif' else imd_records
    primary = _median_splice_f1(src_records.get(primary_label, []))
    log_line(f'{log_tag} primary = {primary_surface.upper()} {primary_label} median splice F1 = {primary:.4f}')
    return primary
