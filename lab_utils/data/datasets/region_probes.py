"""lab_utils.data.datasets.region_probes — eval-only region-probe conditions.

Builders for the BCE-emergence probe conditions. Each condition wraps a PARENT
dataset's **val split** (split hygiene: the study's models train on the train
splits of the same sources) and emits Items that reference the parent's files
plus a fractional ``meta['crop_window']`` — no images are exported; the crop
is applied at load time (dataset.py / eval preprocess / eval metric all honor
the window via crop_conditions.apply_crop_window).

Conditions (registry key = Item.source = condition name):

    ai_interior    parent fake, window INSIDE the eroded mask   → AI content, no boundary
    ai_boundary    parent fake, window straddling the boundary  → AI + real, boundary present
    sp_interior    same, over a real-content splice parent (casia)
    sp_boundary    same
    fr_bg_matched  fr parent, window fully OUTSIDE the mask, (h, w) drawn from
                   the re-derived ai_interior window-size pool  → regen-faithful,
                   no edit, size-matched null (emitted as real)
    real_crop      the SAME interior windows applied to the paired pristine
                   original                                     → matched-pairs null

Pairing: real_crop re-derives the parent item's 'interior' window group with
the same deterministic RNG, so fake crop and real crop share identical
geometry; both carry the same ``meta['pair_stem']`` for matched analysis.

fr_bg_matched replaces the retired 'fr_bg' condition. fr_bg drew window sides
from [floor, 1.6*floor] while interior sides scale with the mask's inscribed
rectangle — two different generating processes, so fr_bg ran ~1.3x larger, and
because image score falls with window size the mismatch inflated interior-vs-
fr_bg detection AUROC (the §5.1 size-artifact finding; post-hoc reweighting
cost effective N). fr_bg_matched instead draws (h, w) iid from the size pool
of an internally re-built ai_interior condition (``size_ref``, default: tgif2
'sp' parents — the n≈300 subset every corrected comparison restricts to), so
the null matches the interior size distribution in law, by construction, at
full N.

Windows / margins / floors / determinism all live in
lab_utils/data/crop_conditions.py (PROBE_WINDOW_SPEC — the eval-probe-only
spec; looser floor than the shared WINDOW_SPEC default used elsewhere, since
these conditions are never trained on) — nothing geometric is decided here.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from lab_utils.data.crop_conditions import (
    PROBE_WINDOW_SPEC,
    ProbeWindow,
    breadth_first_cap,
    sample_boundary_windows,
    sample_interior_windows,
    sample_outside_windows,
    sample_outside_windows_sized,
)
from lab_utils.data.dataset import Dataset
from lab_utils.data.item import Item, make_item_id
from lab_utils.data.resolution import Resolution
from lab_utils.data.sampling import deterministic_subsample
from lab_utils.data.verify import VerifyPolicy
from lab_utils.errors import ConfigError
from lab_utils.logging.text import log_line

_SAMPLERS = {
    'interior': sample_interior_windows,
    'boundary': sample_boundary_windows,
    'outside':  sample_outside_windows,
}

# condition → (window group, emit mask?, image side: 'modified' | 'original')
_CONDITIONS = {
    'ai_interior':   ('interior',        True,  'modified'),
    'ai_boundary':   ('boundary',        True,  'modified'),
    'sp_interior':   ('interior',        True,  'modified'),
    'sp_boundary':   ('boundary',        True,  'modified'),
    'fr_bg_matched': ('outside_matched', False, 'modified'),
    'real_crop':     ('interior',        False, 'original'),
}

# Default reference build for fr_bg_matched's window-size pool: the tgif2-'sp'
# ai_interior pool — the same parents/windows the eval's ai_interior_tgif
# registry entry emits, and the tgif↔tgif n≈300 subset all corrected interior
# comparisons restrict to (fr_bg_matched parents are tgif2 'fr', same root).
_DEFAULT_SIZE_REF: Dict = {'parent': 'tgif2', 'types': {'sp'}}


def _interior_size_pool(
    root: Path,
    *,
    res: Resolution,
    verify_policy: Optional[VerifyPolicy],
    max_parents: int,
    windows_per_item: Optional[int],
    max_probes: int,
    parent: str,
    **parent_kwargs,
) -> List[Tuple[int, int]]:
    """Native-pixel (h, w) pairs of the reference ai_interior condition.

    Re-runs the ai_interior build (deterministic — same RNG keying, same gate,
    same breadth-first cap) and harvests the emitted windows' sizes, so the
    pool IS the realized interior size distribution, not an approximation."""
    _, ref_val = build(
        root, res=res, condition='ai_interior', parent=parent,
        verify_policy=verify_policy, max_parents=max_parents,
        windows_per_item=windows_per_item, max_probes=max_probes,
        **parent_kwargs,
    )
    pool: List[Tuple[int, int]] = []
    for it in ref_val.items:
        w, h = it.meta['window_native_wh']            # meta stores (w, h)
        pool.append((int(h), int(w)))
    if not pool:
        raise ConfigError(
            'region_probes._interior_size_pool: reference ai_interior build '
            f'(parent={parent!r}, root={root}) emitted 0 windows — cannot '
            'size-match fr_bg_matched against an empty pool'
        )
    return pool


def _load_mask_in_image_frame(item: Item) -> Optional[np.ndarray]:
    """Parent GT mask as bool array in the IMAGE's native frame.

    Windows are fractional, so frame choice only matters for the native-pixel
    floor and erosion radius — those are about the image's real resolution,
    not the mask file's. Same-aspect size differences (verify's 'resizable'
    class) are NEAREST-resized to the image frame.
    """
    if item.mask is None:
        return None
    mask = Image.open(item.mask).convert('L')
    img_size = Image.open(item.image).size          # lazy: header only
    if mask.size != img_size:
        mask = mask.resize(img_size, Image.NEAREST)
    return np.asarray(mask) > 127


def _original_path(item: Item) -> Optional[Path]:
    p = item.meta.get('real_path') or item.authentic
    return Path(p) if p else None


def build(
    root: Path,
    *,
    res: Resolution,
    condition: str,
    parent: str,
    verify_policy: Optional[VerifyPolicy] = None,
    max_parents: int = 10000,
    windows_per_item: Optional[int] = None,
    max_probes: int = 300,
    size_ref: Optional[Dict] = None,
    **parent_kwargs,
) -> Tuple[Dataset, Dataset]:
    """Build one probe condition over a parent dataset's val split.

    Args:
        root:             The PARENT dataset's root (probe flags point here).
        condition:        One of the _CONDITIONS keys.
        parent:           Parent registry source ('sagid', 'coco_inpaint',
                          'casia', ... — must yield fakes with GT masks; for
                          real_crop the parent must carry originals).
        max_parents:      Deterministic cap on parent fake items considered
                          (search-breadth control, not an output-size control
                          — see max_probes for that). Default is comfortably
                          above every current parent's val-split size
                          (imd2020 val_split=1.0 ~2424, tgif2 ~9548,
                          sagid=169) so it's effectively "use everything" —
                          affordable because sample_interior_windows rejects
                          the (very common) too-small-mask case via a cheap
                          raw-mask bounding-box pre-filter before paying for
                          erosion + the inscribed-rectangle search.
        windows_per_item: Override PROBE_WINDOW_SPEC.windows_per_item.
        max_probes:       Hard cap on total emitted probe windows — AND on the
                          parent search itself: pass 1 stops gating further
                          parents as soon as max_probes have passed (round 0
                          of the breadth-first flatten below already fills
                          the cap from the first max_probes accepted groups,
                          so scanning more would be discarded work). Since
                          PROBE_WINDOW_SPEC's floor is much looser than the
                          strict default (see its docstring), most candidates
                          now pass — combined with max_parents="basically
                          all" and windows_per_item up to 10, an easy group
                          (boundary/outside) over a large parent pool (e.g.
                          fr_bg's ~8.7k accepted tgif2 parents) could
                          otherwise both emit tens of thousands of probes AND
                          waste time scanning parents whose windows are never
                          used. The flatten itself is breadth-first
                          (round-robin over windows_per_item): every passing
                          parent contributes one window before any parent
                          contributes a second, so hitting the cap thins
                          per-parent depth, never how many distinct source
                          images are represented. Default sits comfortably
                          above the study's ~200-crop-per-condition target.
        size_ref:         fr_bg_matched only — kwargs for the reference
                          ai_interior build whose realized window sizes form
                          the (h, w) draw pool (must include 'parent'; the
                          rest forwards to that parent's builder). Default
                          _DEFAULT_SIZE_REF = tgif2 'sp'. Ignored (with a
                          ConfigError) for other conditions.
        **parent_kwargs:  Forwarded to the parent builder (e.g. val_split).
    """
    if condition not in _CONDITIONS:
        raise ConfigError(
            f'region_probes.build: unknown condition {condition!r}; '
            f'known: {sorted(_CONDITIONS)}'
        )
    group, emit_mask, image_side = _CONDITIONS[condition]
    if group == 'outside_matched':
        ref = dict(_DEFAULT_SIZE_REF if size_ref is None else size_ref)
        size_pool = _interior_size_pool(
            root, res=res, verify_policy=verify_policy,
            max_parents=max_parents, windows_per_item=windows_per_item,
            max_probes=max_probes, parent=ref.pop('parent'), **ref,
        )
        log_line(
            f'[data] {condition}: size pool from ai_interior ref '
            f'(n={len(size_pool)}, median side='
            f'{int(np.median([min(hw) for hw in size_pool]))}px)'
        )
        sampler = functools.partial(sample_outside_windows_sized, size_pool=size_pool)
    elif size_ref is not None:
        raise ConfigError(
            f'region_probes.build: size_ref only applies to fr_bg_matched, '
            f'got condition {condition!r}'
        )
    else:
        sampler = _SAMPLERS[group]

    # Lazy import avoids a registry ↔ region_probes import cycle.
    from lab_utils.data.datasets.registry import REGISTRY
    if parent not in REGISTRY:
        raise ConfigError(f'region_probes.build: unknown parent source {parent!r}')

    _, parent_val = REGISTRY[parent](
        root, res=res, verify_policy=verify_policy, **parent_kwargs
    )
    fakes = [it for it in parent_val.items if not it.is_real]
    fakes = deterministic_subsample(
        fakes, max_parents, seed=f'region_probes|{PROBE_WINDOW_SPEC.version}|{parent}'
    )

    # Pass 1: gate parents and collect each one's full window list (cheap now
    # that sample_interior_windows bbox-pre-filters the too-small case).
    # Stops as soon as max_probes parents have PASSED the gate: breadth_first_
    # cap's round 0 alone already fills the cap from the first max_probes
    # accepted groups, so scanning further parents would be discarded work —
    # real box logs showed fr_bg scanning all ~8.7k accepted tgif2 parents to
    # produce a 300-probe output. Deterministic and identical between
    # real_crop and its paired condition (same fakes order, same per-parent
    # gate), so pairing survives stopping early.
    per_parent: List[Tuple[Item, Path, List[ProbeWindow]]] = []
    n_gated = 0
    n_unpaired = 0
    n_scanned = 0
    for parent_item in fakes:
        n_scanned += 1
        mask = _load_mask_in_image_frame(parent_item)
        if mask is None:
            continue

        if image_side == 'original':
            image_path = _original_path(parent_item)
            if image_path is None:
                n_unpaired += 1
                continue
        else:
            image_path = parent_item.image

        # Window RNG is keyed on the PARENT item_id + group, so real_crop
        # ('interior' group, original image) reproduces ai_interior's windows.
        # PROBE_WINDOW_SPEC (not the shared WINDOW_SPEC default) -- eval-only
        # probes tolerate more upsampling than the train-time fr-background
        # negative sampler, which stays on the strict default.
        windows: List[ProbeWindow] = sampler(
            mask, res, item_id=parent_item.item_id, k=windows_per_item,
            spec=PROBE_WINDOW_SPEC,
        )
        if not windows:
            n_gated += 1
            continue
        per_parent.append((parent_item, image_path, windows))
        if len(per_parent) >= max_probes:
            break

    # Pass 2: breadth-first flatten, capped at max_probes (see docstring) —
    # every passing parent contributes one window before any parent
    # contributes a second, so the cap thins per-parent depth, never parent
    # diversity. Same input order for real_crop vs its paired condition (both
    # share the same fakes/per_parent construction), so pairing is preserved.
    total_available = sum(len(w) for _, _, w in per_parent)
    stopped_early = n_scanned < len(fakes)
    groups = [
        [(parent_item, image_path, win) for win in windows]
        for parent_item, image_path, windows in per_parent
    ]
    items: List[Item] = []
    for parent_item, image_path, win in breadth_first_cap(groups, max_probes):
        meta = win.meta(spec=PROBE_WINDOW_SPEC)
        meta.update({
            'pair_stem':      f'{parent_item.item_id}|{win.index}',
            'parent_item_id': parent_item.item_id,
            'parent_source':  parent_item.source,
            'case_id':        parent_item.meta.get('case_id'),
        })
        items.append(Item(
            image=Path(image_path),
            authentic=None,
            mask=(parent_item.mask if emit_mask else None),
            source=condition,
            item_id=make_item_id(condition, f'{parent_item.image}|w{win.index}'),
            meta=meta,
        ))

    log_line(
        f'[data] {condition} ({parent} @ {root}): {len(items)} probes from '
        f'{n_scanned}/{len(fakes)} parents scanned (gated={n_gated}'
        + (f', unpaired={n_unpaired}' if image_side == 'original' else '')
        + f', available_in_scanned={total_available}, stopped_early={stopped_early}, '
        + f'spec={PROBE_WINDOW_SPEC.version})'
    )

    train_ds = Dataset([], res=res, augment=True)
    val_ds   = Dataset(items, res=res, augment=False)
    return train_ds, val_ds
