"""lab_utils.data.datasets.region_probes — eval-only region-probe conditions.

Builders for the BCE-emergence probe conditions. Each condition wraps a PARENT
dataset's **val split** (split hygiene: the study's models train on the train
splits of the same sources) and emits Items that reference the parent's files
plus a fractional ``meta['crop_window']`` — no images are exported; the crop
is applied at load time (dataset.py / eval preprocess / eval metric all honor
the window via crop_conditions.apply_crop_window).

Conditions (registry key = Item.source = condition name):

    ai_interior   parent fake, window INSIDE the eroded mask   → AI content, no boundary
    ai_boundary   parent fake, window straddling the boundary  → AI + real, boundary present
    sp_interior   same, over a real-content splice parent (casia)
    sp_boundary   same
    fr_bg         fr parent, window fully OUTSIDE the mask     → regen-faithful, no edit (emitted as real)
    real_crop     the SAME interior windows applied to the paired pristine
                  original                                     → matched-pairs null

Pairing: real_crop re-derives the parent item's 'interior' window group with
the same deterministic RNG, so fake crop and real crop share identical
geometry; both carry the same ``meta['pair_stem']`` for matched analysis.

Windows / margins / floors / determinism all live in
lab_utils/data/crop_conditions.py (WINDOW_SPEC) — nothing geometric is decided
here.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from lab_utils.data.crop_conditions import (
    WINDOW_SPEC,
    ProbeWindow,
    sample_boundary_windows,
    sample_interior_windows,
    sample_outside_windows,
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
    'ai_interior': ('interior', True,  'modified'),
    'ai_boundary': ('boundary', True,  'modified'),
    'sp_interior': ('interior', True,  'modified'),
    'sp_boundary': ('boundary', True,  'modified'),
    'fr_bg':       ('outside',  False, 'modified'),
    'real_crop':   ('interior', False, 'original'),
}


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
    **parent_kwargs,
) -> Tuple[Dataset, Dataset]:
    """Build one probe condition over a parent dataset's val split.

    Args:
        root:             The PARENT dataset's root (probe flags point here).
        condition:        One of the _CONDITIONS keys.
        parent:           Parent registry source ('sagid', 'coco_inpaint',
                          'casia', ... — must yield fakes with GT masks; for
                          real_crop the parent must carry originals).
        max_parents:      Deterministic cap on parent fake items (eval-size
                          control; windows_per_item probes emitted per parent).
                          Default is comfortably above every current parent's
                          val-split size (imd2020 val_split=1.0 ~2424,
                          tgif2 ~9548, sagid=169) so it's effectively "use
                          everything" — the floor/erosion gate in
                          crop_conditions.py, not this cap, is what should be
                          limiting final n. Affordable because
                          sample_interior_windows now rejects the (very
                          common) too-small-mask case via a cheap raw-mask
                          bounding-box pre-filter before paying for erosion +
                          the inscribed-rectangle search — searching the
                          full tgif2 pool costs seconds, not minutes.
        windows_per_item: Override WINDOW_SPEC.windows_per_item.
        **parent_kwargs:  Forwarded to the parent builder (e.g. val_split).
    """
    if condition not in _CONDITIONS:
        raise ConfigError(
            f'region_probes.build: unknown condition {condition!r}; '
            f'known: {sorted(_CONDITIONS)}'
        )
    group, emit_mask, image_side = _CONDITIONS[condition]
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
        fakes, max_parents, seed=f'region_probes|{WINDOW_SPEC.version}|{parent}'
    )

    items: List[Item] = []
    n_gated = 0
    n_unpaired = 0
    for parent_item in fakes:
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
        windows: List[ProbeWindow] = sampler(
            mask, res, item_id=parent_item.item_id, k=windows_per_item,
        )
        if not windows:
            n_gated += 1
            continue

        for win in windows:
            meta = win.meta()
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
        f'{len(fakes)} parents (gated={n_gated}'
        + (f', unpaired={n_unpaired}' if image_side == 'original' else '')
        + f', spec={WINDOW_SPEC.version})'
    )

    train_ds = Dataset([], res=res, augment=True)
    val_ds   = Dataset(items, res=res, augment=False)
    return train_ds, val_ds
