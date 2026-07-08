"""lab_utils.data.datasets.registry — source → build() mapping.

Usage::
    from lab_utils.data.datasets.registry import REGISTRY, build
    train_ds, val_ds = build('imd2020', root=paths.imd2020_root, res=res)
    train_ds, val_ds = build('casia',   root=paths.casia_root,   res=res)

Registry keys match Item.source strings.  Unknown keys raise ConfigError.

Each builder signature::
    def build(root, *, res, verify_policy=None, **kwargs) -> (Dataset, Dataset)

Extra kwargs are forwarded to the specific builder (e.g. source= for inpaint).
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from lab_utils.errors import ConfigError
from lab_utils.data.resolution import Resolution
from lab_utils.data.verify import VerifyPolicy
import lab_utils.data.datasets.imd2020  as _imd2020
import lab_utils.data.datasets.casia    as _casia
import lab_utils.data.datasets.inpaint  as _inpaint
import lab_utils.data.datasets.anyedit  as _anyedit
import lab_utils.data.datasets.bfree    as _bfree
import lab_utils.data.datasets.indoor   as _indoor
import lab_utils.data.datasets.tgif2    as _tgif2
import lab_utils.data.datasets.unpaired as _unpaired
import lab_utils.data.datasets.opensdi  as _opensdi
import lab_utils.data.datasets.pico_banana as _pico_banana
import lab_utils.data.datasets.pico_pseudo as _pico_pseudo
import lab_utils.data.datasets.region_probes as _region_probes



REGISTRY: Dict[str, Callable] = {
    'imd2020':      _imd2020.build,
    'casia':        _casia.build,
    'coco_inpaint': lambda root, **kw: _inpaint.build(root, source='coco_inpaint', **kw),
    'sagid':        lambda root, **kw: _inpaint.build(root, source='sagid', **kw),
    # Own builder, NOT an inpaint alias: full re-render source — no paste-back
    # (structural), v2 crop-baked-in format gate, per-pair alignment check.
    'pico_pseudo':  _pico_pseudo.build,
    'anyedit':      _anyedit.build,
    'bfree':        _bfree.build,
    'indoor':       _indoor.build,
    'tgif2':        _tgif2.build,
    'cocoglide':    lambda root, **kw: _unpaired.build(root, source='cocoglide', **kw),
    'opensdi':      _opensdi.build,
    'sid_set':      lambda root, **kw: _unpaired.build(root, source='sid_set', **kw),
    'pico_banana':  _pico_banana.build,
    # Region-probe eval conditions (BCE-emergence study) — eval-only builders
    # over a PARENT dataset's val split; the flag root is the PARENT root.
    # ai_* / real_crop wrap sagid (AI-edited content + its paired original);
    # sp_* wrap imd2020 (more val fakes than casia: ~171 vs ~28, better odds
    # of clearing the interior floor); fr_bg wraps tgif2 restricted to 'fr'
    # manipulations specifically (a held-out OOD fr pool, never sagid's own
    # frs — tgif2 is a pure eval probe with zero train usage anywhere in this
    # study, so there's no split-hygiene concern using its full val pool).
    # Windows/floors/determinism: lab_utils/data/crop_conditions.py.
    'ai_interior':  lambda root, **kw: _region_probes.build(root, condition='ai_interior', parent=kw.pop('parent', 'sagid'), **kw),
    'ai_boundary':  lambda root, **kw: _region_probes.build(root, condition='ai_boundary', parent=kw.pop('parent', 'sagid'), **kw),
    'sp_interior':  lambda root, **kw: _region_probes.build(root, condition='sp_interior', parent=kw.pop('parent', 'imd2020'), **kw),
    'sp_boundary':  lambda root, **kw: _region_probes.build(root, condition='sp_boundary', parent=kw.pop('parent', 'imd2020'), **kw),
    'fr_bg':        lambda root, **kw: _region_probes.build(root, condition='fr_bg',       parent=kw.pop('parent', 'tgif2'), types=kw.pop('types', {'fr'}), **kw),
    'real_crop':    lambda root, **kw: _region_probes.build(root, condition='real_crop',   parent=kw.pop('parent', 'sagid'), **kw),
    # Second, ADDITIONAL parent pool for the same three conditions: tgif2's
    # 'sp' manipulations (paste-back AI edits, NOT real-content splices —
    # unlike casia/imd2020, this belongs with ai_* not sp_*). sagid alone
    # (169 val fakes) starves ai_interior's floor gate; tgif2 has 341
    # coco_ids x up to 3 generator models of sp variants, a much bigger
    # haystack to find large-enough edits in. Items still carry
    # Item.source == the base condition name (e.g. 'ai_interior'), so running
    # both registry keys through eval.py/probe_manifest.py in the same
    # --sources list merges them into one pool automatically — no changes
    # needed downstream.
    'ai_interior_tgif': lambda root, **kw: _region_probes.build(root, condition='ai_interior', parent=kw.pop('parent', 'tgif2'), types=kw.pop('types', {'sp'}), **kw),
    'ai_boundary_tgif': lambda root, **kw: _region_probes.build(root, condition='ai_boundary', parent=kw.pop('parent', 'tgif2'), types=kw.pop('types', {'sp'}), **kw),
    'real_crop_tgif':   lambda root, **kw: _region_probes.build(root, condition='real_crop',   parent=kw.pop('parent', 'tgif2'), types=kw.pop('types', {'sp'}), **kw),
}


def build(
    source: str,
    root: Path,
    *,
    res: Resolution,
    verify_policy: Optional[VerifyPolicy] = None,
    **kwargs: Any,
) -> Tuple:
    """Build (train_dataset, val_dataset) for the named source.

    Args:
        source:        Dataset name — must be a key in REGISTRY.
        root:          Path to the dataset root on disk.
        res:           Resolution for the Datasets.
        verify_policy: Optional override of the default drop-and-log policy.
        **kwargs:      Forwarded to the source-specific build function.

    Raises:
        ConfigError: If source is not in the registry.
    """
    if source not in REGISTRY:
        raise ConfigError(
            f"datasets.registry.build: unknown source {source!r}. "
            f"Known sources: {sorted(REGISTRY)}"
        )
    return REGISTRY[source](root, res=res, verify_policy=verify_policy, **kwargs)
