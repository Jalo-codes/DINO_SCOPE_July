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



REGISTRY: Dict[str, Callable] = {
    'imd2020':      _imd2020.build,
    'casia':        _casia.build,
    'coco_inpaint': lambda root, **kw: _inpaint.build(root, source='coco_inpaint', **kw),
    'sagid':        lambda root, **kw: _inpaint.build(root, source='sagid', **kw),
    'pico_pseudo':  lambda root, **kw: _inpaint.build(root, source='pico_pseudo', **kw),
    'anyedit':      _anyedit.build,
    'bfree':        _bfree.build,
    'indoor':       _indoor.build,
    'tgif2':        _tgif2.build,
    'cocoglide':    lambda root, **kw: _unpaired.build(root, source='cocoglide', **kw),
    'opensdi':      _opensdi.build,
    'sid_set':      lambda root, **kw: _unpaired.build(root, source='sid_set', **kw),
    'pico_banana':  _pico_banana.build,
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
