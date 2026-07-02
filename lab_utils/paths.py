"""Shared dataset and run-root path defaults for DINO_SCOPE experiments."""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Optional


@dataclasses.dataclass(frozen=True)
class DataPaths:
    """Resolved roots used by training and evaluation scripts.

    CLI values should be passed in when present; otherwise environment
    variables keep local runs and the 2080-Ti box pointed at the shared
    data layout without hardcoding paths.
    """

    imd2020_root: Optional[str] = None
    casia_root: Optional[str] = None
    indoor_root: Optional[str] = None
    coco_inpaint_root: Optional[str] = None
    sagid_root: Optional[str] = None
    bfree_root: Optional[str] = None
    anyedit_root: Optional[str] = None
    tgif2_root: Optional[str] = None
    run_root: Optional[str] = None


def _arg_value(args: Any, *names: str) -> Optional[str]:
    if args is None:
        return None
    for name in names:
        if hasattr(args, name):
            value = getattr(args, name)
            if value:
                return str(value)
    return None


def resolve_data_paths(args: Any = None, **overrides: Optional[str]) -> DataPaths:
    """Resolve shared data roots from explicit values, args namespace, then env vars.

    Recognized environment variables:
      DINO_IMD2020_ROOT, DINO_CASIA_ROOT, DINO_INDOOR_ROOT,
      DINO_COCO_INPAINT_ROOT, DINO_SAGID_ROOT, DINO_BFREE_ROOT,
      DINO_ANYEDIT_ROOT, DINO_TGIF2_ROOT, DINO_RUN_ROOT
    """
    return DataPaths(
        imd2020_root=(
            overrides.get("imd2020_root")
            or _arg_value(args, "imd2020_root", "imd_root")
            or os.environ.get("DINO_IMD2020_ROOT")
        ),
        casia_root=(
            overrides.get("casia_root")
            or _arg_value(args, "casia_root")
            or os.environ.get("DINO_CASIA_ROOT")
        ),
        indoor_root=(
            overrides.get("indoor_root")
            or _arg_value(args, "indoor_root")
            or os.environ.get("DINO_INDOOR_ROOT")
        ),
        coco_inpaint_root=(
            overrides.get("coco_inpaint_root")
            or _arg_value(args, "coco_inpaint_root")
            or os.environ.get("DINO_COCO_INPAINT_ROOT")
        ),
        sagid_root=(
            overrides.get("sagid_root")
            or _arg_value(args, "sagid_root")
            or os.environ.get("DINO_SAGID_ROOT")
        ),
        bfree_root=(
            overrides.get("bfree_root")
            or _arg_value(args, "bfree_root")
            or os.environ.get("DINO_BFREE_ROOT")
        ),
        anyedit_root=(
            overrides.get("anyedit_root")
            or _arg_value(args, "anyedit_root")
            or os.environ.get("DINO_ANYEDIT_ROOT")
        ),
        tgif2_root=(
            overrides.get("tgif2_root")
            or _arg_value(args, "tgif2_root", "tgif_root")
            or os.environ.get("DINO_TGIF2_ROOT")
        ),
        run_root=(
            overrides.get("run_root")
            or _arg_value(args, "run_root", "checkpoint_root")
            or os.environ.get("DINO_RUN_ROOT")
        ),
    )
