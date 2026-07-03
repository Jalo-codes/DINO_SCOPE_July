"""experiments.scripts.view_triplets — inline browser for inpaint-triplet datasets.

Walks a root/modified, root/original, root/mask triplet layout (the layout
produced by export_pico_masks.py and consumed by
lab_utils.data.datasets.inpaint.build) and shows, per item: original |
modified | mask | mask-overlaid-on-modified — one row per item, via
experiments.labs.viz.display_image_inline.

Typical usage (notebook cell)::

    from experiments.scripts.view_triplets import browse_triplets
    browse_triplets('/content/pico_gemini_triplets', n=15)

CLI (saves each row to disk instead of displaying, e.g. over SSH)::

    python -m experiments.scripts.view_triplets \\
        --root /content/pico_gemini_triplets --n 15 --out_dir /tmp/triplet_previews
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from PIL import Image

_VALID_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff'})


def _clean_name(filename: str) -> str:
    """Strip extension and common modified/original/mask suffixes for matching.

    Mirrors lab_utils.data.datasets.inpaint._clean_name exactly, so the
    triplets shown here are the same ones the training Dataset would pair.
    """
    stem = os.path.splitext(filename)[0]
    for suf in ('_modified', '_original', '_orig', '_mask', '_fake', '_real',
                '_inpainted', '_gt'):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
            break
    return stem


def _index_dir(folder: Path, exts: frozenset) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for f in sorted(folder.iterdir()):
        if f.is_file() and f.suffix.lower() in exts:
            out[_clean_name(f.name)] = f
    return out


class Triplet(NamedTuple):
    case_id: str
    original: Path
    modified: Path
    mask: Path


def find_triplets(
    root,
    *,
    modified_subdir: str = 'modified',
    original_subdir: str = 'original',
    mask_subdir: str = 'mask',
    valid_exts: Optional[frozenset] = None,
) -> List[Triplet]:
    """Discover matched (original, modified, mask) triplets under root.

    Uses the same basename-matching rule as lab_utils.data.datasets.inpaint,
    so this shows exactly the pairs the training Dataset would see. Case_ids
    missing any of the three files are silently skipped — this is a browsing
    tool, not a dataset validator.
    """
    root = Path(root)
    exts = valid_exts or _VALID_EXTS
    mask_exts = frozenset(exts | {'.png'})

    mods  = _index_dir(root / modified_subdir, exts)
    origs = _index_dir(root / original_subdir, exts)
    masks = _index_dir(root / mask_subdir, mask_exts)

    case_ids = sorted(set(mods) & set(origs) & set(masks))
    return [Triplet(cid, origs[cid], mods[cid], masks[cid]) for cid in case_ids]


def _load_panels(
    t: Triplet,
    *,
    overlay_color: Tuple[int, int, int] = (255, 60, 60),
    overlay_alpha: float = 0.45,
):
    from experiments.labs.viz import mask_overlay

    orig = Image.open(t.original).convert('RGB')
    mod  = Image.open(t.modified).convert('RGB')
    mask_pil = Image.open(t.mask).convert('L')

    mod_arr  = np.array(mod)
    mask_arr = np.array(mask_pil) > 127
    overlay  = mask_overlay(mod_arr, mask_arr, color=overlay_color, alpha=overlay_alpha)

    return orig, mod, mask_pil, overlay


def show_triplet(
    t: Triplet,
    *,
    overlay_color: Tuple[int, int, int] = (255, 60, 60),
    overlay_alpha: float = 0.45,
    figsize: Tuple[float, float] = (14, 4),
    save_path: Optional[Path] = None,
) -> None:
    """Render one (original | modified | mask | overlay) row and display it inline."""
    import matplotlib.pyplot as plt
    from experiments.labs.viz import display_image_inline

    orig, mod, mask_pil, overlay = _load_panels(
        t, overlay_color=overlay_color, overlay_alpha=overlay_alpha,
    )

    fig, axes = plt.subplots(1, 4, figsize=figsize)
    fig.suptitle(t.case_id, fontsize=11)

    axes[0].imshow(orig)
    axes[0].set_title('original')
    axes[0].axis('off')

    axes[1].imshow(mod)
    axes[1].set_title('modified')
    axes[1].axis('off')

    axes[2].imshow(mask_pil, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('mask')
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title('mask on modified')
    axes[3].axis('off')

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=110, bbox_inches='tight')
    display_image_inline(fig)
    plt.close(fig)


def browse_triplets(
    root,
    *,
    n: int = 15,
    shuffle: bool = True,
    seed: int = 42,
    modified_subdir: str = 'modified',
    original_subdir: str = 'original',
    mask_subdir: str = 'mask',
    overlay_color: Tuple[int, int, int] = (255, 60, 60),
    overlay_alpha: float = 0.45,
    out_dir: Optional[str] = None,
) -> List[Triplet]:
    """Find triplets under root and show up to n of them inline, one row each.

    Call this directly from a notebook cell. Set out_dir to also save each
    row as a PNG (useful over SSH / headless boxes, alongside or instead of
    inline display).
    """
    triplets = find_triplets(
        root, modified_subdir=modified_subdir,
        original_subdir=original_subdir, mask_subdir=mask_subdir,
    )
    print(f'[view_triplets] found {len(triplets)} matched triplets under {root}')
    if not triplets:
        return []

    if shuffle:
        rng = random.Random(seed)
        triplets = triplets[:]
        rng.shuffle(triplets)
    chosen = triplets[:n]

    out_path = Path(out_dir) if out_dir else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    for t in chosen:
        save_path = out_path / f'{t.case_id}.png' if out_path is not None else None
        show_triplet(
            t, overlay_color=overlay_color, overlay_alpha=overlay_alpha,
            save_path=save_path,
        )

    return chosen


def main():
    p = argparse.ArgumentParser(
        prog='view_triplets',
        description='Browse inpaint-triplet (original/modified/mask) datasets inline or to disk.',
    )
    p.add_argument('--root', required=True,
                   help='Triplet dataset root (contains modified/, original/, mask/)')
    p.add_argument('--n', type=int, default=15)
    p.add_argument('--shuffle', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--modified_subdir', default='modified')
    p.add_argument('--original_subdir', default='original')
    p.add_argument('--mask_subdir', default='mask')
    p.add_argument('--overlay_alpha', type=float, default=0.45)
    p.add_argument('--out_dir', default=None,
                   help='If set, save each row as a PNG here (for headless/SSH use)')
    args = p.parse_args()

    browse_triplets(
        args.root, n=args.n, shuffle=args.shuffle, seed=args.seed,
        modified_subdir=args.modified_subdir, original_subdir=args.original_subdir,
        mask_subdir=args.mask_subdir, overlay_alpha=args.overlay_alpha,
        out_dir=args.out_dir,
    )


if __name__ == '__main__':
    main()
