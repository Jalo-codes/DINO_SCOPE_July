"""experiments.labs.dino_diff_lab — raw-DINOv3 feature-diff change detection.

No fine-tuned checkpoint, no LoRA, no trained heads — just the pretrained
DINOv3 backbone's own patch tokens. Idea: for a real/modified pair, for each
patch in the modified grid take the max cosine similarity to any patch in a
small neighborhood window of the real grid (absorbs the sub-patch alignment
drift some generators introduce); 1 - max_sim is the change score. Classic
Siamese feature-differencing / block-matching change detection, done in
DINO's own embedding space — cheap, offline, no GroundingDINO/SAM/LLM call.

Known limitations (prototype only — this file makes no correctness claims,
it exists to eyeball whether the signal is there before investing further):
  - DINO's own color-jitter training augmentation makes it partly
    color-invariant, so a pure recolor edit may show little to no signal.
  - Local block matching aliases on repetitive/textured regions (foliage,
    brick, water) — expect false negatives there.
  - A generator that subtly relights/recolors the WHOLE frame can produce a
    diffuse low-level diff everywhere, diluting the localized signal.

Usage (notebook, no checkpoint needed — raw pretrained backbone only)::

    from experiments.labs.dino_diff_lab import run_diff_proto
    results = run_diff_proto(root='/content/pico_banana_native_s3', k=8)
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from lab_utils.data.resolution import Resolution
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.logging.text import log_line

_DEFAULT_MODEL_NAME = 'facebook/dinov3-vith16plus-pretrain-lvd1689m'

_backbone_cache: Dict[str, torch.nn.Module] = {}


def _load_raw_backbone(model_name: str, device) -> torch.nn.Module:
    """Pretrained DINOv3 backbone straight from HuggingFace — no LoRA, no
    trained heads, frozen. Cached per (model_name, device) so repeated calls
    in a notebook don't re-download/re-init."""
    key = f'{model_name}@{device}'
    if key in _backbone_cache:
        return _backbone_cache[key]
    from transformers import AutoModel

    log_line(f'[dd] loading raw backbone: {model_name}')
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device is not None:
        model = model.to(device)
    _backbone_cache[key] = model
    return model


def encode_patches(backbone: torch.nn.Module, x: torch.Tensor, res: Resolution) -> torch.Tensor:
    """(1, 3, S, S) → (num_patches, feat_dim), L2-normalized along feat_dim."""
    with torch.no_grad():
        out = backbone(pixel_values=x).last_hidden_state
    feats = out[:, -res.num_patches:, :].squeeze(0)
    return F.normalize(feats, dim=-1)


def _pool_grid(feats: torch.Tensor, grid_hw: Tuple[int, int], ksize: int) -> torch.Tensor:
    """Average-pool a (rows*cols, D) patch grid over a ksize x ksize window
    (stride 1, same padding), then re-normalize. Softens single-patch noise
    before matching — the "pool neighboring patches" alternative to (or
    combined with) the neighborhood-max search below."""
    if ksize <= 1:
        return feats
    rows, cols = grid_hw
    grid = feats.reshape(1, rows, cols, -1).permute(0, 3, 1, 2)  # (1, D, rows, cols)
    pad = ksize // 2
    pooled = F.avg_pool2d(grid, kernel_size=ksize, stride=1, padding=pad, count_include_pad=False)
    pooled = pooled.permute(0, 2, 3, 1).reshape(rows * cols, -1)
    return F.normalize(pooled, dim=-1)


def neighborhood_max_diff(
    feats_real: torch.Tensor,
    feats_mod: torch.Tensor,
    grid_hw: Tuple[int, int],
    *,
    radius: int = 1,
    pool_ksize: int = 1,
) -> np.ndarray:
    """Change-score map between two (num_patches, D) L2-normalized patch grids.

    For each patch in `feats_mod`, the score is 1 - (max cosine similarity to
    any patch within `radius` patches of the same grid location in
    `feats_real`). radius=1 searches a 3x3 neighborhood. pool_ksize>1
    average-pools both grids first (see _pool_grid) — a softer, cheaper
    alternative/complement to widening the search radius.

    Returns:
        (rows, cols) float32 array in [0, 2] (0 = identical direction).
    """
    rows, cols = grid_hw
    a = _pool_grid(feats_real, grid_hw, pool_ksize).reshape(rows, cols, -1)
    b = _pool_grid(feats_mod, grid_hw, pool_ksize).reshape(rows, cols, -1)

    best_sim = torch.full((rows, cols), -1.0, dtype=a.dtype, device=a.device)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            r0, r1 = max(0, -dr), rows - max(0, dr)
            c0, c1 = max(0, -dc), cols - max(0, dc)
            if r0 >= r1 or c0 >= c1:
                continue
            b_win = b[r0:r1, c0:c1]
            a_win = a[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
            sim = (b_win * a_win).sum(dim=-1)
            best_sim[r0:r1, c0:c1] = torch.maximum(best_sim[r0:r1, c0:c1], sim)

    diff = (1.0 - best_sim).clamp(min=0.0)
    return diff.cpu().numpy().astype(np.float32)


def _crop_edges(img, frac: float):
    """Crop `frac` off each of the four edges (e.g. 0.05 removes a 5%-wide border)."""
    if not frac:
        return img
    w, h = img.size
    dx, dy = int(round(w * frac)), int(round(h * frac))
    return img.crop((dx, dy, w - dx, h - dy))


def _group_case_pairs(items: List) -> Dict[str, Dict[str, object]]:
    """Group Items by meta['case_id'] into {'real': Item, 'modified': Item}."""
    by_case: Dict[str, Dict[str, object]] = {}
    for it in items:
        slot = by_case.setdefault(it.meta.get('case_id', it.item_id), {})
        slot['real' if it.is_real else 'modified'] = it
    return {cid: d for cid, d in by_case.items() if 'real' in d and 'modified' in d}


def diff_one(
    backbone: torch.nn.Module,
    res: Resolution,
    real_path,
    mod_path,
    *,
    device,
    radius: int = 1,
    pool_ksize: int = 1,
    crop_frac: float = 0.0,
) -> Dict:
    """Run the raw backbone on one real/modified pair and diff their patch grids.

    crop_frac: fraction to crop off each of the four edges of BOTH images
    before resizing (e.g. 0.05 discards a 5%-wide border) — use when a
    source has an encode/decode or upload artifact right at the frame edge.
    """
    real_src, mod_src = real_path, mod_path
    if crop_frac:
        from PIL import Image as PILImage
        real_src = _crop_edges(PILImage.open(real_path).convert('RGB'), crop_frac)
        mod_src = _crop_edges(PILImage.open(mod_path).convert('RGB'), crop_frac)
    real_x, real_pil = load_image_tensor(real_src, res, device=device, return_pil=True)
    mod_x, mod_pil = load_image_tensor(mod_src, res, device=device, return_pil=True)

    feats_real = encode_patches(backbone, real_x, res)
    feats_mod = encode_patches(backbone, mod_x, res)

    grid_hw = (res.num_patches_per_side, res.num_patches_per_side)
    diff_map = neighborhood_max_diff(feats_real, feats_mod, grid_hw, radius=radius, pool_ksize=pool_ksize)

    return {
        'real_pil': real_pil, 'mod_pil': mod_pil,
        'diff_map': diff_map, 'grid_hw': grid_hw,
    }


def hot_mask(
    diff_map: np.ndarray,
    grid_hw: Tuple[int, int],
    *,
    percentile='otsu',
    thresh_mult: float = 1.0,
    min_patches: int = 3,
) -> np.ndarray:
    """Boolean hot-patch mask via an adaptive (Otsu/gap) split instead of a
    fixed top-K% cutoff, then drops connected components smaller than
    `min_patches` (kills isolated single-patch aliasing on textured regions).

    Reuses lab_utils.eval.zoom.attention_hot_mask/_label_components as-is —
    both are generic over any 2D score grid, not attention-specific, and
    already power predict.py's --zoom pass on the trained model's attention.
    """
    from lab_utils.eval.zoom import _label_components, attention_hot_mask

    hot = attention_hot_mask(diff_map, grid_hw, percentile=percentile, thresh_mult=thresh_mult)
    if min_patches <= 1:
        return hot
    kept = np.zeros_like(hot)
    for cells in _label_components(hot):
        if len(cells) >= min_patches:
            for (r, c) in cells:
                kept[r, c] = True
    return kept


def _plot_diff(
    real_pil, mod_pil, diff_map: np.ndarray, *, title: str = '',
    percentile='otsu', thresh_mult: float = 1.0, min_patches: int = 3,
):
    """4-panel figure: real | modified | diff heatmap | adaptive hot-mask overlay.
    Follows experiments.labs.viz conventions (figsize, hot cmap, suptitle)."""
    import matplotlib.pyplot as plt

    from experiments.labs.viz import mask_overlay

    mod_arr = np.array(mod_pil.convert('RGB'))
    flag_mask = hot_mask(diff_map, diff_map.shape, percentile=percentile,
                          thresh_mult=thresh_mult, min_patches=min_patches)

    fig, axes = plt.subplots(1, 4, figsize=(3.5 * 4, 4.0))
    if title:
        fig.suptitle(title, fontsize=11)

    axes[0].imshow(np.array(real_pil)); axes[0].set_title('real'); axes[0].axis('off')
    axes[1].imshow(mod_arr); axes[1].set_title('modified'); axes[1].axis('off')
    axes[2].imshow(diff_map, cmap='hot', interpolation='nearest'); axes[2].set_title('diff heatmap'); axes[2].axis('off')
    overlay = mask_overlay(mod_arr, flag_mask, color=(220, 30, 30), alpha=0.45)
    axes[3].imshow(overlay); axes[3].set_title(f'hot ({percentile})'); axes[3].axis('off')

    plt.tight_layout()
    return fig


def run_diff_proto(
    root: str,
    source: str = 'pico_banana',
    k: int = 8,
    image_size: int = 688,
    model_name: str = _DEFAULT_MODEL_NAME,
    radius: int = 1,
    pool_ksize: int = 1,
    hot_percentile='otsu',
    hot_thresh_mult: float = 1.0,
    hot_min_patches: int = 3,
    crop_frac: float = 0.0,
    device: str = 'cuda',
    show: bool = True,
    out_dir: Optional[str] = None,
    seed: int = 42,
) -> List[Dict]:
    """Sample k real/modified pairs from a registered dataset indexer, diff
    them with the RAW (no fine-tuning) DINOv3 backbone, and visualize.

    radius: neighborhood-max search radius in patches (absorbs sub-patch
    alignment drift). pool_ksize: pre-diff average-pool window (softens
    single-patch noise/aliasing) — try both independently before combining.
    hot_percentile/hot_thresh_mult/hot_min_patches: passed to hot_mask() —
    'otsu'/'gap' adaptive threshold (or a numeric percentile) instead of a
    fixed top-K% cutoff, plus connected-component size filtering.
    crop_frac: fraction to crop off each of the four edges of BOTH real and
    modified images before resizing (e.g. 0.05 for a 5% border crop).
    """
    from lab_utils.data.datasets.registry import build as build_source

    dev = torch.device(device if (device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    res = Resolution(image_size=image_size, patch_size=16)

    _, val_ds = build_source(source, Path(root).expanduser(), res=res)
    items = val_ds.items
    if not items:
        raise RuntimeError(f'run_diff_proto: indexer found no items for source={source!r} root={root!r}')

    pairs = list(_group_case_pairs(items).items())
    if not pairs:
        raise RuntimeError(f'run_diff_proto: no case_id in {source!r} had both a real and a modified item')

    rng = random.Random(seed)
    chosen = rng.sample(pairs, k=min(k, len(pairs)))

    backbone = _load_raw_backbone(model_name, dev)

    out_path = Path(out_dir) if out_dir else None
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)

    from experiments.labs.viz import display_image_inline

    results: List[Dict] = []
    for case_id, d in chosen:
        real_it, mod_it = d['real'], d['modified']
        category = real_it.meta.get('category', '')

        r = diff_one(
            backbone, res, real_it.image, mod_it.image,
            device=dev, radius=radius, pool_ksize=pool_ksize, crop_frac=crop_frac,
        )
        r['case_id'] = case_id
        r['category'] = category
        results.append(r)

        log_line(
            f'[dd] {case_id} ({category}): diff_map mean={r["diff_map"].mean():.4f} '
            f'max={r["diff_map"].max():.4f} radius={radius} pool_ksize={pool_ksize}'
        )

        title = f'{case_id} | {category} | radius={radius} pool={pool_ksize}'
        fig = _plot_diff(
            r['real_pil'], r['mod_pil'], r['diff_map'], title=title,
            percentile=hot_percentile, thresh_mult=hot_thresh_mult, min_patches=hot_min_patches,
        )

        if out_path is not None:
            fig.savefig(out_path / f'{case_id}.png', dpi=130, bbox_inches='tight')
        if show:
            display_image_inline(fig)

        import matplotlib.pyplot as plt
        plt.close(fig)

    log_line(f'[dd] done — {len(results)} pair(s) diffed')
    return results
