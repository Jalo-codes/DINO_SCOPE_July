"""lab_utils.eval.cache — freeze and reload ModelInfo bundles.

build_cache(): one GPU pass → ModelInfo arrays saved to disk.
load_cache():  instant, model-free — loads the same contract as a live signal.

Because the cache IS the §2.1 ModelInfo contract, a cached signal and a live
signal are identical — decoders, metric, and aggregate work on both without
modification.  Tests may load a cached fixture to exercise the full pipeline
without a model.

Layout on disk (one .npz per item, or a single archive for bulk):
    <cache_dir>/
        index.json              item_ids in stable order
        <item_id>.npz           ModelInfo arrays for one item
                                    keys: patch_logits, attention, embeddings,
                                          image_logit (0-d or empty), grid_hw, res_*
"""

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from lab_utils.eval.fetch import ModelInfo
from lab_utils.data.resolution import Resolution
from lab_utils.logging.text import log_line


# ── Serialise / deserialise a single ModelInfo ─────────────────────────────────

def _info_to_arrays(info: ModelInfo) -> dict:
    """Pack ModelInfo into a dict of numpy arrays (npz-serialisable)."""
    arrays: dict = {}

    def _pack(name: str, arr: Optional[np.ndarray]) -> None:
        arrays[name] = arr if arr is not None else np.array([], dtype=np.float32)
        arrays[f'{name}_present'] = np.array([arr is not None], dtype=bool)

    _pack('patch_logits', info.patch_logits)
    _pack('attention',    info.attention)
    _pack('embeddings',   info.embeddings)

    arrays['image_logit']         = (
        np.array([info.image_logit], dtype=np.float64)
        if info.image_logit is not None
        else np.array([], dtype=np.float64)
    )
    arrays['image_logit_present'] = np.array([info.image_logit is not None], dtype=bool)
    arrays['grid_hw']             = np.array(list(info.grid_hw), dtype=np.int64)
    arrays['res_image_size']      = np.array([info.res.image_size], dtype=np.int64)
    arrays['res_patch_size']      = np.array([info.res.patch_size], dtype=np.int64)
    return arrays


def _arrays_to_info(arrays: dict) -> ModelInfo:
    """Reconstruct a ModelInfo from the packed arrays dict."""

    def _unpack(name: str) -> Optional[np.ndarray]:
        present = bool(arrays.get(f'{name}_present', np.array([False]))[0])
        if not present:
            return None
        arr = arrays[name]
        return arr if arr.size > 0 else None

    image_logit_arr     = arrays.get('image_logit', np.array([]))
    image_logit_present = bool(arrays.get('image_logit_present', np.array([False]))[0])
    image_logit: Optional[float] = (
        float(image_logit_arr[0]) if image_logit_present and image_logit_arr.size > 0
        else None
    )

    grid_hw = tuple(int(x) for x in arrays['grid_hw'])
    res = Resolution(
        image_size=int(arrays['res_image_size'][0]),
        patch_size=int(arrays['res_patch_size'][0]),
    )
    return ModelInfo(
        patch_logits=_unpack('patch_logits'),
        attention=_unpack('attention'),
        embeddings=_unpack('embeddings'),
        image_logit=image_logit,
        grid_hw=grid_hw,
        res=res,
    )


# ── Build ──────────────────────────────────────────────────────────────────────

def build_cache(
    model,
    loader,
    *,
    device,
    amp: bool = True,
    amp_dtype: str = 'float16',
    cache_dir: Path,
    overwrite: bool = False,
) -> List[str]:
    """Run one GPU pass; save one ModelInfo .npz per item to cache_dir.

    Args:
        model:      MultiHeadDetector (or duck-typed equivalent with .res).
        loader:     DataLoader yielding batches with 'img' and 'meta'
                    (meta must contain 'item_id' per sample).
        device:     torch.device.
        amp:        Use autocast for the forward pass.
        cache_dir:  Directory to write .npz files into.
        overwrite:  If False, skip items that already have a cached file.

    Returns:
        List of item_ids written (in order).
    """
    import torch
    from lab_utils.eval.fetch import model_info

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    written: List[str] = []
    skipped = 0

    for batch in loader:
        if batch is None:
            continue
        imgs   = batch['img']
        metas  = batch['meta']

        n = imgs.shape[0]
        for i in range(n):
            img_i    = imgs[i:i+1]
            meta_i   = (metas[i] if isinstance(metas, list)
                        else {k: v[i] for k, v in metas.items()})
            item_id  = str(meta_i.get('item_id', f'unk_{len(written)}'))

            out_path = cache_dir / f'{item_id}.npz'
            if not overwrite and out_path.exists():
                skipped += 1
                continue

            info = model_info(model, img_i, device=device, amp=amp, amp_dtype=amp_dtype)
            np.savez_compressed(str(out_path), **_info_to_arrays(info))
            written.append(item_id)

    # Write / update the index
    index_path = cache_dir / 'index.json'
    existing: List[str] = []
    if index_path.exists():
        try:
            with open(index_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    all_ids = existing + [x for x in written if x not in set(existing)]
    with open(index_path, 'w') as f:
        json.dump(all_ids, f, indent=2)

    log_line(
        f'[eval] cache built: wrote={len(written)} skipped={skipped} '
        f'total={len(all_ids)} dir={cache_dir}'
    )
    return written


# ── Load ───────────────────────────────────────────────────────────────────────

def load_cache(
    cache_dir: Path,
    *,
    item_ids: Optional[List[str]] = None,
) -> Dict[str, ModelInfo]:
    """Load ModelInfo bundles from cache_dir; return {item_id: ModelInfo}.

    Args:
        cache_dir: Directory written by build_cache.
        item_ids:  If given, only load these item_ids.  Otherwise loads all
                   item_ids listed in index.json.

    Returns:
        Dict mapping item_id → ModelInfo.
    """
    cache_dir = Path(cache_dir)
    index_path = cache_dir / 'index.json'

    if item_ids is None:
        if not index_path.exists():
            raise FileNotFoundError(f'cache index not found: {index_path}')
        with open(index_path) as f:
            item_ids = json.load(f)

    out: Dict[str, ModelInfo] = {}
    missing = 0
    for iid in item_ids:
        p = cache_dir / f'{iid}.npz'
        if not p.exists():
            missing += 1
            continue
        arrays = dict(np.load(str(p), allow_pickle=False))
        out[iid] = _arrays_to_info(arrays)

    if missing:
        log_line(f'[eval] cache load: {missing} item_ids missing from {cache_dir}')
    return out


def iter_cache(
    cache_dir: Path,
    *,
    item_ids: Optional[List[str]] = None,
) -> Iterator[Tuple[str, ModelInfo]]:
    """Lazily iterate (item_id, ModelInfo) tuples from cache_dir.

    Memory-efficient alternative to load_cache() when the full dict is too large.
    """
    cache_dir = Path(cache_dir)
    if item_ids is None:
        index_path = cache_dir / 'index.json'
        if not index_path.exists():
            raise FileNotFoundError(f'cache index not found: {index_path}')
        with open(index_path) as f:
            item_ids = json.load(f)

    for iid in item_ids:
        p = cache_dir / f'{iid}.npz'
        if not p.exists():
            continue
        arrays = dict(np.load(str(p), allow_pickle=False))
        yield iid, _arrays_to_info(arrays)
