"""lab_utils.eval.preprocess — the one image → model-tensor path for eval.

Before this module, three eval sites (train/loop.run_val_eval, scripts/eval.py,
labs/attention_zoom.py) each re-implemented `Image.open → resize → to_tensor →
(x-mean)/std → unsqueeze`.  Three copies of the *exact normalisation the model
was trained with* is a correctness hazard — drift in mean/std/interpolation
silently degrades every score.  This is the single source of truth.

The constants match the training Dataset (lab_utils/data/dataset.py):
    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std  = (0.229, 0.224, 0.225)

Torch-bound (needs tensors); imported only by eval/train tiers, never by the
torch-free data/aggregate layers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from lab_utils.data.item import Item
from lab_utils.data.resolution import Resolution

# ImageNet normalisation — must equal the training Dataset's constants.
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD:  Tuple[float, float, float] = (0.229, 0.224, 0.225)

ImageSource = Union[str, Path, Item, "PILImageType"]  # noqa: F821  (PIL typed lazily)


def _resolve_pil(src: ImageSource):
    """Return a PIL RGB image from a path / Item / open PIL image."""
    from PIL import Image as PILImage

    if isinstance(src, Item):
        path = src.image
        return PILImage.open(path).convert('RGB')
    if isinstance(src, (str, Path)):
        return PILImage.open(src).convert('RGB')
    # Assume it is already a PIL image.
    return src.convert('RGB')


def load_image_tensor(
    src: ImageSource,
    res: Resolution,
    *,
    device=None,
    normalize: bool = True,
    add_batch_dim: bool = True,
    return_pil: bool = False,
):
    """Load an image and turn it into the exact tensor the model expects.

    Args:
        src:           Path / str / Item / open PIL image.
        res:           Resolution — image is resized to (image_size, image_size).
        device:        torch device to move the tensor to (None = leave on CPU).
        normalize:     Apply ImageNet mean/std (set False to inspect raw pixels).
        add_batch_dim: Prepend a batch dim → (1, 3, S, S).  False → (3, S, S).
        return_pil:    Also return the resized PIL image (for visualisation).

    Returns:
        tensor, or (tensor, pil) when return_pil is True.
    """
    import torch
    from PIL import Image as PILImage
    from torchvision.transforms import functional as TF

    pil = _resolve_pil(src)
    S   = res.image_size
    if pil.size != (S, S):
        pil = pil.resize((S, S), PILImage.BILINEAR)

    t = TF.to_tensor(pil)  # (3, S, S) in [0, 1]
    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        t = (t - mean) / std
    if device is not None:
        t = t.to(device)
    if add_batch_dim:
        t = t.unsqueeze(0)

    if return_pil:
        return t, pil
    return t
