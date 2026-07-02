"""lab_utils.data.dataset — one general Dataset class for all experiments.

Output contract (§3.5 DESIGN_GUIDE):
    {
      'img':   FloatTensor (3, S, S),          # S = res.image_size
      'mask':  FloatTensor (1, S, S) or zeros, # GT mask at input res
      'meta':  dict,                            # item_id, source, is_real, ...
    }

Two-stage augmentation pipeline (I6 DESIGN_GUIDE):
  1. Geometric (image+mask jointly): oracle-crop → flip → resize
  2. Appearance (image only, mask unchanged): jpeg → gaussian noise →
     poisson noise → resize-jitter → blur

`apply_light_augmentations` compound is NOT used here (its flip-last ordering
violated I6). Individual ops from light.py are sequenced explicitly.

oracle_mask_crop is imported from augment/crop.py (I1 tripwire: the only
file where the "oracle" token is permitted).
"""

from __future__ import annotations

import hashlib
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from lab_utils.data.item import Item
from lab_utils.data.resolution import (
    Resolution,
    CropResult,
    random_resized_crop_pair,
    resize_only,
    resize_only_mask,
)
from lab_utils.data.augment.light import (
    apply_jpeg,
    apply_gaussian_noise,
    apply_poisson_noise,
    apply_resize_jitter,
    apply_blur,
    apply_flip_h,
)
from lab_utils.data.augment.composite import paste_real_background
from lab_utils.errors import DataError


def _stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)


_DEFAULT_LIGHT_AUG = dict(
    jpeg_prob=0.25, jpeg_q_min=88, jpeg_q_max=98,
    noise_prob=0.15, noise_std_min=0.002, noise_std_max=0.015,
    poisson_prob=0.0, poisson_peak_min=16.0, poisson_peak_max=64.0,
    resize_prob=0.20, resize_scale_min=0.80, resize_scale_max=0.98,
    blur_prob=0.0, blur_sigma_min=0.0, blur_sigma_max=1.0,
)


class Dataset(TorchDataset):
    """General-purpose dataset for all DINO_SCOPE experiments.

    Takes a list of Items (§3.1) and owns the load → augment → tensorize path.
    All datasets are instances of this class, produced by per-dataset builders
    in data/datasets/*.

    Args:
        items:              List of Items (any dataset).
        res:                Resolution — image_size and patch_size.
        augment:            True during training (random crops + light augs).
        normalize_mean:     ImageNet-style channel means.
        normalize_std:      ImageNet-style channel stds.
        crop_scale:         (min_scale, max_scale) for random-resized crop.
        crop_ratio:         (min_ratio, max_ratio) for random-resized crop.
        crop_max_tries:     Max random crop attempts before fallback.
        min_mask_area_frac: Minimum foreground fraction in crop to accept as
                            supervised for a splice item.
        oracle_crop:        If True, use oracle_mask_crop as the crop fallback
                            for splice items that random crops can't surface
                            (train-only; I1 controlled).
        oracle_target_cov:  (lo, hi) coverage target for oracle crop.
        flip_prob:          Probability of horizontal flip (geometric stage).
        paste_background:   If True and item.meta has 'real_path', paste the
                            pristine original background over the un-manipulated
                            region before cropping (inpaint items only).
        paste_frac:         Per-item paste probability (the "sp" share). Pasted
                            items read as local splices (sp); the rest keep the
                            whole-image diffusion fingerprint (fr). 1.0 = always
                            paste; e.g. 0.40 → 40% sp / 60% fr.
        light_aug_kwargs:   Override appearance aug probabilities / ranges.
        deterministic_seed: Seed base for reproducible eval augmentation.
    """

    def __init__(
        self,
        items: List[Item],
        res: Resolution,
        *,
        augment: bool = True,
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std:  Tuple[float, float, float] = (0.229, 0.224, 0.225),
        crop_scale: Tuple[float, float] = (0.18, 1.00),
        crop_ratio: Tuple[float, float] = (0.60, 1.70),
        crop_max_tries: int = 24,
        min_mask_area_frac: float = 0.03,
        oracle_crop: bool = False,
        oracle_target_cov: Tuple[float, float] = (0.10, 0.40),
        flip_prob: float = 0.50,
        paste_background: bool = True,
        paste_frac: float = 1.0,
        light_aug_kwargs: Optional[Dict[str, Any]] = None,
        deterministic_seed: int = 0,
    ):
        self.items              = list(items)
        self.res                = res
        self.augment            = bool(augment)
        self.crop_scale         = tuple(crop_scale)
        self.crop_ratio         = tuple(crop_ratio)
        self.crop_max_tries     = int(crop_max_tries)
        self.min_mask_area_frac = float(min_mask_area_frac)
        self.oracle_crop        = bool(oracle_crop)
        self.oracle_target_cov  = tuple(oracle_target_cov)
        self.flip_prob          = float(flip_prob)
        self.paste_background   = bool(paste_background)
        self.paste_frac         = float(paste_frac)
        self.lak                = dict(_DEFAULT_LIGHT_AUG)
        if light_aug_kwargs:
            self.lak.update(light_aug_kwargs)
        self.deterministic_seed = int(deterministic_seed)
        self.normalize          = transforms.Normalize(
            list(normalize_mean), list(normalize_std)
        )

    def __len__(self) -> int:
        return len(self.items)

    # ── functional builders ──────────────────────────────────────────────────

    def _copy_with_items(self, items: List[Item]) -> "Dataset":
        d = Dataset.__new__(Dataset)
        d.__dict__.update(self.__dict__)
        d.items = list(items)
        return d

    def subsample(self, n: int, *, seed: str) -> "Dataset":
        from lab_utils.data.sampling import deterministic_subsample
        return self._copy_with_items(deterministic_subsample(self.items, n, seed=seed))

    def filter(self, pred: Callable[[Item], bool]) -> "Dataset":
        return self._copy_with_items([it for it in self.items if pred(it)])

    # ── augmentation pipeline ────────────────────────────────────────────────

    def _geometric_stage(
        self,
        img: Image.Image,
        mask: Optional[Image.Image],
        item: Item,
    ) -> Tuple[Image.Image, Optional[Image.Image], bool]:
        """Geometric stage: crop → flip → resize.

        Returns (img, mask, crop_valid).  When crop_valid=False for a splice
        item, the caller should treat the sample as an unsupervised real.
        If img is None, the sample must be dropped entirely.
        """
        S = self.res.image_size

        if not self.augment:
            out_img  = resize_only(img, self.res)
            out_mask = resize_only_mask(mask, self.res) if mask is not None else None
            return out_img, out_mask, True

        # ── Random crop ───────────────────────────────────────────────────────
        is_splice = mask is not None
        crop_result = random_resized_crop_pair(
            img, mask, self.res,
            scale=self.crop_scale,
            ratio=self.crop_ratio,
            max_tries=self.crop_max_tries,
            min_mask_area_frac=self.min_mask_area_frac if is_splice else 0.0,
        )
        out_img  = crop_result.image
        out_mask = crop_result.mask
        crop_valid = crop_result.valid

        # ── Oracle fallback for small splices ─────────────────────────────────
        if (not crop_valid and is_splice and mask is not None and self.oracle_crop):
            from lab_utils.data.augment.crop import oracle_mask_crop
            oc = oracle_mask_crop(
                img, mask, self.res,
                target_cov_range=self.oracle_target_cov,
            )
            if oc.valid:
                out_img, out_mask, crop_valid = oc.image, oc.mask, True
            elif oc.mode == 'oracle_empty':
                return None, None, False

        # ── Geometric flip (operates on image+mask jointly) ───────────────────
        if random.random() < self.flip_prob:
            flip_res = apply_flip_h(out_img, out_mask)
            out_img  = flip_res.image
            out_mask = flip_res.mask

        return out_img, out_mask, crop_valid

    def _appearance_stage(self, img: Image.Image) -> Image.Image:
        """Appearance stage: jpeg → gaussian → poisson → resize → blur.

        Operates on image only; mask is not passed in (geometric stage is done).
        Only called when augment=True.
        """
        lak = self.lak
        out = img

        if random.random() < lak.get('jpeg_prob', 0.0):
            q   = random.randint(int(lak['jpeg_q_min']), int(lak['jpeg_q_max']))
            out = apply_jpeg(out, quality=q, quality_range=(lak['jpeg_q_min'], lak['jpeg_q_max'])).image

        if random.random() < lak.get('noise_prob', 0.0):
            std = random.uniform(lak['noise_std_min'], lak['noise_std_max'])
            out = apply_gaussian_noise(out, std=std, std_range=(lak['noise_std_min'], lak['noise_std_max'])).image

        if random.random() < lak.get('poisson_prob', 0.0):
            peak = random.uniform(lak['poisson_peak_min'], lak['poisson_peak_max'])
            out  = apply_poisson_noise(out, peak=peak, peak_range=(lak['poisson_peak_min'], lak['poisson_peak_max'])).image

        if random.random() < lak.get('resize_prob', 0.0):
            scale = random.uniform(lak['resize_scale_min'], lak['resize_scale_max'])
            out   = apply_resize_jitter(out, scale=scale, scale_range=(lak['resize_scale_min'], lak['resize_scale_max'])).image

        if random.random() < lak.get('blur_prob', 0.0):
            sigma = random.uniform(lak['blur_sigma_min'], lak['blur_sigma_max'])
            out   = apply_blur(out, sigma=sigma, sigma_range=(lak['blur_sigma_min'], lak['blur_sigma_max'])).image

        return out

    # ── __getitem__ ──────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        item = self.items[idx]

        # Deterministic seed for eval
        if not self.augment:
            seed = _stable_seed(f"{item.item_id}|{self.deterministic_seed}")
            py_state = random.getstate()
            np_state = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed & 0xFFFFFFFF)

        try:
            return self._build_sample(item)
        except DataError:
            raise
        except Exception as exc:
            raise DataError(
                f"Dataset.__getitem__ failed "
                f"source={item.source!r} "
                f"item_id={item.item_id!r} "
                f"img={item.image!r}: {exc}"
            ) from exc
        finally:
            if not self.augment:
                random.setstate(py_state)
                np.random.set_state(np_state)

    def _build_sample(self, item: Item) -> Optional[Dict[str, Any]]:
        S = self.res.image_size

        # ── Load ──────────────────────────────────────────────────────────────
        img  = Image.open(item.image).convert('RGB')
        mask = (Image.open(item.mask).convert('L')
                if item.mask is not None else None)

        # ── Pre-stage: composite (inpaint items only) ─────────────────────────
        # Paste the pristine original over the un-manipulated background so that
        # only the inpainted region differs from the original.  This restricts
        # the signal to the splice blob rather than a whole-image VAE fingerprint.
        #
        # paste_frac is the per-item paste probability == the "sp" share: pasted
        # items behave like a local splice (sp); the rest keep the whole-image
        # diffusion fingerprint (fr).  paste_frac=1.0 → always paste (all sp).
        real_path = item.meta.get('real_path')
        if (self.paste_background and real_path is not None and mask is not None
                and random.random() < self.paste_frac):
            real = Image.open(real_path).convert('RGB')
            img  = paste_real_background(img, real, mask)

        # ── Geometric stage ───────────────────────────────────────────────────
        out_img, out_mask, crop_valid = self._geometric_stage(img, mask, item)
        if out_img is None:
            return None

        # Mark splice as unsupervised if crop missed the splice region
        is_supervised = (not item.is_real) and crop_valid

        # ── Appearance stage (train only) ─────────────────────────────────────
        if self.augment:
            out_img = self._appearance_stage(out_img)

        # ── Tensorize ─────────────────────────────────────────────────────────
        img_t = self.normalize(TF.to_tensor(out_img))

        if out_mask is not None:
            mask_t = TF.to_tensor(out_mask)  # (1, S, S) in [0, 1]
        else:
            mask_t = torch.zeros(1, S, S, dtype=torch.float32)

        # ── Shape contract (catches any resize bug early) ─────────────────────
        C, H, W = img_t.shape
        if C != 3 or H != S or W != S:
            raise DataError(
                f"Dataset shape contract violated: expected (3,{S},{S}), "
                f"got {tuple(img_t.shape)} for item_id={item.item_id!r}"
            )

        return {
            'img':  img_t,
            'mask': mask_t,
            'meta': {
                'item_id':       item.item_id,
                'source':        item.source,
                'is_real':       item.is_real,
                'is_supervised': is_supervised,
            },
        }


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def lab_collate_fn(batch: List[Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Filter None samples (dropped by oracle_empty) and stack tensors.

    Raises:
        DataError: If the entire batch collapsed (indicates systematic failure).
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        raise DataError(
            "lab_collate_fn: entire batch is None after filtering. "
            "Check that dataset items are loading correctly."
        )
    return {
        'img':  torch.stack([b['img']  for b in batch]),
        'mask': torch.stack([b['mask'] for b in batch]),
        'meta': [b['meta'] for b in batch],
    }
