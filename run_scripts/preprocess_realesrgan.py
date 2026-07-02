"""Index-driven Real-ESRGAN launder pass for a tgif2 root (CUDA / PyTorch).

Reads <input_dir>/tgif2_index.json (the codebase's standard manifest: every
image is a root-relative path) and produces a laundered <output_dir> that
lab_utils.data.datasets.tgif2.build(root=output_dir) loads unchanged.

  * coco_ids are capped deterministically (md5 ranking, same scheme as
    tgif2._split_coco_ids) so the laundered root holds ~max_images photographic
    images.  Whole coco_ids stay together — an original keeps its fakes + masks.
  * original_512 + every manipulations[].fake_path are UPSCALED on the GPU with
    a native RRDBNet (x2plus for scale=2, x4plus for scale=4) run in BATCHES so
    the GPU stays saturated, with a tqdm bar.
  * masks[...] label-maps are COPIED verbatim — binary ground truth must never
    go through ESRGAN (it interpolates edges and corrupts the labels).
  * a CAPPED tgif2_index.json (only the kept coco_ids) is written to output_dir.

Unlike realesrgan-ncnn-vulkan, this runs on CUDA via PyTorch — no Vulkan/ICD,
so it uses the Colab GPU (e.g. L4) directly.

Why this is fast: the scale matches the model (x2plus does 512->1024 directly
instead of x4plus's 512->2048-then-downsample, ~4x less compute), and inference
is batched across same-shape images rather than one .enhance() call per image.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from typing import Dict, List, Tuple


# (weight_url, native_scale) per requested output scale.
MODEL_URLS = {
    2: ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", 2),
    4: ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", 4),
}


def _img_count(entry: Dict) -> int:
    n = 1 if entry.get("original_512") else 0
    return n + sum(1 for m in entry.get("manipulations", []) if m.get("fake_path"))


def plan_kept_coco_ids(index: Dict, max_images: int, seed: str) -> Dict:
    """Deterministically select whole coco_ids until ~max_images photographic
    images are covered.  Same md5 ranking scheme as tgif2._split_coco_ids so the
    selection is stable and reproducible."""
    ranked = sorted(
        index.keys(),
        key=lambda cid: hashlib.md5(f"{seed}|{cid}".encode("utf-8")).hexdigest(),
    )
    kept: Dict = {}
    used = 0
    for cid in ranked:
        c = _img_count(index[cid])
        if used + c > max_images and kept:  # always keep at least one coco_id
            break
        kept[cid] = index[cid]
        used += c
    return kept


def collect_paths(kept: Dict) -> Tuple[List[str], List[str]]:
    """Return (upscale_rels, mask_rels) — deduped root-relative paths.

    upscale = original_512 + every manipulations[].fake_path
    mask    = every masks[...] value (copied verbatim, never upscaled)
    """
    upscale, masks = [], []
    seen_up, seen_mask = set(), set()
    for entry in kept.values():
        orig = entry.get("original_512")
        if orig and orig not in seen_up:
            seen_up.add(orig)
            upscale.append(orig)
        for man in entry.get("manipulations", []):
            fp = man.get("fake_path")
            if fp and fp not in seen_up:
                seen_up.add(fp)
                upscale.append(fp)
        for rel in (entry.get("masks") or {}).values():
            if rel and rel not in seen_mask:
                seen_mask.add(rel)
                masks.append(rel)
    # A mask must never also be upscaled.
    mask_set = set(masks)
    upscale = [r for r in upscale if r not in mask_set]
    return upscale, masks


class _ImgReadDataset:
    """Map-style dataset that decodes one image per index in a worker process.
    Returns (rel, bgr_uint8_HWC) — or (rel, None) if the file is unreadable."""

    def __init__(self, rels: List[str], input_dir: str):
        self.rels = rels
        self.input_dir = input_dir

    def __len__(self) -> int:
        return len(self.rels)

    def __getitem__(self, i: int):
        import cv2
        rel = self.rels[i]
        img = cv2.imread(os.path.join(self.input_dir, rel), cv2.IMREAD_COLOR)
        return rel, img


def _identity_collate(batch):
    # Images vary in shape — keep the raw list; we batch by shape on the GPU side.
    return batch


def load_model(scale: int, compile_model: bool = False):
    """Load a native RRDBNet for the requested output scale onto CUDA.

    Returns (model, native_scale, device, use_half).  Imports are local so the
    planning helpers / tests don't require torch.
    """
    # basicsr imports torchvision.transforms.functional_tensor, removed in
    # torchvision>=0.17.  rgb_to_grayscale now lives in functional — alias it.
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401
    except ModuleNotFoundError:
        import torchvision.transforms.functional as _F
        sys.modules["torchvision.transforms.functional_tensor"] = _F

    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet

    cuda = torch.cuda.is_available()
    if not cuda and os.environ.get("ALLOW_CPU") != "1":
        raise SystemExit(
            "[preprocess] ERROR: CUDA not available. Use a GPU runtime, "
            "or set ALLOW_CPU=1 to force (very slow)."
        )
    device = torch.device("cuda" if cuda else "cpu")
    print(f"[preprocess] Torch device: {torch.cuda.get_device_name(0) if cuda else 'CPU'}")

    url, native = MODEL_URLS.get(scale, MODEL_URLS[4])
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=native)
    state = torch.hub.load_state_dict_from_url(url, map_location="cpu", progress=True)
    state = state.get("params_ema", state.get("params", state))
    model.load_state_dict(state, strict=True)

    use_half = cuda
    model.eval().to(device)
    if use_half:
        model.half()
    model.to(memory_format=torch.channels_last)   # tensor-core friendly convs
    torch.backends.cudnn.benchmark = True          # fixed input size -> conv autotune
    if compile_model:
        try:
            model = torch.compile(model)           # fuses convs; ~1.3-1.5x once warm
            print("[preprocess] torch.compile enabled (first batch will be slow).")
        except Exception as e:
            print(f"[preprocess] torch.compile unavailable ({e}); continuing eager.")
    return model, native, device, use_half


def upscale_batched(model, native_scale, device, use_half, *, rels, input_dir,
                    output_dir, outscale, batch_size, progress, num_workers, write_workers=4):
    """Pipelined batched GPU upscale.

    Decoding is prefetched by DataLoader worker processes and PNG writes are
    offloaded to a thread pool, so the GPU isn't stalled on disk between batches.
    Images are bucketed by shape from the prefetched stream and a bucket is
    flushed through the net once it reaches batch_size.
    """
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    from concurrent.futures import ThreadPoolExecutor
    from torch.utils.data import DataLoader

    png_opts = [cv2.IMWRITE_PNG_COMPRESSION, 1]   # cheap compression -> fast encode
    counts = {"up": 0, "missing": 0}
    writer = ThreadPoolExecutor(max_workers=write_workers)

    def _save(dst, bgr):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        cv2.imwrite(dst, bgr, png_opts)

    @torch.inference_mode()
    def flush(items):
        # items: list of (rel, bgr_uint8_HWC).  Model expects RGB, NCHW, [0,1].
        arr = np.stack([im[:, :, ::-1] for _, im in items])           # BGR->RGB
        t = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        t = t.permute(0, 3, 1, 2).float().div_(255.0)
        if use_half:
            t = t.half()
        t = t.contiguous(memory_format=torch.channels_last)
        # RRDBNet pixel-unshuffles the input by native_scale, so H/W must be a
        # multiple of it.  Pad the bottom/right, then crop the output back.
        h, w = t.shape[-2:]
        ph, pw = (-h) % native_scale, (-w) % native_scale
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="replicate")
        try:
            out = model(t)
        except RuntimeError as e:                                     # OOM -> split
            if "out of memory" in str(e).lower() and len(items) > 1:
                torch.cuda.empty_cache()
                mid = len(items) // 2
                flush(items[:mid]); flush(items[mid:])
                return
            raise
        out = out[:, :, : h * native_scale, : w * native_scale]       # drop padding
        if outscale != native_scale:
            out = F.interpolate(out, scale_factor=outscale / native_scale,
                                mode="bicubic", align_corners=False)
        out = out.clamp_(0, 1).mul_(255.0).round_().byte()
        out = out.permute(0, 2, 3, 1).cpu().numpy()                   # NHWC RGB
        for (rel, _), o in zip(items, out):
            dst = os.path.join(output_dir, rel)
            writer.submit(_save, dst, np.ascontiguousarray(o[:, :, ::-1]))  # RGB->BGR
            counts["up"] += 1
            progress.update(1)

    loader = DataLoader(
        _ImgReadDataset(rels, input_dir),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_identity_collate,
        prefetch_factor=(4 if num_workers > 0 else None),
        persistent_workers=False,
    )

    pending: Dict[tuple, list] = defaultdict(list)
    for chunk in loader:
        for rel, img in chunk:
            if img is None:
                counts["missing"] += 1
                progress.update(1)
                continue
            bucket = pending[img.shape[:2]]
            bucket.append((rel, img))
            if len(bucket) >= batch_size:
                flush(bucket)
                pending[img.shape[:2]] = []
    for bucket in list(pending.values()):
        if bucket:
            flush(bucket)

    writer.shutdown(wait=True)
    return counts["up"], counts["missing"]


def main() -> None:
    ap = argparse.ArgumentParser(description="CUDA Real-ESRGAN launder for a tgif2 root.")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("scale", type=int, nargs="?", default=2, help="output upscale factor (2 or 4)")
    ap.add_argument("max_images", type=int, nargs="?", default=10000)
    ap.add_argument("--seed", default=os.environ.get("SPLIT_SEED", "tgif_launder"))
    ap.add_argument("--batch-size", type=int, default=16,
                    help="images per GPU batch (auto-halves on OOM)")
    ap.add_argument("--workers", type=int, default=0,
                    help="decode worker processes (0=auto: min(8, cpu_count))")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (~1.3-1.5x once warm; slow first batch)")
    args = ap.parse_args()

    index_path = os.path.join(args.input_dir, "tgif2_index.json")
    if not os.path.isfile(index_path):
        raise SystemExit(f"[preprocess] ERROR: index not found: {index_path}")

    with open(index_path) as f:
        index = json.load(f)

    kept = plan_kept_coco_ids(index, args.max_images, args.seed)
    upscale_rels, mask_rels = collect_paths(kept)

    print("[preprocess] Index-driven Real-ESRGAN launder (CUDA)")
    print(f"  Input : {args.input_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"  Scale : x{args.scale}   Cap: {args.max_images}   Seed: {args.seed}")
    print(f"[preprocess] coco_ids kept {len(kept)}/{len(index)}  "
          f"upscale={len(upscale_rels)}  masks={len(mask_rels)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Capped index + masks first (cheap, and lets us fail fast on a bad root).
    with open(os.path.join(args.output_dir, "tgif2_index.json"), "w") as f:
        json.dump(kept, f)

    n_mask = n_missing = 0
    for rel in mask_rels:
        src = os.path.join(args.input_dir, rel)
        if not os.path.isfile(src):
            n_missing += 1
            continue
        dst = os.path.join(args.output_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)
        n_mask += 1

    # Upscale pass (batched, native-scale model).
    from tqdm.auto import tqdm

    model, native, device, use_half = load_model(args.scale, compile_model=args.compile)
    workers = args.workers or max(2, min(8, (os.cpu_count() or 2)))
    print(f"[preprocess] Model native x{native}, output x{args.scale}, "
          f"batch={args.batch_size}, workers={workers}, half={use_half}")

    with tqdm(total=len(upscale_rels), unit="img", desc="ESRGAN upscale",
              smoothing=0.05) as bar:
        n_up, n_missing_up = upscale_batched(
            model, native, device, use_half,
            rels=upscale_rels, input_dir=args.input_dir, output_dir=args.output_dir,
            outscale=args.scale, batch_size=args.batch_size, progress=bar,
            num_workers=workers,
        )
    n_missing += n_missing_up

    print(f"[preprocess] Upscaled {n_up}, copied {n_mask} masks "
          f"(missing on disk: {n_missing}).")
    print(f"[preprocess] Done! Laundered dataset (capped index) at: {args.output_dir}")


if __name__ == "__main__":
    main()
