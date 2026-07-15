"""experiments.scripts.eval_openfake_by_generator — evaluate DINO_SCOPE checkpoint on downloaded OpenFake images.

Reads images either from `downloaded_manifest.csv` (created by `download_n_per_generator`)
or directly by scanning subdirectories inside `output_dir`.

Computes GT-free activation scores (sigmoid of `image_logit`, i.e., detector probability score)
and outputs both the overall mean activation score and a per-generator breakdown.
Optionally displays and/or saves matplotlib visualizations (input | prediction | attention).

Usage (CLI in Colab or local):
    python -m experiments.scripts.eval_openfake_by_generator \
        --checkpoint /content/drive/MyDrive/DINO_SCOPE_RUNS/gemini_finetune_v3/epoch_0003.pt \
        --output_dir ./openfake_by_generator \
        --viz --viz_per_gen 3
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from lab_utils.eval.decode.hdbscan import decode_hdbscan
from lab_utils.eval.decode.kmeans import decode_kmeans
from lab_utils.eval.decode.threshold import decode_threshold
from lab_utils.eval.fetch import model_info
from lab_utils.eval.load_model import load_eval_model
from lab_utils.eval.preprocess import load_image_tensor
from lab_utils.logging.text import log_line
from lab_utils.train.distributed import unwrap_model
from experiments.labs.viz import display_image_inline, plot_prediction

_DECODERS = {
    "kmeans": decode_kmeans,
    "threshold": decode_threshold,
    "hdbscan": decode_hdbscan,
}


def _sigmoid(logit: Optional[float]) -> float:
    if logit is None or not math.isfinite(logit):
        return float("nan")
    return float(1.0 / (1.0 + math.exp(-logit)))


def find_images_and_generators(
    output_dir: Path, manifest_name: str = "downloaded_manifest.csv"
) -> List[Tuple[Path, str]]:
    """Locate images and their generator labels from manifest CSV or folder structure."""
    manifest_path = output_dir / manifest_name
    items: List[Tuple[Path, str]] = []

    if manifest_path.exists():
        log_line(f"[eval] Found manifest at {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_path_str = row.get("file_path") or row.get("filepath")
                generator = row.get("generator", "unknown")
                if not file_path_str:
                    continue
                p = Path(file_path_str)
                # If relative, resolve against output_dir or cwd
                if not p.is_absolute():
                    if (output_dir / p).exists():
                        p = output_dir / p
                    elif Path(p).exists():
                        p = Path(p)
                    elif (output_dir / p.name).exists():
                        p = output_dir / p.name
                    elif (output_dir / generator / p.name).exists():
                        p = output_dir / generator / p.name
                if p.exists() and p.is_file():
                    items.append((p, generator))
                else:
                    log_line(f"[eval] Warning: file in manifest not found: {file_path_str}")
        if items:
            return items

    log_line(f"[eval] Manifest missing or yielded no valid paths. Scanning subdirectories in {output_dir}...")
    for sub in sorted(output_dir.iterdir()):
        if sub.is_dir():
            gen_name = sub.name
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                for p in sorted(sub.glob(ext)):
                    if p.is_file():
                        items.append((p, gen_name))
        elif sub.is_file() and sub.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"):
            items.append((sub, "root_directory"))

    return items


def evaluate_generator_directory(
    checkpoint_path: str,
    output_dir: Union[str, Path],
    *,
    device: str = "cuda:0",
    use_amp: bool = True,
    amp_dtype: str = "bf16",
    manifest_name: str = "downloaded_manifest.csv",
    viz: bool = False,
    save_viz: bool = False,
    viz_dir: Optional[Union[str, Path]] = None,
    viz_per_gen: int = 3,
    decoder: str = "kmeans",
) -> Dict[str, List[float]]:
    import torch

    output_dir = Path(output_dir)
    items = find_images_and_generators(output_dir, manifest_name=manifest_name)
    if not items:
        raise RuntimeError(f"No images found in {output_dir}")

    log_line(f"[eval] Loading model from {checkpoint_path} across {len(items)} images...")
    model, cfg, res = load_eval_model(checkpoint_path, device=device, strict=False)
    bare_model = unwrap_model(model)
    bare_model.eval()

    # Map generator -> list of activation scores
    scores_by_gen: Dict[str, List[float]] = defaultdict(list)
    all_scores: List[float] = []
    gen_viz_count: Dict[str, int] = defaultdict(int)

    if save_viz and viz_dir is not None:
        viz_dir = Path(viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)

    for i, (path, gen) in enumerate(items, 1):
        try:
            img_t, img_pil = load_image_tensor(path, res, device=device, return_pil=True)
            with torch.no_grad():
                info = model_info(
                    bare_model,
                    img_t,
                    device=device,
                    amp=use_amp,
                    amp_dtype=amp_dtype if use_amp else "float16",
                )
            score = _sigmoid(info.image_logit)
            if not math.isnan(score):
                scores_by_gen[gen].append(score)
                all_scores.append(score)

            # Matplotlib visual display / save
            if (viz or save_viz) and gen_viz_count[gen] < viz_per_gen:
                decoder_fn = _DECODERS.get(decoder, decode_kmeans)
                patch_mask = decoder_fn(info)
                fig = plot_prediction(
                    img_pil,
                    patch_mask,
                    info,
                    title=f"Gen: {gen} | {path.name} | p={score:.3f}",
                )
                if save_viz and viz_dir is not None:
                    out_p = viz_dir / f"{gen}_{path.stem}_p{score:.2f}.png"
                    out_p.parent.mkdir(parents=True, exist_ok=True)
                    if hasattr(fig, "savefig"):
                        fig.savefig(out_p, dpi=130, bbox_inches="tight")
                    else:
                        fig.save(out_p)
                if viz:
                    display_image_inline(fig)
                if hasattr(fig, "clf"):
                    try:
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                    except Exception:
                        pass
                gen_viz_count[gen] += 1

            if i % 20 == 0 or i == len(items):
                log_line(f"[eval] Processed {i}/{len(items)} images...")
        except Exception as e:
            log_line(f"[eval] Error processing {path} ({gen}): {e}")

    # Summary Display
    print("\n" + "=" * 65)
    print("OPENFAKE EVALUATION SUMMARY (NO ZOOM)")
    print("=" * 65)
    if all_scores:
        overall_mean = float(np.mean(all_scores))
        overall_std = float(np.std(all_scores))
        print(f"Overall Mean Activation Score : {overall_mean:.4f} (std: {overall_std:.4f}, n={len(all_scores)})")
    else:
        print("Overall Mean Activation Score : N/A (no valid scores computed)")
    print("-" * 65)
    print(f"{'Generator Model':<32} | {'Count':<5} | {'Mean Score':<10} | {'Std':<8}")
    print("-" * 65)

    for gen, scores in sorted(scores_by_gen.items(), key=lambda x: np.mean(x[1]) if x[1] else -1, reverse=True):
        if scores:
            m = float(np.mean(scores))
            s = float(np.std(scores))
            print(f"{gen:<32} | {len(scores):<5} | {m:<10.4f} | {s:<8.4f}")
        else:
            print(f"{gen:<32} | {0:<5} | {'N/A':<10} | {'N/A':<8}")
    print("=" * 65 + "\n")

    return dict(scores_by_gen)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DINO_SCOPE checkpoint on downloaded OpenFake images by generator.")
    parser.add_argument("--checkpoint", default="/content/drive/MyDrive/DINO_SCOPE_RUNS/gemini_finetune_v3/epoch_0003.pt",
                        help="Path to .pt checkpoint file.")
    parser.add_argument("--output_dir", default="./openfake_by_generator",
                        help="Directory containing downloaded images and manifest CSV.")
    parser.add_argument("--manifest_name", default="downloaded_manifest.csv",
                        help="Manifest CSV filename inside output_dir.")
    parser.add_argument("--device", default="cuda:0", help="Torch device to run inference on.")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision autocast.")
    parser.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"], help="Mixed precision dtype.")
    parser.add_argument("--viz", action="store_true", help="Display matplotlib prediction figures inline.")
    parser.add_argument("--save_viz", action="store_true", help="Save prediction figure PNGs to disk.")
    parser.add_argument("--viz_dir", default="./openfake_by_generator/viz_plots", help="Directory to save visual plots.")
    parser.add_argument("--viz_per_gen", type=int, default=3, help="Max number of samples to visualize per generator.")
    parser.add_argument("--decoder", default="kmeans", choices=["kmeans", "threshold", "hdbscan"], help="Decoder for patch mask overlay.")

    args = parser.parse_args()
    evaluate_generator_directory(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        device=args.device,
        use_amp=not args.no_amp,
        amp_dtype=args.amp_dtype,
        manifest_name=args.manifest_name,
        viz=args.viz,
        save_viz=args.save_viz,
        viz_dir=args.viz_dir,
        viz_per_gen=args.viz_per_gen,
        decoder=args.decoder,
    )


if __name__ == "__main__":
    main()
