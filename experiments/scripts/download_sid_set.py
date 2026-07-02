"""experiments.scripts.download_sid_set — Download saberzl/SID_Set for evaluation.

Downloads images and binary masks from saberzl/SID_Set and saves them to a local
folder in the unpaired format expected by lab_utils.data.datasets.unpaired.

Usage:
    python -m experiments.scripts.download_sid_set --dest_dir ./data --limit 1000
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def clean_key(key: str) -> str:
    """Sanitize the dataset key to be used as a filename."""
    base = os.path.splitext(key)[0]
    for char in ['/', '\\', '.', ' ']:
        base = base.replace(char, '_')
    return base

def save_row(row, images_dir: Path, masks_dir: Path):
    """Saves a single dataset row to disk."""
    try:
        key = clean_key(row['key'])
        label = int(row['label'])
        image = row['image']
        mask = row.get('mask')

        # Real (label == 0)
        if label == 0:
            img_path = images_dir / f"{key}_real.png"
            image.convert("RGB").save(img_path, "PNG")
        # Fake (label == 1 or 2)
        else:
            img_path = images_dir / f"{key}_fake.png"
            image.convert("RGB").save(img_path, "PNG")

            if mask is not None:
                mask_path = masks_dir / f"{key}_fake.png"
                mask.convert("L").save(mask_path, "PNG")
    except Exception as e:
        print(f"Error saving row {row.get('key', 'unknown')}: {e}")

def main() -> None:
    p = argparse.ArgumentParser(description="Download saberzl/SID_Set for DINO_SCOPE evaluation")
    p.add_argument("--dest_dir", default="./data", help="Output root directory")
    p.add_argument("--limit", type=int, default=None,
                   help="Maximum number of items to download (defaults to all)")
    p.add_argument("--workers", type=int, default=16,
                   help="Number of threads for disk writes")
    p.add_argument("--split", default="validation",
                   help="Hugging Face split to download (e.g. validation, test, train)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: The 'datasets' package is required. Run 'pip install datasets' or run this in your Colab.")
        sys.exit(1)

    dest_root = Path(args.dest_dir)
    out_dir = dest_root / "SID_Set"
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to saberzl/SID_Set (split: {args.split})...")
    try:
        dataset = load_dataset("saberzl/SID_Set", split=args.split, streaming=True)
        dataset = dataset.shuffle(seed=args.seed, buffer_size=50)
    except Exception as e:
        print(f"Error loading dataset saberzl/SID_Set: {e}")
        sys.exit(1)

    # Target balanced real and fake images
    limit = args.limit
    target_real = limit // 2 if limit is not None else float('inf')
    target_fake = limit - target_real if limit is not None else float('inf')

    n_real = 0
    n_fake = 0
    futures = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        print(f"Downloading/saving SID_Set images to {out_dir}...")
        for i, row in enumerate(tqdm(dataset, desc="Streaming SID_Set", unit="item")):
            row = dict(row)
            row['key'] = row.get('img_id', f"sid_{i}")
            label = int(row['label'])
            
            if label == 0:
                if n_real < target_real:
                    futures.append(executor.submit(save_row, row, images_dir, masks_dir))
                    n_real += 1
            else:  # label == 1 or 2 are fakes (full synthetic or tampered)
                if n_fake < target_fake:
                    futures.append(executor.submit(save_row, row, images_dir, masks_dir))
                    n_fake += 1
            
            if n_real >= target_real and n_fake >= target_fake:
                break
        
        print("Waiting for disk writes to complete...")
        for fut in tqdm(futures, desc="Writing to disk", unit="file"):
            fut.result()

    print(f"Done! Dataset saved successfully to {out_dir}")
    print(f"Saved {n_real} real and {n_fake} fake items. Total: {n_real + n_fake}")

if __name__ == "__main__":
    main()
