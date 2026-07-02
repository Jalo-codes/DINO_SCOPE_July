"""experiments.scripts.download_hf_eval — standalone download script for CocoGlide and OpenSDI.

Downloads datasets from Hugging Face and saves them to a local folder in the unpaired format
expected by lab_utils.data.datasets.unpaired.

Usage:
    python -m experiments.scripts.download_hf_eval --dataset cocoglide --dest_dir ./data --limit 500
    python -m experiments.scripts.download_hf_eval --dataset opensdi --dest_dir ./data --limit 1000
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


def save_row(row, images_dir: Path, masks_dir: Path, prefix: str = ""):
    """Saves a single dataset row to disk."""
    try:
        key = clean_key(row['key'])
        if prefix:
            key = f"{prefix}_{key}"
        label = int(row['label'])
        image = row['image']
        mask = row.get('mask')

        # Real (label == 0)
        if label == 0:
            img_path = images_dir / f"{key}_real.png"
            image.convert("RGB").save(img_path, "PNG")
        # Fake (label == 1)
        else:
            img_path = images_dir / f"{key}_fake.png"
            image.convert("RGB").save(img_path, "PNG")

            if mask is not None:
                mask_path = masks_dir / f"{key}_fake.png"
                mask.convert("L").save(mask_path, "PNG")
    except Exception as e:
        print(f"Error saving row {row.get('key', 'unknown')}: {e}")


def save_opensdi_row(row, gen_root: Path):
    """Saves a single OpenSDI row, filtering out entire fakes and nesting in generator folders."""
    try:
        key = row['key']  # e.g., "partial/sd15/fake/000651821.png"
        parts = key.split('/')
        if len(parts) < 4:
            category = "unknown"
            filename = clean_key(key)
        else:
            category = parts[0]   # "partial" or "entire"
            filename = os.path.splitext(parts[-1])[0]

        label = int(row['label'])
        
        # Skip fakes that are not partial
        if label != 0 and category != "partial":
            return False

        images_dir = gen_root / "images"
        masks_dir = gen_root / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        image = row['image']
        mask = row.get('mask')

        # Real (label == 0)
        if label == 0:
            img_path = images_dir / f"{category}_{filename}_real.png"
            image.convert("RGB").save(img_path, "PNG")
        # Fake (label == 1)
        else:
            img_path = images_dir / f"{category}_{filename}_fake.png"
            image.convert("RGB").save(img_path, "PNG")

            if mask is not None:
                mask_path = masks_dir / f"{category}_{filename}_fake.png"
                mask.convert("L").save(mask_path, "PNG")
        return True
    except Exception as e:
        print(f"Error saving OpenSDI row {row.get('key', 'unknown')}: {e}")
        return False


def main() -> None:
    p = argparse.ArgumentParser(description="Download HF datasets for DINO_SCOPE evaluation")
    p.add_argument("--dataset", required=True, choices=["cocoglide", "opensdi"],
                   help="Dataset to download")
    p.add_argument("--dest_dir", default="./data", help="Output root directory")
    p.add_argument("--limit", type=int, default=None,
                   help="Maximum number of items to download (defaults to all)")
    p.add_argument("--limit_per_gen", type=int, default=None,
                   help="OpenSDI test only: cap each generator to this many items "
                        "(e.g. 1000 → ~500 real + ~500 partial-fake per generator). "
                        "Overrides --limit's per-generator split when set.")
    p.add_argument("--fakes_only", action="store_true",
                   help="OpenSDI test only: save ONLY partial-fakes (reals are streamed "
                        "past but never written). With --limit_per_gen N this saves N "
                        "fakes/gen, which means streaming ~2N rows/gen since the split is "
                        "~50/50 real/fake. Use when the eval skips reals anyway.")
    p.add_argument("--workers", type=int, default=16,
                   help="Number of threads for disk writes")
    p.add_argument("--opensdi_dataset", default="nebula/OpenSDI_test",
                   help="HF dataset path for OpenSDI (e.g. nebula/OpenSDI_test or nebula/OpenSDI_train)")
    p.add_argument("--opensdi_split", default="test",
                   help="OpenSDI split to use (e.g. test, or sd15 for train)")
    p.add_argument("--generators", nargs="+", default=["sd15", "sd2", "sdxl", "sd3", "flux"],
                   help="OpenSDI generator splits to download from nebula/OpenSDI_test. "
                        "Real names (from HF API): sd15, sd2, sdxl, sd3, flux")
    p.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = p.parse_args()

    # Import datasets here so we don't import it globally if the user runs with --help
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: The 'datasets' package is required. Run 'pip install datasets' or run this in your Colab.")
        sys.exit(1)

    dest_root = Path(args.dest_dir)
    
    # We load the rows and write them to disk using a thread pool
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = []
        
        if args.dataset == "cocoglide":
            print("Connecting to nebula/CocoGlide (split: glide)...")
            ds_name = "nebula/CocoGlide"
            out_dir = dest_root / "CocoGlide"
            images_dir = out_dir / "images"
            masks_dir = out_dir / "masks"
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)

            dataset = load_dataset(ds_name, split="glide", streaming=True)
            # Small buffer — each row contains a decoded image (potentially MBs).
            # buffer_size=50 is enough to shuffle within a parquet shard without hanging.
            dataset = dataset.shuffle(seed=args.seed, buffer_size=50)

            # Target balanced real and fake images
            limit = args.limit
            target_real = limit // 2 if limit is not None else float('inf')
            target_fake = limit - target_real if limit is not None else float('inf')

            n_real = 0
            n_fake = 0
            
            print(f"Downloading/saving CocoGlide images to {out_dir}...")
            for row in tqdm(dataset, desc="Streaming CocoGlide", unit="item"):
                label = int(row['label'])
                if label == 0:
                    if n_real < target_real:
                        futures.append(executor.submit(save_row, row, images_dir, masks_dir))
                        n_real += 1
                else:
                    if n_fake < target_fake:
                        futures.append(executor.submit(save_row, row, images_dir, masks_dir))
                        n_fake += 1
                
                if n_real >= target_real and n_fake >= target_fake:
                    break
            
            print(f"Dispatched {n_real} real and {n_fake} fake items. Total: {n_real + n_fake}")

        else: # opensdi
            out_dir = dest_root / "OpenSDI"
            images_dir = out_dir / "images"
            masks_dir = out_dir / "masks"
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)

            # If user specifies nebula/OpenSDI_train, it doesn't use name configurations, just split
            is_train = "train" in args.opensdi_dataset.lower() or args.opensdi_split != "test"
            
            if is_train:
                # OpenSDI train setup
                print(f"Connecting to {args.opensdi_dataset} (split: {args.opensdi_split})...")
                dataset = load_dataset(args.opensdi_dataset, split=args.opensdi_split, streaming=True)
                dataset = dataset.shuffle(seed=args.seed, buffer_size=50)

                limit = args.limit
                target_real = limit // 2 if limit is not None else float('inf')
                target_fake = limit - target_real if limit is not None else float('inf')

                n_real = 0
                n_fake = 0

                print(f"Downloading/saving OpenSDI (train split) images to {out_dir}...")
                for row in tqdm(dataset, desc="Streaming OpenSDI (train)", unit="item"):
                    label = int(row['label'])
                    if label == 0:
                        if n_real < target_real:
                            futures.append(executor.submit(save_row, row, images_dir, masks_dir))
                            n_real += 1
                    else:
                        if n_fake < target_fake:
                            futures.append(executor.submit(save_row, row, images_dir, masks_dir))
                            n_fake += 1
                    
                    if n_real >= target_real and n_fake >= target_fake:
                        break
                print(f"Dispatched {n_real} real and {n_fake} fake items. Total: {n_real + n_fake}")

            else:
                # OpenSDI test setup (multiple generators)
                generators = args.generators
                if len(generators) == 1 and generators[0].lower() == "all":
                    generators = ["sd15", "sd2", "sdxl", "sd3", "flux"]
                
                print(f"Connecting to {args.opensdi_dataset} across generators: {generators}")
                
                # Determine limits per generator. --limit_per_gen (if set) wins,
                # so "~1k from each generator" is exact regardless of gen count.
                limit = args.limit
                if args.limit_per_gen is not None:
                    limit_per_gen = max(1, args.limit_per_gen)
                elif limit is not None:
                    limit_per_gen = max(1, limit // len(generators))
                else:
                    limit_per_gen = float('inf')
                
                total_dispatched_real = 0
                total_dispatched_fake = 0

                for gen in generators:
                    print(f"Streaming generator: {gen}...")
                    gen_root = out_dir / gen
                    try:
                        # OpenSDI_test has one config ("default"); generators ARE the splits.
                        # e.g. split="sd15", split="flux", etc.
                        gen_dataset = load_dataset(args.opensdi_dataset, split=gen, streaming=True)
                        # Small buffer — big images; 50 is enough to shuffle within a parquet shard.
                        gen_dataset = gen_dataset.shuffle(seed=args.seed, buffer_size=50)
                    except Exception as e:
                        print(f"Error loading generator {gen}: {e}. Skipping.")
                        continue
                    
                    if args.fakes_only:
                        # Stream past reals (never saved); collect limit_per_gen fakes.
                        target_real = 0
                        target_fake = limit_per_gen  # may be inf → whole split
                    else:
                        target_real = limit_per_gen // 2 if limit_per_gen != float('inf') else float('inf')
                        target_fake = limit_per_gen - target_real if limit_per_gen != float('inf') else float('inf')

                    n_real = 0
                    n_fake = 0

                    for row in tqdm(gen_dataset, desc=f"Generator {gen}", unit="item"):
                        label = int(row['label'])
                        key = row['key']
                        parts = key.split('/')
                        category = parts[0] if len(parts) >= 4 else "unknown"

                        if label == 0:
                            if n_real < target_real:
                                futures.append(executor.submit(save_opensdi_row, row, gen_root))
                                n_real += 1
                        else:
                            if category == "partial":
                                if n_fake < target_fake:
                                    futures.append(executor.submit(save_opensdi_row, row, gen_root))
                                    n_fake += 1
                        
                        if n_real >= target_real and n_fake >= target_fake:
                            break
                    
                    print(f"Generator {gen}: Dispatched {n_real} real and {n_fake} fake items.")
                    total_dispatched_real += n_real
                    total_dispatched_fake += n_fake
                
                print(f"Total dispatched: {total_dispatched_real} real and {total_dispatched_fake} fake items.")

        print("Waiting for disk writes to complete...")
        for fut in tqdm(futures, desc="Writing to disk", unit="file"):
            fut.result()

    print(f"Done! Dataset saved successfully to {out_dir}")


if __name__ == "__main__":
    main()
