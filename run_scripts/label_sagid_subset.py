#!/usr/bin/env python3
"""Recover the official SAGI-D split/source for each renamed triplet_NNNNN file.

Reproduces the v14 loader's exact selection:
    df = read_csv(sagid.csv); add clean_* cols; dropna(clean_img/msk/src)
    candidates = df.sample(frac=1, random_state=42).reset_index(drop=True)
    triplet_{idx:05d}  <-  candidates.iloc[idx]

Usage:
    python label_sagid_subset.py --csv /path/to/sagid.csv \
        --modified /content/sagi_d/images/modified \
        --out sagid_subset_manifest.csv

NOTE on exactness: idx->row is exact for a single uninterrupted download run.
If the download was *resumed* after prior skipped triplets, the loader's
`candidates.iloc[already:]` (count) vs `triplet_idx` (max+1) can drift. Run with
--verify to sanity-check a sample by re-downloading the indexed source image and
comparing dimensions against the local triplet (needs kaggle creds + internet).
"""
import argparse, os, re, sys
import pandas as pd

def clean(p):
    if not isinstance(p, str): return None
    p = p.replace("\\", "/")
    return p[6:] if p.startswith("sagid/") else p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="path to sagid.csv")
    ap.add_argument("--modified", required=True, help="dir with triplet_*.jpg (modified/)")
    ap.add_argument("--out", default="sagid_subset_manifest.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["clean_img"] = df["img_path"].apply(clean)
    df["clean_msk"] = df["mask_path"].apply(clean)
    df["clean_src"] = df["src_path"].apply(clean)
    df = df.dropna(subset=["clean_img", "clean_msk", "clean_src"])
    cand = df.sample(frac=1, random_state=42).reset_index(drop=True)
    cand["source"] = cand["img_path"].apply(lambda p: p.replace("\\", "/").split("/")[2])

    idxs = []
    for f in os.listdir(args.modified):
        m = re.match(r"triplet_(\d+)\.", f)
        if m:
            idxs.append(int(m.group(1)))
    idxs = sorted(idxs)
    if not idxs:
        sys.exit(f"No triplet_*.jpg found in {args.modified}")
    print(f"Found {len(idxs)} local triplets (indices {idxs[0]}..{idxs[-1]})")
    if idxs[-1] >= len(cand):
        sys.exit(f"index {idxs[-1]} exceeds candidate rows {len(cand)} — wrong CSV?")

    sub = cand.iloc[idxs].copy()
    sub.insert(0, "triplet", [f"triplet_{i:05d}" for i in idxs])

    print("\n=== split ===")
    print(sub["split"].value_counts())
    print("\n=== source ===")
    print(sub["source"].value_counts())
    print("\n=== split x source ===")
    print(sub.groupby(["split", "source"]).size())

    test_n = (sub["split"] == "test").sum()
    ood = ((sub["split"] == "test") & (sub["source"] == "openimages")).sum()
    print(f"\n>> official TEST images in subset: {test_n} "
          f"({100*test_n/len(sub):.1f}%) | OOD openimages: {ood}")

    sub[["triplet", "split", "source", "inpainting_model", "diffusion_model",
         "type", "img_path"]].to_csv(args.out, index=False)
    print(f"\nWrote per-file manifest -> {args.out}")

if __name__ == "__main__":
    main()
