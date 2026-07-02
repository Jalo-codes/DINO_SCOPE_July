#!/usr/bin/env python3
"""Build a CLEAN SAGI-D validation set on the box: official val split MINUS
anything that leaked into the trained subset, downloaded fresh from Kaggle.

Why this is exact (not fuzzy): the v14 loader selected training triplets as the
first N rows of `sagid.csv.sample(frac=1, random_state=42)`, written sequentially
as triplet_NNNNN. We reproduce that order, read your local triplet indices to get
the EXACT set of rows you trained on, then drop those rows from the official val
split. The remainder are val images the model has never seen.

Output layout (matches lab_utils inpaint.build pairing by shared basename):
    <out>/modified/<stem>.jpg   (inpainted)
    <out>/original/<stem>.jpg   (source/original, resized to modified)
    <out>/mask/<stem>.png       (inpaint mask)
where stem = basename(img_path) without extension (unique per inpainting).

Usage:
    python build_clean_sagid_val.py \
        --csv sagid.csv \
        --trained-modified /content/sagi_d/images/modified \
        --out /content/sagi_d_val_clean \
        --version 8 --max 800 --workers 4

Reads Kaggle creds from $KAGGLE_USERNAME/$KAGGLE_KEY or ~/.kaggle/kaggle.json.
"""
import argparse, io, json, os, re, sys, threading, time, random, urllib.parse, zipfile
import pandas as pd, numpy as np, requests
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

OWNER, DNAME = "giakop", "sagi-d"


def load_creds():
    u, k = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if u and k:
        return u, k
    p = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(p):
        d = json.load(open(p))
        return d["username"], d["key"]
    sys.exit("No Kaggle creds: set KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json")


def clean(p):
    if not isinstance(p, str):
        return None
    p = p.replace("\\", "/")
    return p[6:] if p.startswith("sagid/") else p


_tl = threading.local()


def session(auth):
    if not hasattr(_tl, "s"):
        _tl.s = requests.Session()
        _tl.s.auth = auth
    return _tl.s


def fetch(rel, version, auth):
    q = urllib.parse.quote(rel, safe="")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{OWNER}/{DNAME}/{q}?datasetVersionNumber={version}"
    for attempt in range(6):
        try:
            r = session(auth).get(url, timeout=20)
            if r.status_code == 429:
                time.sleep(min(60, 2 * 2 ** attempt) + random.random())
                continue
            r.raise_for_status()
            data = r.content
            if data.startswith(b"PK\x03\x04"):
                with zipfile.ZipFile(io.BytesIO(data)) as z:
                    return z.read(z.namelist()[0])
            return data
        except Exception as e:
            if attempt == 5:
                print(f"  [fail] {rel}: {e}")
            else:
                time.sleep(1.5 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--trained-modified", required=True,
                    help="dir of your trained triplet_*.jpg (to exclude)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", default="8")
    ap.add_argument("--max", type=int, default=0, help="cap # clean val triplets (0=all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="which official SAGI-D split to draw the clean eval set from. "
                         "'test' is the formal reporting split; openimages (OOD) lives "
                         "ONLY in test.")
    ap.add_argument("--source-filter", default="",
                    help="comma list of sources to keep (coco,raise,openimages). Empty=all. "
                         "Use 'openimages' for OOD-only, 'coco,raise' for in-domain.")
    ap.add_argument("--type-filter", default="",
                    help="comma list of types to keep (sp,fr). Empty=all. sp=spliced "
                         "(localized, _Blended), fr=full-regeneration (_None). For a "
                         "localization model, eval 'sp' separately from 'fr'.")
    ap.add_argument("--by-type", action="store_true",
                    help="route each triplet into <out>/<type>/{modified,original,mask} "
                         "(sp/ and fr/ subfolders) instead of a flat layout. With --max, "
                         "the cap is applied PER type so both subfolders fill.")
    ap.add_argument("--present-only", action="store_true",
                    help="exclude ONLY the triplet files physically present (legacy). "
                         "Default excludes the full contiguous span [min..max], which is "
                         "correct for a single uninterrupted download and immune to a "
                         "truncated/partial copy of the trained dir.")
    args = ap.parse_args()
    auth = load_creds()

    df = pd.read_csv(args.csv)
    df["clean_img"] = df["img_path"].apply(clean)
    df["clean_msk"] = df["mask_path"].apply(clean)
    df["clean_src"] = df["src_path"].apply(clean)
    df = df.dropna(subset=["clean_img", "clean_msk", "clean_src"])
    cand = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # rows you trained on = candidates at your local triplet indices
    idxs = []
    for f in os.listdir(args.trained_modified):
        m = re.match(r"triplet_(\d+)\.", f)
        if m:
            idxs.append(int(m.group(1)))
    idxs = sorted(idxs)
    if not idxs:
        sys.exit(f"No triplet_*.jpg in {args.trained_modified}")
    lo, hi = idxs[0], idxs[-1]
    holes = (hi - lo + 1) - len(idxs)
    if args.present_only:
        sel = idxs
        mode = f"present files only ({len(idxs)} rows)"
    else:
        # single uninterrupted run -> trained set is the contiguous span [lo..hi];
        # any holes are genuine download skips, not untrained rows. Excluding the
        # whole span is robust to a truncated/partial copy of the trained dir.
        sel = list(range(lo, hi + 1))
        mode = f"contiguous span [{lo}..{hi}] = {len(sel)} rows"
    trained_keys = set(cand.iloc[sel]["img_path"])
    print(f"Trained subset: {len(idxs)} files present, span [{lo}..{hi}] "
          f"({holes} holes). Excluding {mode}.")

    pool = df[df["split"] == args.split].copy()
    pool["source"] = pool["img_path"].apply(lambda p: p.replace("\\", "/").split("/")[2])
    if args.source_filter:
        keep = {s.strip() for s in args.source_filter.split(",") if s.strip()}
        pool = pool[pool["source"].isin(keep)]
        print(f"Source filter: {sorted(keep)}")
    if args.type_filter:
        keept = {s.strip() for s in args.type_filter.split(",") if s.strip()}
        pool = pool[pool["type"].isin(keept)]
        print(f"Type filter: {sorted(keept)}  (sp=spliced/localized, fr=full-regen)")
    clean_val = pool[~pool["img_path"].isin(trained_keys)]
    leaked = len(pool) - len(clean_val)
    print(f"Official {args.split} rows: {len(pool)} | leaked into training: {leaked} | "
          f"clean available: {len(clean_val)}")
    print("  by source:", clean_val["source"].value_counts().to_dict())
    print("  by type:  ", clean_val["type"].value_counts().to_dict())

    # deterministic order for reproducible capped subsets
    clean_val = clean_val.sample(frac=1, random_state=123).reset_index(drop=True)
    if args.max:
        if args.by_type:
            clean_val = clean_val.groupby("type", group_keys=False).head(args.max)
        else:
            clean_val = clean_val.iloc[:args.max]
    print(f"Downloading {len(clean_val)} clean triplets -> {args.out}"
          + (" (split into sp/ fr/ subfolders)" if args.by_type else ""))

    # --by-type routes each triplet into <out>/<type>/{modified,original,mask};
    # otherwise everything goes flat under <out>/{modified,original,mask}.
    def dirs_for(row):
        base = os.path.join(args.out, row["type"]) if args.by_type else args.out
        return {k: os.path.join(base, k) for k in ("modified", "original", "mask")}

    type_roots = sorted(clean_val["type"].unique()) if args.by_type else [None]
    for tr in type_roots:
        base = os.path.join(args.out, tr) if tr else args.out
        for k in ("modified", "original", "mask"):
            os.makedirs(os.path.join(base, k), exist_ok=True)

    def work(row):
        dirs = dirs_for(row)
        stem = os.path.splitext(os.path.basename(row["clean_img"]))[0]
        if os.path.exists(os.path.join(dirs["modified"], stem + ".jpg")):
            return "skip", row
        ib = fetch(row["clean_img"], args.version, auth)
        sb = fetch(row["clean_src"], args.version, auth)
        mb = fetch(row["clean_msk"], args.version, auth)
        if not (ib and sb and mb):
            return "fail", row
        try:
            im = np.array(Image.open(io.BytesIO(ib)).convert("RGB"))
            sr = np.array(Image.open(io.BytesIO(sb)).convert("RGB"))
            mk = np.array(Image.open(io.BytesIO(mb)).convert("L"))
        except Exception:
            return "fail", row
        if sr.shape[:2] != im.shape[:2]:
            sr = np.array(Image.fromarray(sr).resize((im.shape[1], im.shape[0])))
        if mk.shape[:2] != im.shape[:2]:
            mk = np.array(Image.fromarray(mk).resize((im.shape[1], im.shape[0])))
        Image.fromarray(im).save(os.path.join(dirs["modified"], stem + ".jpg"), quality=95)
        Image.fromarray(sr).save(os.path.join(dirs["original"], stem + ".jpg"), quality=95)
        Image.fromarray(mk).save(os.path.join(dirs["mask"], stem + ".png"))
        return "ok", row

    ok = fail = skip = 0
    manifest = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r for _, r in clean_val.iterrows()}
        for i, fut in enumerate(as_completed(futs), 1):
            status, row = fut.result()
            ok += status == "ok"; fail += status == "fail"; skip += status == "skip"
            if status in ("ok", "skip"):
                manifest.append({
                    "stem": os.path.splitext(os.path.basename(row["clean_img"]))[0],
                    "split": row["split"],
                    "source": row["img_path"].replace("\\", "/").split("/")[2],
                    "type": row.get("type"),
                    "inpainting_model": row.get("inpainting_model"),
                    "img_path": row["img_path"],
                })
            if i % 50 == 0:
                print(f"  {i}/{len(futs)}  ok={ok} skip={skip} fail={fail}")

    pd.DataFrame(manifest).to_csv(os.path.join(args.out, "clean_val_manifest.csv"), index=False)
    print(f"\nDone. ok={ok} skip={skip} fail={fail}")
    print(f"Layout: {args.out}/{{modified,original,mask}}  +  clean_val_manifest.csv")
    print("Point your eval at it with --sagid_root " + args.out)


if __name__ == "__main__":
    main()
