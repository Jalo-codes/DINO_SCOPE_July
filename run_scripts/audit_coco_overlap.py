# ============================================================
#  COCO-overlap audit  —  "what can we honestly call OOD?"
#
#  The InpaintCOCO downloader renamed every triplet to triplet_NNNNN and
#  discarded the coco_id. But phiyodr/InpaintCOCO has a SINGLE 'test' split
#  (1260 rows) and the downloader assigned global_idx in row order, so
#       triplet_i  <->  test-row i  <->  coco_details.coco_url  <->  coco_id.
#  We re-stream the dataset (metadata only, images not decoded) to rebuild that
#  map, recompute the EXACT train/val split inpaint.build() uses (seed 42, 10%),
#  and intersect coco_ids against the tgif2 OOD index (which IS keyed by coco_id).
#
#  Output: how many COCO scenes in the TGIF/FLUX "OOD" eval were also seen in
#  training via coco_inpaint — i.e. the contamination that weakens an OOD claim.
#
#  Usage (Colab/box, where the data lives):
#    python run_scripts/audit_coco_overlap.py \
#       --coco_inpaint_root /content/inpaint_coco/images \
#       --tgif2_index       /content/dataset_root/content/flux_originals/tgif2_index.json \
#       --out               /content/coco_overlap_report
#  Box paths:
#       --coco_inpaint_root /media/ssd/DINO_SCOPE_DATA/INPAINT_COCO/content/inpaint_coco/images
#       --tgif2_index       /media/ssd/DINO_SCOPE_DATA/content/flux_originals/tgif2_index.json
# ============================================================

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path


def _pip(*p):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *p])


# ── coco_id helpers ──────────────────────────────────────────────────────────

_URL_RE = re.compile(r"/(\d+)\.jpg", re.IGNORECASE)


def _coco_id_from_url(url):
    """http://images.cocodataset.org/val2017/000000397133.jpg -> (397133, 'val2017')."""
    if not url:
        return None, None
    m = _URL_RE.search(url)
    if not m:
        return None, None
    subset = None
    parts = url.rstrip("/").split("/")
    for p in parts:
        if p in ("val2017", "train2017", "test2017", "val2014", "train2014"):
            subset = p
            break
    return int(m.group(1)), subset


def _norm_id(x):
    """Normalize a coco_id (int / '397133' / '000000397133' / url) -> int."""
    if isinstance(x, int):
        return x
    s = str(x)
    cid, _ = _coco_id_from_url(s)
    if cid is not None:
        return cid
    digits = re.sub(r"\D", "", s)
    return int(digits) if digits else None


def _get_coco_url(details):
    """coco_details may be a dict or a JSON string; pull coco_url out of it."""
    if details is None:
        return None
    if isinstance(details, str):
        try:
            details = json.loads(details)
        except Exception:
            m = re.search(r"http[s]?://\S+?\.jpg", details)
            return m.group(0) if m else None
    if isinstance(details, dict):
        return details.get("coco_url") or details.get("url")
    return None


# ── 1. Recover triplet_i -> coco_id by re-streaming InpaintCOCO ──────────────

def recover_inpaint_coco_ids():
    _pip("datasets", "huggingface_hub")
    from datasets import load_dataset
    print("[INFO] Streaming phiyodr/InpaintCOCO test split (metadata only)...")
    ds = load_dataset("phiyodr/InpaintCOCO", split="test", streaming=True,
                      trust_remote_code=True)
    row_to_id = {}            # row index -> coco_id
    subsets = {}
    for i, ex in enumerate(ds):
        url = _get_coco_url(ex.get("coco_details"))
        cid, subset = _coco_id_from_url(url)
        row_to_id[i] = cid
        if subset:
            subsets[subset] = subsets.get(subset, 0) + 1
    print(f"   [OK] recovered {sum(v is not None for v in row_to_id.values())}"
          f"/{len(row_to_id)} coco_ids. COCO subsets: {subsets}")
    return row_to_id


# ── 2. Recompute the EXACT inpaint.build train/val split ─────────────────────

def _clean_name(filename):
    stem = os.path.splitext(filename)[0]
    for suf in ("_modified", "_original", "_orig", "_mask", "_fake", "_real",
                "_inpainted", "_gt"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def split_triplets(coco_inpaint_root, val_split=0.10, split_seed=42):
    """Mirror lab_utils/data/datasets/inpaint.build's split exactly."""
    root = Path(coco_inpaint_root)
    mod, org, msk = root / "modified", root / "original", root / "mask"
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def index(folder, extra=()):
        return {_clean_name(f.name): f for f in sorted(folder.iterdir())
                if f.is_file() and f.suffix.lower() in (set(exts) | set(extra))}

    mods = index(mod)
    origs = index(org)
    masks = index(msk, extra={".png"})
    bases = sorted(set(mods) & set(origs) & set(masks))

    rng = random.Random(int(split_seed))
    shuffled = list(bases)
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * float(val_split))
    val_bases = set(shuffled[:n_val])
    train_bases = [b for b in bases if b not in val_bases]
    print(f"[INFO] coco_inpaint triplets: {len(bases)} "
          f"(train={len(train_bases)}, val={len(val_bases)}; seed={split_seed}, "
          f"val_split={val_split})")
    return bases, train_bases, sorted(val_bases)


def _triplet_index(base):
    """'triplet_00037' -> 37 (the InpaintCOCO test-split row index)."""
    m = re.search(r"(\d+)", base)
    return int(m.group(1)) if m else None


# ── 3. Load tgif2 OOD coco_ids ───────────────────────────────────────────────

def load_tgif2_ids(tgif2_index):
    with open(tgif2_index) as fh:
        index = json.load(fh)
    ids = set()
    for k in index.keys():
        nid = _norm_id(k)
        if nid is not None:
            ids.add(nid)
    print(f"[INFO] tgif2 OOD index: {len(index)} coco_ids ({len(ids)} normalized)")
    return ids


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Audit COCO-scene overlap between "
                                             "coco_inpaint (train) and tgif2 (OOD eval).")
    ap.add_argument("--coco_inpaint_root", required=True,
                    help="the images/ parent holding modified/ original/ mask/")
    ap.add_argument("--tgif2_index", required=True,
                    help="path to tgif2_index.json")
    ap.add_argument("--val_split", type=float, default=0.10)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--out", default="coco_overlap_report",
                    help="output dir for JSON/CSV report")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    row_to_id = recover_inpaint_coco_ids()
    bases, train_bases, val_bases = split_triplets(
        args.coco_inpaint_root, args.val_split, args.split_seed)
    tgif_ids = load_tgif2_ids(args.tgif2_index)

    def ids_for(base_list):
        out_ids, missing = {}, 0
        for b in base_list:
            ridx = _triplet_index(b)
            cid = row_to_id.get(ridx) if ridx is not None else None
            if cid is None:
                missing += 1
                continue
            out_ids[b] = cid
        return out_ids, missing

    all_map, miss_all = ids_for(bases)
    train_map, _ = ids_for(train_bases)
    val_map, _ = ids_for(val_bases)

    all_ids = set(all_map.values())
    train_ids = set(train_map.values())
    val_ids = set(val_map.values())

    ov_all = all_ids & tgif_ids
    ov_train = train_ids & tgif_ids
    ov_val = val_ids & tgif_ids
    train_val_dup = train_ids & val_ids   # should be empty

    print("\n" + "=" * 60)
    print("  COCO-overlap audit")
    print("=" * 60)
    print(f"  coco_inpaint unique coco_ids : {len(all_ids)}"
          f"   (unrecovered triplets: {miss_all})")
    print(f"  tgif2 (OOD) unique coco_ids  : {len(tgif_ids)}")
    print("-" * 60)
    print(f"  OVERLAP  coco_inpaint ∩ tgif2          : {len(ov_all)}")
    print(f"    ├─ via coco_inpaint TRAIN ∩ tgif2    : {len(ov_train)}  "
          f"<-- weakens OOD claim")
    print(f"    └─ via coco_inpaint VAL   ∩ tgif2    : {len(ov_val)}")
    print(f"  sanity: coco_inpaint train ∩ val       : {len(train_val_dup)}  "
          f"(expect 0)")
    if len(all_ids):
        print(f"  => {100.0 * len(ov_all) / len(all_ids):.1f}% of coco_inpaint "
              f"scenes reappear in the tgif2 OOD eval")
    if len(tgif_ids):
        print(f"  => {100.0 * len(ov_train) / len(tgif_ids):.1f}% of tgif2 OOD "
              f"scenes were SEEN in training (via coco_inpaint train)")
    print("=" * 60)

    report = {
        "coco_inpaint_unique_ids": len(all_ids),
        "tgif2_unique_ids": len(tgif_ids),
        "overlap_all": sorted(ov_all),
        "overlap_train": sorted(ov_train),
        "overlap_val": sorted(ov_val),
        "train_val_dup": sorted(train_val_dup),
        "unrecovered_triplets": miss_all,
        "params": {"val_split": args.val_split, "split_seed": args.split_seed},
    }
    with open(out / "overlap_report.json", "w") as fh:
        json.dump(report, fh, indent=2)

    # CSV of the train-side leak (triplet -> coco_id) for spot-checking.
    import csv
    with open(out / "train_leak_triplets.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["triplet", "coco_id", "in_tgif2_ood"])
        for b, cid in sorted(train_map.items()):
            if cid in tgif_ids:
                w.writerow([b, cid, 1])
    print(f"[DONE] wrote {out}/overlap_report.json and train_leak_triplets.csv")


if __name__ == "__main__":
    main()
