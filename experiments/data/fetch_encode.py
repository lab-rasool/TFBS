"""Fetch the exact ENCODE-690 experiments for the family-aware split from
``huggingface.co/datasets/Lab-Rasool/ENCODE-TFBS`` into ``data/{train,test,ood}/``.

Driven entirely by ``tfbs.constants`` so the data selection is reproducible and
cell-line controlled. For each TF we pick the experiment in ``PREFERRED_CELL[tf]``
(falling back to the most-replicated cell line), download its ``_AC`` (positives,
training) and/or ``_B`` (balanced 500/500 curated test) file, and pin the exact
filename + cell line + antibody + lab in ``data/encode_manifest.json``.

Roles:
  * TRAIN_TFS                          -> _AC into data/train/, _B into data/test/
  * OOD (within+cross family, nonmotif)-> _B into data/ood/

The OOD_CELLLINE stratum (same TF, unseen line) is handled separately (it needs
TF@CELL keying) and is NOT fetched here.

Run from the repo root:
    python -m experiments.data.fetch_encode            # fetch missing files
    python -m experiments.data.fetch_encode --refetch  # re-pick + overwrite OOD
    python -m experiments.data.fetch_encode --dry_run   # just print the plan
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

from huggingface_hub import HfApi, hf_hub_download

from tfbs.constants import (
    TRAIN_TFS, OOD_WITHIN_FAMILY, OOD_CROSS_FAMILY, OOD_NONMOTIF,
    PREFERRED_CELL, DBD_FAMILY, DATA_DIR, TRAIN_DIR, TEST_DIR, OOD_DIR,
)

REPO = "Lab-Rasool/ENCODE-TFBS"
PREF_ORDER = ["K562", "GM12878", "HepG2", "H1-hESC", "HeLa-S3", "A549", "HUVEC"]


def _inventory():
    """{tf: {cell: {'AC': fname, 'B': fname, 'antibody': str, 'lab': str}}}."""
    api = HfApi()
    files = api.list_repo_files(REPO, repo_type="dataset")
    inv = defaultdict(lambda: defaultdict(dict))
    for f in files:
        if not f.endswith(".seq.gz"):
            continue
        role = "AC" if f.endswith("_AC.seq.gz") else "B" if f.endswith("_B.seq.gz") else None
        if role is None:
            continue
        base = f[: -len(f"_{role}.seq.gz")]
        parts = base.split("_")
        tf, cell = parts[0], parts[1]
        antibody = parts[2] if len(parts) > 2 else ""
        lab = parts[-1] if len(parts) > 3 else ""
        inv[tf][cell][role] = f
        inv[tf][cell]["antibody"] = antibody
        inv[tf][cell]["lab"] = lab
    return inv


def _pick_cell(tf, cells, need):
    """Pick the cell line for ``tf`` that has the required role(s); prefer the
    designated PREFERRED_CELL, then the common-line order, then most-replicated."""
    want = PREFERRED_CELL.get(tf)
    ok = [c for c in cells if all(r in cells[c] for r in need)]
    if not ok:
        return None
    if want in ok:
        return want
    for c in PREF_ORDER:
        if c in ok:
            return c
    return sorted(ok)[0]


def _download(fname, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(fname))
    if os.path.exists(dest):
        return dest, False
    cached = hf_hub_download(REPO, fname, repo_type="dataset")
    shutil.copy(cached, dest)
    return dest, True


def _existing_tf(folder):
    if not os.path.isdir(folder):
        return set()
    return {f.split("_")[0] for f in os.listdir(folder) if f.endswith(".seq.gz")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refetch", action="store_true",
                    help="clear data/ood and re-pick all OOD files")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    inv = _inventory()
    manifest = {}
    plan = []  # (tf, role, dest_dir, fname, cell)

    # ---- training TFs: _AC -> train, _B -> test ----
    have_train = _existing_tf(TRAIN_DIR)
    for tf in TRAIN_TFS:
        if tf in have_train:  # keep the original curated training files as-is
            manifest[tf] = {"role": "train", "status": "preexisting"}
            continue
        if tf not in inv:
            print(f"!! {tf}: not in dataset"); continue
        cell = _pick_cell(tf, inv[tf], need=("AC", "B"))
        if cell is None:
            print(f"!! {tf}: no experiment with both _AC and _B"); continue
        e = inv[tf][cell]
        plan.append((tf, "AC", str(TRAIN_DIR), e["AC"], cell))
        plan.append((tf, "B", str(TEST_DIR), e["B"], cell))
        manifest[tf] = {"role": "train", "cell": cell, "family": DBD_FAMILY.get(tf),
                        "AC": e["AC"], "B": e["B"], "antibody": e["antibody"], "lab": e["lab"]}

    # ---- OOD TFs: _B -> ood ----
    if args.refetch and OOD_DIR.exists() and not args.dry_run:
        bak = str(OOD_DIR) + "_old"
        if not os.path.exists(bak):
            shutil.move(str(OOD_DIR), bak)
            print(f"  moved old OOD -> {bak}")
    have_ood = set() if args.refetch else _existing_tf(OOD_DIR)
    for tf in OOD_WITHIN_FAMILY + OOD_CROSS_FAMILY + OOD_NONMOTIF:
        if tf in have_ood:
            manifest[tf] = {"role": "ood", "status": "preexisting"}; continue
        if tf not in inv:
            print(f"!! {tf}: not in dataset"); continue
        cell = _pick_cell(tf, inv[tf], need=("B",))
        if cell is None:
            print(f"!! {tf}: no _B file"); continue
        e = inv[tf][cell]
        stratum = ("within_family" if tf in OOD_WITHIN_FAMILY else
                   "cross_family" if tf in OOD_CROSS_FAMILY else "nonmotif")
        plan.append((tf, "B", str(OOD_DIR), e["B"], cell))
        manifest[tf] = {"role": "ood", "stratum": stratum, "cell": cell,
                        "family": DBD_FAMILY.get(tf), "B": e["B"],
                        "antibody": e["antibody"], "lab": e["lab"]}

    print(f"\nPlan: {len(plan)} files to download "
          f"({sum(1 for p in plan if p[2]==str(TRAIN_DIR))} train, "
          f"{sum(1 for p in plan if p[2]==str(TEST_DIR))} test, "
          f"{sum(1 for p in plan if p[2]==str(OOD_DIR))} ood)")
    for tf, role, dd, fname, cell in plan:
        print(f"  {tf:8s} {role:2s} {cell:10s} -> {os.path.basename(dd)}/  {os.path.basename(fname)}")
    if args.dry_run:
        return

    n = 0
    for tf, role, dd, fname, cell in plan:
        _, fetched = _download(fname, dd)
        n += int(fetched)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DATA_DIR / "encode_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDownloaded {n} new files; manifest -> {DATA_DIR/'encode_manifest.json'}")


if __name__ == "__main__":
    main()
