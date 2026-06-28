"""Data-driven justification for the family-aware OOD split.

For every training and OOD TF, build an average k-mer frequency profile over its
positive sequences, then compute the cosine similarity of each OOD TF to each
training TF. The claim the split rests on -- that within-DBD-family held-out TFs
are sequence-closer to a training TF than cross-family ones -- is testable here:
within-family OOD should have higher max-similarity-to-train than cross-family,
which in turn should beat the non-motif stratum.

    python -m experiments.analysis.family_similarity   # -> results/analysis/family_similarity.{csv,md}
"""
import gzip
import os
from collections import Counter
from itertools import product

import numpy as np
import pandas as pd

from tfbs.constants import (TRAIN_TFS, OOD_WITHIN_FAMILY, OOD_CROSS_FAMILY,
                            OOD_NONMOTIF, DBD_FAMILY, TRAIN_DIR, OOD_DIR)
from tfbs.utils import get_tf_name, load_files_from_folder

K = 4
KMERS = ["".join(p) for p in product("ACGT", repeat=K)]
KIDX = {k: i for i, k in enumerate(KMERS)}


def _profile(path, max_seqs=500):
    """Mean L1-normalised k-mer frequency vector over the positive sequences."""
    vecs = []
    with gzip.open(path, "rt") as fh:
        next(fh, None)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            seq, bound = parts[2].upper(), parts[3]
            if bound != "1":
                continue
            v = np.zeros(len(KMERS))
            for i in range(len(seq) - K + 1):
                j = KIDX.get(seq[i:i + K])
                if j is not None:
                    v[j] += 1
            s = v.sum()
            if s > 0:
                vecs.append(v / s)
            if len(vecs) >= max_seqs:
                break
    return np.mean(vecs, axis=0) if vecs else None


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    out = "results/analysis"
    os.makedirs(out, exist_ok=True)
    train_files = {get_tf_name(f): f for f in load_files_from_folder(str(TRAIN_DIR))}
    ood_files = {get_tf_name(f): f for f in load_files_from_folder(str(OOD_DIR))}

    train_prof = {tf: _profile(train_files[tf]) for tf in TRAIN_TFS if tf in train_files}
    rows = []
    for stratum, tfs in [("within_family", OOD_WITHIN_FAMILY),
                         ("cross_family", OOD_CROSS_FAMILY),
                         ("non_motif", OOD_NONMOTIF)]:
        for tf in tfs:
            if tf not in ood_files:
                continue
            p = _profile(ood_files[tf])
            if p is None:
                continue
            sims = {t: _cos(p, train_prof[t]) for t in train_prof}
            best_t = max(sims, key=sims.get)
            # is there a same-family training TF, and is it the closest?
            fam = DBD_FAMILY.get(tf)
            same_fam_train = [t for t in train_prof if DBD_FAMILY.get(t) == fam]
            rows.append({
                "tf": tf, "family": fam, "stratum": stratum,
                "max_sim_to_train": max(sims.values()),
                "closest_train_tf": best_t,
                "closest_is_same_family": best_t in same_fam_train,
                **{f"sim_{t}": round(sims[t], 4) for t in TRAIN_TFS if t in sims},
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "family_similarity.csv"), index=False)

    L = ["# Train-vs-OOD k-mer (4-mer) profile similarity", "",
         f"Cosine similarity of each OOD TF's positive-sequence {K}-mer profile to each "
         "training TF. Higher = sequence-closer (a proxy for motif-family relatedness).", ""]
    for stratum in ["within_family", "cross_family", "non_motif"]:
        sub = df[df.stratum == stratum]
        if sub.empty:
            continue
        L.append(f"## {stratum}  (mean max-sim-to-train = {sub.max_sim_to_train.mean():.3f})")
        L.append("| OOD TF | family | max sim | closest train | same family? |")
        L.append("|---|---|---|---|---|")
        for _, r in sub.iterrows():
            L.append(f"| {r.tf} | {r.family} | {r.max_sim_to_train:.3f} | "
                     f"{r.closest_train_tf} | {'yes' if r.closest_is_same_family else 'no'} |")
        L.append("")
    L.append("## Summary")
    for stratum in ["within_family", "cross_family", "non_motif"]:
        sub = df[df.stratum == stratum]
        if not sub.empty:
            L.append(f"- **{stratum}**: mean max-sim {sub.max_sim_to_train.mean():.3f}; "
                     f"closest-train-TF-is-same-family for {int(sub.closest_is_same_family.sum())}/{len(sub)}")
    with open(os.path.join(out, "family_similarity.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
