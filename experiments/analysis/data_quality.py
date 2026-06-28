"""Data-quality track (Reviewer 2, comments 4 & 6): the negative-set mismatch.

The experts train on DINUCLEOTIDE-SHUFFLE negatives (utils/data.ChipDataLoader.dinucshuffle)
but are evaluated on REAL curated genomic negatives (the ``_B`` sets). Tourne et al.
(Bioinformatics 2026, btag048) show this shuffle-train -> real-negative-test mismatch
systematically depresses AUC. This module:

  1. DIAGNOSTIC (runs now, no genome needed): quantifies the distribution shift between
     dinucleotide-shuffle negatives and the real ``_B`` negatives (GC content + dinucleotide
     frequencies + a simple logistic "negative-type detector" AUC). A large gap is direct
     evidence for the reviewer point and motivates the fix.

  2. FIX (needs a genome FASTA): build_gc_matched_negatives() draws GC/length-matched real
     genomic windows as negatives so train and test negatives come from the same distribution.
     Retrain experts on these and report the OOD delta. Requires pyfaidx + an hg19/hg38 FASTA
     (single assembly; download from hgdownload.soe.ucsc.edu). Leakage control: exclude the 6
     OOD TFs' peak intervals and the ENCODE blacklist via ``exclude_bed``.

    python data_quality.py diagnose
    python data_quality.py build-negs --genome hg38.fa --tf ARID3A --n 17000 --out data/train_genomicneg
"""
import argparse
import csv
import glob
import gzip
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tfbs.data import ChipDataLoader

BASES = "ACGT"
DINUCS = [a + b for a in BASES for b in BASES]


def gc(seq):
    s = seq.upper()
    return (s.count("G") + s.count("C")) / max(1, len(s))


def dinuc_freq(seq):
    s = seq.upper()
    v = np.zeros(16)
    for i in range(len(s) - 1):
        d = s[i:i + 2]
        if d in DINUCS:
            v[DINUCS.index(d)] += 1
    return v / max(1, v.sum())


def _read_pos_neg(path):
    """Real positives + real negatives from a curated _B file (label col 3)."""
    pos, neg = [], []
    with gzip.open(path, "rt") as fh:
        next(fh)
        for row in csv.reader(fh, delimiter="\t"):
            (pos if int(row[3]) == 1 else neg).append(row[2])
    return pos, neg


def diagnose():
    """Compare dinuc-shuffle negatives (train scheme) vs real _B negatives (test scheme)."""
    np.random.seed(42)  # deterministic dinucleotide-shuffle negatives -> reproducible detector AUC
    cdl = ChipDataLoader("")
    rows = []
    for f in sorted(glob.glob("./data/test/*_B.seq.gz") + glob.glob("./data/ood/*_B.seq.gz")):
        tf = os.path.basename(f).split("_")[0]
        pos, real_neg = _read_pos_neg(f)
        shuf_neg = [cdl.dinucshuffle(s) for s in pos]  # the train-time negative scheme
        gc_real = np.mean([gc(s) for s in real_neg]); gc_shuf = np.mean([gc(s) for s in shuf_neg])
        # detector: can a logistic model tell shuffle-neg from real-neg by dinuc freq?
        X = np.array([dinuc_freq(s) for s in real_neg + shuf_neg])
        y = np.array([0] * len(real_neg) + [1] * len(shuf_neg))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        det = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        det_auc = roc_auc_score(yte, det.predict_proba(Xte)[:, 1])
        rows.append((tf, gc_real, gc_shuf, gc_real - gc_shuf, det_auc))
    print(f"{'TF':10s} {'GC_real':>8s} {'GC_shuf':>8s} {'dGC':>7s} {'detector_AUC':>13s}")
    for tf, gr, gs, dg, da in rows:
        print(f"{tf:10s} {gr:8.3f} {gs:8.3f} {dg:+7.3f} {da:13.3f}")
    mean_auc = np.mean([r[4] for r in rows])
    print(f"\nMean negative-type detector AUC = {mean_auc:.3f} "
          f"(1.0 = shuffle & real negatives are trivially separable -> train/test mismatch).")
    print("Implication: experts trained to reject dinuc-shuffle negatives face a DIFFERENT "
          "negative distribution at test; GC-matched genomic negatives close this gap.")
    import csv
    os.makedirs("results/analysis", exist_ok=True)
    out = "results/analysis/negative_set_diagnostic.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tf", "GC_real", "GC_shuf", "dGC", "detector_AUC"])
        for tf, gr, gs, dg, da in rows:
            w.writerow([tf, f"{gr:.4f}", f"{gs:.4f}", f"{dg:+.4f}", f"{da:.4f}"])
        w.writerow(["MEAN", "", "", "", f"{mean_auc:.4f}"])
    print(f"wrote {out}")
    return rows


def build_gc_matched_negatives(genome_fa, pos_seqs, n, length=101, gc_tol=0.03,
                               exclude_bed=None, seed=42, max_tries_factor=200):
    """Sample n real genomic windows whose GC matches the positive GC histogram.
    Requires pyfaidx and a genome FASTA. exclude_bed: BED of intervals to avoid (OOD-TF
    peaks + ENCODE blacklist) for leakage control."""
    from pyfaidx import Fasta  # noqa: requires `pip install pyfaidx`
    rng = np.random.default_rng(seed)
    genome = Fasta(genome_fa)
    chroms = [c for c in genome.keys() if "_" not in c and c not in ("chrM", "chrEBV")]
    target_gc = np.array([gc(s) for s in pos_seqs])
    excl = _load_bed(exclude_bed) if exclude_bed else {}
    negs, tries = [], 0
    while len(negs) < n and tries < n * max_tries_factor:
        tries += 1
        c = chroms[rng.integers(len(chroms))]
        L = len(genome[c]); start = int(rng.integers(0, max(1, L - length)))
        if _overlaps(excl.get(c, []), start, start + length):
            continue
        s = str(genome[c][start:start + length]).upper()
        if "N" in s or len(s) != length:
            continue
        want = target_gc[rng.integers(len(target_gc))]
        if abs(gc(s) - want) <= gc_tol:
            negs.append(s)
    return negs


def _load_bed(path):
    d = {}
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for line in fh:
            p = line.split("\t")
            if len(p) >= 3:
                d.setdefault(p[0], []).append((int(p[1]), int(p[2])))
    for c in d:
        d[c].sort()
    return d


def _overlaps(intervals, a, b):
    for s, e in intervals:
        if s < b and a < e:
            return True
        if s >= b:
            break
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("diagnose")
    bp = sub.add_parser("build-negs")
    bp.add_argument("--genome", required=True); bp.add_argument("--tf", required=True)
    bp.add_argument("--n", type=int, default=17000); bp.add_argument("--out", default="./data/train_genomicneg")
    bp.add_argument("--exclude_bed", default=None)
    args = ap.parse_args()

    if args.cmd == "diagnose":
        diagnose()
    else:
        f = glob.glob(f"./data/train/{args.tf}_*_AC.seq.gz")[0]
        pos = [r[2] for r in ChipDataLoader(f).load_data_with_seq(shuffle=False)]
        negs = build_gc_matched_negatives(args.genome, pos, args.n, exclude_bed=args.exclude_bed)
        os.makedirs(args.out, exist_ok=True)
        op = os.path.join(args.out, os.path.basename(f))
        with gzip.open(op, "wt") as out:
            out.write("idx\tsource\tseq\tBound\n")
            for i, s in enumerate(pos):
                out.write(f"{i}\tpos\t{s}\t1\n")
            for i, s in enumerate(negs):
                out.write(f"{i}\tgcmatched\t{s}\t0\n")
        print(f"wrote {len(pos)} pos + {len(negs)} GC-matched negs -> {op}")


if __name__ == "__main__":
    main()
