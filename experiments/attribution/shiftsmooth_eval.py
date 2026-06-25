"""Controlled attribution study for the ShiftSmooth method (Reviewer 2, Comment 8 /
interpretability). Benchmarks a ladder of attribution methods on the differentiable
ConvNet experts, with and without the Koo-lab simplex gradient correction, and reports
quantitative faithfulness + stability rather than qualitative heatmaps.

Methods (all reduced to a per-position importance over the 101 bp core):
  vanilla  : saliency |dscore/dx|
  gradxinp : grad x input  (first-order approximation of ISM)
  smoothgrad: mean gradient over Gaussian-noised inputs (SmoothGrad, Smilkov 2017)
  shiftsmooth: mean gradient over small TRANSLATIONS of the input (kept valid one-hot,
               motivated by CNN translation-equivariance / motif-position uncertainty)
  ISM      : in-silico mutagenesis -- the genomics faithfulness gold standard

Gradient correction (Majdandzic et al., Genome Biology 2023): subtract the per-position
mean across A/C/G/T to project the gradient onto the simplex tangent (removes off-simplex
noise). Reported as the `_corr` variant of every gradient method.

Metrics:
  faithfulness  = Spearman corr of the method's importance to ISM (higher = better)
  stability     = mean attribution correlation under small input perturbation
                  (higher = more robust; ShiftSmooth's core selling point)

    python shiftsmooth_eval.py --n_seqs 60 --shift 2 --noise 0.1 --reps 8
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from tfbs.data import ChipDataLoader
from tfbs.models import ConvNet
from tfbs.utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_TFS = ["ARID3A", "FOXM1", "GATA3"]
CORE0, CORE1 = 23, 124  # 101 bp core inside the 4x147 padded one-hot (motiflen-1=23 pad)
OUT_DIR = "./results/attribution"


def load_expert(tf):
    cfg = torch.load(f"./models/hyperparams/{tf}.pth")
    m = ConvNet(cfg).to(device)
    m.load_state_dict(torch.load(f"./models/experts/{tf}.pth", map_location=device))
    m.eval()
    return m


def score(model, x):
    return model(x, training=False).squeeze()


def raw_grad(model, x):
    x = x.clone().detach().requires_grad_(True)
    s = score(model, x)
    s.backward()
    return x.grad.detach()[0]  # (4,147)


def correct(g):  # Majdandzic simplex-tangent correction
    return g - g.mean(dim=0, keepdim=True)


def reduce_core(gmap, x0):
    """grad x input over the 4 bases, restricted to the 101 bp core -> (101,)."""
    gi = (gmap * x0[0]).sum(0)
    return gi[CORE0:CORE1].cpu().numpy()


def ism(model, x0):
    """In-silico mutagenesis importance over the core (score_orig - mean_mut score)."""
    base = score(model, x0).item()
    imp = np.zeros(CORE1 - CORE0)
    x0c = x0.clone()
    for j, pos in enumerate(range(CORE0, CORE1)):
        orig = x0c[0, :, pos].clone()
        deltas = []
        for b in range(4):
            if orig[b] > 0.5:
                continue
            x0c[0, :, pos] = 0.0
            x0c[0, b, pos] = 1.0
            deltas.append(base - score(model, x0c).item())
        x0c[0, :, pos] = orig
        imp[j] = np.mean(deltas) if deltas else 0.0
    return imp


def smoothgrad(model, x0, noise, reps, rng):
    acc = torch.zeros_like(x0[0])
    for _ in range(reps):
        xn = x0 + noise * torch.from_numpy(rng.standard_normal(x0.shape).astype(np.float32)).to(device)
        acc += raw_grad(model, xn)
    return acc / reps


def shiftsmooth(model, x0, shift):
    """Mean gradient over integer translations s in [-shift, shift] of the core, each a
    valid one-hot; gradients are translated back before averaging."""
    acc = torch.zeros_like(x0[0])
    shifts = list(range(-shift, shift + 1))
    for s in shifts:
        xs = x0.clone()
        xs[0, :, CORE0:CORE1] = torch.roll(x0[0, :, CORE0:CORE1], shifts=s, dims=1)
        g = raw_grad(model, xs)
        g[:, CORE0:CORE1] = torch.roll(g[:, CORE0:CORE1], shifts=-s, dims=1)
        acc += g
    return acc / len(shifts)


def method_maps(model, x0, args, rng):
    """Return {name: (4,147) gradient map} for the gradient-based methods."""
    return {
        "vanilla": raw_grad(model, x0),
        "smoothgrad": smoothgrad(model, x0, args.noise, args.reps, rng),
        "shiftsmooth": shiftsmooth(model, x0, args.shift),
    }


def evaluate_tf(tf, seqs, args):
    model = load_expert(tf)
    cdl = ChipDataLoader("")  # only used for seqtopad (no file is opened)
    rng = np.random.default_rng(args.seed)
    rows = []
    methods = ["vanilla", "smoothgrad", "shiftsmooth"]
    variants = [(m, corr) for m in methods for corr in (False, True)]
    agg = {f"{m}{'_corr' if c else ''}": {"faith": [], "stab": []} for m, c in variants}
    agg["gradxinp"] = {"faith": [], "stab": []}

    for s in seqs[: args.n_seqs]:
        x0 = torch.from_numpy(cdl.seqtopad(s)).float().unsqueeze(0).to(device)
        ism_imp = ism(model, x0)
        # perturbed copy for stability (tiny gaussian noise, renormalised not needed)
        x0p = x0 + 0.02 * torch.from_numpy(rng.standard_normal(x0.shape).astype(np.float32)).to(device)

        maps = method_maps(model, x0, args, rng)
        maps_p = method_maps(model, x0p, args, rng)
        # plain grad x input faithfulness (no smoothing)
        gi = reduce_core(maps["vanilla"], x0)
        agg["gradxinp"]["faith"].append(_safe_spear(gi, ism_imp))
        agg["gradxinp"]["stab"].append(_safe_spear(gi, reduce_core(maps_p["vanilla"], x0p)))

        for m, c in variants:
            name = f"{m}{'_corr' if c else ''}"
            g = correct(maps[m]) if c else maps[m]
            gp = correct(maps_p[m]) if c else maps_p[m]
            imp = reduce_core(g, x0)
            imp_p = reduce_core(gp, x0p)
            agg[name]["faith"].append(_safe_spear(imp, ism_imp))
            agg[name]["stab"].append(_safe_spear(imp, imp_p))

    for name, d in agg.items():
        rows.append({"tf": tf, "method": name,
                     "faithfulness_to_ISM": float(np.nanmean(d["faith"])),
                     "stability": float(np.nanmean(d["stab"])),
                     "n_seqs": len(d["faith"])})
    return rows


def _safe_spear(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return spearmanr(a, b).correlation


def positives_for(tf, n):
    """Positive (bound) sequences from the TF's curated _B test set."""
    import glob
    f = glob.glob(f"./data/test/{tf}_*_B.seq.gz")[0]
    rows = ChipDataLoader(f).load_data_with_seq(shuffle=False)
    return [s for x, y, s in rows if y[0] == 1][:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_seqs", type=int, default=60)
    ap.add_argument("--shift", type=int, default=2)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    all_rows = []
    for tf in TRAIN_TFS:
        seqs = positives_for(tf, args.n_seqs)
        print(f"[attrib] {tf}: {len(seqs)} positive sequences", flush=True)
        all_rows += evaluate_tf(tf, seqs, args)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "attribution_per_tf.csv"), index=False)
    summ = df.groupby("method")[["faithfulness_to_ISM", "stability"]].mean().reset_index()
    summ = summ.sort_values("faithfulness_to_ISM", ascending=False)
    summ.to_csv(os.path.join(OUT_DIR, "attribution_summary.csv"), index=False)
    print("\n=== Attribution method comparison (mean over 3 TFs) ===")
    print(summ.round(4).to_string(index=False))
    print(f"\nWrote {OUT_DIR}/attribution_summary.csv")
    print("Interpretation: ShiftSmooth should match/beat SmoothGrad on faithfulness_to_ISM "
          "and win on stability; the _corr (simplex-corrected) variants should improve both.")


if __name__ == "__main__":
    main()
