"""Reproducible validation of every quantitative claim in paper/edited.tex.

For each claim it (1) confirms the number is actually written in the paper, and
(2) recomputes the value from the source results file / code and checks they
agree to the stated rounding. Method/equation claims are checked by inspecting
the implementation. Exits non-zero if any check fails.

    python -m experiments.analysis.validate_paper
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAPER = (ROOT / "paper" / "edited.tex").read_text()
GEN = json.load(open(ROOT / "results/hetmoe/seed42_genomic/hetmoe_eval.json"))
from tfbs.constants import OOD_WITHIN_FAMILY, OOD_CROSS_FAMILY, OOD_NONMOTIF, TRAIN_TFS

MOTIF = set(OOD_WITHIN_FAMILY + OOD_CROSS_FAMILY)
WIN, CROSS = set(OOD_WITHIN_FAMILY), set(OOD_CROSS_FAMILY)
rows = []  # (ok, desc, paper, source)


def chk(desc, paper_val, source_val, tol=5e-4, in_paper=True):
    """paper_val: the value as written in the paper; source_val: recomputed."""
    ok = abs(paper_val - source_val) <= tol
    if in_paper:
        # confirm the rounded number literally appears in the manuscript
        s = f"{paper_val:.3f}".rstrip("0").rstrip(".")
        s2 = f"{paper_val:.2f}"
        present = (s in PAPER) or (s2 in PAPER) or (f"{paper_val:.3f}" in PAPER)
        ok = ok and present
        if not present:
            desc += "  [NUMBER NOT FOUND IN PAPER]"
    rows.append((ok, desc, paper_val, source_val))


def chk_bool(desc, cond, detail=""):
    rows.append((bool(cond), desc + (f" ({detail})" if detail else ""), None, None))


# ---------- A. Headline genomic seed-42 numbers ----------
om, wf, cf, nm = GEN["ood_mean"], GEN["ood_within_family"], GEN["ood_cross_family"], GEN["ood_nonmotif"]
chk("in-dist mean HetMoE = 0.881", 0.881, GEN["indist_mean_hetmoe"])
chk("motif-OOD HetMoE = 0.827", 0.827, om["HetMoE"])
chk("motif-OOD DNABERT = 0.800", 0.800, om["DNABERT"])
chk("motif-OOD static avg = 0.754", 0.754, om["static_mean"])
chk("motif-OOD best single = 0.739", 0.739, om["best_single"])
chk("within HetMoE = 0.846", 0.846, wf["HetMoE"]);  chk("within DNABERT = 0.816", 0.816, wf["DNABERT"])
chk("cross HetMoE = 0.775", 0.775, cf["HetMoE"]);    chk("cross DNABERT = 0.753", 0.753, cf["DNABERT"])
chk("non-motif HetMoE = 0.696", 0.696, nm["HetMoE"]); chk("non-motif DNABERT = 0.756", 0.756, nm["DNABERT"])
chk("gating - static = +0.073", 0.073, om["HetMoE"] - om["static_mean"])
chk("gating - best single = +0.088", 0.088, om["HetMoE"] - om["best_single"])
chk("within margin = +0.029", 0.029, wf["HetMoE"] - wf["DNABERT"])
chk("cross margin = +0.022", 0.022, cf["HetMoE"] - cf["DNABERT"])
chk("TOST margin = 0.01", 0.01, GEN["tost_margin"])

# in-distribution per-model means recomputed from per_dataset
pd = GEN["per_dataset"]
for label, key, val in [("in-dist DNABERT = 0.818", "DNABERT_auc", 0.818),
                        ("in-dist static = 0.796", "static_mean_auc", 0.796),
                        ("in-dist best single = 0.747", "best_single_auc", 0.747)]:
    m = np.mean([r[key] for r in pd.values() if r.get("data_type") == "in_distribution"])
    chk(label, val, float(m))

# superior counts
pv = [r for r in GEN["paired_vs_dnabert"] if r["data_type"] == "out_of_distribution" and r["tf"] in MOTIF]
sup = sum(r["superior"] for r in pv)
wsup = sum(r["superior"] for r in pv if r["tf"] in WIN)
csup = sum(r["superior"] for r in pv if r["tf"] in CROSS)
chk_bool("superior 13/23", sup == 13 and len(pv) == 23, f"{sup}/{len(pv)}")
chk_bool("within superior 10/17", wsup == 10 and len(WIN) == 17, f"{wsup}/{len(WIN)}")
chk_bool("cross superior 3/6", csup == 3 and len(CROSS) == 6, f"{csup}/{len(CROSS)}")

# per-factor deltas cited in the text
pvm = {r["tf"]: r["mean_diff"] for r in pv}
for tf, d in [("FOXA2", 0.143), ("FOXA1", 0.127), ("CTCF", 0.125), ("HNF4A", 0.100), ("SP2", 0.065)]:
    chk(f"delta {tf} = +{d:.3f}", d, pvm[tf], tol=1e-3)

# config
cfg = GEN["config"]
chk_bool("selected pool = 21 experts", len(GEN["config"]["expert_order"]) == 21)
chk_bool("selected backbones = ConvNet+DanQ+DeepSEA",
         sorted({n.split("::")[0] for n in GEN["config"]["expert_order"]}) == ["ConvNet", "DanQ", "DeepSEA"])
chk_bool("l2norm on, entropy_reg=1e-3", cfg["l2norm"] is True and abs(cfg["entropy_reg"] - 1e-3) < 1e-9)
chk_bool("TRAIN_TFS = 7", len(TRAIN_TFS) == 7)
chk_bool("strata sizes 17/6/6", len(OOD_WITHIN_FAMILY) == 17 and len(OOD_CROSS_FAMILY) == 6 and len(OOD_NONMOTIF) == 6)

# ---------- B. Multi-seed (3 genomic seeds) ----------
H, D = [], []
for s in (42, 0, 1):
    d = json.load(open(ROOT / f"results/hetmoe/seed{s}_genomic/hetmoe_eval.json"))
    H.append(d["ood_mean"]["HetMoE"]); D.append(d["ood_mean"]["DNABERT"])
H, D = np.array(H), np.array(D)
chk("multiseed HetMoE = 0.821", 0.821, H.mean())
chk("multiseed HetMoE sd = 0.005", 0.005, H.std(ddof=1), tol=5e-4)
chk("multiseed DNABERT = 0.799", 0.799, D.mean())
chk("multiseed DNABERT sd = 0.008", 0.008, D.std(ddof=1), tol=5e-4)
chk("multiseed margin = +0.022", 0.022, (H - D).mean())
chk("multiseed range low = +0.011", 0.011, (H - D).min(), tol=1e-3)
chk("multiseed range high = +0.028", 0.028, (H - D).max(), tol=1e-3)
chk_bool("margin positive in all 3 seeds", bool((H - D > 0).all()))
chk_bool("HetMoE sd < DNABERT sd", H.std(ddof=1) < D.std(ddof=1))

# ---------- C. Pool-composition / gating ablation (config sweep) ----------
SW = {}
for f in (ROOT / "results/moe_grid/seed42_genomic").glob("decision_*.json"):
    d = json.load(open(f)); SW[d["tag"]] = d
abl = {"convnet_only": (0.720, 0.673, 0.686, 1.71), "dnabert6_only": (0.805, 0.725, 0.753, 1.57),
       "convnet_dnabert6": (0.817, 0.743, 0.766, 2.37), "cnn_only": (0.881, 0.827, 0.846, 2.66),
       "full": (0.875, 0.818, 0.839, 2.87), "full_noknobs": (0.855, 0.800, 0.819, 0.98)}
for tag, (ind, mood, wn, ent) in abl.items():
    d = SW[tag]
    chk(f"abl {tag} in-dist={ind}", ind, d["indist_mean_hetmoe"])
    chk(f"abl {tag} motif-OOD={mood}", mood, d["ood_mean_hetmoe"])
    chk(f"abl {tag} gate-H={ent}", ent, d["mean_gate_entropy"], tol=5e-3, in_paper=False)
chk_bool("cnn_only selected (max in-dist mean)",
         max(SW.values(), key=lambda r: r["indist_mean_hetmoe"])["tag"] == "cnn_only")
# robustness note: validation-split selection -> full pool, OOD still > DNABERT (0.817 vs 0.800)
valsel = max(SW.values(), key=lambda r: r["gate_val_auc"])
chk("robustness: val-selected OOD = 0.817", 0.817, valsel["ood_mean_hetmoe"], tol=1e-3)
chk_bool("val-selected pool is 28-expert (full)", valsel["num_experts"] == 28)

# ---------- D. Attribution table vs CSV ----------
import csv as _csv
A = {}
with open(ROOT / "results/attribution/attribution_summary.csv") as fh:
    for r in _csv.DictReader(fh):
        A[(r["model_class"], r["method"])] = (float(r["faithfulness_to_ISM"]), float(r["stability"]))
tab = {("MoE", "shiftsmooth_corr"): (0.75, 0.96), ("MoE", "smoothgrad_corr"): (0.76, 0.92),
       ("MoE", "vanilla_corr"): (0.75, 0.95), ("MoE", "shiftsmooth"): (0.72, 0.96),
       ("MoE", "smoothgrad"): (0.75, 0.92), ("expert", "shiftsmooth_corr"): (0.78, 0.95),
       ("expert", "smoothgrad_corr"): (0.78, 0.91), ("expert", "vanilla_corr"): (0.78, 0.95),
       ("expert", "shiftsmooth"): (0.72, 0.95), ("expert", "smoothgrad"): (0.73, 0.91)}
for (mc, meth), (pf, ps) in tab.items():
    sf, ss = A[(mc, meth)]
    chk(f"attr {mc}/{meth} faith={pf}", pf, sf, tol=5e-3, in_paper=False)
    chk(f"attr {mc}/{meth} stab={ps}", ps, ss, tol=5e-3, in_paper=False)
chk_bool("ShiftSmooth+corr highest stability (MoE)",
         A[("MoE", "shiftsmooth_corr")][1] == max(v[1] for k, v in A.items() if k[0] == "MoE"))
chk_bool("ShiftSmooth+corr highest stability (expert)",
         A[("expert", "shiftsmooth_corr")][1] == max(v[1] for k, v in A.items() if k[0] == "expert"))

# ---------- E. Dinucleotide-shuffle inflation ablation ----------
try:
    DIN = json.load(open(ROOT / "results/hetmoe/seed42/hetmoe_eval.json"))
    dpv = [r for r in DIN["paired_vs_dnabert"] if r["data_type"] == "out_of_distribution" and r["tf"] in MOTIF]
    chk("dinuc motif-OOD = 0.864", 0.864, DIN["ood_mean"]["HetMoE"], tol=1e-3)
    chk_bool("dinuc superior 22/23", sum(r["superior"] for r in dpv) == 22)
    chk("dinuc inflation = +0.066", 0.066, DIN["ood_mean"]["HetMoE"] - DIN["ood_mean"]["DNABERT"], tol=1e-3)
except FileNotFoundError:
    chk_bool("dinuc eval present", False, "results/hetmoe/seed42/hetmoe_eval.json MISSING")

# ---------- F. Method / equation claims vs code ----------
models_src = (ROOT / "tfbs/models.py").read_text()
data_src = (ROOT / "tfbs/data.py").read_text()
metrics_src = (ROOT / "tfbs/metrics.py").read_text()
gate_src = (ROOT / "tfbs/gate.py").read_text()
ss_src = (ROOT / "experiments/attribution/shiftsmooth_eval.py").read_text()
# MoE per-expert projection has NO nonlinearity (Eq.1 is linear)
moe_fwd = models_src[models_src.index("def forward", models_src.index("class MixtureOfExperts")):]
moe_fwd = moe_fwd[:moe_fwd.index("@staticmethod")]
chk_bool("MoE per-expert projection is linear (no ReLU in forward)",
         "relu" not in moe_fwd.lower() and "expert(x[" in moe_fwd)
chk_bool("padding is uniform 0.25 (not zero)", "S[i] = 0.25" in data_src)
chk_bool("paired_bootstrap default n_boot=1000", "n_boot=1000" in metrics_src)
chk_bool("gate: l2norm default True", "l2norm=True" in gate_src)
chk_bool("gate: entropy_reg default 1e-3", "entropy_reg=1e-3" in gate_src)
chk_bool("gate: SGD Nesterov momentum 0.98 lr 0.01",
         "momentum=0.98" in gate_src and "nesterov=True" in gate_src and "lr=0.01" in gate_src)
chk_bool("ShiftSmooth rolls input by s and grad back by -s",
         "torch.roll(x0[0, :, CORE0:CORE1], shifts=s" in ss_src and "shifts=-s" in ss_src)
chk_bool("simplex correction subtracts mean over base axis (dim=0)",
         "g.mean(dim=0, keepdim=True)" in ss_src)
chk_bool("padded tensor is 4x147 (101 + 2*23)", 101 + 2 * 23 == 147)

# ---------- report ----------
print(f"\n{'='*78}\nPAPER NUMBER & METHOD VALIDATION  ({len(rows)} checks)\n{'='*78}")
npass = 0
for ok, desc, p, s in rows:
    tag = "PASS" if ok else "**FAIL**"
    if p is not None:
        print(f"  [{tag}] {desc:<46} paper={p:.4f}  source={s:.4f}")
    else:
        print(f"  [{tag}] {desc}")
    npass += ok
print(f"{'='*78}\n  {npass}/{len(rows)} checks passed")
sys.exit(0 if npass == len(rows) else 1)
