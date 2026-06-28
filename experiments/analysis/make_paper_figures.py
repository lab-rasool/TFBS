"""Canonical paper figure generator (analysis figures 3-9): one figure per result,
regenerated from the existing genomic eval JSON -- NO experiments re-run.

Writes results/figures/paper/fig_{3..9}_*.{pdf,png}. Run with the genomic negative
protocol so every path resolves to *_genomic:

    TFBS_TRAIN_NEG=genomic python -m experiments.analysis.make_paper_figures --seed 42
"""
import argparse
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from tfbs import figstyle as fs
from tfbs.figstyle import COL1, COL1_5, COL2, OKABE_ITO, color, save
from tfbs.constants import (TRAIN_TFS, OOD_WITHIN_FAMILY, OOD_CROSS_FAMILY,
                            OOD_NONMOTIF, DBD_FAMILY, MODE_SUFFIX)

fs.apply()
PAPER = os.path.join("results", "figures", "paper")
MOTIF = OOD_WITHIN_FAMILY + OOD_CROSS_FAMILY          # 23 motif-bearing OOD factors
REPR = ["JUN", "MYC", "ELF1", "SP2", "GATA2", "CTCF"]  # gate-heatmap rows
PRETTY = {"HetMoE": "HetMoE", "DNABERT": "DNABERT-6",
          "best_single": "best single expert", "static_mean": "static average"}


def load(seed):
    return json.load(open(f"results/hetmoe/seed{seed}{MODE_SUFFIX}/hetmoe_eval.json"))


def od(d, key):
    return d["per_dataset"][f"out_of_distribution::{key}"]


def idr(d, key):
    return d["per_dataset"][f"in_distribution::{key}"]


def _baseline_motif_ood(seed, backbone):
    """Motif-OOD mean AUC of a standalone backbone ensemble (mean of the per-training-factor
    predictions), derived from the genomic embedding cache. DNABERT6 uses its cached
    classifier predictions; DeepSEA/DanQ use their probe predictions."""
    cd = f"results/cache/seed{seed}{MODE_SUFFIX}"
    aucs = []
    for tf in MOTIF:
        preds, y = [], None
        for tfe in TRAIN_TFS:
            fn = (f"dnabert6base_{tfe}__out_of_distribution_{tf}.npz" if backbone == "DNABERT6"
                  else f"{backbone}_{tfe}__out_of_distribution_{tf}.npz")
            dd = np.load(os.path.join(cd, fn)); preds.append(dd["pred"]); y = dd["y"]
        aucs.append(roc_auc_score(y, np.mean(preds, axis=0)))
    return float(np.mean(aucs))


def _stars(p):
    """Significance level for a p-value (Nature convention)."""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"


def _baseline_indist(seed, backbone, tf, B=1000):
    """In-distribution AUC + 95% percentile-bootstrap CI for a standalone backbone ensemble."""
    cd = f"results/cache/seed{seed}{MODE_SUFFIX}"
    preds, y = [], None
    for tfe in TRAIN_TFS:
        dd = np.load(os.path.join(cd, f"{backbone}_{tfe}__in_distribution_{tf}.npz"))
        preds.append(dd["pred"]); y = dd["y"]
    p = np.mean(preds, axis=0); auc = roc_auc_score(y, p)
    rng = np.random.default_rng(0); n = len(y); bs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) > 1:
            bs.append(roc_auc_score(y[idx], p[idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return auc, (float(lo), float(hi))


def _motif_factor_aucs(seed, key, d):
    """Per-factor motif-OOD AUC over the 23 motif factors (HetMoE/DNABERT from the eval
    dict d; DanQ/DeepSEA derived from the cache)."""
    cd = f"results/cache/seed{seed}{MODE_SUFFIX}"
    out = {}
    for tf in MOTIF:
        if key in ("HetMoE", "DNABERT"):
            out[tf] = d["per_dataset"][f"out_of_distribution::{tf}"][f"{key}_auc"]
        else:
            preds, y = [], None
            for tfe in TRAIN_TFS:
                dd = np.load(os.path.join(cd, f"{key}_{tfe}__out_of_distribution_{tf}.npz"))
                preds.append(dd["pred"]); y = dd["y"]
            out[tf] = roc_auc_score(y, np.mean(preds, axis=0))
    return out


def _bracket_p(het, base, B=2000):
    """Two-sided p that HetMoE's mean motif-OOD differs from a baseline's, via a paired
    bootstrap over the motif factors (the units of analysis)."""
    tfs = list(het)
    diff = np.array([het[t] - base[t] for t in tfs])
    rng = np.random.default_rng(0); n = len(diff)
    md = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
    return max(2 * min((md <= 0).mean(), (md >= 0).mean()), 1.0 / B)


def fig_3_indist_performance(d):
    """Per-TF in-distribution AUROC (dot + 95% CI), HetMoE vs DNABERT-6, 7 train TFs."""
    models = [("HetMoE", "HetMoE"), ("DNABERT-6", "DNABERT"), ("DanQ", "DanQ"), ("DeepSEA", "DeepSEA")]
    pmap = {r["tf"]: r["p_value"] for r in d["paired_vs_dnabert"] if r["data_type"] == "in_distribution"}
    nf, nm = len(TRAIN_TFS), len(models)
    fig, ax = plt.subplots(figsize=(COL1_5, 0.20 * nf * nm * COL1 / 4 + 0.3))
    bw = 0.78 / nm
    offs = (np.arange(nm) - (nm - 1) / 2) * bw
    yc = np.arange(nf)[::-1]
    for mi, (lab, key) in enumerate(models):
        for fi, tf in enumerate(TRAIN_TFS):
            y = yc[fi] - offs[mi]
            if key in ("HetMoE", "DNABERT"):
                r = idr(d, tf); auc = r[f"{key}_auc"]; lo, hi = r[f"{key}_ci"]
            else:
                auc, (lo, hi) = _baseline_indist(42, key, tf)
            ax.barh(y, auc, height=bw, color=color(key), edgecolor="black", linewidth=0.4, zorder=2,
                    label=lab if fi == 0 else None)
            ax.plot([lo, hi], [y, y], color="black", lw=0.6, zorder=4)
    for fi, tf in enumerate(TRAIN_TFS):
        ax.text(1.012, yc[fi], _stars(pmap.get(tf, 1.0)), ha="left", va="center", fontsize=6, clip_on=False)
    ax.axvline(0.5, ls=":", lw=0.6, color=OKABE_ITO["grey"], zorder=0)
    ax.set_yticks(yc); ax.set_yticklabels(TRAIN_TFS, fontsize=6.5)
    ax.set_xlim(0.0, 1.0); ax.set_xlabel("in-distribution AUROC"); ax.set_ylim(-0.6, nf - 0.4)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=5.5,
              frameon=False, columnspacing=1.2, handlelength=1.1, handletextpad=0.3)
    save(fig, "fig_3_indist_performance", outdir=PAPER)


def fig_4_ood_headline(d):
    """Motif-OOD mean AUROC, 4 models -> gain is gating, not ensembling. ``ood_mean``
    is the motif-strata mean (within+cross), == 0.827 for HetMoE."""
    order = ["HetMoE", "DNABERT", "best_single", "static_mean"]
    vals = [d["ood_mean"][m] for m in order]
    fig, ax = plt.subplots(figsize=(COL1, 0.78 * COL1))
    x = np.arange(len(order))
    ax.bar(x, vals, width=0.66, color=[color(m) for m in order], zorder=3)
    ax.axhline(0.5, ls=":", lw=0.6, color=OKABE_ITO["grey"], zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in order], rotation=20, ha="right")
    ax.set_ylabel("OOD AUROC (motif strata, mean)"); ax.set_ylim(0.0, 1.0)
    save(fig, "fig_4_ood_headline", outdir=PAPER)


def fig_5_ood_forest(d):
    """Per-factor HetMoE-DNABERT-6 ΔAUROC ± 95% CI for the 23 motif-OOD factors, grouped
    by DBD family; blue where the CI excludes 0 in HetMoE's favor."""
    rows = [r for r in d["paired_vs_dnabert"]
            if r["data_type"] == "out_of_distribution" and r["tf"] in MOTIF]
    fam_order, seen = [], set()
    for tf in MOTIF:
        f = DBD_FAMILY.get(tf, "?")
        if f not in seen:
            seen.add(f); fam_order.append(f)
    rows.sort(key=lambda r: (fam_order.index(DBD_FAMILY.get(r["tf"], "?")), -r["mean_diff"]))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(COL1_5, 0.30 * n * COL1 / 6))
    ypos = np.arange(n)[::-1]; last = None
    for y, r in zip(ypos, rows):
        c = color("HetMoE") if r["superior"] else OKABE_ITO["grey"]
        ax.plot([r["ci95_low"], r["ci95_high"]], [y, y], color=c, lw=1.1, zorder=2)
        ax.plot(r["mean_diff"], y, "o", color=c, ms=3, zorder=3)
        fam = DBD_FAMILY.get(r["tf"], "?")
        if last is not None and fam != last:
            ax.axhline(y + 0.5, color=OKABE_ITO["grey"], lw=0.3, alpha=0.5, zorder=0)
        last = fam
    ax.axvline(0, ls="-", lw=0.6, color="black", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['tf']}  ({DBD_FAMILY.get(r['tf'], '?')})" for r in rows], fontsize=6)
    ax.set_xlabel("HetMoE − DNABERT-6   ΔAUROC (95% CI)"); ax.set_ylim(-0.7, n - 0.3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=color("HetMoE"), marker="o", ms=3, lw=1.1, label="superior (95% CI > 0)"),
                       Line2D([0], [0], color=OKABE_ITO["grey"], marker="o", ms=3, lw=1.1, label="n.s.")],
              loc="lower right", fontsize=5.5, frameon=False)
    save(fig, "fig_5_ood_forest", outdir=PAPER)


def fig_6_ood_strata(d):
    """OOD AUROC by stratum (within / cross / non-motif), HetMoE vs DNABERT-6."""
    strata = [("within-family", "ood_within_family"),
              ("cross-family", "ood_cross_family"),
              ("non-motif\n(reported sep.)", "ood_nonmotif")]
    fig, ax = plt.subplots(figsize=(COL1_5, 0.62 * COL1_5))
    x = np.arange(len(strata)); w = 0.38
    h = [d[k]["HetMoE"] for _, k in strata]; b = [d[k]["DNABERT"] for _, k in strata]
    ax.bar(x - w / 2, h, w, color=color("HetMoE"), label="HetMoE", zorder=3)
    ax.bar(x + w / 2, b, w, color=color("DNABERT"), label="DNABERT-6", zorder=3)
    for xi, v in zip(x - w / 2, h):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)
    for xi, v in zip(x + w / 2, b):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)
    ax.axhline(0.5, ls=":", lw=0.6, color=OKABE_ITO["grey"], zorder=0)
    ax.axvspan(1.5, 2.5, color=OKABE_ITO["grey"], alpha=0.10, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in strata])
    ax.set_ylabel("OOD AUROC"); ax.set_ylim(0.0, 1.0); ax.legend(loc="upper right", ncol=2)
    save(fig, "fig_6_ood_strata", outdir=PAPER)


def fig_7_multiseed(seeds=(0, 1, 42)):
    """Motif-OOD mean AUROC per genomic seed, HetMoE vs DNABERT-6 (3 seeds)."""
    S = {"HetMoE": [], "DNABERT": [], "DanQ": [], "DeepSEA": []}
    for s in seeds:
        dd = json.load(open(f"results/hetmoe/seed{s}{MODE_SUFFIX}/hetmoe_eval.json"))
        S["HetMoE"].append(dd["ood_mean"]["HetMoE"]); S["DNABERT"].append(dd["ood_mean"]["DNABERT"])
        S["DanQ"].append(_baseline_motif_ood(s, "DanQ")); S["DeepSEA"].append(_baseline_motif_ood(s, "DeepSEA"))
    order = [("HetMoE", "HetMoE"), ("DNABERT-6", "DNABERT"), ("DanQ", "DanQ"), ("DeepSEA", "DeepSEA")]
    d42 = json.load(open(f"results/hetmoe/seed42{MODE_SUFFIX}/hetmoe_eval.json"))
    hf = _motif_factor_aucs(42, "HetMoE", d42)
    fig, ax = plt.subplots(figsize=(COL1_5, 0.82 * COL1_5))
    for xi, (lab, key) in enumerate(order):
        v = np.array(S[key]); m = v.mean(); sem = v.std(ddof=1) / np.sqrt(len(v))
        ax.bar(xi, m, width=0.62, color=color(key), edgecolor="black", linewidth=0.8, zorder=2)
        ax.errorbar(xi, m, yerr=sem, color="black", capsize=2.5, lw=0.8, zorder=4)
        jit = np.linspace(-0.13, 0.13, len(v))
        ax.plot(xi + jit, v, "o", mfc="white", mec="black", mew=0.5, ms=3.5, zorder=5)
    top = max(np.mean(S[k]) for _, k in order)
    for j, (lab, key) in enumerate(order[1:], start=1):
        p = _bracket_p(hf, _motif_factor_aucs(42, key, d42))
        y = top + 0.035 + 0.05 * (j - 1)
        ax.plot([0, 0, j, j], [y - 0.013, y, y, y - 0.013], color="black", lw=0.7, zorder=6)
        ax.text(j / 2.0, y + 0.004, _stars(p), ha="center", va="bottom", fontsize=7, zorder=6)
    ax.axhline(0.5, ls=":", lw=0.6, color=OKABE_ITO["grey"], zorder=0)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([l for l, _ in order], fontsize=6.5)
    ax.set_ylabel("motif-OOD AUROC"); ax.set_xlim(-0.6, len(order) - 0.4); ax.set_ylim(0.0, 1.0)
    save(fig, "fig_7_multiseed", outdir=PAPER)


def fig_8_gate_routing(d):
    """Mean gate weight, expert (cols, config.expert_order) x representative OOD TF (rows)."""
    order = d["config"]["expert_order"]

    def abbr(n):
        bb, tf = n.split("::")
        bb = {"ConvNet": "Conv", "DeepSEA": "DSEA", "DanQ": "DanQ", "DNABERT6": "BERT"}.get(bb, bb)
        return f"{bb}:{tf}"

    M = np.array([od(d, tf)["gate_weights"] for tf in REPR])
    bb_of = [n.split("::")[0] for n in order]
    tf_of = [n.split("::")[1] for n in order]
    BB = {"ConvNet": "ConvNet", "DeepSEA": "DeepSEA", "DanQ": "DanQ", "DNABERT6": "DNABERT-6"}
    fig, ax = plt.subplots(figsize=(COL2, 0.42 * COL2))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.12, M.max()))
    ax.set_xticks(range(len(order))); ax.set_xticklabels(tf_of, rotation=90, fontsize=5)
    ax.set_yticks(range(len(REPR))); ax.set_yticklabels(REPR, fontsize=6.5)
    bounds = [i for i in range(1, len(order)) if bb_of[i] != bb_of[i - 1]]
    for b in bounds:
        ax.axvline(b - 0.5, color="white", lw=1.3)
    starts = [0] + bounds + [len(order)]
    for a, c in zip(starts[:-1], starts[1:]):
        ax.text((a + c - 1) / 2, -0.9, BB.get(bb_of[a], bb_of[a]),
                ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.set_ylim(len(REPR) - 0.5, -1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.01)
    cb.set_label("mean gate weight", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    save(fig, "fig_8_gate_routing", outdir=PAPER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    d = load(args.seed)
    fig_3_indist_performance(d)
    fig_4_ood_headline(d)
    fig_5_ood_forest(d)
    fig_6_ood_strata(d)
    fig_7_multiseed()
    fig_8_gate_routing(d)
    print("paper figs:", sorted(f for f in os.listdir(PAPER) if f.endswith(".pdf")))


if __name__ == "__main__":
    main()
