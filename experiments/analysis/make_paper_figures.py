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


def fig_3_indist_performance(d):
    """Per-TF in-distribution AUROC (dot + 95% CI), HetMoE vs DNABERT-6, 7 train TFs."""
    fig, ax = plt.subplots(figsize=(COL1_5, 0.55 * COL1_5))
    x = np.arange(len(TRAIN_TFS)); off = 0.16
    for m, dx in [("HetMoE", -off), ("DNABERT", +off)]:
        for i, tf in enumerate(TRAIN_TFS):
            r = idr(d, tf); lo, hi = r[f"{m}_ci"]
            ax.plot([i + dx, i + dx], [lo, hi], color=color(m), lw=1.1, zorder=2)
            ax.plot(i + dx, r[f"{m}_auc"], "o", color=color(m), ms=3.5, zorder=3,
                    label=PRETTY[m] if i == 0 else None)
    ax.axhline(0.5, ls=":", lw=0.6, color=OKABE_ITO["grey"], zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(TRAIN_TFS, fontsize=6); ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("AUROC (mean, 95% CI)"); ax.legend(ncol=2, loc="lower right")
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
    ax.set_yticklabels([f"{r['tf']}  ({DBD_FAMILY.get(r['tf'], '?')})" for r in rows], fontsize=5)
    ax.set_xlabel("HetMoE − DNABERT-6   ΔAUROC (95% CI)"); ax.set_ylim(-0.7, n - 0.3)
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
    H, D = [], []
    for s in seeds:
        dd = json.load(open(f"results/hetmoe/seed{s}{MODE_SUFFIX}/hetmoe_eval.json"))
        H.append(dd["ood_mean"]["HetMoE"]); D.append(dd["ood_mean"]["DNABERT"])
    H, D = np.array(H), np.array(D)
    fig, ax = plt.subplots(figsize=(COL1, 0.85 * COL1))
    for xi, (lab, key, v) in enumerate([("HetMoE", "HetMoE", H), ("DNABERT-6", "DNABERT", D)]):
        ax.plot([xi] * len(v), v, "o", color=color(key), ms=5, zorder=3)
        ax.plot([xi - 0.2, xi + 0.2], [v.mean(), v.mean()], color="black", lw=1.2, zorder=4)
        ax.text(xi, v.max() + 0.004, f"{v.mean():.3f}\n±{v.std(ddof=1):.3f}",
                ha="center", va="bottom", fontsize=6)
    ax.axhline(0.5, ls=":", lw=0.6, color=OKABE_ITO["grey"], zorder=0)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HetMoE", "DNABERT-6"])
    ax.set_ylabel("motif-OOD AUROC"); ax.set_xlim(-0.5, 1.5); ax.set_ylim(0.0, 1.0)
    save(fig, "fig_7_multiseed", outdir=PAPER)


def fig_8_gate_routing(d):
    """Mean gate weight, expert (cols, config.expert_order) x representative OOD TF (rows)."""
    order = d["config"]["expert_order"]

    def abbr(n):
        bb, tf = n.split("::")
        bb = {"ConvNet": "Conv", "DeepSEA": "DSEA", "DanQ": "DanQ", "DNABERT6": "BERT"}.get(bb, bb)
        return f"{bb}:{tf}"

    M = np.array([od(d, tf)["gate_weights"] for tf in REPR])
    fig, ax = plt.subplots(figsize=(COL2, 0.34 * COL2))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.12, M.max()))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([abbr(n) for n in order], rotation=90, fontsize=3.6)
    ax.set_yticks(range(len(REPR))); ax.set_yticklabels(REPR, fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cb.set_label("mean gate weight", fontsize=6); cb.ax.tick_params(labelsize=5)
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
