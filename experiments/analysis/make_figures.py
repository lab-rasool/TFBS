"""Regenerate the manuscript figures at publication quality (Reviewer 2, Comment 8).

Produces, from the canonical evaluation outputs, decluttered, colorblind-safe,
>=300 dpi (PNG) + vector (PDF) figures with larger fonts and direct numeric labels
on the (thin) confidence intervals:

* ``fig3_bars_in_distribution`` / ``fig6_bars_out_of_distribution`` -- grouped AUC
  bar charts with 95% CI error bars and value labels.
* ``fig4_ci_in_distribution`` / ``fig7_ci_out_of_distribution`` -- the ANOVA 95%
  confidence-interval "box" figures (one box per model per dataset).
* ``fig5_roc_out_of_distribution`` (and the in-distribution ROC) -- decluttered ROC
  panels, one subplot per TF.
* ``fig1_visual_abstract_mockup`` / ``fig8_shiftsmooth_mockup`` -- redesign mockups
  for the two hand-made figures whose editable sources are not in the repo, plus
  written guidance in ``results/figures/FIGURE_GUIDANCE.md``.

Inputs (written by ``evaluate.py --protocol rigorous`` and ``stats.py``):
``results/stats/stats_rigorous_modelci.csv``, ``results/bootstrap_paired.csv``,
``results/evaluation_results.json``, ``results/evaluation_summary.csv``.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

FIG_DIR = "./results/figures"
# Okabe-Ito colorblind-safe palette
COLORS = {"MoE": "#0072B2", "ARID3A": "#E69F00", "FOXM1": "#009E73", "GATA3": "#CC79A7"}
MODEL_ORDER = ["MoE", "ARID3A", "FOXM1", "GATA3"]
JSON_KEY = {"MoE": "moe", "ARID3A": "expert_ARID3A",
            "FOXM1": "expert_FOXM1", "GATA3": "expert_GATA3"}
IN_TFS = ["ARID3A", "FOXM1", "GATA3"]
OOD_TFS = ["BCLAF1", "CTCF", "POLR2A", "RBBP5", "SAP30", "STAT3"]

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_modelci():
    p = "./results/stats/stats_rigorous_modelci.csv"
    if os.path.exists(p):
        return pd.read_csv(p)
    # fall back: recompute mean + percentile CI from the summary replicate columns
    df = pd.read_csv("./results/evaluation_summary.csv")
    rep = [c for c in df.columns if (c.startswith("boot_") or c.startswith("trial_")) and c.endswith("_auc")]
    lbl = {"moe": "MoE", "expert_ARID3A": "ARID3A", "expert_FOXM1": "FOXM1", "expert_GATA3": "GATA3"}
    rows = []
    for _, r in df.iterrows():
        arr = r[rep].values.astype(float)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append({"data_type": r["data_type"], "dataset_tf": r["dataset_tf"],
                     "model": lbl[r["model"]], "mean_auc": arr.mean(),
                     "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def _tfs(data_type):
    return IN_TFS if data_type == "in_distribution" else OOD_TFS


def fig_bars(modelci, data_type, name, title):
    tfs = _tfs(data_type)
    x = np.arange(len(tfs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(2.4 * len(tfs) + 2, 5.2))
    for j, m in enumerate(MODEL_ORDER):
        means, los, his = [], [], []
        for tf in tfs:
            r = modelci[(modelci.data_type == data_type) & (modelci.dataset_tf == tf) &
                        (modelci.model == m)].iloc[0]
            means.append(r.mean_auc); los.append(r.mean_auc - r.ci95_low); his.append(r.ci95_high - r.mean_auc)
        xpos = x + (j - 1.5) * w
        ax.bar(xpos, means, w, yerr=[los, his], capsize=3, color=COLORS[m],
               label=m, edgecolor="black", linewidth=0.5, error_kw={"linewidth": 1})
        for xp, mv in zip(xpos, means):
            ax.text(xp, mv + 0.012, f"{mv:.3f}", ha="center", va="bottom",
                    fontsize=9, rotation=90)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.text(len(tfs) - 0.5, 0.505, "chance", color="grey", fontsize=10, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(tfs)
    ax.set_ylabel("AUC (mean, 95% CI)"); ax.set_ylim(0.45, 1.0)
    ax.set_title(title)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    _save(fig, name)


def fig_ci_boxes(modelci, data_type, name, title):
    """95% CI 'box' figure: a box spanning each model's CI, mean marked + labelled."""
    tfs = _tfs(data_type)
    x = np.arange(len(tfs))
    w = 0.2
    fig, ax = plt.subplots(figsize=(2.6 * len(tfs) + 2, 5.4))
    for j, m in enumerate(MODEL_ORDER):
        for i, tf in enumerate(tfs):
            r = modelci[(modelci.data_type == data_type) & (modelci.dataset_tf == tf) &
                        (modelci.model == m)].iloc[0]
            xc = x[i] + (j - 1.5) * w
            lo, hi, mv = r.ci95_low, r.ci95_high, r.mean_auc
            # CI box (widened minimum height so thin CIs stay visible)
            h = max(hi - lo, 0.004)
            yc = mv - h / 2
            ax.add_patch(plt.Rectangle((xc - w * 0.42, yc), w * 0.84, h,
                                       facecolor=COLORS[m], edgecolor="black",
                                       linewidth=0.6, alpha=0.85,
                                       label=m if i == 0 else None))
            ax.plot([xc - w * 0.42, xc + w * 0.42], [mv, mv], color="black", lw=1.1)
            ax.text(xc, hi + 0.006, f"{mv:.3f}", ha="center", va="bottom",
                    fontsize=8.5, rotation=90)
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(tfs)
    ax.set_ylabel("AUC (95% confidence interval)"); ax.set_ylim(0.45, 1.0)
    ax.set_title(title)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    _save(fig, name)


def fig_roc(results, data_type, name, title):
    tfs = _tfs(data_type)
    ncol = 3
    nrow = (len(tfs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 4.6 * nrow))
    axes = np.atleast_2d(axes)
    for idx, tf in enumerate(tfs):
        ax = axes[idx // ncol, idx % ncol]
        for m in MODEL_ORDER:
            mk = JSON_KEY[m]
            entry = None
            for ds in results[mk][data_type]:
                if ds and ds[0]["dataset_tf"] == tf:
                    entry = ds[0]; break
            if entry is None:
                continue
            fpr, tpr, _ = roc_curve(entry["targets"], entry["predictions"])
            ls = "--" if m == "MoE" else "-"
            ax.plot(fpr, tpr, color=COLORS[m], ls=ls, lw=2,
                    label=f"{m} ({entry['auc']:.3f})")
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=1)
        ax.set_title(tf); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        if idx % ncol == 0:
            ax.set_ylabel("True positive rate")
        if idx // ncol == nrow - 1:
            ax.set_xlabel("False positive rate")
        ax.legend(loc="lower right", fontsize=11, frameon=True)
        ax.grid(alpha=0.25)
    for idx in range(len(tfs), nrow * ncol):
        fig.delaxes(axes[idx // ncol, idx % ncol])
    fig.suptitle(title, y=1.02, fontsize=17)
    _save(fig, name)


def _box(ax, xy, wh, text, fc):
    ax.add_patch(FancyBboxPatch(xy, wh[0], wh[1], boxstyle="round,pad=0.02,rounding_size=0.04",
                                fc=fc, ec="black", lw=1.2))
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=12)


def _arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.4, color="#333333"))


def fig1_mockup():
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
    _box(ax, (0.2, 2), (2.0, 1.1), "ChIP-seq\n101 bp seqs\n(+ dinuc-shuffle\nnegatives)", "#E8F0FE")
    for i, (tf, c) in enumerate(zip(IN_TFS, ["#E69F00", "#009E73", "#CC79A7"])):
        _box(ax, (3.0, 0.4 + i * 1.55), (2.2, 1.1), f"{tf} expert\n(modified DeepBIND)", c + "55")
        _arrow(ax, (2.2, 2.55), (3.0, 0.95 + i * 1.55))
    _box(ax, (6.0, 1.6), (2.0, 1.6), "32-d embeddings\n(3x32 = 96)\n+ gating softmax", "#D0E8FF")
    for i in range(3):
        _arrow(ax, (5.2, 0.95 + i * 1.55), (6.0, 2.4))
    _box(ax, (8.6, 1.9), (2.0, 1.1), "MoE\nweighted sum\n-> prediction", "#0072B2AA")
    _arrow(ax, (8.0, 2.4), (8.6, 2.45))
    _box(ax, (11.0, 2.7), (1.8, 1.0), "AUC eval\n(50/50 sets)", "#EEEEEE")
    _box(ax, (11.0, 1.2), (1.8, 1.0), "Attribution\n(ShiftSmooth)", "#EEEEEE")
    _arrow(ax, (10.6, 2.6), (11.0, 3.1)); _arrow(ax, (10.6, 2.3), (11.0, 1.8))
    ax.set_title("Figure 1 (mockup): MoE training, evaluation and attribution pipeline",
                 fontsize=15)
    _save(fig, "fig1_visual_abstract_mockup")


def fig8_mockup():
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
    _box(ax, (0.2, 1.7), (1.9, 1.1), "Input\nsequence x", "#E8F0FE")
    for i, s in enumerate([-1, 0, 1]):
        _box(ax, (2.7, 0.3 + i * 1.45), (2.1, 1.0), f"shift by {s}\n-> grad", "#FFE9C7")
        _arrow(ax, (2.1, 2.2), (2.7, 0.8 + i * 1.45))
    _box(ax, (5.6, 1.6), (2.4, 1.2), "align back +\naverage gradients", "#D0E8FF")
    for i in range(3):
        _arrow(ax, (4.8, 0.8 + i * 1.45), (5.6, 2.2))
    _box(ax, (8.8, 1.7), (2.8, 1.1), "ShiftSmooth\nattribution map", "#009E7355")
    _arrow(ax, (8.0, 2.2), (8.8, 2.25))
    ax.set_title("Figure 8 (mockup): ShiftSmooth gradient averaging over circular shifts (N=2 shown)",
                 fontsize=14)
    _save(fig, "fig8_shiftsmooth_mockup")


GUIDANCE = """# Figure redesign guidance (Reviewer 2, Comment 8)

The reviewer flagged Figures 1, 5 and 8 as crowded / low-resolution. Figures 3, 4,
6 and 7 (bar charts and ANOVA CI boxes) are regenerated directly from data at >=300
dpi with a colorblind-safe palette, larger fonts and direct numeric labels.

## Figure 1 - Visual abstract (hand-made; editable source not in repo)
`fig1_visual_abstract_mockup.pdf` is a decluttered layout proposal. For the final
version: (a) keep a single left-to-right flow (data -> 3 experts -> embeddings +
gating -> MoE -> evaluation/attribution); (b) >=12 pt sans-serif labels; (c) drop
decorative artwork; (d) one accent color per stage; (e) export vector PDF.
ACTION NEEDED: drop the editable source (.ai/.pptx/.svg) into paper/figures/ for a
production pass.

## Figure 5 - OOD ROC panel
`fig5_roc_out_of_distribution.pdf`: 2x3 grid, one TF per subplot, 4 model curves,
larger fonts, AUC in the legend, MoE dashed. Much less crowded than a single packed
panel.

## Figure 8 - ShiftSmooth illustration (hand-made; editable source not in repo)
`fig8_shiftsmooth_mockup.pdf` is a schematic proposal (shift -> gradient -> align +
average). ACTION NEEDED: provide the editable source for a production pass.
"""


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    modelci = load_modelci()
    results = json.load(open("./results/evaluation_results.json"))

    fig_bars(modelci, "in_distribution", "fig3_bars_in_distribution",
             "Mean AUC by model -- in-distribution")
    fig_bars(modelci, "out_of_distribution", "fig6_bars_out_of_distribution",
             "Mean AUC by model -- out-of-distribution")
    fig_ci_boxes(modelci, "in_distribution", "fig4_ci_in_distribution",
                 "95% confidence intervals -- in-distribution")
    fig_ci_boxes(modelci, "out_of_distribution", "fig7_ci_out_of_distribution",
                 "95% confidence intervals -- out-of-distribution")
    fig_roc(results, "out_of_distribution", "fig5_roc_out_of_distribution",
            "ROC curves -- out-of-distribution")
    fig_roc(results, "in_distribution", "roc_in_distribution",
            "ROC curves -- in-distribution")
    fig1_mockup()
    fig8_mockup()
    open(os.path.join(FIG_DIR, "FIGURE_GUIDANCE.md"), "w").write(GUIDANCE)
    print(f"Wrote figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
