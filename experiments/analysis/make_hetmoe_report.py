"""Publication tables + figures for the heterogeneous embedding-gated MoE.

Consumes Phase-C outputs (results/hetmoe/hetmoe_eval.json) and the Phase-B config
sweep (results/moe_grid/decision_*.json) and writes:

  results/stats/hetmoe_report.md / .tex   -- headline, paired-vs-DNABERT, ablation tables
  results/figures/hetmoe_ood_bars.{png,pdf}  -- per-OOD-TF AUC (HetMoE vs DNABERT vs static) + CIs
  results/figures/hetmoe_gate_usage.{png,pdf} -- expert x OOD-TF gate-weight heatmap (Reviewer 1)
  results/figures/hetmoe_ablation.{png,pdf}  -- OOD AUC by backbone set / knobs (Reviewer 7)

300 dpi, colorblind-safe Okabe-Ito palette, large fonts (Reviewer 8).
    python make_hetmoe_report.py
"""
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OKABE = {"black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
         "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
         "vermillion": "#D55E00", "purple": "#CC79A7"}
OOD_TFS = ["BCLAF1", "CTCF", "POLR2A", "RBBP5", "SAP30", "STAT3"]
OOD_LEARNABLE = ["CTCF", "STAT3"]
FIG_DIR = "./results/figures"
STATS_DIR = "./results/stats"
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 11, "figure.dpi": 300})


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def headline_table(R):
    om = R["ood_mean"]
    rows = [("HetMoE (ours)", om.get("HetMoE"), R["indist_mean_hetmoe"]),
            ("DNABERT-6", om.get("DNABERT", R["references"]["DNABERT6"]), None),
            ("Static-mean (same zoo)", om.get("static_mean"), None),
            ("Best-single expert", om.get("best_single"), None),
            ("DeepSEA (baseline)", R["references"]["DeepSEA"], None),
            ("DanQ (baseline)", R["references"]["DanQ"], None),
            ("Original MoE (paper)", R["references"]["orig_MoE"], None)]
    md = ["### Headline: OOD AUC (balanced 50/50, B=1000 bootstrap)", "",
          "| Model | OOD mean AUC | In-dist AUC |", "|---|---|---|"]
    for name, ood, indist in rows:
        ind = f"{indist:.4f}" if indist is not None else "--"
        md.append(f"| {name} | {ood:.4f} | {ind} |")
    md += ["",
           f"Stratified OOD HetMoE: learnable (CTCF,STAT3) = **{R['ood_learnable_hetmoe']:.4f}**, "
           f"indirect = {R['ood_indirect_hetmoe']:.4f}.  "
           f"OOD calibration: ECE = {R['ood_mean_ece']:.3f}, Brier = {R['ood_mean_brier']:.3f}."]
    return "\n".join(md)


def paired_table(R):
    pr = [r for r in R["paired_vs_dnabert"] if r["data_type"] == "out_of_distribution"]
    if not pr:
        return "_(no DNABERT baseline cached -- paired comparison unavailable.)_"
    md = ["### HetMoE vs DNABERT-6 (paired bootstrap on identical resamples)", "",
          f"TOST non-inferiority margin = {R['tost_margin']}.  "
          "SUP = 95% CI excludes 0 (superior); NI = non-inferior (TOST).", "",
          "| OOD TF | ΔAUC (HetMoE−DNABERT) | 95% CI | p | verdict |", "|---|---|---|---|---|"]
    nsup = nni = 0
    for r in pr:
        v = "SUP" if r["superior"] else ("NI" if r["non_inferior_TOST"] else "—")
        nsup += r["superior"]; nni += r["non_inferior_TOST"]
        md.append(f"| {r['tf']} | {r['mean_diff']:+.4f} | "
                  f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['p_value']:.3f} | {v} |")
    md.append(f"\n**Superior on {nsup}/6, non-inferior on {nni}/6 OOD TFs.**")
    return "\n".join(md)


def ablation_table(grid):
    md = ["### Ablation: backbone set / N_e / gate knobs (selection on in-dist only)", "",
          "| Config | N_e | OOD AUC | OOD oracle | in-dist | gate entropy |",
          "|---|---|---|---|---|---|"]
    for g in sorted(grid, key=lambda x: -x["ood_mean_hetmoe"]):
        md.append(f"| {g['tag']} | {g['num_experts']} | {g['ood_mean_hetmoe']:.4f} | "
                  f"{g['ood_mean_oracle']:.4f} | {g['indist_mean_hetmoe']:.4f} | "
                  f"{g['mean_gate_entropy']:.2f} |")
    return "\n".join(md)


def fig_ood_bars(R):
    pd_ = R["per_dataset"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(OOD_TFS)); w = 0.27
    series = [("HetMoE", "HetMoE", OKABE["blue"]),
              ("DNABERT", "DNABERT-6", OKABE["vermillion"]),
              ("static_mean", "static-mean", OKABE["green"])]
    for i, (key, lbl, col) in enumerate(series):
        vals, los, his = [], [], []
        for tf in OOD_TFS:
            r = pd_[f"out_of_distribution::{tf}"]
            a = r.get(f"{key}_auc"); ci = r.get(f"{key}_ci", [a, a]) if a is not None else None
            if a is None:
                vals.append(np.nan); los.append(0); his.append(0); continue
            vals.append(a); los.append(a - ci[0]); his.append(ci[1] - a)
        ax.bar(x + (i - 1) * w, vals, w, label=lbl, color=col,
               yerr=[los, his], capsize=3, error_kw={"elinewidth": 1})
    ax.axhline(0.5, ls=":", c="grey", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(OOD_TFS)
    ax.set_ylabel("OOD AUC"); ax.set_ylim(0.5, 0.95)
    ax.set_title("Per-OOD-TF AUC: HetMoE vs DNABERT-6 vs static-mean (95% CI)")
    for tf in OOD_LEARNABLE:
        ax.get_xticklabels()[OOD_TFS.index(tf)].set_fontweight("bold")
    ax.legend(loc="upper right")
    _save(fig, "hetmoe_ood_bars")


def fig_gate_usage(R):
    order = R["config"]["expert_order"]
    M = np.array([R["per_dataset"][f"out_of_distribution::{tf}"]["gate_weights"] for tf in OOD_TFS])
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.7), 5))
    im = ax.imshow(M.T, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(OOD_TFS))); ax.set_xticklabels(OOD_TFS)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=9)
    ax.set_title("Mean gate weight per expert across OOD TFs")
    fig.colorbar(im, ax=ax, label="gate weight")
    _save(fig, "hetmoe_gate_usage")


def fig_ablation(grid):
    g = sorted(grid, key=lambda x: x["ood_mean_hetmoe"])
    fig, ax = plt.subplots(figsize=(9, max(4, len(g) * 0.5)))
    y = np.arange(len(g))
    ax.barh(y, [x["ood_mean_hetmoe"] for x in g], color=OKABE["skyblue"])
    ax.axvline(0.7492, ls="--", c=OKABE["vermillion"], label="DNABERT-6 (0.749)")
    ax.set_yticks(y); ax.set_yticklabels([x["tag"] for x in g], fontsize=9)
    ax.set_xlabel("OOD mean AUC"); ax.set_xlim(0.5, 0.8)
    ax.set_title("Ablation: OOD AUC by configuration")
    ax.legend()
    _save(fig, "hetmoe_ablation")


def main():
    os.makedirs(STATS_DIR, exist_ok=True)
    R = json.load(open("./results/hetmoe/hetmoe_eval.json"))
    grid = [json.load(open(f)) for f in glob.glob("./results/moe_grid/decision_*.json")]

    report = "\n\n".join(["# Heterogeneous Embedding-Gated MoE — results report",
                          headline_table(R), paired_table(R),
                          ablation_table(grid) if grid else ""])
    open(os.path.join(STATS_DIR, "hetmoe_report.md"), "w").write(report)
    print(report)

    fig_ood_bars(R)
    fig_gate_usage(R)
    if grid:
        fig_ablation(grid)
    print(f"\nWrote {STATS_DIR}/hetmoe_report.md and figures to {FIG_DIR}/")


if __name__ == "__main__":
    main()
