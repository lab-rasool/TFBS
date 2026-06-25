"""Nature-style publication figure pack for the TFBS HetMoE paper.

Produces ~12 standalone figures (vector PDF + 600 dpi PNG) in
``results/figures/nature/`` plus ``stats_summary.md`` with the numbers behind them.
Beyond re-plotting, it computes three new analyses:
  * reliability/calibration curves from recomputed per-sample HetMoE probabilities
    (the selected ConvNet+DNABERT-6 gate, reproduced on the seed-42 cache);
  * standardized effect sizes for HetMoE-minus-DNABERT per OOD TF;
  * multi-seed mean +/- SD of HetMoE vs DNABERT OOD AUC over seeds 0-3, 42.

Run from the repo root:  python -m experiments.analysis.make_paper_figures
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tfbs import figstyle as fs
from tfbs.figstyle import COL1, COL1_5, COL2, OKABE_ITO, color, panel_label, save
from tfbs.constants import OOD_TFS, OOD_LEARNABLE, OOD_INDIRECT, TRAIN_TFS

fs.apply()

HET = "results/hetmoe/seed42"
SEEDS = [0, 1, 2, 3, 42]
DNABERT_REF = 0.7492  # published fine-tuned DNABERT-6 OOD bar

PRETTY = {"DNABERT-6": "DNABERT-6", "static_mean": "static mean",
          "best_single": "best single", "orig-MoE": "orig MoE", "HetMoE": "HetMoE",
          "DeepSEA": "DeepSEA", "DanQ": "DanQ"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def het_eval(seed=42):
    return json.load(open(f"results/hetmoe/seed{seed}/hetmoe_eval.json"))


def ood_table():
    """Per-OOD-TF AUC for every model: {model: {tf: auc}} (+ CIs where available)."""
    base = pd.read_csv("results/baselines/baseline_comparison_ood.csv").set_index("model")
    ev = het_eval()["per_dataset"]

    def het(field):
        return {tf: ev[f"out_of_distribution::{tf}"][field] for tf in OOD_TFS}

    def het_ci(field):
        return {tf: ev[f"out_of_distribution::{tf}"][field] for tf in OOD_TFS}

    table = {
        "HetMoE": het("HetMoE_auc"),
        "DNABERT-6": {tf: float(base.loc["DNABERT", tf]) for tf in OOD_TFS},
        "best_single": het("best_single_auc"),
        "DeepSEA": {tf: float(base.loc["DeepSEA", tf]) for tf in OOD_TFS},
        "DanQ": {tf: float(base.loc["DanQ", tf]) for tf in OOD_TFS},
        "static_mean": het("static_mean_auc"),
        "orig-MoE": {tf: float(base.loc["MoE", tf]) for tf in OOD_TFS},
    }
    cis = {"HetMoE": het_ci("HetMoE_ci"), "DNABERT-6": het_ci("DNABERT_ci"),
           "best_single": het_ci("best_single_ci"), "static_mean": het_ci("static_mean_ci")}
    return table, cis


def indist_table():
    """Per-in-dist-TF AUC: {model: {tf: auc}} for HetMoE + DNABERT + baselines."""
    base = pd.read_csv("results/baselines/baseline_comparison_indist.csv").set_index("model")
    ev = het_eval()["per_dataset"]
    het = {tf: ev[f"in_distribution::{tf}"]["HetMoE_auc"] for tf in TRAIN_TFS}
    out = {"HetMoE": het}
    for m, key in [("DNABERT-6", "DNABERT"), ("DeepSEA", "DeepSEA"), ("DanQ", "DanQ"),
                   ("orig-MoE", "MoE")]:
        if key in base.index:
            out[m] = {tf: float(base.loc[key, tf]) for tf in TRAIN_TFS if tf in base.columns}
    return out


# ---------------------------------------------------------------------------
# New analysis: recompute per-sample HetMoE probabilities (selected config)
# ---------------------------------------------------------------------------
def hetmoe_persample(seed=42):
    """Reproduce the selected ConvNet+DNABERT-6 gate on the cache and return
    per-OOD-TF (probs, y) for HetMoE and the DNABERT-6 baseline."""
    from sklearn.model_selection import train_test_split
    from tfbs.experts import load_zoo_cache, subset_zoo, _load_dnabert_baseline, _cdir
    from tfbs.gate import train_gate, gate_predict

    zoo = subset_zoo(load_zoo_cache(seed), backbones=["ConvNet", "DNABERT6"])
    emb, Y = zoo["emb"], zoo["y"]
    E, ne = zoo["embedding_dim"], len(zoo["expert_order"])
    tr = [f"train::{tf}" for tf in TRAIN_TFS]
    emb_all = np.concatenate([emb[k] for k in tr], 0)
    y_all = np.concatenate([Y[k] for k in tr], 0)
    itr, iva = train_test_split(np.arange(len(y_all)), test_size=0.2, random_state=seed)
    moe, _ = train_gate(emb_all[itr], y_all[itr], emb_all[iva], y_all[iva], ne, E, seed,
                        l2norm=True, entropy_reg=1e-3, gate_temperature=1.0)
    het = {}
    for tf in OOD_TFS:
        k = f"out_of_distribution::{tf}"
        p, _ = gate_predict(moe, emb[k])
        het[tf] = (p, Y[k])
    keys = [f"out_of_distribution::{tf}" for tf in OOD_TFS]
    dna_raw = _load_dnabert_baseline(keys, Y, _cdir(seed))
    dna = {tf: (dna_raw[f"out_of_distribution::{tf}"], Y[f"out_of_distribution::{tf}"])
           for tf in OOD_TFS} if dna_raw else None
    return het, dna


def reliability(probs, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ws = [], [], []
    for b in range(bins):
        m = (probs >= edges[b]) & (probs < edges[b + 1] if b < bins - 1 else probs <= edges[b + 1])
        if m.sum() == 0:
            continue
        xs.append(probs[m].mean()); ys.append(y[m].mean()); ws.append(m.sum())
    ece = sum(w * abs(x - yv) for x, yv, w in zip(xs, ys, ws)) / max(1, sum(ws))
    return np.array(xs), np.array(ys), np.array(ws), float(ece)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_ood_headline(table):
    order = ["HetMoE", "DNABERT-6", "best_single", "DeepSEA", "DanQ", "static_mean", "orig-MoE"]
    means, errs, cols, labs = [], [], [], []
    for m in order:
        v = np.array([table[m][tf] for tf in OOD_TFS])
        means.append(v.mean()); errs.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
        cols.append(color(m)); labs.append(PRETTY.get(m, m))
    fig, ax = plt.subplots(figsize=(COL1, 0.62 * COL1))
    yp = np.arange(len(order))[::-1]
    ax.barh(yp, means, xerr=errs, color=cols, height=0.7, error_kw=dict(lw=0.6, capsize=1.5))
    for y, m in zip(yp, means):
        ax.text(m + 0.004, y, f"{m:.3f}", va="center", ha="left", fontsize=5.5)
    ax.axvline(DNABERT_REF, ls="--", lw=0.6, color=OKABE_ITO["grey"], zorder=0)
    ax.set_yticks(yp); ax.set_yticklabels(labs)
    ax.set_xlabel("OOD AUROC (mean over 6 TFs ± 95% CI)")
    ax.set_xlim(0.60, 0.84)
    save(fig, "fig_ood_headline")


def fig_forest_vs_dnabert():
    pr = pd.read_csv(f"{HET}/bootstrap_paired_vs_dnabert.csv")
    pr = pr[pr.data_type == "out_of_distribution"].set_index("tf").reindex(OOD_TFS)
    fig, ax = plt.subplots(figsize=(COL1_5, 0.58 * COL1_5))
    yp = np.arange(len(OOD_TFS))[::-1]
    for y, tf in zip(yp, OOD_TFS):
        r = pr.loc[tf]
        c = color("HetMoE") if r["superior"] else OKABE_ITO["grey"]
        ax.plot([r["ci95_low"], r["ci95_high"]], [y, y], color=c, lw=1.1, zorder=2)
        ax.plot(r["mean_diff"], y, "o", color=c, ms=3.5, zorder=3)
    ax.axvline(0, ls="-", lw=0.6, color="black", zorder=1)
    strata = {tf: ("●" if tf in OOD_LEARNABLE else "○") for tf in OOD_TFS}
    ax.set_yticks(yp); ax.set_yticklabels([f"{strata[tf]} {tf}" for tf in OOD_TFS])
    ax.set_xlabel("HetMoE − DNABERT-6  ΔAUROC (95% CI)")
    ax.set_xlim(-0.006, 0.10)
    ax.text(0.98, 0.04, "● sequence-specific    ○ indirect", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=5.5, color=OKABE_ITO["grey"])
    save(fig, "fig_forest_vs_dnabert")


def fig_multiseed():
    rows = []
    for s in SEEDS:
        try:
            om = het_eval(s)["ood_mean"]
            rows.append((s, om["HetMoE"], om.get("DNABERT", np.nan)))
        except FileNotFoundError:
            continue
    arr = np.array([(h, d) for _, h, d in rows])
    fig, ax = plt.subplots(figsize=(COL1, 0.7 * COL1))
    for j, (m, lab) in enumerate([("HetMoE", "HetMoE"), ("DNABERT", "DNABERT-6")]):
        v = arr[:, j]
        x = np.full(len(v), j) + np.linspace(-0.06, 0.06, len(v))
        ax.scatter(x, v, s=12, color=color(lab), zorder=3, edgecolor="white", linewidth=0.3)
        ax.plot([j - 0.18, j + 0.18], [v.mean(), v.mean()], color="black", lw=1.0, zorder=4)
        ax.text(j, v.mean() + 0.012, f"{v.mean():.3f}\n±{v.std(ddof=1):.3f}",
                ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HetMoE", "DNABERT-6"])
    ax.set_ylabel("OOD AUROC"); ax.set_xlim(-0.5, 1.5)
    ax.set_title(f"{len(rows)} seeds", fontsize=6, loc="right", color=OKABE_ITO["grey"])
    save(fig, "fig_multiseed")


def fig_indist_vs_ood(indist, table):
    fig, ax = plt.subplots(figsize=(COL1, 0.8 * COL1))
    for m in ["HetMoE", "DNABERT-6", "DeepSEA", "DanQ", "orig-MoE"]:
        if m not in indist:
            continue
        xi = np.mean([indist[m][tf] for tf in indist[m]])
        yo = np.mean([table[m][tf] for tf in OOD_TFS])
        ax.plot([0, 1], [xi, yo], "-", color=color(m), lw=1.0, alpha=0.9)
        ax.scatter([0, 1], [xi, yo], s=14, color=color(m), zorder=3,
                   edgecolor="white", linewidth=0.3, label=PRETTY.get(m, m))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["in-dist", "OOD"])
    ax.set_ylabel("AUROC (mean)"); ax.set_xlim(-0.2, 1.2)
    ax.legend(loc="lower left", ncol=1, fontsize=5.5)
    save(fig, "fig_indist_vs_ood")


def fig_gate_heatmap():
    ev = het_eval()
    order = ev["config"]["expert_order"]
    M = np.array([ev["per_dataset"][f"out_of_distribution::{tf}"]["gate_weights"] for tf in OOD_TFS])
    fig, ax = plt.subplots(figsize=(COL1_5, 0.62 * COL1_5))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.35, M.max()))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([n.replace("::", "\n") for n in order], fontsize=5)
    ax.set_yticks(range(len(OOD_TFS))); ax.set_yticklabels(OOD_TFS)
    for i in range(len(OOD_TFS)):
        for j in range(len(order)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=4.5,
                    color="white" if M[i, j] > 0.22 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("mean gate weight", fontsize=6); cb.ax.tick_params(labelsize=5)
    ax.set_title("Gate routing (expert × OOD TF)", fontsize=7)
    save(fig, "fig_gate_heatmap")


def fig_calibration(het, dna):
    fig, ax = plt.subplots(figsize=(COL1, COL1))
    ax.plot([0, 1], [0, 1], ls=":", lw=0.6, color="black", zorder=1)
    out = {}
    series = [("HetMoE", het)]
    if dna is not None:
        series.append(("DNABERT-6", dna))
    for lab, d in series:
        p = np.concatenate([d[tf][0] for tf in OOD_TFS])
        y = np.concatenate([d[tf][1] for tf in OOD_TFS])
        xs, ys, ws, ece = reliability(p, y)
        out[lab] = ece
        ax.plot(xs, ys, "-o", color=color(lab), ms=3, lw=1.0, label=f"{lab} (ECE={ece:.3f})")
    ax.set_xlabel("mean predicted P(bound)"); ax.set_ylabel("observed fraction bound")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=5.5)
    ax.set_title("OOD calibration (6 TFs pooled)", fontsize=7)
    save(fig, "fig_calibration")
    return out


def fig_ood_strata(table):
    groups = [("sequence-specific", OOD_LEARNABLE), ("indirect / recruited", OOD_INDIRECT)]
    fig, ax = plt.subplots(figsize=(COL1_5, 0.6 * COL1_5))
    models = ["HetMoE", "DNABERT-6", "DeepSEA", "DanQ"]
    x = 0; ticks, ticklabs = [], []
    for gname, tfs in groups:
        for tf in tfs:
            for k, m in enumerate(models):
                ax.bar(x + k * 0.8 / len(models), table[m][tf], width=0.8 / len(models),
                       color=color(m), label=PRETTY.get(m, m) if (x == 0 and tf == tfs[0]) else None)
            ticks.append(x + 0.4 * (len(models) - 1) / len(models)); ticklabs.append(tf)
            x += 1
        x += 0.4
    ax.axhspan(0.5, 0.70, color=OKABE_ITO["grey"], alpha=0.10, zorder=0)
    ax.text(len(OOD_LEARNABLE) + 0.05, 0.60, "seq-only ceiling", fontsize=5,
            va="center", ha="center", color=OKABE_ITO["grey"], rotation=90)
    ax.set_xticks(ticks); ax.set_xticklabels(ticklabs, fontsize=5.5)
    ax.set_ylabel("OOD AUROC"); ax.set_ylim(0.5, 0.95)
    ax.legend(loc="upper right", ncol=4, fontsize=5)
    ax.set_title("learnable (CTCF, STAT3)  vs  indirect", fontsize=7, loc="left")
    save(fig, "fig_ood_strata")


def _abl(csv, xcol, ycols, name, xlabel, title, logx=False):
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(COL1, 0.72 * COL1))
    pal = [color("HetMoE"), OKABE_ITO["orange"], OKABE_ITO["grey"]]
    for i, (yc, lab) in enumerate(ycols):
        ax.plot(df[xcol], df[yc], "-o", color=pal[i % len(pal)], ms=3.5, label=lab)
    if logx:
        ax.set_xscale("log", base=2); ax.set_xticks(df[xcol]); ax.set_xticklabels(df[xcol])
    ax.set_xlabel(xlabel); ax.set_ylabel("AUROC")
    if len(ycols) > 1:
        ax.legend(fontsize=5.5)
    ax.set_title(title, fontsize=7)
    save(fig, name)


def fig_ablations():
    _abl("results/ablation/ablation_embedding_size.csv", "embedding_dim",
         [("ood_meanAUC", "OOD"), ("in_distribution_meanAUC", "in-dist")],
         "fig_abl_embedding", "embedding dim E", "Embedding size", logx=True)
    _abl("results/ablation/ablation_num_experts.csv", "num_experts",
         [("ood_meanAUC_mean", "mean"), ("ood_meanAUC_best", "best")],
         "fig_abl_nexperts", "number of experts $N_e$", "Number of experts")
    df = pd.read_csv("results/ablation/ablation_frozen_vs_finetuned.csv")
    fig, ax = plt.subplots(figsize=(COL1, 0.72 * COL1))
    xp = np.arange(len(df))
    ax.bar(xp - 0.18, df["in_distribution_meanAUC"], width=0.34, color=OKABE_ITO["sky"], label="in-dist")
    ax.bar(xp + 0.18, df["ood_meanAUC"], width=0.34, color=color("HetMoE"), label="OOD")
    ax.set_xticks(xp); ax.set_xticklabels(df["experts"])
    ax.set_ylabel("AUROC"); ax.legend(fontsize=5.5); ax.set_title("Frozen vs fine-tuned experts", fontsize=7)
    save(fig, "fig_abl_frozen")


def fig_roc_ood(het):
    from sklearn.metrics import roc_curve
    res = json.load(open("results/evaluation_results.json"))
    n = len(OOD_TFS); ncol = 3; nrow = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(COL2, 0.62 * COL2))
    for ax, tf in zip(axes.ravel(), OOD_TFS):
        # orig MoE from saved predictions
        for entry in res["moe"]["out_of_distribution"]:
            e = entry[0]
            if e["dataset_tf"] == tf:
                fpr, tpr, _ = roc_curve(e["targets"], e["predictions"])
                ax.plot(fpr, tpr, color=color("orig-MoE"), lw=0.9, label="orig MoE")
        # HetMoE recomputed
        p, y = het[tf]
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color("HetMoE"), lw=1.1, label="HetMoE")
        ax.plot([0, 1], [0, 1], ls=":", lw=0.5, color=OKABE_ITO["grey"])
        ax.set_title(tf, fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    for ax in axes[-1]:
        ax.set_xlabel("FPR")
    for ax in axes[:, 0]:
        ax.set_ylabel("TPR")
    axes.ravel()[0].legend(fontsize=5, loc="lower right")
    fig.suptitle("OOD ROC: HetMoE vs original MoE", fontsize=8)
    save(fig, "fig_roc_ood")


def fig_config_sweep():
    rows = []
    for f in glob.glob("results/moe_grid/seed42/decision_*_seed42.json"):
        d = json.load(open(f))
        tag = os.path.basename(f).replace("decision_", "").replace("_seed42.json", "")
        rows.append((tag, d.get("num_experts"), d.get("ood_mean_hetmoe", np.nan),
                     d.get("ood_mean_oracle", np.nan)))
    rows.sort(key=lambda r: r[2])
    tags = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(COL1_5, 0.7 * COL1_5))
    yp = np.arange(len(rows))
    sel = "convnet_dnabert6_6"
    cols = [color("HetMoE") if t == sel else OKABE_ITO["sky"] for t in tags]
    ax.barh(yp, [r[2] for r in rows], color=cols, height=0.66)
    ax.plot([r[3] for r in rows], yp, "D", color=OKABE_ITO["grey"], ms=2.5, label="oracle ceiling")
    ax.axvline(DNABERT_REF, ls="--", lw=0.6, color=OKABE_ITO["orange"], label="DNABERT-6 (0.749)")
    ax.set_yticks(yp); ax.set_yticklabels(tags, fontsize=5)
    ax.set_xlabel("OOD AUROC"); ax.set_xlim(0.65, 0.81)
    ax.legend(fontsize=5, loc="lower right")
    ax.set_title("Config sweep (selected = blue)", fontsize=7)
    save(fig, "fig_config_sweep")


# ---------------------------------------------------------------------------
def main():
    table, cis = ood_table()
    indist = indist_table()
    print("[fig] recomputing per-sample HetMoE probabilities (selected config)...", flush=True)
    het_ps, dna_ps = hetmoe_persample(42)

    figs = [
        ("fig_ood_headline", lambda: fig_ood_headline(table)),
        ("fig_forest_vs_dnabert", fig_forest_vs_dnabert),
        ("fig_multiseed", fig_multiseed),
        ("fig_indist_vs_ood", lambda: fig_indist_vs_ood(indist, table)),
        ("fig_gate_heatmap", fig_gate_heatmap),
        ("fig_ood_strata", lambda: fig_ood_strata(table)),
        ("fig_ablations", fig_ablations),
        ("fig_roc_ood", lambda: fig_roc_ood(het_ps)),
        ("fig_config_sweep", fig_config_sweep),
    ]
    ece = {}
    ok, fail = [], []
    for name, fn in figs:
        try:
            fn(); ok.append(name)
        except Exception as e:  # keep going; report at end
            fail.append((name, repr(e)))
            print(f"  FAILED {name}: {e}", flush=True)
    try:
        ece = fig_calibration(het_ps, dna_ps); ok.append("fig_calibration")
    except Exception as e:
        fail.append(("fig_calibration", repr(e)))

    # ---- stats summary ----
    ev = het_eval()
    het_ood = np.array([table["HetMoE"][tf] for tf in OOD_TFS])
    dna_ood = np.array([table["DNABERT-6"][tf] for tf in OOD_TFS])
    diff = het_ood - dna_ood
    cohend = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else float("nan")
    ms = []
    for s in SEEDS:
        try:
            ms.append(het_eval(s)["ood_mean"])
        except FileNotFoundError:
            pass
    het_ms = np.array([m["HetMoE"] for m in ms]); dna_ms = np.array([m["DNABERT"] for m in ms])
    lines = [
        "# Figure pack — stats summary",
        "",
        f"Generated into `results/figures/nature/` ({len(ok)} figures). Palette: Okabe-Ito (CB-safe).",
        "",
        "## OOD headline (mean over 6 TFs)",
        "| model | OOD AUROC |",
        "|---|---|",
    ]
    for m in ["HetMoE", "DNABERT-6", "best_single", "DeepSEA", "DanQ", "static_mean", "orig-MoE"]:
        lines.append(f"| {PRETTY.get(m, m)} | {np.mean([table[m][tf] for tf in OOD_TFS]):.4f} |")
    lines += [
        "",
        "## HetMoE vs DNABERT-6 (OOD)",
        f"- mean ΔAUROC = {diff.mean():+.4f} (per-TF: " +
        ", ".join(f"{tf} {d:+.3f}" for tf, d in zip(OOD_TFS, diff)) + ")",
        f"- standardized effect size (Cohen's d, paired across TFs) = {cohend:.2f}",
        f"- selected config: {ev['config']['num_experts']} experts "
        f"({'+'.join(sorted(set(n.split('::')[0] for n in ev['config']['expert_order'])))}), "
        f"l2norm={ev['config']['l2norm']}, entropy={ev['config']['entropy_reg']}, tau={ev['config']['gate_temperature']}",
        "",
        "## Multi-seed robustness (OOD mean ± SD)",
        f"- HetMoE   = {het_ms.mean():.4f} ± {het_ms.std(ddof=1):.4f}  (n={len(het_ms)} seeds)",
        f"- DNABERT-6 = {dna_ms.mean():.4f} ± {dna_ms.std(ddof=1):.4f}",
        "",
        "## OOD calibration (pooled, ECE)",
    ] + [f"- {k} = {v:.4f}" for k, v in ece.items()]
    if fail:
        lines += ["", "## Figures that failed"] + [f"- {n}: {e}" for n, e in fail]
    os.makedirs(fs.FIG_DIR, exist_ok=True)
    open(os.path.join(fs.FIG_DIR, "stats_summary.md"), "w").write("\n".join(lines) + "\n")

    print(f"\n[fig] wrote {len(ok)} figures + stats_summary.md to {fs.FIG_DIR}")
    if fail:
        print(f"[fig] {len(fail)} failed: {[n for n, _ in fail]}")


if __name__ == "__main__":
    main()
