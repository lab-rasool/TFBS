"""Statistical analysis for the TFBS Mixture-of-Experts study (Reviewer 2, Comment 3).

Reads an evaluation summary CSV (per-model AUC replicates) plus, when available, a
paired-bootstrap differences CSV, and reports for each of the 9 datasets:

* one-way ANOVA over the 4 models (3 experts + MoE): F, exact p, and effect sizes
  eta^2 (= SS_between / SS_total) and omega^2;
* per-model mean AUC with a 95% confidence interval (percentile CI for the
  bootstrap protocol, t-based CI for the legacy MC-dropout trials);
* a multiplicity-aware post-hoc test of the MoE against each expert -- the paired
  bootstrap difference + 95% CI (rigorous protocol) when ``bootstrap_paired.csv``
  is present, otherwise Tukey HSD.

The script auto-detects the protocol from the replicate-column prefix:
``boot_*_auc`` (rigorous 50/50 paired bootstrap, written by ``evaluate.py
--protocol rigorous``) or ``trial_*_auc`` (legacy MC-dropout trials).  Only scipy
is required (Tukey via ``scipy.stats.studentized_range``).

Usage::

    python stats.py                       # canonical rigorous results
    python stats.py --legacy-check        # reproduce the published Section-7 ANOVA table
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, studentized_range, t as t_dist

MODELS = ["moe", "expert_ARID3A", "expert_FOXM1", "expert_GATA3"]
MODEL_LABEL = {"moe": "MoE", "expert_ARID3A": "ARID3A",
               "expert_FOXM1": "FOXM1", "expert_GATA3": "GATA3"}
EXPERTS = ["expert_ARID3A", "expert_FOXM1", "expert_GATA3"]
IN_DIST_TFS = ["ARID3A", "FOXM1", "GATA3"]
OOD_TFS = ["BCLAF1", "CTCF", "POLR2A", "RBBP5", "SAP30", "STAT3"]
STATS_DIR = "./results/stats"


# ----------------------------------------------------------------------------
def one_way_anova(groups):
    groups = [np.asarray(g, dtype=float) for g in groups]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    grand = np.concatenate(groups).mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_w = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_t = ss_b + ss_w
    df_b, df_w = k - 1, n_total - k
    ms_w = ss_w / df_w
    F, p = f_oneway(*groups)
    eta2 = ss_b / ss_t if ss_t > 0 else np.nan
    omega2 = (ss_b - df_b * ms_w) / (ss_t + ms_w) if (ss_t + ms_w) > 0 else np.nan
    return {"F": float(F), "p": float(p), "df_between": df_b, "df_within": df_w,
            "ms_within": ms_w, "eta2": float(eta2), "omega2": float(omega2),
            "k": k, "n_per_group": len(groups[0])}


def tukey_hsd(mean_a, mean_b, n, ms_within, df_within, k, alpha=0.05):
    diff = mean_a - mean_b
    se = np.sqrt(ms_within / n)
    q = diff / se if se > 0 else np.inf
    p_adj = float(studentized_range.sf(abs(q), k, df_within))
    qcrit = float(studentized_range.ppf(1 - alpha, k, df_within))
    return diff, float(q), p_adj, diff - qcrit * se, diff + qcrit * se


def _fmt_p(p):
    return "<1e-15" if p < 1e-15 else f"{p:.2e}"


def _rank_of_moe(means):
    order = sorted(means, key=means.get, reverse=True)
    return order.index("moe") + 1, order[0]


# ----------------------------------------------------------------------------
def analyse(summary_csv, paired_csv, alpha=0.05):
    df = pd.read_csv(summary_csv)
    rep_cols = [c for c in df.columns
                if (c.startswith("boot_") or c.startswith("trial_")) and c.endswith("_auc")]
    is_boot = rep_cols[0].startswith("boot_")
    paired = None
    if paired_csv and os.path.exists(paired_csv):
        paired = pd.read_csv(paired_csv)

    ordered = [("in_distribution", tf) for tf in IN_DIST_TFS] + \
              [("out_of_distribution", tf) for tf in OOD_TFS]
    summaries, posthoc, cis = [], [], []
    for data_type, tf in ordered:
        sub = df[(df["data_type"] == data_type) & (df["dataset_tf"] == tf)]
        m2a = {}
        for m in MODELS:
            r = sub[sub["model"] == m]
            if r.empty:
                raise ValueError(f"missing {m}/{tf}/{data_type} in {summary_csv}")
            m2a[m] = r[rep_cols].values.astype(float).ravel()
        aov = one_way_anova([m2a[m] for m in MODELS])
        means = {m: float(np.mean(m2a[m])) for m in MODELS}
        n = len(m2a["moe"])
        moe_rank, best = _rank_of_moe(means)
        row = {"data_type": data_type, "dataset_tf": tf, "F": aov["F"],
               "df_between": aov["df_between"], "df_within": aov["df_within"],
               "p": aov["p"], "eta2": aov["eta2"], "omega2": aov["omega2"],
               "best_model": MODEL_LABEL[best], "moe_rank": f"{moe_rank}/4",
               "mean_MoE": means["moe"], "mean_ARID3A": means["expert_ARID3A"],
               "mean_FOXM1": means["expert_FOXM1"], "mean_GATA3": means["expert_GATA3"]}

        # Per-model 95% CI
        for m in MODELS:
            arr = np.asarray(m2a[m])
            if is_boot:
                lo, hi = np.percentile(arr, [2.5, 97.5])
            else:
                tc = t_dist.ppf(0.975, n - 1)
                se = arr.std(ddof=1) / np.sqrt(n)
                lo, hi = arr.mean() - tc * se, arr.mean() + tc * se
            cis.append({"data_type": data_type, "dataset_tf": tf, "model": MODEL_LABEL[m],
                        "mean_auc": float(arr.mean()), "std_auc": float(arr.std(ddof=1)),
                        "ci95_low": float(lo), "ci95_high": float(hi)})

        # Post-hoc MoE vs each expert
        for e in EXPERTS:
            lbl = MODEL_LABEL[e]
            if paired is not None:
                pr = paired[(paired["data_type"] == data_type) & (paired["dataset_tf"] == tf) &
                            (paired["comparison"] == f"MoE - {lbl}")]
                pr = pr.iloc[0]
                diff, lo, hi, p_adj = pr["mean_diff"], pr["ci95_low"], pr["ci95_high"], pr["p_value"]
                sig = bool(pr["significant"])
                method = "paired-bootstrap"
            else:
                diff, q, p_adj, lo, hi = tukey_hsd(means["moe"], means[e], n,
                                                   aov["ms_within"], aov["df_within"], aov["k"], alpha)
                sig = p_adj < alpha
                method = "Tukey-HSD"
            posthoc.append({"data_type": data_type, "dataset_tf": tf,
                            "comparison": f"MoE - {lbl}", "method": method, "diff": float(diff),
                            "ci95_low": float(lo), "ci95_high": float(hi),
                            "p": float(p_adj), "significant": sig})
            row[f"diff_vs_{lbl}"] = float(diff)
            row[f"sig_vs_{lbl}"] = "*" if sig else "ns"
        summaries.append(row)

    return (pd.DataFrame(summaries), pd.DataFrame(posthoc), pd.DataFrame(cis),
            "rigorous" if is_boot else "legacy")


# ----------------------------------------------------------------------------
def to_markdown(sdf, title):
    out = [f"### {title}", "",
           "| Dataset | F(3,{}) | p | eta^2 | omega^2 | Best | MoE rank | "
           "MoE-ARID3A | MoE-FOXM1 | MoE-GATA3 |".format(int(sdf.iloc[0]["df_within"])),
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in sdf.iterrows():
        out.append(
            f"| {r['dataset_tf']} | {r['F']:.1f} | {_fmt_p(r['p'])} | {r['eta2']:.3f} | "
            f"{r['omega2']:.3f} | {r['best_model']} | {r['moe_rank']} | "
            f"{r['diff_vs_ARID3A']:+.4f}{r['sig_vs_ARID3A']} | "
            f"{r['diff_vs_FOXM1']:+.4f}{r['sig_vs_FOXM1']} | "
            f"{r['diff_vs_GATA3']:+.4f}{r['sig_vs_GATA3']} |")
    return "\n".join(out)


def to_latex(sdf, caption, label):
    out = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{caption}}}",
           f"\\label{{{label}}}", "\\begin{tabular}{lrrrrclrrr}", "\\hline",
           "Dataset & $F$ & $p$ & $\\eta^2$ & $\\omega^2$ & Best & MoE rank & "
           "$\\Delta$ARID3A & $\\Delta$FOXM1 & $\\Delta$GATA3 \\\\", "\\hline"]
    for _, r in sdf.iterrows():
        p = "<10^{-15}" if r["p"] < 1e-15 else f"{r['p']:.1e}"
        out.append(
            f"{r['dataset_tf']} & {r['F']:.1f} & ${p}$ & {r['eta2']:.3f} & {r['omega2']:.3f} & "
            f"{r['best_model']} & {r['moe_rank']} & {r['diff_vs_ARID3A']:+.4f} & "
            f"{r['diff_vs_FOXM1']:+.4f} & {r['diff_vs_GATA3']:+.4f} \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def legacy_sanity_check(summary_csv):
    """Reproduce the published Section-7 ANOVA values from the archived legacy CSV."""
    sdf, _, _, _ = analyse(summary_csv, paired_csv=None)
    ref = {"ARID3A": (2865.1, 0.987), "FOXM1": (14155.7, 0.997), "GATA3": (1494.5, 0.975),
           "CTCF": (18776.2, 0.998), "STAT3": (334.8, 0.897)}
    print("Legacy reproduction vs handoff Section 7:")
    ok = True
    for _, r in sdf.iterrows():
        if r["dataset_tf"] in ref:
            Fr, er = ref[r["dataset_tf"]]
            good = abs(r["F"] - Fr) / Fr < 0.02 and abs(r["eta2"] - er) < 0.005
            ok = ok and good
            print(f"  {r['dataset_tf']:<8} F={r['F']:.1f} (ref {Fr}) eta2={r['eta2']:.3f} "
                  f"(ref {er})  [{'OK' if good else 'MISMATCH'}]")
    print("Sanity check:", "PASSED" if ok else "FAILED")
    return sdf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default="./results/evaluation_summary.csv")
    ap.add_argument("--paired", default="./results/bootstrap_paired.csv")
    ap.add_argument("--legacy-check", action="store_true",
                    help="Reproduce the published Section-7 ANOVA from legacy/results/.")
    args = ap.parse_args()
    os.makedirs(STATS_DIR, exist_ok=True)

    if args.legacy_check:
        sdf = legacy_sanity_check("./archive/legacy/results/evaluation_summary_published.csv")
        in_df = sdf[sdf["data_type"] == "in_distribution"]
        ood_df = sdf[sdf["data_type"] == "out_of_distribution"]
        md = ("# Published (legacy) statistics -- Tukey HSD post-hoc\n\n"
              + to_markdown(in_df, "In-distribution (published)") + "\n\n"
              + to_markdown(ood_df, "Out-of-distribution (published)"))
        open(os.path.join(STATS_DIR, "stats_legacy_published.md"), "w").write(md)
        print("\n" + md)
        return

    sdf, posthoc, cis, tag = analyse(args.summary, args.paired)
    sdf.to_csv(os.path.join(STATS_DIR, f"stats_{tag}_summary.csv"), index=False)
    posthoc.to_csv(os.path.join(STATS_DIR, f"stats_{tag}_posthoc.csv"), index=False)
    cis.to_csv(os.path.join(STATS_DIR, f"stats_{tag}_modelci.csv"), index=False)

    in_df = sdf[sdf["data_type"] == "in_distribution"]
    ood_df = sdf[sdf["data_type"] == "out_of_distribution"]
    posthoc_method = posthoc.iloc[0]["method"]
    title = (f"Rigorous 50/50 protocol -- ANOVA + {posthoc_method} post-hoc "
             "(* = MoE-expert 95% CI excludes 0)" if tag == "rigorous"
             else f"Legacy MC-dropout protocol -- ANOVA + {posthoc_method}")
    md = (f"# Statistical analysis ({tag})\n\n_{title}_\n\n"
          + to_markdown(in_df, "In-distribution") + "\n\n"
          + to_markdown(ood_df, "Out-of-distribution"))
    open(os.path.join(STATS_DIR, f"stats_{tag}.md"), "w").write(md)
    cap = ("One-way ANOVA over the four models with paired-bootstrap post-hoc comparisons "
           "of the MoE against each expert on the balanced 50/50 held-out sets.")
    open(os.path.join(STATS_DIR, f"stats_{tag}.tex"), "w").write(
        to_latex(in_df, cap + " In-distribution.", f"tab:{tag}_indist") + "\n\n"
        + to_latex(ood_df, cap + " Out-of-distribution.", f"tab:{tag}_ood"))
    print(f"Wrote {STATS_DIR}/stats_{tag}_*\n")
    print(md)


if __name__ == "__main__":
    main()
