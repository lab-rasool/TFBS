"""Aggregate the multi-seed HetMoE robustness runs (results/hetmoe/seed*/hetmoe_eval.json)
into a mean +/- std summary of the OOD win over DNABERT-6, per OOD TF and overall.

    python aggregate_seeds.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd

OOD_TFS = ["BCLAF1", "CTCF", "POLR2A", "RBBP5", "SAP30", "STAT3"]
OUT = "./results/hetmoe"


def main():
    files = sorted(glob.glob("results/hetmoe/seed*/hetmoe_eval.json"))
    if not files:
        print("no per-seed results found yet")
        return
    rows, per_tf = [], {tf: [] for tf in OOD_TFS}
    hm, db, sup_counts, configs = [], [], [], []
    for f in files:
        R = json.load(open(f))
        seed = R["config"]["seed"]
        hm.append(R["ood_mean"]["HetMoE"]); db.append(R["ood_mean"].get("DNABERT", np.nan))
        configs.append("+".join(sorted({n.split("::")[0] for n in R["config"]["expert_order"]})))
        pr = [r for r in R["paired_vs_dnabert"] if r["data_type"] == "out_of_distribution"]
        sup_counts.append(sum(r["superior"] for r in pr))
        for r in pr:
            per_tf[r["tf"]].append(r["mean_diff"])
        rows.append({"seed": seed, "selected_backbones": configs[-1],
                     "ood_hetmoe": R["ood_mean"]["HetMoE"],
                     "ood_dnabert": R["ood_mean"].get("DNABERT"),
                     "margin": R["ood_mean"]["HetMoE"] - R["ood_mean"].get("DNABERT", np.nan),
                     "ood_ece": R["ood_mean_ece"], "n_superior_6": sup_counts[-1]})
    df = pd.DataFrame(rows).sort_values("seed")
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "multiseed_per_seed.csv"), index=False)

    hm, db = np.array(hm), np.array(db)
    margin = hm - db
    lines = ["# Multi-seed robustness (HetMoE vs DNABERT-6, OOD)", "",
             f"Seeds: {sorted(df['seed'])}  (n={len(df)})", "",
             "| metric | mean | std | min | max |", "|---|---|---|---|---|",
             f"| HetMoE OOD AUC | {hm.mean():.4f} | {hm.std(ddof=1):.4f} | {hm.min():.4f} | {hm.max():.4f} |",
             f"| DNABERT OOD AUC | {db.mean():.4f} | {db.std(ddof=1):.4f} | {db.min():.4f} | {db.max():.4f} |",
             f"| margin (HetMoE-DNABERT) | {margin.mean():+.4f} | {margin.std(ddof=1):.4f} | {margin.min():+.4f} | {margin.max():+.4f} |",
             f"| #OOD-TFs superior (of 6) | {np.mean(sup_counts):.1f} | {np.std(sup_counts, ddof=1):.1f} | {min(sup_counts)} | {max(sup_counts)} |",
             "", "Selected config per seed: " + ", ".join(f"s{r.seed}:{r.selected_backbones}" for r in df.itertuples()),
             "", "### Per-OOD-TF mean margin (HetMoE - DNABERT) across seeds", "",
             "| OOD TF | mean ΔAUC | std | seeds superior |", "|---|---|---|---|"]
    for tf in OOD_TFS:
        d = np.array(per_tf[tf])
        nsup = int((d > 0).sum())
        lines.append(f"| {tf} | {d.mean():+.4f} | {d.std(ddof=1):.4f} | {nsup}/{len(d)} |")
    report = "\n".join(lines)
    open(os.path.join(OUT, "multiseed_summary.md"), "w").write(report)
    print(report)
    print(f"\nWrote {OUT}/multiseed_summary.md and multiseed_per_seed.csv")


if __name__ == "__main__":
    main()
