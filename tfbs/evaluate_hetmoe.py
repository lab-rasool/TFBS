"""Phase-0 decision gate and Phase-C publication evaluation for HetMoE: train
the gate over a (cached) zoo, compute per-dataset metrics, paired bootstrap vs
the DNABERT-6 baseline, TOST non-inferiority, calibration, and OOD strata.
"""
import json
import os

import numpy as np
from sklearn.model_selection import train_test_split

from tfbs.constants import TRAIN_TFS, OOD_TFS, OOD_LEARNABLE, OOD_INDIRECT
from tfbs.experts import build_zoo, _cdir, _load_dnabert_baseline
from tfbs.gate import train_gate, gate_predict
from tfbs.metrics import auc, brier, ece, paired_bootstrap


def run_config(zoo, seed=42, l2norm=True, entropy_reg=1e-3, gate_temperature=1.0,
               out="./results/phase0", tag=None):
    """Train the gate for ONE config over an (already built or cached) zoo and report
    the full per-dataset metrics + the GO/NO-GO decision number."""
    os.makedirs(out, exist_ok=True)
    order = zoo["expert_order"]
    emb, pred, Y, meta = zoo["emb"], zoo["pred"], zoo["y"], zoo["meta"]
    embedding_dim = zoo["embedding_dim"]
    num_experts = len(order)

    # ---- gate training data: combined 3 training-TF _AC, 80/20 split ----
    tr_keys = [f"train::{tf}" for tf in TRAIN_TFS]
    emb_all = np.concatenate([emb[k] for k in tr_keys], axis=0)
    y_all = np.concatenate([Y[k] for k in tr_keys], axis=0)
    idx_tr, idx_va = train_test_split(np.arange(len(y_all)), test_size=0.2, random_state=seed)
    moe, gate_val_auc = train_gate(emb_all[idx_tr], y_all[idx_tr], emb_all[idx_va],
                                   y_all[idx_va], num_experts, embedding_dim, seed,
                                   l2norm=l2norm, entropy_reg=entropy_reg,
                                   gate_temperature=gate_temperature)

    # ---- evaluate on the 9 eval datasets ----
    rows, gate_entropies = {}, []
    for k, (dtype, tf) in meta.items():
        if dtype == "train":
            continue
        y = Y[k]
        moe_p, gw = gate_predict(moe, emb[k])
        # controls on the SAME zoo (Reviewer 1): static mean + per-TF oracle + best-single
        static = pred[k].mean(axis=0)
        per_expert_auc = [auc(y, pred[k][j]) for j in range(num_experts)]
        oracle = max(a for a in per_expert_auc if a == a)
        rows[k] = {
            "data_type": dtype, "tf": tf,
            "hetmoe_auc": auc(y, moe_p),
            "static_mean_auc": auc(y, static),
            "oracle_auc": oracle,
            "best_expert": order[int(np.nanargmax(per_expert_auc))],
            "per_expert_auc": dict(zip(order, [round(a, 4) for a in per_expert_auc])),
            "hetmoe_brier": brier(y, moe_p), "hetmoe_ece": ece(y, moe_p),
            "gate_entropy": float(-(gw * np.log(gw + 1e-9)).sum(1).mean()),
        }
        gate_entropies.append(rows[k]["gate_entropy"])

    def mean_over(tfs, field):
        vals = [rows[f"out_of_distribution::{tf}"][field] for tf in tfs]
        return float(np.mean(vals))

    tag = tag or "hetmoe"
    summary = {
        "tag": tag, "seed": seed, "num_experts": num_experts,
        "config": {"l2norm": l2norm, "entropy_reg": entropy_reg,
                   "gate_temperature": gate_temperature, "embedding_dim": embedding_dim},
        "expert_order": order, "gate_val_auc": gate_val_auc,
        "mean_gate_entropy": float(np.mean(gate_entropies)),
        "ood_mean_hetmoe": mean_over(OOD_TFS, "hetmoe_auc"),
        "ood_mean_oracle": mean_over(OOD_TFS, "oracle_auc"),
        "ood_mean_static": mean_over(OOD_TFS, "static_mean_auc"),
        "ood_learnable_hetmoe": mean_over(OOD_LEARNABLE, "hetmoe_auc"),
        "ood_learnable_oracle": mean_over(OOD_LEARNABLE, "oracle_auc"),
        "ood_indirect_hetmoe": mean_over(OOD_INDIRECT, "hetmoe_auc"),
        "indist_mean_hetmoe": float(np.mean([rows[f"in_distribution::{tf}"]["hetmoe_auc"]
                                             for tf in TRAIN_TFS])),
        "dnabert_ref_ood": 0.7492, "deepsea_ref_ood": 0.6918, "danq_ref_ood": 0.6881,
        "per_dataset": rows,
    }
    with open(os.path.join(out, f"decision_{tag}_seed{seed}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ DECISION GATE ================", flush=True)
    print(f"tag={tag} seed={seed} experts={num_experts} "
          f"l2norm={l2norm} entropy={entropy_reg} tau={gate_temperature}")
    print(f"  OOD mean   HetMoE={summary['ood_mean_hetmoe']:.4f}  "
          f"ORACLE={summary['ood_mean_oracle']:.4f}  static={summary['ood_mean_static']:.4f}")
    print(f"  OOD learnable (CTCF,STAT3)  HetMoE={summary['ood_learnable_hetmoe']:.4f}  "
          f"ORACLE={summary['ood_learnable_oracle']:.4f}")
    print(f"  in-dist HetMoE={summary['indist_mean_hetmoe']:.4f}  "
          f"gate_entropy={summary['mean_gate_entropy']:.3f}")
    print(f"  reference: DNABERT-6 OOD=0.7492  DeepSEA=0.6918  DanQ=0.6881")
    verdict = ("PROCEED" if summary["ood_mean_oracle"] > 0.749 else "STOP/PARITY")
    print(f"  >>> oracle {'>' if summary['ood_mean_oracle']>0.749 else '<='} 0.749  -> {verdict}")
    print("===============================================\n", flush=True)
    return summary


def decision_gate(fm_name="dnabert2", embedding_dim=32, seed=42, l2norm=True,
                  entropy_reg=1e-3, gate_temperature=1.0, out="./results/phase0",
                  backbones=None, dnabert_epochs=3):
    """Phase 0: build the zoo (and cache it), then train+evaluate the gate for one
    config. ``tag`` encodes the backbone set so CNN-only and full-zoo runs don't clash."""
    zoo = build_zoo(fm_name=fm_name, embedding_dim=embedding_dim, seed=seed,
                    backbones=backbones, dnabert_epochs=dnabert_epochs)
    seen, ordered_bb = set(), []
    for n in zoo["expert_order"]:
        b = n.split("::")[0]
        if b not in seen:
            seen.add(b); ordered_bb.append(b)
    tag = "-".join(ordered_bb)
    return run_config(zoo, seed=seed, l2norm=l2norm, entropy_reg=entropy_reg,
                      gate_temperature=gate_temperature, out=out, tag=tag)


def full_evaluation(zoo, seed=42, l2norm=True, entropy_reg=1e-3, gate_temperature=1.0,
                    n_boot=1000, base_seed=42, tost_margin=0.01, out="./results/hetmoe"):
    """Phase C: publication-grade rigorous evaluation of the selected HetMoE config.

    Paired bootstrap (common resample indices per dataset) for HetMoE, the static-mean
    and best-single-expert controls (Reviewer 1), and -- when a DNABERT-6 baseline is
    cached -- the paired HetMoE-minus-DNABERT difference with a percentile CI, a
    two-sided p, and a TOST non-inferiority verdict at +/-``tost_margin``. Writes
    hetmoe_summary.csv, bootstrap_paired_vs_dnabert.csv, hetmoe_eval.json."""
    os.makedirs(out, exist_ok=True)
    order = zoo["expert_order"]
    emb, pred, Y, meta = zoo["emb"], zoo["pred"], zoo["y"], zoo["meta"]
    embedding_dim, num_experts = zoo["embedding_dim"], len(order)
    keys = [k for k in meta if meta[k][0] != "train"]

    tr_keys = [f"train::{tf}" for tf in TRAIN_TFS]
    emb_all = np.concatenate([emb[k] for k in tr_keys], axis=0)
    y_all = np.concatenate([Y[k] for k in tr_keys], axis=0)
    idx_tr, idx_va = train_test_split(np.arange(len(y_all)), test_size=0.2, random_state=seed)
    moe, _ = train_gate(emb_all[idx_tr], y_all[idx_tr], emb_all[idx_va], y_all[idx_va],
                        num_experts, embedding_dim, seed, l2norm=l2norm,
                        entropy_reg=entropy_reg, gate_temperature=gate_temperature)

    dnabert = _load_dnabert_baseline(keys, Y, _cdir(seed))
    summary_rows, paired_rows, per_ds = [], [], {}
    best_single_idx = int(np.nanargmax([np.nanmean(
        [auc(Y[f"in_distribution::{tf}"], pred[f"in_distribution::{tf}"][j]) for tf in TRAIN_TFS])
        for j in range(num_experts)]))  # selected on in-dist only

    for k in keys:
        dtype, tf = meta[k]
        y = Y[k]
        moe_p, gw = gate_predict(moe, emb[k])
        preds = {"HetMoE": moe_p, "static_mean": pred[k].mean(0),
                 "best_single": pred[k][best_single_idx]}
        if dnabert is not None:
            preds["DNABERT"] = dnabert[k]
        boot = paired_bootstrap(preds, y, base_seed + keys.index(k), n_boot)
        rec = {"data_type": dtype, "tf": tf}
        for m in preds:
            lo, hi = np.percentile(boot[m], [2.5, 97.5])
            rec[f"{m}_auc"] = auc(y, preds[m]); rec[f"{m}_ci"] = [float(lo), float(hi)]
            summary_rows.append({"model": m, "data_type": dtype, "tf": tf,
                                 "point_auc": auc(y, preds[m]), "mean_auc": float(boot[m].mean()),
                                 "ci95_low": float(lo), "ci95_high": float(hi)})
        rec["HetMoE_brier"] = brier(y, moe_p); rec["HetMoE_ece"] = ece(y, moe_p)
        rec["gate_entropy"] = float(-(gw * np.log(gw + 1e-9)).sum(1).mean())
        rec["gate_weights"] = gw.mean(0).tolist()  # per-expert mean gate weight (heatmap)
        if dnabert is not None:
            d = boot["HetMoE"] - boot["DNABERT"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
            paired_rows.append({"data_type": dtype, "tf": tf, "mean_diff": float(d.mean()),
                                "ci95_low": float(lo), "ci95_high": float(hi), "p_value": p,
                                "superior": bool(lo > 0),
                                "non_inferior_TOST": bool(lo > -tost_margin)})
        per_ds[k] = rec

    def ood_mean(field, tfs=OOD_TFS):
        return float(np.mean([per_ds[f"out_of_distribution::{tf}"][field] for tf in tfs]))

    result = {
        "config": {"l2norm": l2norm, "entropy_reg": entropy_reg,
                   "gate_temperature": gate_temperature, "seed": seed,
                   "num_experts": num_experts, "expert_order": order},
        "ood_mean": {m: ood_mean(f"{m}_auc") for m in
                     (["HetMoE", "static_mean", "best_single"] + (["DNABERT"] if dnabert else []))},
        "ood_learnable_hetmoe": ood_mean("HetMoE_auc", OOD_LEARNABLE),
        "ood_indirect_hetmoe": ood_mean("HetMoE_auc", OOD_INDIRECT),
        "indist_mean_hetmoe": float(np.mean([per_ds[f"in_distribution::{tf}"]["HetMoE_auc"]
                                             for tf in TRAIN_TFS])),
        "ood_mean_ece": ood_mean("HetMoE_ece"), "ood_mean_brier": ood_mean("HetMoE_brier"),
        "tost_margin": tost_margin, "per_dataset": per_ds, "paired_vs_dnabert": paired_rows,
        "references": {"DNABERT6": 0.7492, "DeepSEA": 0.6918, "DanQ": 0.6881, "orig_MoE": 0.6829},
    }
    import pandas as pd
    pd.DataFrame(summary_rows).to_csv(os.path.join(out, "hetmoe_summary.csv"), index=False)
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(os.path.join(out, "bootstrap_paired_vs_dnabert.csv"), index=False)
    with open(os.path.join(out, "hetmoe_eval.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\n============== FULL EVALUATION (Phase C) ==============", flush=True)
    for m, v in result["ood_mean"].items():
        print(f"  OOD mean {m:12s} = {v:.4f}")
    print(f"  OOD learnable(CTCF,STAT3) HetMoE = {result['ood_learnable_hetmoe']:.4f}")
    print(f"  in-dist HetMoE = {result['indist_mean_hetmoe']:.4f} | "
          f"OOD ECE={result['ood_mean_ece']:.3f} Brier={result['ood_mean_brier']:.3f}")
    if paired_rows:
        ood_pr = [r for r in paired_rows if r["data_type"] == "out_of_distribution"]
        nsup = sum(r["superior"] for r in ood_pr); nni = sum(r["non_inferior_TOST"] for r in ood_pr)
        print(f"  vs DNABERT (per-OOD-TF): superior CI-excludes-0 on {nsup}/6, "
              f"non-inferior(TOST +/-{tost_margin}) on {nni}/6")
        for r in ood_pr:
            flag = "SUP" if r["superior"] else ("NI" if r["non_inferior_TOST"] else "—")
            print(f"     {r['tf']:8s} diff={r['mean_diff']:+.4f} CI[{r['ci95_low']:+.4f},{r['ci95_high']:+.4f}] {flag}")
    print("======================================================\n", flush=True)
    return result
