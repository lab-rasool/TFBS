"""Phase B + C over the cached zoo (cheap; no expert forward).

B: train the gate for an enumerated config grid (N_e subset x grouping x l2norm x
   entropy x temperature), writing each config's in-dist + stratified-OOD metrics.
C: select the SINGLE config by IN-DISTRIBUTION validation AUC only (pre-registered;
   OOD is never used for selection), then run the publication-grade full_evaluation
   (bootstrap CIs + paired HetMoE-vs-DNABERT + TOST + calibration) on it.

    python sweep_and_eval.py --seed 42
"""
import argparse

from tfbs.evaluate_hetmoe import full_evaluation, run_config
from tfbs.experts import load_zoo_cache, subset_zoo

# (backbones subset, grouped, l2norm, entropy, temperature, tag)
# Tags are count-agnostic: the actual #experts = (#backbones in the subset) x
# (#training TFs). With the 7-TF family-aware set, "full" = 4x7 = 28 experts.
CONFIGS = [
    dict(backbones=None, group=False, l2norm=True,  entropy=1e-3, temp=1.0, tag="full"),
    dict(backbones=None, group=False, l2norm=False, entropy=0.0,  temp=1.0, tag="full_noknobs"),
    dict(backbones=None, group=False, l2norm=True,  entropy=1e-3, temp=2.0, tag="full_tau2"),
    dict(backbones=None, group=False, l2norm=True,  entropy=1e-2, temp=1.0, tag="full_ent1e-2"),
    dict(backbones=None, group=True,  l2norm=True,  entropy=1e-3, temp=1.0, tag="grouped_by_backbone"),
    dict(backbones=["ConvNet", "DNABERT6"], group=False, l2norm=True, entropy=1e-3, temp=1.0, tag="convnet_dnabert6"),
    dict(backbones=["ConvNet", "DeepSEA", "DanQ"], group=False, l2norm=True, entropy=1e-3, temp=1.0, tag="cnn_only"),
    dict(backbones=["DNABERT6"], group=False, l2norm=True, entropy=0.0, temp=1.0, tag="dnabert6_only"),
    dict(backbones=["ConvNet"], group=False, l2norm=True, entropy=0.0, temp=1.0, tag="convnet_only"),
]


def _zoo_for(base, c):
    if c["backbones"] or c["group"]:
        return subset_zoo(base, backbones=c["backbones"], group_by_backbone=c["group"])
    return base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    from tfbs.constants import MODE_SUFFIX
    grid_out = f"./results/moe_grid/seed{args.seed}{MODE_SUFFIX}"
    hetmoe_out = f"./results/hetmoe/seed{args.seed}{MODE_SUFFIX}"
    base = load_zoo_cache(args.seed)
    results = []
    for c in CONFIGS:
        s = run_config(_zoo_for(base, c), seed=args.seed, l2norm=c["l2norm"],
                       entropy_reg=c["entropy"], gate_temperature=c["temp"],
                       out=grid_out, tag=c["tag"])
        results.append((c, s))

    # ---- pre-registered selection: IN-DISTRIBUTION mean AUC only (never OOD) ----
    best_c, best_s = max(results, key=lambda r: r[1]["indist_mean_hetmoe"])
    print(f"\n[select] best config by in-dist AUC: {best_c['tag']} "
          f"(in-dist {best_s['indist_mean_hetmoe']:.4f}); its OOD {best_s['ood_mean_hetmoe']:.4f}")
    print(f"[select] (#configs screened = {len(CONFIGS)}; Bonferroni-note this in the paper)")

    full_evaluation(_zoo_for(base, best_c), seed=args.seed, l2norm=best_c["l2norm"],
                    entropy_reg=best_c["entropy"], gate_temperature=best_c["temp"],
                    n_boot=args.n_boot, out=hetmoe_out)


if __name__ == "__main__":
    main()
