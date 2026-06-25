"""Phase B: train the embedding-gate for ONE configuration over the cached expert zoo
(no GPU / no expert forward needed) and write its in-distribution + stratified-OOD
metrics. Dozens of these run cheaply in parallel reusing the single Phase-A cache.

The config grid (N_e subset x gate l2norm x entropy x temperature x 12-way/grouped) is
enumerated by the Phase-B SLURM array; selection across configs uses IN-DISTRIBUTION
validation AUC only (Phase C), never OOD.

    python main_hetmoe.py --seed 42 --backbones ConvNet,DNABERT6 --l2norm 1 --entropy_reg 1e-3
    python main_hetmoe.py --seed 42 --group 1 --tag grouped4        # 4-way grouped gate
"""
import argparse

from tfbs.evaluate_hetmoe import run_config
from tfbs.experts import load_zoo_cache, subset_zoo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbones", default=None,
                    help="comma subset of {ConvNet,DeepSEA,DanQ,DNABERT6}; default = all cached")
    ap.add_argument("--group", type=int, default=0, help="1 = 4-way grouped-by-backbone gate")
    ap.add_argument("--l2norm", type=int, default=1)
    ap.add_argument("--entropy_reg", type=float, default=1e-3)
    ap.add_argument("--gate_temperature", type=float, default=1.0)
    ap.add_argument("--out", default="./results/moe_grid")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    zoo = load_zoo_cache(args.seed)
    bb = args.backbones.split(",") if args.backbones else None
    if bb or args.group:
        zoo = subset_zoo(zoo, backbones=bb, group_by_backbone=bool(args.group))

    if args.tag:
        tag = args.tag
    else:
        bbset = "-".join(sorted({n.split("::")[0] for n in zoo["expert_order"]}))
        tag = (f"{'group_' if args.group else ''}{bbset}"
               f"_l2{args.l2norm}_e{args.entropy_reg:g}_t{args.gate_temperature:g}")
    run_config(zoo, seed=args.seed, l2norm=bool(args.l2norm), entropy_reg=args.entropy_reg,
               gate_temperature=args.gate_temperature, out=args.out, tag=tag)


if __name__ == "__main__":
    main()
