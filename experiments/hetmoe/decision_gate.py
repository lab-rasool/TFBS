"""Phase-0 decision gate CLI: build the heterogeneous zoo, train the gate over the
concatenated expert embeddings, and report whether the per-TF ORACLE over the zoo can
clear the DNABERT-6 0.749 OOD bar before launching the full sweep. Thin wrapper over
``tfbs.evaluate_hetmoe.decision_gate``.

Run from the repo root:
    python -m experiments.hetmoe.decision_gate --backbones ConvNet,DeepSEA,DanQ,DNABERT6
"""
import argparse

from tfbs.experts import _FM_REGISTRY
from tfbs.evaluate_hetmoe import decision_gate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fm", default="dnabert2", choices=list(_FM_REGISTRY))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embedding_dim", type=int, default=32)
    ap.add_argument("--l2norm", type=int, default=1)
    ap.add_argument("--entropy_reg", type=float, default=1e-3)
    ap.add_argument("--gate_temperature", type=float, default=1.0)
    ap.add_argument("--out", default="./results/phase0")
    ap.add_argument("--backbones", default=None,
                    help="comma list, e.g. 'ConvNet,DeepSEA,DanQ,DNABERT6' "
                         "(default) or 'ConvNet,DeepSEA,DanQ' for a fast CNN-only run")
    ap.add_argument("--dnabert_epochs", type=int, default=3)
    args = ap.parse_args()
    backbones = args.backbones.split(",") if args.backbones else None
    decision_gate(fm_name=args.fm, embedding_dim=args.embedding_dim, seed=args.seed,
                  l2norm=bool(args.l2norm), entropy_reg=args.entropy_reg,
                  gate_temperature=args.gate_temperature, out=args.out,
                  backbones=backbones, dnabert_epochs=args.dnabert_epochs)


if __name__ == "__main__":
    main()
