"""Phase A: build the heterogeneous expert zoo ONCE and cache every expert's E-dim
embedding (+ scalar prediction) for all train/eval datasets to results/cache/.

This is the only GPU-heavy step (it fine-tunes DeepSEA/DanQ/DNABERT-6 per TF and runs
their forwards). The cheap gate sweep (main_hetmoe.py) then reuses the cache with no
GPU and no expert forwards, so dozens of gate configs cost minutes total.

    python cache_embeddings.py --seed 42 --backbones ConvNet,DeepSEA,DanQ,DNABERT6
"""
import argparse

from tfbs.experts import build_zoo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--embedding_dim", type=int, default=32)
    ap.add_argument("--backbones", default="ConvNet,DeepSEA,DanQ,DNABERT6")
    ap.add_argument("--dnabert_epochs", type=int, default=3)
    args = ap.parse_args()
    backbones = args.backbones.split(",")
    zoo = build_zoo(embedding_dim=args.embedding_dim, seed=args.seed, backbones=backbones,
                    dnabert_epochs=args.dnabert_epochs, save_cache=True, save_models=True)
    print(f"[cache] cached {len(zoo['expert_order'])} experts x "
          f"{len(zoo['emb'])} datasets to results/cache/ (seed {args.seed})")
    print(f"[cache] expert order: {zoo['expert_order']}")


if __name__ == "__main__":
    main()
