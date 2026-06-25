"""CLI: train DeepSEA / DanQ / DNABERT baselines and compare them against the
expert/MoE models on the rigorous 50/50 protocol. Thin wrapper over the library
(``tfbs.baselines``).

Run from the repo root:
    python -m experiments.baselines.baselines [--skip_dnabert]
"""

from tfbs.baselines import main

if __name__ == "__main__":
    main()
