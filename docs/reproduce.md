# Reproducing the results

All commands are run **from the repository root** with the package installed
(`pip install -e .`). The HF environment variables avoid slow online metadata calls
once weights are cached:

```sh
export HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## 1. Original Mixture-of-Experts pipeline (3 ConvNet experts + soft gate)

```sh
# train experts + MoE reusing the documented hyperparameters (no Optuna):
python -m experiments.train.main --seed 42 --use_saved_hyperparams
# canonical evaluation (real 50/50 sets, dropout off, paired bootstrap):
python -m experiments.train.evaluate --protocol rigorous
# -> results/evaluation_summary.csv, evaluation_results.json, bootstrap_paired.csv
python -m experiments.analysis.stats          # ANOVA + eta^2/omega^2 + post-hoc
```

## 2. Heterogeneous MoE (the OOD-winning extension)

```sh
# Phase A (GPU): build + cache the zoo (ConvNet + DeepSEA + DanQ + fine-tuned DNABERT-6)
python -m experiments.hetmoe.cache_embeddings --seed 42 --backbones ConvNet,DeepSEA,DanQ,DNABERT6
# Phase B+C (cheap, reuses the cache): sweep configs, select on in-dist val, full eval
python -m experiments.hetmoe.sweep --seed 42
python -m experiments.hetmoe.aggregate_seeds  # multi-seed mean +/- std
# On the cluster (chained):
#   sbatch slurm/phaseA_cache.sbatch 42
#   sbatch --dependency=afterok:<A_jobid> slurm/phaseBC.sbatch 42
```

## 3. Other analyses

```sh
python -m experiments.baselines.baselines [--skip_dnabert]   # DeepSEA/DanQ/DNABERT comparison
python -m experiments.ablation.ablation                      # embedding / frozen / num_experts sweeps
python -m experiments.attribution.shiftsmooth_eval --n_seqs 60
python -m experiments.analysis.make_paper_figures            # Nature figure pack -> results/figures/nature/
```

---

## Reproducibility caveats (important — read before re-running)

The pipeline is deterministic **for a fixed (seed, machine/device, expert-construction
order)**, but it is **not portable bit-for-bit across machines**, for two structural reasons
discovered during the refactor:

1. **`ConvNet.wRect` (the conv bias) is a plain tensor, not an `nn.Parameter`.** It is
   re-randomised on every construction and is **not saved in the checkpoint**, so
   `load_state_dict` never restores it. Expert (and therefore MoE) predictions depend on the
   RNG state at construction time — which depends on the seed *and* on how much RNG was consumed
   earlier in the run (e.g. building data loaders, expert ordering). Consequence: the committed
   `results/evaluation_summary.csv` (MoE OOD AUC **0.6829**) is a specific GPU run; a fresh run of
   the same code yields a *nearby but not identical* number (≈0.678 on H200, ≈0.658 on CPU).
   *Fully* pinning the result requires promoting `wRect` to a saved `nn.Parameter` and retraining
   (this would change the reported numbers, so it was intentionally **not** done in the cleanup).

2. **Expert order comes from unsorted `os.listdir`**, which is filesystem-dependent. The MoE
   concatenates per-expert embeddings in that order, so the gate must be evaluated in the same
   order it was trained in. The canonical order is `[ARID3A, FOXM1, GATA3]`
   (`tfbs.constants.TRAIN_TFS`); if you reorganise `data/train`, pin this order explicitly.

3. **Device numerics.** CPU (oneDNN) vs CUDA (cuDNN) forward differences are amplified by the
   random bias above, so AUCs shift by up to a few points between CPU and GPU. Run the canonical
   reproduction on a **GPU** node (see `slurm/`).

### How the refactor was verified (behavior-neutral)

The cleanup changed only file layout, dead-code removal, and import paths — never model logic.
Neutrality was checked by running the **pre-** and **post-** refactor code in identical conditions:
- core `evaluate --protocol rigorous` on CPU: byte-identical `evaluation_summary.csv` before/after;
- HetMoE `full_evaluation` on the seed-42 cache (CPU): byte-identical `hetmoe_summary.csv` before/after;
- a GPU SLURM job re-running the current `evaluate` reproduces the fresh-run baseline.
