# Transcription-Factor Binding-Site (TFBS) prediction with a heterogeneous Mixture-of-Experts

Prediction of transcription-factor binding sites with a dense, soft **Mixture-of-Experts (MoE)**
that gates over per-expert *embeddings*, plus a **heterogeneous expert zoo** (modified-DeepBIND
ConvNet + DeepSEA + DanQ + fine-tuned DNABERT-6) that improves **out-of-distribution (OOD)**
generalization, and a "ShiftSmooth" attribution method. This code backs the paper (LaTeX in
`paper/`).

**Headline result.** Feeding the unchanged embedding-gating MoE a *heterogeneous* expert pool beats
DNABERT-6 on OOD: **0.777 ± 0.003 vs 0.736 ± 0.009** over 5 seeds. See
[`docs/RESULTS_HETMOE.md`](docs/RESULTS_HETMOE.md).

## Repository layout

```
tfbs/            importable library
  constants.py   TF lists, OOD stratification, path roots
  data.py        ChIP-seq loaders, one-hot, dinuc-shuffle negatives
  models.py      ConvNet expert, FeatureProbeExpert, MixtureOfExperts gate
  experts.py     heterogeneous zoo: feature extractors, probes, build/cache/load/subset
  gate.py        gate training / prediction over cached embeddings
  metrics.py     bootstrap / paired / TOST / ECE / Brier helpers
  evaluate_hetmoe.py  HetMoE decision gate + publication evaluation
  baselines.py   DeepSEA / DanQ / DNABERT models + trainers
  utils.py       seeding, early stopping, file discovery
experiments/     runnable CLIs (thin wrappers; run as `python -m experiments.<group>.<name>`)
  train/         main.py (two-stage training), evaluate.py (canonical evaluation)
  hetmoe/        cache_embeddings, sweep, decision_gate, aggregate_seeds
  baselines/     baselines.py (comparison CLI)
  ablation/      ablation.py
  attribution/   shiftsmooth_eval.py + notebooks
  analysis/      stats.py, make_paper_figures.py, data_quality.py
data/            ChIP-seq inputs (see data/README.md for conventions)
models/          checkpoints — gitignored, kept local (data + models to be HuggingFace-hosted)
results/         summaries + figures tracked; cache/ gitignored (see docs/results_layout.md)
slurm/           cluster job scripts
docs/            results report, reviewer responses, reproduce.md, results_layout.md
paper/           LaTeX (not under git)
archive/         deprecated legacy/ + old/ material (gitignored)
```

## Installation

```sh
git clone https://github.com/Aakash-Tripathi/TFBS.git
cd TFBS
python -m venv venv && source venv/bin/activate
# install the CUDA build of torch for your system (see requirements.txt), then:
pip install -e .          # installs the `tfbs` package + dependencies
```

Tested on Python 3.13, PyTorch 2.8 / CUDA 12.8 (original experiments on an RTX 3090; HetMoE on
cluster H100/H200). `optuna` is only needed for the per-expert hyperparameter search (skip it with
`--use_saved_hyperparams`).

## Usage

Run everything **from the repository root**. Set the HuggingFace cache once weights are local:

```sh
export HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

```sh
# Original MoE: train (reusing saved hyperparameters) then evaluate
python -m experiments.train.main --seed 42 --use_saved_hyperparams
python -m experiments.train.evaluate --protocol rigorous
python -m experiments.analysis.stats

# Heterogeneous MoE (the OOD-winning extension)
python -m experiments.hetmoe.cache_embeddings --seed 42 --backbones ConvNet,DeepSEA,DanQ,DNABERT6   # Phase A (GPU)
python -m experiments.hetmoe.sweep --seed 42                                                        # Phase B+C
python -m experiments.hetmoe.aggregate_seeds

# Baselines, ablations, attribution, figures
python -m experiments.baselines.baselines
python -m experiments.ablation.ablation
python -m experiments.attribution.shiftsmooth_eval --n_seqs 60
python -m experiments.analysis.make_paper_figures
```

On the cluster, submit the chained SLURM jobs in `slurm/` (see `docs/reproduce.md`).

## Results & reproducibility

The original MoE and the HetMoE pipelines are reproducible from the saved checkpoints/cache.
**Model checkpoints (`models/`) are not committed** — they're kept local for now and will be hosted on
HuggingFace (together with the data); regenerate them via training, or request them. See
[`docs/reproduce.md`](docs/reproduce.md) for exact commands **and important reproducibility caveats**
(the conv bias `wRect` is not persisted, and expert order derives from `os.listdir`, so headline
numbers are deterministic per machine but not bit-portable across machines). Attribution results are
reproduced by `experiments/attribution/attributes_motif.ipynb` and `attributes_ood.ipynb`.

## License

See the [LICENSE](LICENSE) file.
