# Transcription-Factor Binding-Site (TFBS) prediction with a heterogeneous Mixture-of-Experts

Prediction of transcription-factor binding sites with a dense, soft **Mixture-of-Experts (MoE)**
that gates over per-expert *embeddings*, plus a **heterogeneous expert zoo** (modified-DeepBIND
ConvNet + DeepSEA + DanQ + fine-tuned DNABERT-6) that improves **out-of-distribution (OOD)**
generalization, and a "ShiftSmooth" attribution method. This code backs the paper (LaTeX in
`paper/`).

**Headline result.** On the genomic, fair-negative protocol over the seven training factors, feeding
the unchanged embedding-gating MoE a *heterogeneous* expert pool (ConvNet + DeepSEA + DanQ) beats a
fine-tuned DNABERT-6 baseline on the motif-bearing OOD strata: **0.821 ± 0.005 vs 0.799 ± 0.008**
over seeds 0/1/42 (`results/hetmoe/genomic_multiseed_summary.txt`).

**Data & weights on the Hub.** The ENCODE ChIP-seq data are hosted at
[`Lab-Rasool/ENCODE-TFBS` (dataset)](https://huggingface.co/datasets/Lab-Rasool/ENCODE-TFBS) and the
trained checkpoints at [`Lab-Rasool/ENCODE-TFBS` (model)](https://huggingface.co/Lab-Rasool/ENCODE-TFBS).

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
data/            ChIP-seq inputs (see data/README.md for conventions; hosted on the Hub, below)
models/          checkpoints — gitignored, hosted on the Hub (Lab-Rasool/ENCODE-TFBS model repo)
results/         summaries + figures tracked; cache/ gitignored
```

## Installation

```sh
git clone https://github.com/lab-rasool/TFBS.git
cd TFBS
python -m venv venv && source venv/bin/activate
# install the CUDA build of torch for your system (see requirements.txt), then:
pip install -e .          # installs the `tfbs` package + dependencies
```

Tested on Python 3.13, PyTorch 2.8 / CUDA 12.8 (expert training on an RTX 3090; HetMoE on
cluster H100/H200). `optuna` is only needed for the per-expert hyperparameter search (skip it with
`--use_saved_hyperparams`).

## Usage

Run everything **from the repository root**. Set the HuggingFace cache once weights are local:

```sh
export HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

```sh
# Train the ConvNet experts (reusing saved hyperparameters) + canonical evaluation harness
python -m experiments.train.main --seed 42 --use_saved_hyperparams
python -m experiments.train.evaluate --protocol rigorous
python -m experiments.analysis.stats

# HetMoE — the proposed heterogeneous Mixture-of-Experts
python -m experiments.hetmoe.cache_embeddings --seed 42 --backbones ConvNet,DeepSEA,DanQ,DNABERT6   # Phase A (GPU)
python -m experiments.hetmoe.sweep --seed 42                                                        # Phase B+C
python -m experiments.hetmoe.aggregate_seeds

# Baselines, ablations, attribution, figures
python -m experiments.baselines.baselines
python -m experiments.ablation.ablation
python -m experiments.attribution.shiftsmooth_eval --n_seqs 60
python -m experiments.analysis.make_paper_figures
```

## Data & model checkpoints

Both the data and the trained weights live on the Hugging Face Hub under the **`Lab-Rasool/ENCODE-TFBS`**
namespace:

- **Dataset** — [`datasets/Lab-Rasool/ENCODE-TFBS`](https://huggingface.co/datasets/Lab-Rasool/ENCODE-TFBS):
  the ENCODE ChIP-seq `*_{AC,B}.seq.gz` files. Place them under `data/`.
- **Models** — [`Lab-Rasool/ENCODE-TFBS`](https://huggingface.co/Lab-Rasool/ENCODE-TFBS): the canonical
  checkpoints — 7 ConvNet experts (`experts/`), their hyperparameters (`hyperparams/`), the homogeneous
  ConvNet MoE (`moe/`), and the genomic heterogeneous zoo probes (`zoo/seed{0,1,42}/`, DeepSEA + DanQ).
  Download into `models/` (which is gitignored), e.g. `hf download Lab-Rasool/ENCODE-TFBS --local-dir models`.

## Reproducibility

The HetMoE pipeline (and the underlying ConvNet-expert training) is reproducible from the checkpoints
above. The conv bias `wRect` is a saved `nn.Parameter` and expert order is pinned to
`tfbs.constants.TRAIN_TFS`, so re-running `evaluate.py` is byte-identical on a given machine (minor
device-numerics differences may remain across machines). Attribution figures are reproduced by
`experiments/attribution/shiftsmooth_eval.py` and `experiments/attribution/make_attribution_figures.py`.

## License

See the [LICENSE](LICENSE) file.
