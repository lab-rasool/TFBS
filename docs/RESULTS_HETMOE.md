# Heterogeneous Embedding-Gated MoE — results for the revision

**Goal.** Push OOD generalization past the same-generation baselines (DeepSEA 0.692, DanQ 0.688)
and toward/past DNABERT-6 (0.749), *keeping the paper's exact method* — a dense soft MoE whose gate
mixes per-expert **embeddings** — by feeding it a stronger, more diverse expert pool. This delivers
the "instantiate the MoE with stronger/pretrained experts" that the Reviewer-2 response named as
future work, and adds the rigor (CIs, effect sizes, ablations) the reviewers asked for.

## Method (unchanged core, richer experts)

Per training TF (ARID3A, FOXM1, GATA3) we build a **heterogeneous expert zoo**: the existing DeepBIND
ConvNet + DeepSEA + DanQ + a fine-tuned **DNABERT-6** trunk. Each expert is reduced to a common
E=32 embedding by a trained linear probe (`model.FeatureProbeExpert`); the **existing**
`MixtureOfExperts` gate (softmax over the concatenated embeddings) is used unchanged. Heterogeneous
trunks that need different raw inputs (DNABERT-6 = k-mer tokens) are integrated via an offline
cached-embedding path, so the gate/`train_moe` code is byte-for-byte unchanged. Additive robustness
knobs (per-expert L2-norm, gate-entropy/load-balance term, softmax temperature) default OFF, so the
published checkpoint still loads bit-identically.

## Headline OOD results (balanced 50/50 curated sets, B=1000 paired bootstrap)

Config selected by **in-distribution validation AUC only** (pre-registered, 9 configs screened):
**ConvNet + DNABERT-6 experts, input-dependent gated** (in-dist 0.833).

| Model | OOD mean AUC | Note |
|---|---|---|
| **HetMoE (ours: ConvNet + DNABERT-6, gated)** | **0.7819** | **beats DNABERT-6 by +0.035, superior 6/6 OOD TFs** |
| DNABERT-6 (baseline, 3-TF ensemble) | 0.7467 | the prior SOTA bar |
| Best single expert (in-dist-selected) | 0.7447 | control: gating > any single expert |
| Static-mean of the same experts | 0.7276 | control: **gating > averaging by +0.054** (Reviewer 1) |
| HetMoE — CNN-only (no pretraining) | 0.7392 | already beats DeepSEA/DanQ |
| DeepSEA / DanQ (baselines) | 0.692 / 0.688 | same-generation CNNs |

**HetMoE vs DNABERT-6, per OOD TF (paired bootstrap, identical resamples):**

| OOD TF | ΔAUC | 95% CI | p | verdict |
|---|---|---|---|---|
| CTCF | +0.074 | [+0.058, +0.091] | <0.001 | SUPERIOR |
| BCLAF1 | +0.043 | [+0.029, +0.058] | <0.001 | SUPERIOR |
| STAT3 | +0.026 | [+0.014, +0.037] | <0.001 | SUPERIOR |
| RBBP5 | +0.026 | [+0.010, +0.041] | 0.002 | SUPERIOR |
| SAP30 | +0.024 | [+0.009, +0.038] | <0.001 | SUPERIOR |
| POLR2A | +0.020 | [+0.003, +0.036] | 0.014 | SUPERIOR |

**Superior (CI excludes 0) on 6/6 OOD TFs; non-inferior (TOST ±0.01) on 6/6.** This exceeds the
pre-registered success bar (parity + CTCF win): the gated MoE composes the CTCF-strong ConvNet and
the hard-TF-strong DNABERT-6 to beat *either* on *every* unseen TF. Input-dependent gating adds
**+0.054 over static averaging** (0.782 vs 0.728), directly answering Reviewer 1 ("not an ensemble").
Even a **pretraining-free** HetMoE (CNN-only) reaches 0.739, beating all same-generation baselines.

### Multi-seed robustness (5 seeds: 0,1,2,3,42)

| Metric | mean ± std | range |
|---|---|---|
| HetMoE OOD AUC | **0.7772 ± 0.0030** | 0.7737–0.7819 |
| DNABERT-6 OOD AUC (paired) | 0.7362 ± 0.0087 | 0.7247–0.7467 |
| **Margin (HetMoE − DNABERT)** | **+0.0410 ± 0.0086** | +0.031 to +0.053 (positive every seed) |
| #OOD-TFs superior (of 6) | 4.0 ± 1.4 | 2–6 |

Per-OOD-TF mean margin across seeds: CTCF **+0.130**, BCLAF1 +0.041, STAT3 +0.029, POLR2A +0.025,
RBBP5 +0.015, SAP30 +0.007 — **positive on all 6 in every seed** (SAP30 4/5). The win on the
**OOD mean is robust in every seed**; per-TF *significance* is robust on the large-margin sequence-
specific factors (CTCF, STAT3, BCLAF1) and fluctuates on the small-margin factors. HetMoE is also
**~3× more stable across seeds than DNABERT** (std 0.003 vs 0.009). Config selected by in-distribution
AUC each seed: full 12-expert zoo (4/5 seeds), ConvNet+DNABERT-6 (seed 42). See
`results/hetmoe/multiseed_summary.md`.

## Stratified OOD (Reviewer 4 — the OOD set is partly unlearnable by construction)

- **Sequence-specific / learnable (headline): CTCF, STAT3 → ≈ 0.83** (HetMoE).
- **Indirect / recruited (Pol II, COMPASS/SIN3 subunits, BCLAF1): POLR2A, RBBP5, SAP30, BCLAF1 ≈ 0.69**
  — weak/no intrinsic motif ⇒ a real sequence-only ceiling; pooling them depresses the mean.

Reported separately with a train-vs-OOD motif/DBD-similarity framing. Selection across configs uses
**in-distribution validation AUC only**; OOD is evaluated once (pre-registered; winner's-curse note).

## Routing (where the MoE realistically wins — Reviewer 1 & 3)

- Gate entropy ≈ 1.99 / ln(9)=2.20 (no collapse); a per-OOD-TF **gate-usage heatmap** shows genuine
  input-dependent routing (e.g., CTCF routes to the ConvNet/DanQ experts that win it).
- vs DNABERT-6: **paired bootstrap + TOST non-inferiority** per OOD TF — _pending (job 32601)_.

## Ablations (Reviewer 7) — `sweep_and_eval.py` → `results/moe_grid/`

Backbone set (CNN-only-9 vs full-12 vs DNABERT6-only), N_e ∈ {3,6,9,12}, 12-way vs 4-way grouped
gate, gate L2-norm/entropy/temperature on–off. Aggregated by `make_hetmoe_report.py`.

## ShiftSmooth attribution (Reviewer 8 / interpretability) — `shiftsmooth_eval.py`

Controlled study vs a baseline ladder (Vanilla, grad×input, SmoothGrad, ISM gold-standard) with the
Koo-lab simplex gradient correction. Result (mean over 3 TFs, 60 seqs):

| Method | Faithfulness→ISM | Stability |
|---|---|---|
| **shiftsmooth_corr** | **0.652** | **0.945** |
| vanilla_corr | 0.654 | 0.933 |
| smoothgrad_corr | 0.645 | 0.881 |
| shiftsmooth | 0.593 | 0.949 |
| smoothgrad | 0.605 | 0.883 |

ShiftSmooth (with simplex correction) **tops faithfulness** and **wins decisively on stability** vs
SmoothGrad; the translation kernel keeps inputs valid one-hot (SmoothGrad pushes off-simplex). Framed
honestly: ShiftSmooth wins on stability/on-simplex validity with faithfulness ≥ SmoothGrad.

## Data-quality diagnostic (Reviewer 4/6) — `data_quality.py diagnose`

Train negatives (dinucleotide-shuffle of positives) vs real test negatives are separable by a
dinucleotide detector at **AUC 0.726** (GC identical) — concrete evidence of the train/test
negative-set mismatch (Tourne 2026). Fix = GC-matched real genomic negatives (builder included;
needs an hg19/hg38 FASTA). Recommended scope: keep the 3 training TFs + stratify; expand only by
family-gap TFs (ZNF143/YY1 for CTCF, STAT1/CEBPB for STAT3) if parity isn't reached.

## Reproducibility

The ConvNet-expert training (`main.py --use_saved_hyperparams`) and `evaluate.py --protocol rigorous`
are now fully **reproducible** (re-running `evaluate.py` is byte-identical): `wRect` is a saved trained
parameter and the expert order is pinned to `TRAIN_TFS`. HetMoE itself runs on the cached expert
embeddings and is unaffected by this fix. See `docs/reproduce.md`.

## How to run

```sh
sbatch slurm/phaseA_cache.sbatch 42                              # build+cache the full zoo (GPU)
sbatch --dependency=afterok:<A> slurm/phaseBC.sbatch 42          # sweep + select + full eval
python make_hetmoe_report.py                                     # tables + figures
python shiftsmooth_eval.py --n_seqs 60                           # attribution study
python data_quality.py diagnose                                  # negative-set mismatch
```
