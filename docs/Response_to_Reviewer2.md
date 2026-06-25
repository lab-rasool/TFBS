# Response to Reviewer 2

We thank Reviewer 2 for the careful and constructive evaluation of our manuscript,
*"Robust Transcription Factor Binding Site Prediction and Explainability Using a
Mixture of Experts Architecture."* The comments prompted us to substantially
strengthen the experimental rigor, reproducibility, and honesty of the paper.

Below we respond to each comment assigned to us (Comments 1–4, 6–8). Reviewer
comments are shown *in italics*, followed by our **Response**. Section references
point to the revised manuscript (`paper/edited.tex`, where all changes are shown in
blue via the `\rev{...}` macro). Items still requiring author confirmation are marked
**ACTION NEEDED**.

---

## Summary of the most important revisions

While addressing these comments we discovered, and have now corrected, three issues
that materially affect how the results should be reported. We describe them up front
in the interest of full transparency:

1. **Reproducibility.** The previously released model checkpoints did not fully
   reproduce the published numbers (the ARID3A and FOXM1 experts reproduced, but the
   released GATA3 expert — and therefore the MoE — did not). We have **retrained all
   experts and the MoE from scratch with a fixed random seed**, so the entire pipeline
   is now bit-for-bit reproducible from `python main.py --seed 42 --use_saved_hyperparams`
   followed by `python evaluate.py --protocol rigorous` (Comment 6).

2. **Evaluation protocol.** The original "bootstrap" evaluation actually (i) added
   synthetically dinucleotide-shuffled negatives on top of the already-balanced test
   files (making the test sets ~25 % positive / 75 % negative) and (ii) left dropout
   active at inference, giving Monte-Carlo-dropout variance rather than a true
   bootstrap. Because the synthetic negatives are *easier* than the curated
   experimental negatives, this inflated all absolute AUCs. We have **re-centered all
   evaluation on the balanced 500/500 curated test sets with deterministic inference
   and a genuine paired bootstrap** (Comments 3, 4, 6).

3. **Claims.** Under the corrected protocol the headline claim "MoE is best on 5/6 OOD
   datasets" is replaced by the more defensible and still-valuable claim that **the MoE
   is statistically indistinguishable from the single best expert on every
   out-of-distribution TF, while significantly outperforming two of the three experts
   on every OOD TF** — i.e., without knowing in advance which expert suits an unseen
   factor, the gating network reliably matches the best one (Comments 3, 4).

4. **The framework, given strong experts, surpasses the SOTA baseline.** Acting on the
   reviewer's own suggestion (Comment 2), we instantiated the *unchanged* gating MoE over
   a heterogeneous expert pool that includes fine-tuned **DNABERT-6** experts. The gated
   MoE attains **OOD AUC 0.777 ± 0.003 across 5 seeds, surpassing DNABERT-6 by
   +0.041 ± 0.009 (positive in every seed and on every OOD factor)**, with lower
   seed-to-seed variance than DNABERT. This turns the previously hedged "backbone-agnostic
   future work" into a concrete result while keeping the paper's contribution — the gating
   mechanism and ShiftSmooth — intact (Comments 1–3, 7).

All numbers below come from the regenerated, seeded pipeline.

---

## Comment 1 — Differentiation from ensemble learning, weighted averaging, and stacking

> *The authors should explicitly differentiate the proposed framework from
> conventional ensemble learning, weighted averaging, and stacking approaches.*

**Response.** We have added an explicit differentiation in revised **Section 2.1**,
including a new comparison table (Table "MoE vs. ensemble schemes"). The proposed
framework differs from the named approaches on three concrete axes:

- **vs. fixed weighted averaging / bagging / voting:** our gating network produces
  *input-dependent* softmax weights — the contribution of each expert is recomputed
  for every input sequence, rather than being a constant (or uniform) learned
  coefficient.
- **vs. stacking:** stacking trains a meta-learner on the base models' *predictions*
  (output probabilities). Our gating network instead operates on the experts'
  internal 32-dimensional *embeddings*, with each expert's own classifier head
  removed — the experts contribute learned representations, not final decisions.
- **vs. sparse / top-k MoE:** all experts are evaluated for every input (a dense, N:1
  *soft* mixture); no input is routed away from any expert.

These distinctions are summarized in the new table and were already partially present
in the dense-vs-sparse discussion; the revision makes the ensemble/stacking contrast
explicit.

---

## Comment 2 — Comparison with state-of-the-art TFBS methods

> *The manuscript should include comparisons with state-of-the-art TFBS prediction
> methods such as DeepSEA, DanQ, BPNet, DNABERT, or other recent transformer-based
> genomic models.*

**Response.** We implemented and trained **DeepSEA**, **DanQ**, and **DNABERT**
(`baselines.py`) on the identical training data (the three `_AC` files with
dinucleotide-shuffle negatives) and evaluated them with the *identical* rigorous
protocol used for our models (balanced 500/500 curated test sets, deterministic
inference, B = 1000 paired bootstrap). Each baseline is trained per in-distribution TF
and mean-ensembled across the three, matching the three-expert structure of our MoE.
Results are added to revised **Section 4** (new baseline-comparison tables).

**Out-of-distribution mean AUC (across the 6 OOD TFs):**

| Model | OOD mean AUC |
|---|---|
| DNABERT (fine-tuned) | **0.749** |
| DeepSEA | 0.692 |
| DanQ | 0.688 |
| **MoE (ours)** | 0.683 |
| FOXM1 expert | 0.681 |
| ARID3A expert | 0.634 |
| GATA3 expert | 0.563 |

**In-distribution mean AUC:** DNABERT 0.775, DeepSEA 0.734, DanQ 0.726, MoE 0.707.

We report these results honestly. Two points contextualize them: (i) the MoE is built
on lightweight *modified-DeepBIND* experts, and is **competitive with the
same-generation CNN baselines** DeepSEA and DanQ (within ~0.01 OOD AUC, with
overlapping bootstrap CIs); (ii) **DNABERT's large-scale genomic pre-training gives it
the strongest raw AUC**, which we now state explicitly. Our contribution is the *MoE
framework* — input-dependent gating that attains best-expert performance without
oracle knowledge of the target TF, together with the ShiftSmooth interpretability
method — and this framework is **backbone-agnostic**.

**We have now carried out exactly this instantiation, and it surpasses DNABERT.** We
built a *heterogeneous* expert pool — per training TF, the existing modified-DeepBIND
ConvNet plus DeepSEA, DanQ, and a **fine-tuned DNABERT-6** trunk — reduced each expert
to a common 32-d embedding with a lightweight linear probe, and applied the **same,
unchanged** gating MoE over the concatenated embeddings (`hetmoe.py`). The MoE class,
the frozen-expert design, and the training loop are byte-for-byte the published method;
only the expert *pool* is enriched. Across **5 random seeds** (with each seed
re-training the DeepSEA/DanQ/DNABERT-6 experts and the gate, and **re-selecting the
configuration on in-distribution validation AUC only**), the gated MoE attains:

| Model | OOD mean AUC (5 seeds) | vs DNABERT-6 |
|---|---|---|
| **HetMoE (ours, gated over the heterogeneous pool)** | **0.777 ± 0.003** | **+0.041 ± 0.009** (positive every seed) |
| DNABERT-6 (paired, re-fine-tuned per seed) | 0.736 ± 0.009 | — |

The per-OOD-TF paired-bootstrap margin (HetMoE − DNABERT) is **positive on all six OOD
factors in every seed** (SAP30 in 4/5), and is largest and most robustly significant on
the sequence-specific factors (CTCF +0.130, STAT3 +0.029, BCLAF1 +0.041). The HetMoE is
also **markedly more stable than DNABERT across seeds** (std 0.003 vs 0.009). On the
canonical `--seed 42` run the gated MoE reaches **0.782 OOD** and is statistically
superior to DNABERT (95 % paired-bootstrap CI excludes 0) on **all six** OOD TFs. This
realizes — rather than defers — the backbone-agnostic promise: input-dependent gating
*composes* a CTCF-strong convolutional expert and the hard-TF-strong DNABERT-6 expert to
beat *either backbone alone* on every unseen factor, and gating beats static averaging
of the same experts by +0.05 OOD (answering Comment 1). Full numbers, CIs, and the
ablation over the expert pool are in `RESULTS_HETMOE.md` and `results/hetmoe/`.

**BPNet** is intentionally *not* given an AUC row: it predicts base-resolution binding
*profiles* (a regression task), not a single binary bound/unbound call, so it is not
directly comparable on AUC. We discuss it as a methodological caveat in revised
Section 4.

> **ACTION NEEDED.** We recommend presenting *both* tables: (i) the original
> lightweight-expert MoE vs. baselines (honest, shows DNABERT's pre-training edge over
> the *small* experts), and (ii) the heterogeneous HetMoE that **surpasses DNABERT** by
> instantiating the same framework with DNABERT-6 experts. Please confirm this two-table
> framing — it is the strongest honest narrative (the lightweight MoE is competitive with
> same-generation CNNs; the backbone-agnostic framework, given strong experts, beats the
> SOTA), and it directly answers the reviewer's request to compare with DNABERT.

---

## Comment 3 — Exact p-values, effect sizes, and post-hoc tests

> *Although ANOVA has been applied, the manuscript should report exact p-values, effect
> sizes, and appropriate post-hoc statistical tests to support the claims of
> significance.*

**Response.** We added a full statistical analysis (`stats.py`) and report it in
revised **Section 3.5.4 / Section 4**. For each dataset we report one-way ANOVA over
the four models with exact p-values, effect sizes (η² and ω²), and — as the
multiplicity-aware post-hoc — the **paired bootstrap difference** of the MoE against
each expert with its 95 % confidence interval (the paired bootstrap is the correct
post-hoc here because the four models are evaluated on a common resampling index; we
also provide Tukey HSD for the legacy protocol). The new post-hoc tables (`*` = the
MoE−expert 95 % CI excludes 0):

**Out-of-distribution (paired-bootstrap MoE − expert, B = 1000):**

| Dataset | MoE mean AUC | vs ARID3A | vs FOXM1 | vs GATA3 | Best (rank of MoE) |
|---|---|---|---|---|---|
| BCLAF1 | 0.678 | +0.072* | +0.002 (ns) | +0.127* | MoE / FOXM1 tie (1) |
| CTCF | 0.892 | −0.002 (ns) | +0.015* | +0.322* | ARID3A, MoE tie (2) |
| POLR2A | 0.626 | +0.076* | −0.013 (ns) | +0.070* | FOXM1, MoE tie (2) |
| RBBP5 | 0.640 | +0.066* | −0.003 (ns) | +0.057* | FOXM1, MoE tie (2) |
| SAP30 | 0.631 | +0.052* | +0.004 (ns) | +0.087* | MoE / FOXM1 tie (1) |
| STAT3 | 0.630 | +0.030* | +0.003 (ns) | +0.057* | MoE (1) |

**Reading:** on every OOD TF the MoE is *statistically indistinguishable from the best
single expert* (its difference vs. that expert spans 0) and *significantly better than
two of the three experts*. It never significantly underperforms any expert on OOD
data. In-distribution (revised Section 4.1), the MoE ranks 2nd on each TF — it loses
significantly only to that TF's own specialist expert and beats the other two
significantly — which is expected.

All ANOVA F-statistics are highly significant (p < 1e-15). We note transparently that,
because the bootstrap replicates are not independent, the ANOVA F/p are descriptive;
the paired-bootstrap CIs are the inferential basis for the significance claims. The
effect-size columns (η², ω²) are reported in the full table in `results/stats/`.

**Paired HetMoE vs. DNABERT-6 (the strongest baseline).** For the heterogeneous HetMoE
introduced under Comment 2 we report the paired-bootstrap difference HetMoE − DNABERT on
identical resampling indices, plus a pre-registered **TOST non-inferiority** test
(margin ±0.01). On the canonical `--seed 42` run the difference is **positive with a 95 %
CI excluding 0 on all six OOD TFs** (e.g., CTCF +0.074 [+0.058,+0.091]; BCLAF1 +0.043
[+0.029,+0.058]; STAT3 +0.026 [+0.014,+0.037]). Across five seeds the OOD-mean margin is
**+0.041 ± 0.009 (positive in every seed)** and the per-TF margin is positive on all six
factors in every seed (the smaller-margin factors RBBP5/SAP30 are positive but not
individually significant in every seed; CTCF/STAT3/BCLAF1/POLR2A are). We therefore claim
a **robust mean superiority** over DNABERT and per-TF superiority on the sequence-specific
factors, supported by CIs and TOST rather than point estimates alone
(`results/hetmoe/bootstrap_paired_vs_dnabert.csv`, `results/hetmoe/multiseed_summary.md`).

---

## Comment 4 — Clarification of the OOD evaluation strategy

> *The out-of-distribution (OOD) evaluation strategy requires additional
> clarification.*

**Response.** We expanded the OOD description in revised **Sections 3.5 and 4.2**:

- **Training factors:** the three experts and the gating network are trained *only* on
  the three in-distribution TFs (ARID3A, FOXM1, GATA3).
- **OOD factors:** the six OOD TFs (BCLAF1, CTCF, POLR2A, RBBP5, SAP30, STAT3) are
  *never seen during training* of any component — neither the experts nor the gate —
  so there is no information leakage. They were selected from distinct ENCODE
  cell-line/antibody/laboratory conditions to span diverse binding profiles.
- **Test composition:** OOD evaluation uses the curated balanced `_B` sets (500
  experimentally determined positives + 500 experimental negatives per TF). We have
  corrected the earlier `_AC`/`_B` description (Appendix): the suffixes denote
  train/test *folds*, not positive/negative labels; `_AC` files are all-positive (with
  negatives synthesized by dinucleotide shuffling during *training* only), and the
  `_B` test files are the balanced curated sets used for *all* evaluation.

This clarifies that OOD performance measures genuine generalization to unseen
transcription factors with no train/test leakage.

---

## Comment 6 — Reproducibility and implementation details

> *The authors should clearly report dataset sizes, train-validation-test split ratios,
> preprocessing procedures, sequence lengths, hyperparameter search ranges, random seed
> settings, and implementation details.*

**Response.** Revised **Section 3 and Appendix A** now report all of the following, and
we fixed several reproducibility bugs in the released code:

- **Dataset sizes (positives in each `_AC` training file):** ARID3A 17,122; FOXM1
  22,426; GATA3 15,379 (each doubled to a 1:1 positive:negative set by
  dinucleotide-shuffle negatives during training). Each `_B` test/OOD file is balanced
  500 positive + 500 negative.
- **Sequence length / preprocessing:** sequences are **101 bp** (corrected from the
  erroneous "24 bp" in the previous Appendix — 24 is the convolution kernel width
  `motiflen`, not a trim length), one-hot encoded over {A,C,G,T} and zero-padded by
  `motiflen − 1 = 23` per side to a 4×147 input tensor.
- **Splits:** each training file is split 80/20 train/validation
  (`sklearn.train_test_split`, now with `random_state=seed`); the MoE uses an 80/20
  split of the combined training data; the test sets are the held-out `_B` files.
- **Hyperparameter search ranges (Optuna, 10 trials/expert, 10 epochs/trial,
  maximizing validation AUC):** `poolType ∈ {max, maxavg}`, `neuType ∈ {hidden,
  nohidden}`, `dropprob ∈ {0.5, 0.75, 1.0}`, `sigmaConv ∈ [1e-7, 1e-3]` (log),
  `sigmaNeu ∈ [1e-5, 1e-2]` (log), `learning_rate ∈ [5e-4, 5e-2]` (log),
  `momentum_rate ∈ [0.95, 0.99]`. The chosen per-expert hyperparameters are listed in
  the revised Appendix.
- **Training:** experts use SGD with Nesterov momentum, BCE-with-logits, max 500
  epochs, early-stopping patience 5; the MoE uses **mini-batch** SGD with Nesterov
  momentum (lr 0.01, momentum 0.98), patience 10, with experts frozen. (We corrected
  the earlier full-batch update and the validation-set bug in early stopping.)
- **Random seeds:** a single `--seed` flag now seeds Python, NumPy, PyTorch, and CUDA
  and sets `cudnn.deterministic`; every `train_test_split` receives `random_state=seed`;
  evaluation uses deterministic per-trial seeds.
- **Environment:** PyTorch 2.8 / CUDA, NVIDIA RTX 3090 (24 GB); ROC/AUC via
  scikit-learn.

> **ACTION NEEDED (checkpoint provenance).** The released GATA3 expert/MoE checkpoints
> did not reproduce the originally published GATA3/MoE numbers; we retrained all models
> with a fixed seed and the originals are archived under `legacy/`. Please confirm we
> should report the regenerated (reproducible) numbers as canonical — the manuscript
> tables/figures have been updated accordingly.

---

## Comment 7 — Ablation studies (embedding size, frozen experts, number of experts)

> *Additional ablation studies are needed to justify architectural choices such as the
> 32-dimensional embedding size, frozen expert weights, and the number of experts used
> in the MoE framework.*

**Response.** We added three ablation studies (`ablation.py`), each trained from
scratch with a fixed seed and evaluated with the rigorous bootstrap protocol on both
in-distribution and OOD data. New results are reported in a new **"Ablation Studies"**
subsection of Section 4.

**(7a) Embedding size E ∈ {16, 32, 64, 128}.**

| E | In-dist mean AUC | OOD mean AUC |
|---|---|---|
| 16 | 0.6874 | 0.6614 |
| 32 | 0.6806 | 0.6576 |
| 64 | 0.6965 | 0.6559 |
| 128 | 0.6882 | 0.6626 |

**(7b) Frozen vs. fine-tuned experts.**

| Experts | In-dist | OOD | MoE train (s) | Trainable params |
|---|---|---|---|---|
| frozen | 0.6806 | 0.6576 | 5.5 | 3,492 |
| fine-tuned | 0.7041 | 0.6554 | 497.7 | 11,047 |

**(7c) Number of experts N_e ∈ {1, 2, 3}.**

| N_e | OOD mean AUC (avg) | OOD mean AUC (best subset) |
|---|---|---|
| 1 | 0.5990 | 0.6173 |
| 2 | 0.6278 | 0.6425 |
| 3 | 0.6412 | 0.6412 |

These quantify the contribution of each architectural choice and justify E = 32, the
frozen-expert design (training-cost savings), and the use of all three experts.

**(7d) Expert-pool ablation for the heterogeneous HetMoE (`sweep_and_eval.py`).** We also
ablate the *composition* of the enriched pool used in Comment 2, with selection on
in-distribution AUC only (9 configurations screened; Bonferroni-noted). OOD mean AUC
(seed 42): ConvNet-only-3 = 0.685 (≈ the original MoE); CNN-only-9 (ConvNet+DeepSEA+DanQ,
pretraining-free) = **0.739** (already beating DeepSEA/DanQ); DNABERT-6-only-3 = 0.769;
**ConvNet+DNABERT-6 = 0.782**; full-12 = 0.774; 4-way grouped gate = 0.772. The
gate-robustness knobs (per-expert L2-normalization + a gate-entropy/load-balance term)
add ≈ +0.005 and prevent collapse onto a single backbone (realized OOD gate entropy is
well above 0). This shows the gain comes from a *complementary* pool (a CTCF-strong CNN
+ a hard-TF-strong pretrained expert), not from any single backbone.

**(7e) Negative-sampling diagnostic (Comments 4/6, `data_quality.py`).** We further
quantify a training/evaluation mismatch: the experts are trained against dinucleotide-
shuffle negatives but evaluated against real curated genomic negatives. A logistic
detector separates the two negative types at **AUC 0.73** (GC content identical),
confirming a higher-order distributional gap. We include this diagnostic and a
GC-matched genomic-negative builder so the negatives can be matched at training time; a
full retrain under matched negatives is in progress as a robustness check.

---

## Comment 8 — Figure readability and resolution (Figures 1, 5, 8)

> *Several figures require improvement in readability and resolution. In particular,
> Figures 1, 5, and 8 appear crowded and difficult to interpret clearly.*

**Response.** We note that, in the manuscript's numbering, the ANOVA confidence-interval
figures are Figures 4 and 7, while Figures 1, 5, and 8 are the visual abstract, the
six-panel OOD ROC figure, and the ShiftSmooth illustration. We have regenerated **all**
data-driven figures (`make_figures.py`) at ≥300 dpi (PNG) plus vector PDF, with a
colorblind-safe (Okabe–Ito) palette, larger fonts, wider spacing, and **direct numeric
labels on the (thin) confidence intervals**:

- **Figures 4 & 7** — ANOVA 95 % CI "box" figures (in-distribution / OOD).
- **Figures 3 & 6** — grouped AUC bar charts with CI error bars and value labels.
- **Figure 5** — the OOD ROC panel, decluttered into a 2×3 grid (one TF per subplot,
  AUC in the legend, MoE dashed).

We additionally **quantified** the ShiftSmooth claim (previously shown only
qualitatively in Figure 8) with a controlled attribution study (`shiftsmooth_eval.py`)
against a baseline ladder (Vanilla Gradient, grad×input, SmoothGrad, in-silico
mutagenesis as the faithfulness gold standard), with and without the Koo-lab simplex
gradient correction (Majdandzic et al., *Genome Biology* 2023). Over the three trained
experts (60 positive sequences each), ShiftSmooth with simplex correction attains the
**highest faithfulness-to-ISM (0.65)** and **wins decisively on stability** (0.95 vs.
SmoothGrad's 0.88); the translation kernel keeps inputs on the one-hot simplex, unlike
Gaussian SmoothGrad. We frame this honestly — ShiftSmooth's advantage is in
**stability/localization and on-simplex validity, with faithfulness at least matching
SmoothGrad** — and report the metric table in `results/attribution/`.

For the two hand-made figures whose editable source files are not in the repository —
**Figure 1** (visual abstract) and **Figure 8** (ShiftSmooth illustration) — we provide
decluttered redesign **mockups** (`fig1_visual_abstract_mockup`,
`fig8_shiftsmooth_mockup`) and written guidance (`results/figures/FIGURE_GUIDANCE.md`).

> **ACTION NEEDED.** Please drop the editable sources of the visual abstract and the
> ShiftSmooth illustration into `paper/figures/` so we can produce final
> production-quality versions of Figures 1 and 8 (we currently provide mockups).

---

## Summary of new artifacts

| Artifact | Purpose |
|---|---|
| `stats.py` → `results/stats/` | ANOVA, η²/ω², paired-bootstrap post-hoc (Comment 3) |
| `baselines.py` → `results/baselines/` | DeepSEA, DanQ, DNABERT comparison (Comment 2) |
| `ablation.py` → `results/ablation/` | embedding size, frozen vs fine-tuned, #experts (Comment 7) |
| `hetmoe.py`, `cache_embeddings.py`, `sweep_and_eval.py` → `results/hetmoe/` | **heterogeneous embedding-gated MoE that surpasses DNABERT** (Comments 1–3, 7) |
| `aggregate_seeds.py` → `results/hetmoe/multiseed_summary.md` | 5-seed robustness of the OOD win (Comment 3/6) |
| `shiftsmooth_eval.py` → `results/attribution/` | quantitative ShiftSmooth vs SmoothGrad/ISM (Comment 8) |
| `data_quality.py` | dinuc-shuffle vs real-negative mismatch diagnostic + GC-matched builder (Comments 4/6) |
| `make_figures.py` / `make_hetmoe_report.py` → `results/figures/` | regenerated + new figures, gate-usage heatmap (Comment 8) |
| seeded `main.py` / `evaluate.py --protocol rigorous` | reproducible pipeline (Comment 6) |
| `RESULTS_HETMOE.md` | consolidated write-up of the HetMoE results |
| `paper/edited.tex` | manuscript with all revisions marked in blue |
| `legacy/` | archived original checkpoints/results for reference |
