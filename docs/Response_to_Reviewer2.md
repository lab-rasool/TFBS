# Response to Reviewer 2

We thank Reviewer 2 for the careful and constructive evaluation. The comments prompted a
substantial strengthening of the experimental rigor, baselines, statistics, OOD design,
explainability, and reproducibility of the work. Reviewer comments are shown *in
italics*, followed by our **Response**. Section references point to the revised manuscript
(`paper/edited.tex`, revisions in blue).

## Summary of the most important revisions

1. **The proposed model is now HetMoE**, a heterogeneous, embedding-gated
   Mixture-of-Experts. The unchanged dense-soft gate operates over a pool of complementary
   backbones (a modified-DeepBIND ConvNet, DeepSEA, DanQ, and a fine-tuned **DNABERT-6**)
   per training factor. Because the gate sees only embeddings, architecturally different
   experts are combined; non-convolutional trunks are frozen feature extractors reduced to
   a common 32-dimensional embedding by a linear probe.

2. **A redesigned, family-aware OOD evaluation.** We train on **seven factors spanning six
   DBD families** (ARID3A, FOXM1, GATA3, JUND, MAX, GABPA, SP1) and evaluate on a held-out
   set **stratified by DBD family**: within-family (17), cross-family (6), and a separately
   reported non-motif appendix (6). Training and most OOD factors are anchored to the K562
   cell line to avoid confounding the train/OOD boundary with cell type.

3. **A fair negative-set protocol.** Experts are also retrained on GC- and repeat-matched
   real genomic negatives (hg19), addressing the train/evaluation mismatch documented in
   the negative-sampling literature; the dinucleotide-shuffle configuration is retained as
   an explicit ablation. All evaluation uses the balanced curated test sets (500
   experimental positives + 500 experimental negatives per factor).

Headline result (fair GC- and repeat-matched genomic-negative protocol; real-negative
balanced test sets): on the motif-bearing OOD strata HetMoE attains mean AUC **0.821 ±
0.005** vs **0.799 ± 0.008** for the strongest baseline (fine-tuned DNABERT-6) across three
seeds (margin **+0.022**, positive in every seed), superior on **12–13 of 23** factors
(95% CI excludes 0), most decisively on the sequence-specific and within-family factors,
with in-distribution mean **0.881**. We additionally report that the common dinucleotide-
shuffle negative protocol *inflates* this comparison to +0.066 (0.864, 22/23); we therefore
report the fair-negative result as our headline and the shuffle result as an explicit
negative-set ablation (a contribution in its own right, consistent with Tourne et al. 2026).

---

## Comment 1: Differentiate from ensemble learning, weighted averaging, and stacking

> *The authors should explicitly differentiate the proposed framework from conventional
> ensemble learning, weighted averaging, and stacking approaches.*

**Response.** Revised **Section 2.1** adds an explicit differentiation and a comparison
table. The framework differs on three concrete axes: (i) vs. fixed weighted
averaging/bagging/voting, the gate produces **input-dependent** softmax weights,
recomputed per sequence, not constant coefficients; (ii) vs. stacking, the gate operates
on the experts' internal **embeddings** (each expert's own classifier head removed), not
on their output predictions; (iii) vs. sparse/top-k MoE, **all** experts are evaluated for
every input (dense, soft mixture). Empirically, input-dependent gating beats a **static
(uniform) average of the same experts by +0.073 AUC** on the motif-OOD strata (0.827 vs.
0.754) and the best single expert by +0.088 (0.827 vs. 0.739), so the gain is from routing,
not ensembling.

---

## Comment 2: Comparison with state-of-the-art TFBS methods

> *The manuscript should include comparisons with state-of-the-art TFBS prediction methods
> such as DeepSEA, DanQ, BPNet, DNABERT, or other recent transformer-based genomic models.*

**Response.** DeepSEA, DanQ, and a fine-tuned DNABERT-6 are trained on the identical data
and evaluated with the identical protocol; they are also included as experts in the HetMoE
pool. Revised **Section 4** reports a per-stratum comparison table:

| Model | In-dist mean | Within-family | Cross-family | Non-motif | Motif-OOD mean |
|---|---|---|---|---|---|
| **HetMoE (ours)** | **0.881** | **0.846** | **0.775** | 0.696 | **0.827** |
| Best single expert | 0.747 | 0.754 | 0.693 | 0.654 | 0.739 |
| DNABERT-6 (baseline) | 0.818 | 0.816 | 0.753 | **0.756** | 0.800 |
| Static average | 0.796 | 0.770 | 0.707 | 0.675 | 0.754 |

(Numbers are the canonical-seed fair-negative results; across three seeds the motif-OOD
mean is 0.821 ± 0.005 for HetMoE vs. 0.799 ± 0.008 for DNABERT-6.) HetMoE attains the best
in-distribution and best motif-OOD mean, and **surpasses the strongest single model,
fine-tuned DNABERT-6, on 13 of the 23 motif-bearing OOD factors**; on the non-motif stratum
(reported separately, near a sequence-only ceiling) DNABERT-6 is stronger. **BPNet** is
discussed but not given an AUC row because it predicts base-resolution binding *profiles*
(regression), not a binary call, and is not directly comparable on AUC.

---

## Comment 3: Exact p-values, effect sizes, and post-hoc tests

> *Although ANOVA has been applied, the manuscript should report exact p-values, effect
> sizes, and appropriate post-hoc statistical tests to support the claims of significance.*

**Response.** Revised **Sections 3.5.4 / 4** report one-way ANOVA with exact p-values and
effect sizes (η², ω²), and, as the multiplicity-aware post-hoc, the **paired-bootstrap
difference** of HetMoE against each comparator with a 95% percentile confidence interval
(the correct post-hoc, since all models are evaluated on a common resampling index). For
HetMoE − DNABERT-6 we report, per factor, the mean ΔAUC, its 95% CI, a two-sided p, and a
pre-registered **TOST non-inferiority** verdict (±0.01). Under the fair-negative protocol
the CI excludes zero in favor of HetMoE on **13 of 23** motif factors (10/17 within-family,
3/6 cross-family); on several C2H2 zinc-finger factors (YY1, ZNF143, REST, ZBTB33) DNABERT-6
is stronger. We additionally report per-factor auPRC.

---

## Comment 4: Clarification of the OOD evaluation strategy

> *The out-of-distribution (OOD) evaluation strategy requires additional clarification.*

**Response.** This is the most substantial revision. Revised **Section "Training Factors,
Out-of-Distribution Stratification, and Negative Sets"** specifies a leakage-free,
family-aware design:

- **Training factors:** seven factors spanning six DBD families; experts and gate see
  *only* these.
- **Stratified OOD:** (i) **within-family** held-out members of the trained families
  (motif-family transfer; e.g. JUN/FOSL1 for JUND, MYC/USF1 for MAX, GATA1/GATA2 for
  GATA3); (ii) **cross-family** sequence-specific factors from unseen families (CTCF, STAT3,
  NRF1, HNF4A, TCF7L2, ZBTB33); (iii) **cell-line transfer** (a training factor in an unseen
  line); (iv) a **non-motif** appendix (POLR2A, EP300, EZH2, TAF1, RBBP5, SAP30) reported
  separately because their occupancy has a low sequence-prediction ceiling.
- **Cell-line confound control:** training and most OOD factors are anchored to K562, so
  the train/OOD boundary is not confounded with cell type.
- The headline OOD result is the mean over the motif-bearing strata; the non-motif factors
  are never folded into it.

This replaces the earlier single undifferentiated OOD pool and prevents an easy, motif-rich
factor from dominating the average.

---

## Comment 5: Quantitative explainability metrics for ShiftSmooth

> *… its validation remains primarily qualitative. Quantitative explainability metrics such
> as faithfulness, localization accuracy, insertion/deletion analysis, or motif recovery
> performance should be included.*

**Response.** We added a controlled attribution study against a baseline ladder (Vanilla
Gradient, gradient×input, SmoothGrad) with **in-silico mutagenesis (ISM)** as the
faithfulness gold standard, with and without the Koo-lab **simplex gradient correction**
(Majdandzic et al., 2023). We report three quantitative metrics: **faithfulness** (rank
correlation to ISM), **motif localization** (top-k positional overlap with ISM), and
**stability** (attribution correlation under perturbation). ShiftSmooth keeps inputs on the
one-hot simplex (unlike Gaussian SmoothGrad) and attains the highest stability (0.96 on the
MoE, 0.95 on the experts, vs. 0.92/0.91 for SmoothGrad), with faithfulness and localization
on par with the simplex-corrected gradient baselines. We state honestly that stability alone
is not evidence of faithfulness, and that the simplex correction accounts for most of the
faithfulness improvement.

---

## Comment 6: Dataset sizes, splits, preprocessing, hyperparameters, seeds, implementation

> *The authors should clearly report dataset sizes, train-validation-test split ratios,
> preprocessing procedures, sequence lengths, hyperparameter search ranges, random seed
> settings, and implementation details.*

**Response.** Revised **Section 3 and the Appendix** now report: the seven training factors
and the exact ENCODE experiment per factor (cell line, antibody, laboratory) pinned in a
released data manifest; 101-bp sequences, one-hot encoded and padded to a 4×147 tensor with
uniform 0.25 flanks of 23 positions per side; 80/20 train/validation splits with a fixed
random seed; the balanced curated 500/500 test sets; the per-expert hyperparameters;
SGD/Nesterov training with early stopping; a single random seed seeding Python, NumPy, and
the deep-learning framework with deterministic backends; and the two negative-set protocols
(dinucleotide-shuffle and GC/repeat-matched genomic negatives from hg19). All code, the data
manifest, and the figure-generation scripts are released.

---

## Comment 7: Ablation studies

> *Additional ablation studies are needed to justify architectural choices such as the
> 32-dimensional embedding size, frozen expert weights, and the number of experts used in
> the MoE framework.*

**Response.** Revised **Section 4 (Ablation Studies)** reports: embedding size
E ∈ {16,32,64,128} (the OOD mean varies by under 0.01 AUC, so E=32 is retained as the
smallest competitive value); frozen vs. fine-tuned experts (freezing matches fine-tuning at
roughly two orders of magnitude lower training cost); and an **expert-pool and
gate-stabilizer ablation** under the fair-negative protocol, which shows that the
convolutional ConvNet+DeepSEA+DanQ pool is the configuration chosen by in-distribution mean
AUC, that adding the DNABERT-6 experts does not raise the mean further (so the selected pool
is pretraining-free), and that removing the two gate stabilizers (L2-normalized embeddings
and an entropy term) collapses the gate's mean entropy from 2.87 to 0.98 nats. Selection
uses in-distribution mean AUC only and the OOD strata are scored once. We also add a
**negative-set ablation** (dinucleotide-shuffle vs. GC-matched genomic negatives), motivated
by a diagnostic in which a detector separates the two training-negative types at AUC 0.73
with matched GC content. These quantify each architectural choice and the impact of the
fair-negative protocol.

---

## Comment 8: Figure readability and resolution

> *Several figures require improvement in readability and resolution. In particular,
> Figures 1, 5, and 8 appear crowded and difficult to interpret clearly.*

**Response.** All data-driven figures were regenerated at high resolution (vector PDF +
600 dpi PNG) with a colorblind-safe palette, for the revised stratified study: per-stratum
bars, a forest plot of HetMoE − DNABERT-6 per factor (grouped by DBD family), the gate-usage
heatmap; internal plot titles were removed (Reviewer 1, Comment
6). The attribution sequence figures were redesigned (the redundant y-axis removed, the
panel height reduced, and the motif region clearly boxed). The visual abstract was rebuilt
from an editable vector source to match the original layout, and the ShiftSmooth schematic
uses the original vector illustration. All figures follow the journal's figure
specifications: embedded fonts, vector PDF at ≥600 dpi, and a colorblind-safe palette.

---

We thank Reviewer 2 for comments that substantially strengthened the rigor, honesty, and
reproducibility of the manuscript.
