# Figure de-duplication & canonical figure set — design spec

- **Date:** 2026-06-27
- **Status:** Approved (design); pending spec review
- **Scope:** Consolidate the HetMoE paper's figures into one canonical figure per
  result, regenerate them from existing results via a single generator per figure
  family, delete dead/duplicate generators and figure files, and rewire the paper.

## 1. Background / problem

The paper currently embeds **24 `\includegraphics`** that collapse to a handful of
distinct results. Concrete problems established during review:

- **Byte-identical duplicates** included under different figure numbers:
  `roc_in_distribution.pdf` ≡ `fig4_ci_in_distribution.pdf`;
  `fig5_roc_out_of_distribution.pdf` ≡ `fig7_ci_out_of_distribution.pdf` ≡
  the (uncited) `fig_roc_ood.pdf`.
- **Same numbers shown three ways** (bars *and* CI *and* "ROC") for in-distribution
  and OOD performance.
- **`fig6_bars_out_of_distribution` ≡ `fig_ood_strata`** (same stratum comparison).
- **Caption/figure mismatches:** files named `roc_*` are actually CI dot-plots;
  `fig7` caption says "by stratum" but shows per-factor; `fig_calibration` caption
  says "reliability diagram for HetMoE and DNABERT-6" but the committed file is an
  ECE/Brier bar chart.
- **Redundancy in the code, not just output:** there are ~7 figure generators with
  overlapping responsibilities:
  - `experiments/analysis/make_paper_figures.py` — **dead** (crashes with
    `KeyError('JUN')`; reads stale `baseline_comparison_*.csv`; draws 5–6 model
    variants).
  - `experiments/analysis/make_figures_v2.py` — **current/working**; sourced entirely
    from `results/hetmoe/seed*_genomic/hetmoe_eval.json`.
  - `experiments/analysis/make_multiseed_fig.py` — `fig_multiseed` from the genomic seeds.
  - `experiments/analysis/make_schematics.py` — matplotlib `_mockup` schematics (unused).
  - `paper/generate_fig1_drawio.py` — drawio source for the real Fig 1 asset.
  - `experiments/attribution/make_attribution_figures.py` — the 10 attribution panels.
  - `experiments/attribution/make_sequence_logos.py` — regenerates 2 of those 10.

## 2. Decisions (locked)

1. **Production approach:** one consolidated generator per figure family that
   **regenerates from existing results** (`hetmoe_eval.json` + cache); **no
   experiments re-run**; delete the dead/duplicate generators; rewire the paper.
2. **Figure set:** **Set B — one figure per result** (~10 figures, no composite
   multi-panel grouping except the attribution grid).
3. **Schematics (Figs 1–2):** **keep the existing assets, rename only** (no
   regeneration). Their known issues are recorded as caveats (§8).
4. **Defaults approved:** single flat output dir `results/figures/paper/`; **Fig 9 is
   a reliability diagram** (HetMoE vs DNABERT-6), not the ECE/Brier bar chart.

## 3. Canonical figure list

All final figures live in **`results/figures/paper/`** as `<name>.pdf` (vector) +
`<name>.png` (600 dpi). The paper `\graphicspath` points only at that dir.

| # | Name | Shows | Source |
|---|------|-------|--------|
| 1 | `fig_1_method_schematic` | Pipeline / visual abstract | existing `fig1_visual_abstract.pdf`, renamed |
| 2 | `fig_2_shiftsmooth_schematic` | shift → grad → back → avg | existing `ShiftSmoothExample.pdf`, renamed |
| 3 | `fig_3_indist_performance` | Per-TF in-dist AUROC (dot + 95% CI), HetMoE vs DNABERT-6, 7 train TFs | JSON `per_dataset["in_distribution::*"]` |
| 4 | `fig_4_ood_headline` | **Motif**-OOD mean AUROC, 4 models (HetMoE / DNABERT-6 / best-single / static-avg) — shows gain is gating, not ensembling | per-model mean of JSON `ood_within_family` + `ood_cross_family` (NOT `ood_mean`, which includes non-motif) |
| 5 | `fig_5_ood_forest` | Per-factor ΔAUROC (HetMoE − DNABERT-6) ± 95% CI, 23 motif factors grouped by DBD family; CI-excludes-0 highlighted | JSON `paired_vs_dnabert` (filtered to 23 motif OOD) |
| 6 | `fig_6_ood_strata` | OOD AUROC by stratum (within / cross / non-motif), HetMoE vs DNABERT-6, non-motif marked as seq-only ceiling | JSON `ood_within_family`, `ood_cross_family`, `ood_nonmotif` |
| 7 | `fig_7_multiseed` | Motif-OOD mean AUROC per seed (HetMoE vs DNABERT-6), 3 genomic seeds | `results/hetmoe/seed{0,1,42}_genomic/hetmoe_eval.json` |
| 8 | `fig_8_gate_routing` | Mean gate weight, expert (cols) × OOD factor (rows) | `config.expert_order` + recomputed gate weights (§5) |
| 9 | `fig_9_calibration` | Reliability diagram, motif-OOD pooled, HetMoE vs DNABERT-6, ECE in legend | cached per-sample probs (§5) |
| 10 | `fig_10_attribution` | Grid: rows = {input seq, VG-expert, SS-expert, VG-MoE, SS-MoE}; cols = {positive GATA3 \| random} | cache + `models/zoo`/`models/experts` via Captum + logomaker |

**Net: 24 includes → 10 figures.** Each distinct result appears exactly once.

## 4. Data sources (precise)

Canonical results = the **fair GC-matched-negative ("genomic") protocol**:
`results/hetmoe/seed42_genomic/hetmoe_eval.json` (used by `validate_paper.py`).

Verified top-level keys: `config` (`l2norm`, `entropy_reg`, `gate_temperature`,
`seed`, `num_experts`, `expert_order`), `ood_mean`, `ood_mean_auprc`,
`ood_within_family`, `ood_cross_family`, `ood_nonmotif` (each
`{HetMoE, static_mean, best_single, DNABERT}`), `indist_mean_hetmoe` (0.8806),
`ood_mean_ece` (0.2167), `ood_mean_brier` (0.2403), `tost_margin` (0.01),
`per_dataset` (36 entries: 7 `in_distribution::*` + 29 `out_of_distribution::*`),
`paired_vs_dnabert` (36 entries).

- Fig 7 multiseed: `seed{0,1,42}_genomic` exist; motif-OOD per-seed mean = mean over
  `ood_within_family` + `ood_cross_family`.
- Figs 9–10: per-sample probabilities / attributions come from the offline cache
  `results/cache/seed*/` and the saved models; no re-training.

## 5. Two implementation decision points (resolved, not TBD)

- **Fig 8 gate weights:** `hetmoe_eval.json` has no top-level `gate_weights` field, so
  the generator **recomputes** mean per-expert gate weights over each OOD dataset from
  the cached embeddings + the selected config's saved gate
  (`tfbs.gate.gate_predict`), then averages per factor. This is deterministic and uses
  only existing results. If a `gate_weights` field is present for the selected config,
  use it directly as a shortcut.
- **Fig 9 calibration:** reliability diagram computed from cached per-sample probs for
  HetMoE and DNABERT-6 pooled over the motif-OOD factors; ECE annotated in the legend.
  Uses cache, not re-inference.

## 6. Generator architecture

- **`experiments/analysis/make_paper_figures.py`** — rewritten as *the* analysis
  generator. Emits **Figs 3–9** to `results/figures/paper/`. Folds in
  `make_figures_v2.py` and `make_multiseed_fig.py`. Single command:
  `python -m experiments.analysis.make_paper_figures` (optional `--seed`,
  default 42 → `seed42_genomic`).
- **`experiments/attribution/make_attribution_figures.py`** — refactored to emit the
  single **Fig 10** grid. Folds in `make_sequence_logos.py`. Single command:
  `python -m experiments.attribution.make_attribution_figures`.
- **Figs 1–2** — static assets copied/renamed into `results/figures/paper/`. Not
  generated by the pipeline.
- All figures use `tfbs.figstyle` (Okabe–Ito palette, Nature column widths
  89/120/183 mm, despined thin axes, `save()` → PDF + 600 dpi PNG). AUROC axes span
  full 0–1 with a 0.5 chance line; difference/weight axes (Figs 5, 8) are exempt.

## 7. Paper rewiring (`paper/paper_hetmoe_clean.tex`)

- Replace the 24 `\includegraphics` with the 10 new ones; delete the duplicate
  `figure`/`figure*` environments.
- Rewrite each caption to describe the single figure it now labels (fixing the
  calibration / fig7 / "ROC" mismatches).
- New labels: `fig:method`, `fig:shiftsmooth`, `fig:indist`, `fig:ood_headline`,
  `fig:ood_forest`, `fig:ood_strata`, `fig:multiseed`, `fig:gate`, `fig:calib`,
  `fig:attribution`. Update every in-text `\ref`/`\cref`.
- Set `\graphicspath` to `{../results/figures/paper/}{./}`.
- Regenerate the track-changes copy: `cd paper && ./make_trackchanges.sh` (never hand-edit
  `edited.tex`). Recompile `paper_hetmoe_clean.pdf`.
- Ablation **tables** are unchanged.

## 8. Cleanup / deletion (after `paper/` dir verified complete)

- **Code (delete):** `experiments/analysis/make_figures_v2.py`,
  `experiments/analysis/make_multiseed_fig.py`,
  `experiments/analysis/make_schematics.py`,
  `experiments/attribution/make_sequence_logos.py`, and the old
  `make_paper_figures.py` body (replaced). `paper/generate_fig1_drawio.py` is kept (it
  is the source of the retained Fig 1 asset).
- **Figures (delete):** `results/figures/{nature,performance,supplementary}/` and the
  `_mockup` schematics. These are git-tracked → recoverable. Deletion happens only
  after the 10 `paper/` figures exist and the paper compiles.

## 9. Caveats (recorded; not fixed here)

- **Fig 8 vs Fig 1 inconsistency:** the selected gate config has 21 experts
  (ConvNet + DeepSEA + DanQ; no DNABERT-6 column), while the kept Fig 1 schematic shows
  4 backbones including DNABERT-6. The generator will print the exact `expert_order`
  so the author can reconcile later.
- **Fig 1 text-overlap glitch** ("Heterogeneous frozen experts…" behind the ConvNet
  box) remains (rename-only choice).
- **Fig 7 uses 3 seeds** (0/1/42), so its caption must say 3 (not 5).

## 10. Out of scope

No experiment re-runs; no BioRender automation; ablation tables unchanged; no changes
to `tfbs/` model/eval code beyond reusing existing helpers (`tfbs.gate`,
`tfbs.metrics`, `tfbs.figstyle`).

## 11. Acceptance criteria

1. `results/figures/paper/` contains exactly the 10 `fig_N_*` figures (PDF + PNG).
2. `python -m experiments.analysis.make_paper_figures` (Figs 3–9) and
   `python -m experiments.attribution.make_attribution_figures` (Fig 10) run clean and
   emit **only** their canonical figures — no extras. Figs 1–2 are copied static assets.
3. `paper_hetmoe_clean.tex` references exactly those 10, each caption matches its
   figure, all `\ref`/`\cref` resolve, and the PDF compiles.
4. The deleted generators/figure dirs are gone; no remaining reference to old names.
5. `python -m experiments.analysis.validate_paper` still passes (number claims intact).
