# `results/` layout

What each subdirectory holds and which command regenerates it. Small summary artifacts
(CSV/JSON/figures the paper cites) are tracked in git; large regenerable caches are
**gitignored** (see `.gitignore`) and rebuilt by the commands below.

| Path | Tracked? | Produced by | Contents |
|------|----------|-------------|----------|
| `evaluation_summary.csv`, `evaluation_results.json`, `bootstrap_paired.csv` | yes | `experiments.train.evaluate --protocol rigorous` | per-model bootstrap AUC + 95% CI; deterministic predictions; paired MoE−expert diffs |
| `figures/paper/` | yes | `experiments.analysis.make_paper_figures` (Figs 3–8) + `experiments.attribution.make_attribution_figures` (Fig 9) | the **9 paper figures** `fig_1`…`fig_9` (method & ShiftSmooth schematics, in-distribution performance, OOD headline/forest/strata, multiseed, gate-routing heatmap, attribution grid); all AUROC axes 0–1; PDF + 600 dpi PNG. Figs 1–2 are static schematic assets. |
| `figures/schematics/` | yes | hand-made (no generator) | the two authored schematics: `fig1_visual_abstract_mockup`, `fig8_shiftsmooth_mockup` |
| `stats/` | yes | `experiments.analysis.stats` | ANOVA + η²/ω² + post-hoc tables |
| `baselines/` | yes | `experiments.baselines.baselines` | DeepSEA/DanQ/DNABERT comparison (CSV + LaTeX) |
| `ablation/` | yes | `experiments.ablation.ablation` | embedding-size / frozen / N_e sweeps |
| `attribution/` | yes | `experiments.attribution.shiftsmooth_eval` (CSVs) + `experiments.attribution.make_attribution_figures` (logos) | ShiftSmooth faithfulness/stability CSVs + the 10 sequence-logo attribution figures used by the paper |
| `hetmoe/seed<N>/` | yes (small) | `experiments.hetmoe.sweep` | `hetmoe_summary.csv`, `hetmoe_eval.json`, paired-vs-DNABERT |
| `moe_grid/seed<N>/` | yes (small) | `experiments.hetmoe.sweep` | per-config `decision_*.json` (sweep + selection) |
| **`cache/seed<N>/`** | **no (gitignored)** | `experiments.hetmoe.cache_embeddings` (Phase A, GPU) | per-(expert,dataset) embedding `.npz` + DNABERT-6 baseline preds + `manifest.json` (hundreds of MB/seed) |
| `_verify/` | no (gitignored) | refactor verification jobs | temporary; safe to delete |

> **Paper figures.** `paper/edited.tex` resolves every figure straight from `results/`
> (`figures/{nature,performance,supplementary,schematics}/` + `attribution/`) via `\graphicspath` —
> there are no figure copies under `paper/figures/`. `main.tex` is untouched.

Other large regenerable artifacts kept out of git:
- `.hf_cache/` — HuggingFace weights (multi-GB), rebuilt on first download.
- `models/zoo/seed<N>/` — DeepSEA/DanQ/DNABERT-6 zoo checkpoints, rebuilt by Phase A.
- `archive/` — deprecated `legacy/`/`old/` material kept on disk for reference only. The 149 MB
  `archive/legacy/results/evaluation_results_published.json` is a non-regenerable raw-prediction
  dump from the original (non-reproducing) published model; move it to external storage
  (Zenodo/OSF/release asset) if a clean tree is needed.
