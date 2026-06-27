# Figure de-duplication — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the paper's ~24 overlapping `\includegraphics` and ~7 figure
generators with **10 canonical figures (one per result)**, regenerated from existing
results by **one generator per family**, and rewire the paper to them.

**Architecture:** A single analysis generator (`experiments/analysis/make_paper_figures.py`,
rewritten) emits Figs 3–9 from `results/hetmoe/seed42_genomic/hetmoe_eval.json` (+ the 3
genomic seeds for Fig 7, + the offline cache for Fig 9). A single attribution generator
(`experiments/attribution/make_attribution_figures.py`, refactored) emits the Fig 10
grid. Figs 1–2 are copied static schematic assets. All final figures live in
`results/figures/paper/`. The paper is rewired and the dead/duplicate code + figure dirs
are deleted.

**Tech Stack:** Python 3.13, numpy, matplotlib (`tfbs.figstyle`), logomaker + Captum
(attribution), scikit-learn (`train_test_split`), `tfbs.gate`/`tfbs.experts` (gate
reconstruction), pdflatex + the bundled `latexdiff.pl` (paper).

## Global Constraints

- **Negative protocol:** run every generator with `TFBS_TRAIN_NEG=genomic` so
  `tfbs.constants.MODE_SUFFIX == "_genomic"` and all JSON/cache paths resolve to
  `*_genomic`. Without it they resolve to the wrong (non-genomic) results.
- **Canonical data:** `results/hetmoe/seed42_genomic/hetmoe_eval.json`; Fig 7 also reads
  `seed{0,1}_genomic`; Fig 9 also reads `results/cache/seed42_genomic/*.npz`.
- **No experiments re-run.** Fig 9 retrains only the lightweight *gate* (a linear layer)
  over already-cached embeddings — no expert training, no new data.
- **Output dir:** `results/figures/paper/` only. Names: `fig_1_method_schematic` …
  `fig_10_attribution`.
- **Style:** every figure uses `tfbs.figstyle` — Okabe–Ito palette via `color()`, column
  widths `COL1`/`COL1_5`/`COL2`, `save(fig, name, outdir=PAPER)` → PDF + 600 dpi PNG.
  AUROC axes span 0–1 with a 0.5 chance line; the ΔAUROC axis (Fig 5) and gate-weight
  axis (Fig 8) are exempt.
- **Selected config = 21 experts** (`ConvNet`+`DeepSEA`+`DanQ`, no `DNABERT6`). Fig 8
  reflects this. The Fig 1 schematic still shows 4 backbones incl. DNABERT-6 — this
  mismatch is a **recorded caveat; do NOT edit Fig 1** (rename-only decision).
- **Commits:** no `Co-Authored-By` / Claude attribution lines.

---

## Task 1: Consolidated analysis generator — Figs 3, 4, 5, 6, 8

**Files:**
- Modify (full rewrite): `experiments/analysis/make_paper_figures.py`

**Interfaces:**
- Consumes: `results/hetmoe/seed42_genomic/hetmoe_eval.json` keys `per_dataset`
  (`in_distribution::TF` / `out_of_distribution::TF`, each with `HetMoE_auc`,
  `HetMoE_ci`, `DNABERT_auc`, `DNABERT_ci`, `gate_weights` [len 21]), `ood_mean`,
  `ood_within_family`, `ood_cross_family`, `ood_nonmotif`, `paired_vs_dnabert`,
  `config.expert_order`.
- Produces: module with `load(seed)`, `od(d,key)`, `idr(d,key)`,
  `fig_3_indist_performance(d)`, `fig_4_ood_headline(d)`, `fig_5_ood_forest(d)`,
  `fig_6_ood_strata(d)`, `fig_8_gate_routing(d)`, `main()`. Tasks 2–3 extend this file.

- [ ] **Step 1: Replace the file with the consolidated generator (Figs 3–6, 8)**

Overwrite `experiments/analysis/make_paper_figures.py` with exactly:

```python
"""Canonical paper figure generator (analysis figures 3-9): one figure per result,
regenerated from the existing genomic eval JSON -- NO experiments re-run.

Writes results/figures/paper/fig_{3..9}_*.{pdf,png}. Run with the genomic negative
protocol so every path resolves to *_genomic:

    TFBS_TRAIN_NEG=genomic python -m experiments.analysis.make_paper_figures --seed 42
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from tfbs import figstyle as fs
from tfbs.figstyle import COL1, COL1_5, COL2, OKABE_ITO, color, save
from tfbs.constants import (TRAIN_TFS, OOD_WITHIN_FAMILY, OOD_CROSS_FAMILY,
                            OOD_NONMOTIF, DBD_FAMILY, MODE_SUFFIX)

fs.apply()
PAPER = os.path.join("results", "figures", "paper")
MOTIF = OOD_WITHIN_FAMILY + OOD_CROSS_FAMILY          # 23 motif-bearing OOD factors
REPR = ["JUN", "MYC", "ELF1", "SP2", "GATA2", "CTCF"]  # gate-heatmap rows
PRETTY = {"HetMoE": "HetMoE", "DNABERT": "DNABERT-6",
          "best_single": "best single expert", "static_mean": "static average"}


def load(seed):
    return json.load(open(f"results/hetmoe/seed{seed}{MODE_SUFFIX}/hetmoe_eval.json"))


def od(d, key):
    return d["per_dataset"][f"out_of_distribution::{key}"]


def idr(d, key):
    return d["per_dataset"][f"in_distribution::{key}"]


def fig_3_indist_performance(d):
    """Per-TF in-distribution AUROC (dot + 95% CI), HetMoE vs DNABERT-6, 7 train TFs."""
    fig, ax = plt.subplots(figsize=(COL1_5, 0.55 * COL1_5))
    x = np.arange(len(TRAIN_TFS)); off = 0.16
    for m, dx in [("HetMoE", -off), ("DNABERT", +off)]:
        for i, tf in enumerate(TRAIN_TFS):
            r = idr(d, tf); lo, hi = r[f"{m}_ci"]
            ax.plot([i + dx, i + dx], [lo, hi], color=color(m), lw=1.1, zorder=2)
            ax.plot(i + dx, r[f"{m}_auc"], "o", color=color(m), ms=3.5, zorder=3,
                    label=PRETTY[m] if i == 0 else None)
    ax.axhline(0.5, ls=":", lw=0.6, color=OKABE_ITO["grey"], zorder=0)
    ax.set_xticks(x); ax.set_xticklabels(TRAIN_TFS, fontsize=6); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC (mean, 95% CI)"); ax.legend(ncol=2, loc="lower right")
    save(fig, "fig_3_indist_performance", outdir=PAPER)


def fig_4_ood_headline(d):
    """Motif-OOD mean AUROC, 4 models -> gain is gating, not ensembling. ``ood_mean``
    is the motif-strata mean (within+cross), == 0.827 for HetMoE."""
    order = ["HetMoE", "DNABERT", "best_single", "static_mean"]
    vals = [d["ood_mean"][m] for m in order]
    fig, ax = plt.subplots(figsize=(COL1, 0.78 * COL1))
    x = np.arange(len(order))
    ax.bar(x, vals, width=0.66, color=[color(m) for m in order], zorder=3)
    ax.axhline(0.5, ls=":", lw=0.6, color=OKABE_ITO["grey"], zorder=0)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x); ax.set_xticklabels([PRETTY[m] for m in order], rotation=20, ha="right")
    ax.set_ylabel("OOD AUROC (motif strata, mean)"); ax.set_ylim(0.0, 1.0)
    save(fig, "fig_4_ood_headline", outdir=PAPER)


def fig_5_ood_forest(d):
    """Per-factor HetMoE-DNABERT-6 ΔAUROC ± 95% CI for the 23 motif-OOD factors, grouped
    by DBD family; blue where the CI excludes 0 in HetMoE's favor."""
    rows = [r for r in d["paired_vs_dnabert"]
            if r["data_type"] == "out_of_distribution" and r["tf"] in MOTIF]
    fam_order, seen = [], set()
    for tf in MOTIF:
        f = DBD_FAMILY.get(tf, "?")
        if f not in seen:
            seen.add(f); fam_order.append(f)
    rows.sort(key=lambda r: (fam_order.index(DBD_FAMILY.get(r["tf"], "?")), -r["mean_diff"]))
    n = len(rows)
    fig, ax = plt.subplots(figsize=(COL1_5, 0.30 * n * COL1 / 6))
    ypos = np.arange(n)[::-1]; last = None
    for y, r in zip(ypos, rows):
        c = color("HetMoE") if r["superior"] else OKABE_ITO["grey"]
        ax.plot([r["ci95_low"], r["ci95_high"]], [y, y], color=c, lw=1.1, zorder=2)
        ax.plot(r["mean_diff"], y, "o", color=c, ms=3, zorder=3)
        fam = DBD_FAMILY.get(r["tf"], "?")
        if last is not None and fam != last:
            ax.axhline(y + 0.5, color=OKABE_ITO["grey"], lw=0.3, alpha=0.5, zorder=0)
        last = fam
    ax.axvline(0, ls="-", lw=0.6, color="black", zorder=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r['tf']}  ({DBD_FAMILY.get(r['tf'], '?')})" for r in rows], fontsize=5)
    ax.set_xlabel("HetMoE − DNABERT-6   ΔAUROC (95% CI)"); ax.set_ylim(-0.7, n - 0.3)
    save(fig, "fig_5_ood_forest", outdir=PAPER)


def fig_6_ood_strata(d):
    """OOD AUROC by stratum (within / cross / non-motif), HetMoE vs DNABERT-6."""
    strata = [("within-family", "ood_within_family"),
              ("cross-family", "ood_cross_family"),
              ("non-motif\n(reported sep.)", "ood_nonmotif")]
    fig, ax = plt.subplots(figsize=(COL1_5, 0.62 * COL1_5))
    x = np.arange(len(strata)); w = 0.38
    h = [d[k]["HetMoE"] for _, k in strata]; b = [d[k]["DNABERT"] for _, k in strata]
    ax.bar(x - w / 2, h, w, color=color("HetMoE"), label="HetMoE", zorder=3)
    ax.bar(x + w / 2, b, w, color=color("DNABERT"), label="DNABERT-6", zorder=3)
    for xi, v in zip(x - w / 2, h):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)
    for xi, v in zip(x + w / 2, b):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)
    ax.axhline(0.5, ls=":", lw=0.6, color=OKABE_ITO["grey"], zorder=0)
    ax.axvspan(1.5, 2.5, color=OKABE_ITO["grey"], alpha=0.10, zorder=0)
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in strata])
    ax.set_ylabel("OOD AUROC"); ax.set_ylim(0.0, 1.0); ax.legend(loc="upper right", ncol=2)
    save(fig, "fig_6_ood_strata", outdir=PAPER)


def fig_8_gate_routing(d):
    """Mean gate weight, expert (cols, config.expert_order) x representative OOD TF (rows)."""
    order = d["config"]["expert_order"]

    def abbr(n):
        bb, tf = n.split("::")
        bb = {"ConvNet": "Conv", "DeepSEA": "DSEA", "DanQ": "DanQ", "DNABERT6": "BERT"}.get(bb, bb)
        return f"{bb}:{tf}"

    M = np.array([od(d, tf)["gate_weights"] for tf in REPR])
    fig, ax = plt.subplots(figsize=(COL2, 0.34 * COL2))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=max(0.12, M.max()))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([abbr(n) for n in order], rotation=90, fontsize=3.6)
    ax.set_yticks(range(len(REPR))); ax.set_yticklabels(REPR, fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cb.set_label("mean gate weight", fontsize=6); cb.ax.tick_params(labelsize=5)
    save(fig, "fig_8_gate_routing", outdir=PAPER)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    d = load(args.seed)
    fig_3_indist_performance(d)
    fig_4_ood_headline(d)
    fig_5_ood_forest(d)
    fig_6_ood_strata(d)
    fig_8_gate_routing(d)
    print("paper figs:", sorted(f for f in os.listdir(PAPER) if f.endswith(".pdf")))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generator**

Run: `TFBS_TRAIN_NEG=genomic python -m experiments.analysis.make_paper_figures --seed 42`
Expected: prints `paper figs: ['fig_3_indist_performance.pdf', 'fig_4_ood_headline.pdf', 'fig_5_ood_forest.pdf', 'fig_6_ood_strata.pdf', 'fig_8_gate_routing.pdf']`

- [ ] **Step 3: Verify the 5 figures exist as PDF + PNG**

Run:
```bash
ls results/figures/paper/fig_{3,4,5,6,8}_*.pdf results/figures/paper/fig_{3,4,5,6,8}_*.png
```
Expected: 10 files listed, no "No such file".

- [ ] **Step 4: Spot-check values are sane**

Run:
```bash
python3 -c "import json; d=json.load(open('results/hetmoe/seed42_genomic/hetmoe_eval.json')); print('ood_mean HetMoE', round(d['ood_mean']['HetMoE'],3), 'DNABERT', round(d['ood_mean']['DNABERT'],3)); print('experts', d['config']['num_experts'])"
```
Expected: `ood_mean HetMoE 0.827 DNABERT 0.8` and `experts 21`.

- [ ] **Step 5: Commit**

```bash
git add experiments/analysis/make_paper_figures.py
git commit -m "figs: consolidated analysis generator (Figs 3-6,8) -> results/figures/paper"
```

---

## Task 2: Add Fig 7 (multi-seed robustness)

**Files:**
- Modify: `experiments/analysis/make_paper_figures.py`

**Interfaces:**
- Consumes: `results/hetmoe/seed{0,1,42}_genomic/hetmoe_eval.json` key `ood_mean`.
- Produces: `fig_7_multiseed(seeds=(0,1,42))`; called from `main()`.

- [ ] **Step 1: Add `fig_7_multiseed` above `def main()`**

```python
def fig_7_multiseed(seeds=(0, 1, 42)):
    """Motif-OOD mean AUROC per genomic seed, HetMoE vs DNABERT-6 (3 seeds)."""
    H, D = [], []
    for s in seeds:
        dd = json.load(open(f"results/hetmoe/seed{s}{MODE_SUFFIX}/hetmoe_eval.json"))
        H.append(dd["ood_mean"]["HetMoE"]); D.append(dd["ood_mean"]["DNABERT"])
    H, D = np.array(H), np.array(D)
    fig, ax = plt.subplots(figsize=(COL1, 0.85 * COL1))
    for xi, (lab, key, v) in enumerate([("HetMoE", "HetMoE", H), ("DNABERT-6", "DNABERT", D)]):
        ax.plot([xi] * len(v), v, "o", color=color(key), ms=5, zorder=3)
        ax.plot([xi - 0.2, xi + 0.2], [v.mean(), v.mean()], color="black", lw=1.2, zorder=4)
        ax.text(xi, v.max() + 0.004, f"{v.mean():.3f}\n±{v.std(ddof=1):.3f}",
                ha="center", va="bottom", fontsize=6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HetMoE", "DNABERT-6"])
    ax.set_ylabel("motif-OOD AUROC"); ax.set_xlim(-0.5, 1.5); ax.set_ylim(0.5, 0.9)
    save(fig, "fig_7_multiseed", outdir=PAPER)
```

- [ ] **Step 2: Call it in `main()`** — add after the `fig_6_ood_strata(d)` line:

```python
    fig_7_multiseed()
```

- [ ] **Step 3: Run and verify**

Run: `TFBS_TRAIN_NEG=genomic python -m experiments.analysis.make_paper_figures --seed 42`
Then: `ls results/figures/paper/fig_7_multiseed.pdf results/figures/paper/fig_7_multiseed.png`
Expected: both files exist; the figure shows HetMoE mean ≈ 0.821 ±0.005 and DNABERT-6 ≈ 0.799 ±0.008.

- [ ] **Step 4: Commit**

```bash
git add experiments/analysis/make_paper_figures.py
git commit -m "figs: add Fig 7 multi-seed robustness (3 genomic seeds)"
```

---

## Task 3: Add Fig 9 (reliability diagram, reconstructed per-sample)

**Files:**
- Modify: `experiments/analysis/make_paper_figures.py`

**Interfaces:**
- Consumes: `tfbs.experts.load_zoo_cache`, `tfbs.experts.subset_zoo`,
  `tfbs.gate.train_gate`, `tfbs.gate.gate_predict`; cache
  `results/cache/seed42_genomic/DNABERT6_<trainTF>__out_of_distribution_<TF>.npz`
  (keys `pred`, `y`); JSON `ood_mean_ece` (sanity check, ~0.217).
- Produces: `reliability(probs, y, bins=10)`, `_persample_motif(seed)`,
  `fig_9_calibration(seed)`; called from `main()`.

- [ ] **Step 1: Add the reliability helper + per-sample reconstruction + figure**

Add above `def main()`:

```python
def reliability(probs, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    xs, ys, ws = [], [], []
    for b in range(bins):
        m = (probs >= edges[b]) & (probs < edges[b + 1] if b < bins - 1 else probs <= edges[b + 1])
        if m.sum() == 0:
            continue
        xs.append(probs[m].mean()); ys.append(y[m].mean()); ws.append(m.sum())
    ece = sum(w * abs(x - yv) for x, yv, w in zip(xs, ys, ws)) / max(1, sum(ws))
    return np.array(xs), np.array(ys), np.array(ws), float(ece)


def _persample_motif(seed):
    """Per-sample (probs, y) pooled over the 23 motif-OOD factors for HetMoE (the selected
    21-expert gate, retrained deterministically over the cached embeddings -- same path that
    produced the JSON) and the DNABERT-6 mean-ensemble baseline (cache). No expert training."""
    from sklearn.model_selection import train_test_split
    from tfbs.experts import load_zoo_cache, subset_zoo
    from tfbs.gate import train_gate, gate_predict
    zoo = subset_zoo(load_zoo_cache(seed), backbones=["ConvNet", "DeepSEA", "DanQ"])
    emb, Y = zoo["emb"], zoo["y"]
    E, ne = zoo["embedding_dim"], len(zoo["expert_order"])
    tr = [f"train::{tf}" for tf in TRAIN_TFS]
    Xtr = np.concatenate([emb[k] for k in tr], 0)
    ytr = np.concatenate([Y[k] for k in tr], 0)
    itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.2, random_state=seed)
    moe, _ = train_gate(Xtr[itr], ytr[itr], Xtr[iva], ytr[iva], ne, E, seed,
                        l2norm=True, entropy_reg=1e-3, gate_temperature=1.0)
    hp, hy = [], []
    for tf in MOTIF:
        p, _ = gate_predict(moe, emb[f"out_of_distribution::{tf}"])
        hp.append(p); hy.append(Y[f"out_of_distribution::{tf}"])
    cdir = f"results/cache/seed{seed}{MODE_SUFFIX}"
    dp, dy = [], []
    for tf in MOTIF:
        ps = [np.load(f"{cdir}/DNABERT6_{t}__out_of_distribution_{tf}.npz")["pred"]
              for t in TRAIN_TFS]
        dp.append(np.mean(ps, 0))
        dy.append(np.load(f"{cdir}/DNABERT6_{TRAIN_TFS[0]}__out_of_distribution_{tf}.npz")["y"])
    return np.concatenate(hp), np.concatenate(hy), np.concatenate(dp), np.concatenate(dy)


def fig_9_calibration(seed):
    """Reliability diagram (motif-OOD pooled), HetMoE vs DNABERT-6, ECE in legend."""
    hp, hy, dp, dy = _persample_motif(seed)
    fig, ax = plt.subplots(figsize=(COL1, COL1))
    ax.plot([0, 1], [0, 1], ls=":", lw=0.6, color="black", zorder=1)
    eces = {}
    for lab, key, p, y in [("HetMoE", "HetMoE", hp, hy), ("DNABERT-6", "DNABERT", dp, dy)]:
        xs, ys, ws, ece = reliability(p, y)
        eces[lab] = ece
        ax.plot(xs, ys, "-o", color=color(key), ms=3, lw=1.0, label=f"{lab} (ECE={ece:.3f})")
    ax.set_xlabel("mean predicted P(bound)"); ax.set_ylabel("observed fraction bound")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=5.5)
    save(fig, "fig_9_calibration", outdir=PAPER)
    return eces
```

- [ ] **Step 2: Call it in `main()`** — add after `fig_8_gate_routing(d)`:

```python
    eces = fig_9_calibration(args.seed)
    print("recomputed ECE:", eces, "| json ood_mean_ece:", round(d["ood_mean_ece"], 3))
```

- [ ] **Step 3: Run and verify the reconstruction is faithful**

Run: `TFBS_TRAIN_NEG=genomic python -m experiments.analysis.make_paper_figures --seed 42`
Expected: `recomputed ECE: {'HetMoE': ~0.217, 'DNABERT-6': ~0.13x} | json ood_mean_ece: 0.217`.
The HetMoE ECE must match the JSON `ood_mean_ece` (0.217) within ±0.02.

> **Contingency (only if HetMoE ECE deviates > 0.02 from 0.217):** the gate reconstruction
> did not reproduce the selected config. Do NOT ship a wrong curve. Instead replace
> `fig_9_calibration` with a JSON-backed honest panel — two bars `d["ood_mean_ece"]`
> (0.217) and `d["ood_mean_brier"]` (0.240), ylabel "OOD calibration error", title
> "Calibration under shift (HetMoE)" — and note the fallback in the commit message.
> Stop and surface this to the user before continuing.

- [ ] **Step 4: Verify file exists**

Run: `ls results/figures/paper/fig_9_calibration.pdf results/figures/paper/fig_9_calibration.png`
Expected: both exist.

- [ ] **Step 5: Commit**

```bash
git add experiments/analysis/make_paper_figures.py
git commit -m "figs: add Fig 9 reliability diagram (per-sample reconstructed from cache)"
```

---

## Task 4: Refactor attribution into the single Fig 10 grid

**Files:**
- Modify: `experiments/attribution/make_attribution_figures.py`

**Interfaces:**
- Consumes (unchanged, already in the module): `load_expert`, `load_moe`, `shiftsmooth`,
  `CORE0`, `CORE1`, `positives_for`, `device` (from `shiftsmooth_eval`); `vanilla_gradxinput`,
  `shiftsmooth_gradxinput`, `core_seq`, `motif_ranges`, `random_seq`, `BASES`, `MOTIF`.
- Produces: `draw_logo(ax, data4, seq, highlight, show_yaxis)`,
  `fig_10_attribution(expert, moe, cdl, rng)`; rewritten `main()`. Output goes to
  `results/figures/paper/fig_10_attribution.{pdf,png}`.

- [ ] **Step 1: Add a draw-into-axis logo helper**

After the existing `make_sequence_logo` function, add:

```python
def draw_logo(ax, data4, seq, highlight=True, show_yaxis=True):
    """Render a (4, L) per-base array as a letter logo INTO an existing axis."""
    df = pd.DataFrame({nuc: data4[i] for i, nuc in enumerate(BASES)})
    logo = logomaker.Logo(df, ax=ax, color_scheme="classic")
    logo.style_spines(visible=False)
    logo.style_spines(spines=["bottom"] + (["left"] if show_yaxis else []), visible=True)
    if not show_yaxis:
        ax.set_yticks([])
    ax.set_xticks([])
    if highlight:
        for k, (a, b) in enumerate(motif_ranges(seq)):
            logo.highlight_position_range(
                pmin=a, pmax=b, color="lightcyan" if k == 0 else "honeydew",
                edgecolor="blue" if k == 0 else "green", padding=0.05)
```

- [ ] **Step 2: Add the grid figure builder**

```python
def fig_10_attribution(expert, moe, cdl, rng):
    """One figure: rows = {input seq, VG-expert, SS-expert, VG-MoE, SS-MoE};
    cols = {GATA3 positive | random negative}. -> results/figures/paper/fig_10_attribution."""
    import os
    from tfbs import figstyle as fs
    from tfbs.figstyle import COL2
    PAPER = os.path.join("results", "figures", "paper")

    pos = positives_for("GATA3", 60)
    both = next((s for s in pos if "GATAA" in s[CORE0:CORE1] or "TTATC" in s[CORE0:CORE1]), pos[0])
    columns = [("both", both, True), ("random", random_seq(rng), False)]
    row_labels = ["input sequence", "VG (expert)", "ShiftSmooth (expert)",
                  "VG (MoE)", "ShiftSmooth (MoE)"]
    fig, axes = plt.subplots(5, 2, figsize=(COL2, 1.05 * COL2))
    for c, (tag, seq, hl) in enumerate(columns):
        x0 = torch.from_numpy(cdl.seqtopad(seq)).float().unsqueeze(0).to(device)
        cs = core_seq(x0)
        oh = x0[0, :, CORE0:CORE1].cpu().numpy()
        data = [oh,
                vanilla_gradxinput(expert, x0)[:, CORE0:CORE1],
                shiftsmooth_gradxinput(expert, x0)[:, CORE0:CORE1],
                vanilla_gradxinput(moe, x0)[:, CORE0:CORE1],
                shiftsmooth_gradxinput(moe, x0)[:, CORE0:CORE1]]
        for r in range(5):
            draw_logo(axes[r][c], data[r], cs, highlight=hl, show_yaxis=(r > 0))
        axes[0][c].set_title("GATA3 (positive)" if tag == "both" else "random (negative)",
                             fontsize=8)
    for r, lab in enumerate(row_labels):
        axes[r][0].set_ylabel(lab, fontsize=6)
    axes[4][0].set_xlabel("nucleotide position", fontsize=7)
    axes[4][1].set_xlabel("nucleotide position", fontsize=7)
    fs.save(fig, "fig_10_attribution", outdir=PAPER)
```

- [ ] **Step 3: Replace `main()` body** with:

```python
def main():
    set_seed(42)
    cdl = ChipDataLoader("")
    rng = np.random.default_rng(42)
    expert = load_expert("GATA3")
    moe = load_moe()
    fig_10_attribution(expert, moe, cdl, rng)
    print("[attrib-fig] wrote results/figures/paper/fig_10_attribution.{pdf,png}")
```

- [ ] **Step 4: Run and verify**

Run: `TFBS_TRAIN_NEG=genomic python -m experiments.attribution.make_attribution_figures`
Then: `ls results/figures/paper/fig_10_attribution.pdf results/figures/paper/fig_10_attribution.png`
Expected: both exist; the figure is a 5×2 grid (GATA / TTATC boxes highlighted in the positive column).

- [ ] **Step 5: Commit**

```bash
git add experiments/attribution/make_attribution_figures.py
git commit -m "figs: single Fig 10 attribution grid (5x2) -> results/figures/paper"
```

---

## Task 5: Place the two schematic assets (Figs 1–2)

**Files:**
- Create: `results/figures/paper/fig_1_method_schematic.pdf`,
  `results/figures/paper/fig_2_shiftsmooth_schematic.pdf` (copies)

- [ ] **Step 1: Copy + rename the existing assets**

```bash
mkdir -p results/figures/paper
cp results/figures/schematics/fig1_visual_abstract.pdf results/figures/paper/fig_1_method_schematic.pdf
cp results/figures/schematics/ShiftSmoothExample.pdf results/figures/paper/fig_2_shiftsmooth_schematic.pdf
```

- [ ] **Step 2: Verify all 10 figures are now present**

Run:
```bash
ls results/figures/paper/fig_{1,2,3,4,5,6,7,8,9,10}_*.pdf | wc -l
```
Expected: `10`.

- [ ] **Step 3: Commit**

```bash
git add results/figures/paper
git commit -m "figs: place schematic assets as Fig 1/2; full 10-figure paper set"
```

---

## Task 6: Rewire the paper to the 10 figures

**Files:**
- Modify: `paper/paper_hetmoe_clean.tex`
- Regenerate (do not hand-edit): `paper/edited.tex` via `paper/make_trackchanges.sh`

**Old → new mapping** (every `\includegraphics{<old>}` filename → new):

| old filename | new | label |
|---|---|---|
| `fig1_visual_abstract.pdf` | `fig_1_method_schematic.pdf` | `fig:method` |
| `ShiftSmoothExample.pdf` | `fig_2_shiftsmooth_schematic.pdf` | `fig:shiftsmooth` |
| `roc_in_distribution.pdf` **and** `fig4_ci_in_distribution.pdf` | `fig_3_indist_performance.pdf` (keep ONE figure, delete the other env) | `fig:indist` |
| `fig_ood_headline.pdf` | `fig_4_ood_headline.pdf` | `fig:ood_headline` |
| `fig_forest_vs_dnabert.pdf` **and** `fig5_roc_out_of_distribution.pdf` **and** `fig7_ci_out_of_distribution.pdf` | `fig_5_ood_forest.pdf` (keep ONE, delete the other two envs) | `fig:ood_forest` |
| `fig6_bars_out_of_distribution.pdf` **and** `fig_ood_strata.pdf` | `fig_6_ood_strata.pdf` (keep ONE, delete the other env) | `fig:ood_strata` |
| `fig_multiseed.pdf` | `fig_7_multiseed.pdf` | `fig:multiseed` |
| `fig_gate_heatmap.pdf` | `fig_8_gate_routing.pdf` | `fig:gate` |
| `fig_calibration.pdf` | `fig_9_calibration.pdf` | `fig:calib` |
| `fig3_bars_in_distribution.pdf` | *(delete env — redundant with Fig 3)* | — |
| the 10 attribution panels (`sequence_both_highlighted.png`, `vg_both.png`, `shiftsmooth_both.png`, `moe_vg_both.png`, `moe_shiftsmooth_both.png`, and the 5 `*_random.png`) | `fig_10_attribution.pdf` (one figure replacing all 10 envs) | `fig:attribution` |

- [ ] **Step 1: Set the graphicspath** (replace the existing `\graphicspath{...}` line, ~line 38):

```latex
\graphicspath{{../results/figures/paper/}{./}}
```

- [ ] **Step 2: Collapse in-distribution figures to one (Fig 3)**

Delete the `figure*` env containing `roc_in_distribution.pdf` (label `fig:roc`) and the
`figure` env containing `fig3_bars_in_distribution.pdf` (label `fig:in_distribution_bar_plot`).
Replace the `fig4_ci_in_distribution.pdf` env with:

```latex
\begin{figure}[ht]
    \centering
    \includegraphics[width=\linewidth]{fig_3_indist_performance.pdf}
    \caption{Per-factor in-distribution AUROC (point $=$ mean, whisker $=95\%$ bootstrap CI) for HetMoE versus fine-tuned DNABERT-6 across the seven training factors. Full $0$--$1$ axis; deterministic inference on the balanced curated test sets.}
    \label{fig:indist}
\end{figure}
```

- [ ] **Step 3: Collapse OOD per-factor figures to one (Fig 5)**

Delete the `figure*` envs for `fig5_roc_out_of_distribution.pdf` (`fig:rocood`) and
`fig7_ci_out_of_distribution.pdf` (`fig:anova_out`). Replace the `fig_forest_vs_dnabert.pdf`
env (`fig:hetmoe_forest`) with the relabelled version:

```latex
\begin{figure}[ht]
    \centering
    \includegraphics[width=\linewidth]{fig_5_ood_forest.pdf}
    \caption{Per-factor HetMoE\,$-$\,DNABERT-6 $\Delta$AUROC with $95\%$ paired-bootstrap confidence intervals for the $23$ motif-bearing out-of-distribution factors, grouped by DNA-binding-domain family. The interval excludes zero in favour of HetMoE on $13$ of the $23$ factors (blue), with the largest gains on the sequence-specific factors (FOXA2, FOXA1, CTCF, HNF4A).}
    \label{fig:ood_forest}
\end{figure}
```

- [ ] **Step 4: Collapse OOD-stratum figures to one (Fig 6)**

Delete the `fig6_bars_out_of_distribution.pdf` env (`fig:ood_distribution_bar_plot`).
Replace the `fig_ood_strata.pdf` env (`fig:hetmoe_strata`) with label `fig:ood_strata` and
caption kept as-is (it already matches the figure).

- [ ] **Step 5: Update the remaining single-figure includes + labels**

In place, change filenames and labels (caption text already matches each figure):
- `fig1_visual_abstract.pdf` → `fig_1_method_schematic.pdf`, label `fig:method`
- `ShiftSmoothExample.pdf` → `fig_2_shiftsmooth_schematic.pdf`, label `fig:shiftsmooth`
- `fig_ood_headline.pdf` → `fig_4_ood_headline.pdf`, label `fig:ood_headline`
- `fig_multiseed.pdf` → `fig_7_multiseed.pdf`, label `fig:multiseed` (caption: change "in every seed" to "in all three genomic seeds")
- `fig_gate_heatmap.pdf` → `fig_8_gate_routing.pdf`, label `fig:gate`

- [ ] **Step 6: Fix the calibration figure (Fig 9)**

Replace the `fig_calibration.pdf` env with:

```latex
\begin{figure}[ht]
    \centering
    \includegraphics[width=\linewidth]{fig_9_calibration.pdf}
    \caption{Reliability diagram (motif-bearing out-of-distribution factors pooled) for HetMoE and fine-tuned DNABERT-6, with the expected calibration error (ECE) in the legend. Both models are mis-calibrated under distribution shift, reported honestly.}
    \label{fig:calib}
\end{figure}
```

- [ ] **Step 7: Replace the 10 attribution envs with one (Fig 10)**

Delete the ten `figure*` envs for `sequence_both_highlighted.png`, `vg_both.png`,
`shiftsmooth_both.png`, `moe_vg_both.png`, `moe_shiftsmooth_both.png`, `sequence_random.png`,
`vg_random.png`, `shiftsmooth_random.png`, `moe_vg_random.png`, `moe_shiftsmooth_random.png`.
Insert one figure where the first attribution figure was:

```latex
\begin{figure*}[ht]
    \centering
    \includegraphics[width=\textwidth]{fig_10_attribution.pdf}
    \caption{ShiftSmooth attribution. Rows: input sequence, then Vanilla Gradient and ShiftSmooth attributions for the GATA3 ConvNet expert and for the MoE. Columns: a GATA3-positive sequence (GATA / TTATC motifs highlighted) and a random negative. ShiftSmooth concentrates importance on the motif and suppresses off-motif noise.}
    \label{fig:attribution}
\end{figure*}
```

- [ ] **Step 8: Fix all in-text references**

Run to find every cross-reference to an old label:
```bash
grep -nE "\\\\(ref|cref|Cref)\{fig:(roc|in_distribution_bar_plot|anova_in|rocood|anova_out|ood_distribution_bar_plot|hetmoe_bars|hetmoe_forest|hetmoe_multiseed|hetmoe_strata|hetmoe_gate|hetmoe_calib|visualabstract|shiftsmoothexample|seq_GATA|GATA_VG|GATA_ShSm|MoE_VG|MoE_ShSm|seq_random|random_VG|random_ShSm|moe_random_VG|moe_random_ShSm)\}" paper/paper_hetmoe_clean.tex
```
For each hit, remap to the new label: in-dist (`fig:roc`/`fig:in_distribution_bar_plot`/`fig:anova_in`)→`fig:indist`; OOD per-factor (`fig:rocood`/`fig:anova_out`/`fig:hetmoe_forest`)→`fig:ood_forest`; strata (`fig:ood_distribution_bar_plot`/`fig:hetmoe_strata`)→`fig:ood_strata`; `fig:hetmoe_bars`→`fig:ood_headline`; `fig:hetmoe_multiseed`→`fig:multiseed`; `fig:hetmoe_gate`→`fig:gate`; `fig:hetmoe_calib`→`fig:calib`; `fig:visualabstract`→`fig:method`; `fig:shiftsmoothexample`→`fig:shiftsmooth`; all attribution labels→`fig:attribution`.

- [ ] **Step 9: Confirm no old filenames or labels remain**

Run:
```bash
grep -nE "roc_in_distribution|fig3_bars|fig4_ci|fig5_roc|fig6_bars|fig7_ci|fig_ood_headline|fig_forest_vs_dnabert|fig_multiseed|fig_ood_strata|fig_gate_heatmap|fig_calibration|fig1_visual_abstract|ShiftSmoothExample|vg_both|shiftsmooth_both|moe_vg|moe_shiftsmooth|sequence_both|sequence_random|vg_random|shiftsmooth_random" paper/paper_hetmoe_clean.tex
```
Expected: no output (all replaced).

- [ ] **Step 10: Compile + regenerate track-changes**

```bash
cd paper && pdflatex -interaction=nonstopmode paper_hetmoe_clean && bibtex paper_hetmoe_clean && pdflatex -interaction=nonstopmode paper_hetmoe_clean && pdflatex -interaction=nonstopmode paper_hetmoe_clean && ./make_trackchanges.sh
```
Expected: `paper_hetmoe_clean.pdf` rebuilds with no "Undefined references"/"File not found"; `edited.pdf` regenerates.

- [ ] **Step 11: Commit**

```bash
cd /mnt/f/Projects/TFBS && git add paper/paper_hetmoe_clean.tex
git commit -m "paper: rewire to 10 canonical figures; fix captions/labels/refs"
```
(Note: `paper/` is gitignored except tracked sources — `git add` only what is tracked; if the tex is untracked, skip the commit and note it.)

---

## Task 7: Delete dead/duplicate generators and figure dirs

**Files:**
- Delete: `experiments/analysis/make_figures_v2.py`, `experiments/analysis/make_multiseed_fig.py`,
  `experiments/analysis/make_schematics.py`, `experiments/attribution/make_sequence_logos.py`
- Delete dirs: `results/figures/nature/`, `results/figures/performance/`, `results/figures/supplementary/`

- [ ] **Step 1: Confirm nothing imports the to-be-deleted modules**

```bash
grep -rnE "make_figures_v2|make_multiseed_fig|make_schematics|make_sequence_logos" --include=*.py --include=*.sbatch --include=*.md . | grep -v "docs/superpowers"
```
Expected: no output outside the spec/plan docs (if any references exist, update them first).

- [ ] **Step 2: Delete the dead generators**

```bash
git rm experiments/analysis/make_figures_v2.py experiments/analysis/make_multiseed_fig.py \
       experiments/analysis/make_schematics.py experiments/attribution/make_sequence_logos.py
```

- [ ] **Step 3: Delete the superseded figure dirs**

```bash
git rm -r results/figures/nature results/figures/performance results/figures/supplementary
```

- [ ] **Step 4: Verify the two canonical generators still run from clean**

```bash
TFBS_TRAIN_NEG=genomic python -m experiments.analysis.make_paper_figures --seed 42 && \
TFBS_TRAIN_NEG=genomic python -m experiments.attribution.make_attribution_figures && \
ls results/figures/paper/fig_*.pdf | wc -l
```
Expected: both run clean; `10` (Fig 1/2 are static, already present; generators write 3–10).

- [ ] **Step 5: Re-validate paper numbers**

Run: `python -m experiments.analysis.validate_paper`
Expected: exits 0 (all quantitative claims still hold; we changed only figures, not numbers).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "figs: remove dead generators and superseded figure dirs"
```

---

## Self-review (completed by plan author)

- **Spec coverage:** Figs 1–10 → Tasks 5,5,1,1,1,1,2,1,3,4; one generator per family →
  Tasks 1–4; paper rewire → Task 6; deletions → Task 7; caveats (Fig 8 21-expert, Fig 1
  overlap, 3 seeds) honored in Global Constraints + Task 6 caption note; acceptance
  criteria 1–5 map to Task 5 Step 2, Task 7 Step 4, Task 6 Step 10, Task 7 Steps 1–3,
  Task 7 Step 5. No gaps.
- **Placeholder scan:** none — every code step is complete; Task 6 prose edits give exact
  strings; the Fig 9 contingency is a defined branch, not a TODO.
- **Type consistency:** generator functions `fig_3_indist_performance`/`fig_4_ood_headline`/
  `fig_5_ood_forest`/`fig_6_ood_strata`/`fig_7_multiseed`/`fig_8_gate_routing`/
  `fig_9_calibration` and helpers `od`/`idr`/`load`/`reliability`/`_persample_motif` are
  defined once and called consistently; attribution `draw_logo`/`fig_10_attribution`
  reuse the module's existing `vanilla_gradxinput`/`shiftsmooth_gradxinput`/`core_seq`/
  `motif_ranges`/`random_seq` with their current signatures.
