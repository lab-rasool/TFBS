"""Fig 9 — single attribution grid for the paper's Explainability section.

Builds a 5×2 grid (rows: input sequence / VG-expert / ShiftSmooth-expert / VG-MoE /
ShiftSmooth-MoE; cols: GATA3 positive | random negative) as a nucleotide-letter sequence
logo using Captum's InputXGradient and the ShiftSmooth average-over-translations gradient.

Output: results/figures/paper/fig_9_attribution.{pdf,png}

Run from the repo root:
  TFBS_TRAIN_NEG=genomic python -m experiments.attribution.make_attribution_figures
"""
import os

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logomaker
from captum.attr import InputXGradient

from tfbs.data import ChipDataLoader
from tfbs.utils import set_seed
from tfbs import figstyle as fs
from tfbs.figstyle import COL2
from experiments.attribution.shiftsmooth_eval import (
    load_expert, load_moe, shiftsmooth, CORE0, CORE1, positives_for, device,
)

fs.apply()  # Nature-style rcParams (Arial, embedded fonts, despined)
BASES = "ACGT"
MOTIF = "GATAA"  # GATA3 core; its reverse complement TTATC is the other orientation


def fwd(model):
    return lambda x: model(x, training=False).reshape(-1, 1)


def vanilla_gradxinput(model, x0):
    """Captum InputXGradient = input x grad of the logit w.r.t. the one-hot input -> (4,147)."""
    attr = InputXGradient(fwd(model)).attribute(x0.clone().detach().requires_grad_(True), target=0)
    return attr.detach()[0].cpu().numpy()


def shiftsmooth_gradxinput(model, x0, shift=2):
    """ShiftSmooth gradient (averaged over small translations) times input -> (4,147)."""
    g = shiftsmooth(model, x0, shift)            # (4,147) averaged gradient
    return (g * x0[0]).detach().cpu().numpy()


def core_seq(x0):
    oh = x0[0, :, CORE0:CORE1].cpu().numpy()
    return "".join(BASES[i] for i in oh.argmax(0))


def motif_ranges(seq):
    """0-based [min,max] index ranges of GATAA / TTATC occurrences for logo highlighting."""
    rc = MOTIF.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    out = []
    for m in (MOTIF, rc):
        i = seq.find(m)
        while i != -1:
            out.append((i, i + len(m) - 1))
            i = seq.find(m, i + 1)
    return out[:2]  # logomaker highlight supports up to two ranges here


def draw_logo(ax, data4, seq, highlight=True):
    """Render a (4, L) per-base array as a letter logo INTO an existing axis (no y-axis;
    rows are identified by Nature-style panel letters instead, see the caption)."""
    df = pd.DataFrame({nuc: data4[i] for i, nuc in enumerate(BASES)})
    logo = logomaker.Logo(df, ax=ax, color_scheme="classic")
    logo.style_spines(visible=False)
    logo.style_spines(spines=["bottom"], visible=True)
    ax.set_yticks([])
    ax.set_xticks([])
    if highlight:
        # k=0 is GATAA (green), k=1 its reverse complement TTATC (blue), matching the text.
        for k, (a, b) in enumerate(motif_ranges(seq)):
            logo.highlight_position_range(
                pmin=a, pmax=b, color="honeydew" if k == 0 else "lightcyan",
                edgecolor="green" if k == 0 else "blue", padding=0.05)


def random_seq(rng, n=101):
    return "".join(rng.choice(list(BASES), size=n))


def fig_9_attribution(expert, moe, cdl, rng):
    """One figure: rows = {input seq, VG-expert, SS-expert, VG-MoE, SS-MoE};
    cols = {GATA3 positive | random negative}. -> results/figures/paper/fig_9_attribution."""
    PAPER = os.path.join("results", "figures", "paper")

    pos = positives_for("GATA3", 60)
    both = next((s for s in pos if "GATAA" in s[CORE0:CORE1] or "TTATC" in s[CORE0:CORE1]), pos[0])
    columns = [("both", both, True), ("random", random_seq(rng), False)]
    panel_letters = ["A", "B", "C", "D", "E"]  # one per row; meanings given in the caption

    # The full 101 bp core renders as ~100 unreadable letters, so zoom every row to a
    # window that covers the positive sequence's highlighted GATA sites (+flank) and crops
    # the long uninformative flanks. The same width is used for both columns so the letter
    # size matches across the figure.
    FLANK = 6
    both_cs = core_seq(torch.from_numpy(cdl.seqtopad(both)).float().unsqueeze(0).to(device))
    mr = motif_ranges(both_cs)
    lo = max(0, min(a for a, _ in mr) - FLANK) if mr else 0
    hi = min(len(both_cs), max(b for _, b in mr) + 1 + FLANK) if mr else len(both_cs)
    WIN = hi - lo

    fig, axes = plt.subplots(5, 2, figsize=(COL2, 0.66 * COL2),
                             gridspec_kw={"height_ratios": [0.3, 1, 1, 1, 1]})
    for c, (tag, seq, hl) in enumerate(columns):
        x0 = torch.from_numpy(cdl.seqtopad(seq)).float().unsqueeze(0).to(device)
        cs = core_seq(x0)
        ranges = motif_ranges(cs) if hl else []
        center = ((min(a for a, _ in ranges) + max(b for _, b in ranges)) // 2
                  if ranges else len(cs) // 2)
        w0 = max(0, min(center - WIN // 2, len(cs) - WIN))
        w1 = w0 + WIN
        cw = cs[w0:w1]
        oh = x0[0, :, CORE0:CORE1].cpu().numpy()[:, w0:w1]
        data = [oh,
                vanilla_gradxinput(expert, x0)[:, CORE0:CORE1][:, w0:w1],
                shiftsmooth_gradxinput(expert, x0)[:, CORE0:CORE1][:, w0:w1],
                vanilla_gradxinput(moe, x0)[:, CORE0:CORE1][:, w0:w1],
                shiftsmooth_gradxinput(moe, x0)[:, CORE0:CORE1][:, w0:w1]]
        for r in range(5):
            draw_logo(axes[r][c], data[r], cw, highlight=hl)
        axes[0][c].set_title("GATA3 (positive)" if tag == "both" else "random (negative)",
                             fontsize=8)
    for r, letter in enumerate(panel_letters):
        fs.panel_label(axes[r][0], letter, x=-0.03, y=1.0)
    axes[4][0].set_xlabel("nucleotide position", fontsize=7)
    axes[4][1].set_xlabel("nucleotide position", fontsize=7)
    fig.tight_layout()
    fs.save(fig, "fig_9_attribution", outdir=PAPER)


def main():
    set_seed(42)
    cdl = ChipDataLoader("")
    rng = np.random.default_rng(42)
    expert = load_expert("GATA3")
    moe = load_moe()
    fig_9_attribution(expert, moe, cdl, rng)
    print("[attrib-fig] wrote results/figures/paper/fig_9_attribution.{pdf,png}")


if __name__ == "__main__":
    main()
