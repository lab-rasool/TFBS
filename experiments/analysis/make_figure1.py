"""Figure 1 — the HetMoE method overview (data -> experts -> MoE -> explainability).

A hand-laid-out schematic built in matplotlib so it renders reproducibly to a
vector PDF (+600 dpi PNG) and matches the rest of the paper figure pack
(``tfbs.figstyle``). Replaces the old hand-drawn ``paper/figure1_hetmoe.drawio``
export, fixing the clipped expert label, the disconnected panels, and the empty
explainability panel (now showing real Vanilla-Gradients vs ShiftSmooth logos).

Run from the repo root::

    python -m experiments.analysis.make_figure1

Writes ``paper/figure1_hetmoe.{pdf,png}`` by default (``--outdir`` to redirect).
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

from tfbs import figstyle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "paper" / "Special_Issue__Explainable_AI_in_Genomics_Revised"

# ---- palette ----
OI = figstyle.OKABE_ITO
EXPERT_COLORS = {
    "ConvNet": OI["sky"],
    "DeepSEA": OI["green"],
    "DanQ": OI["vermillion"],
    "DNABERT-6": OI["orange"],
}
# nucleotide colours (colourblind-safe, standard A=green C=blue G=orange T=red)
BASE_COLOR = {"A": OI["green"], "C": OI["blue"], "G": OI["orange"], "T": OI["vermillion"]}

# panel background tints
TINT = {
    "A": "#f4f4f4",
    "B": "#fdf3e7",   # faint orange (individual experts)
    "C": "#eef7f1",   # faint green (the hero MoE panel)
    "D": "#f2f6fa",   # faint blue (explainability)
}
PANEL_EDGE = "#b8b8b8"
INK = "#2b2b2b"

INK_LW = 1.1          # standard arrow / edge weight
EDGE_LW = 0.8


def lighten(hexcolor, amount=0.55):
    """Blend a hex colour toward white by ``amount`` (0=orig, 1=white)."""
    h = hexcolor.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------------------------------------- #
#  drawing primitives (all in data coords on a single full-figure axes)
# --------------------------------------------------------------------------- #
def panel(ax, x, y, w, h, tag, title, fill, title_center=False):
    """Rounded background container with a bold A/B/C/D tag and a title.

    Narrow panels left-align the title right after the tag (avoids the tag
    overlapping a centred title); wide panels can centre it.
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=2.5",
        linewidth=EDGE_LW, edgecolor=PANEL_EDGE, facecolor=fill, zorder=1))
    ax.text(x + 2.6, y + h - 2.7, tag, fontsize=11, fontweight="bold",
            color=INK, va="top", ha="left", zorder=6)
    if title_center:
        ax.text(x + w / 2, y + h - 3.1, title, fontsize=8.5, fontweight="bold",
                color=INK, va="top", ha="center", zorder=6)
    else:
        ax.text(x + 8.5, y + h - 3.1, title, fontsize=8.0, fontweight="bold",
                color=INK, va="top", ha="left", zorder=6)


def box(ax, cx, cy, w, h, text, fc="white", ec=INK, fs=6.6, bold=False,
        rounding=1.2, lw=EDGE_LW, fontcolor=INK, z=4):
    """Rounded rectangle node centred at (cx, cy). Returns the centre."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z))
    ax.text(cx, cy, text, fontsize=fs, ha="center", va="center",
            color=fontcolor, fontweight="bold" if bold else "normal", zorder=z + 1)
    return (cx, cy)


def expert(ax, cx, cy, w, h, name, fc, fs=6.4, taper_frac=0.14):
    """A conv-style trapezoid (narrows downward) for an expert backbone.

    The label sits slightly above centre, where the trapezoid is widest, so
    long names (DNABERT-6, DeepSEA) clear the sloped sides.
    """
    taper = w * taper_frac
    pts = [(cx - w / 2, cy + h / 2), (cx + w / 2, cy + h / 2),
           (cx + w / 2 - taper, cy - h / 2), (cx - w / 2 + taper, cy - h / 2)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=INK,
                         linewidth=EDGE_LW, zorder=4))
    ax.text(cx, cy + h * 0.12, name, fontsize=fs, fontweight="bold",
            ha="center", va="center", color=INK, zorder=5)


def arrow(ax, p0, p1, color=INK, lw=INK_LW, style="-", connect=None, z=3,
          mut=7):
    """Arrow from p0 to p1. ``connect`` overrides the FancyArrowPatch path."""
    cs = connect or "arc3,rad=0"
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=mut, linewidth=lw,
        color=color, linestyle=style, connectionstyle=cs,
        shrinkA=1.5, shrinkB=1.5, zorder=z, joinstyle="round",
        capstyle="round"))


def onehot(ax, x0, y0, cellw, cellh, seq, show_rowlabels=True, show_seq=True):
    """One-hot grid: rows A,C,G,T x len(seq) columns. Filled cell = base color."""
    rows = ["A", "C", "G", "T"]
    n = len(seq)
    for j, base in enumerate(seq):
        if show_seq:
            ax.text(x0 + (j + 0.5) * cellw, y0 + 4 * cellh + 1.1, base,
                    fontsize=5.4, ha="center", va="bottom",
                    color=BASE_COLOR[base], fontweight="bold")
        for i, r in enumerate(rows):
            yy = y0 + (3 - i) * cellh   # A on top
            on = (base == r)
            ax.add_patch(Rectangle(
                (x0 + j * cellw, yy), cellw, cellh,
                facecolor=BASE_COLOR[r] if on else "white",
                edgecolor="#c9c9c9", linewidth=0.35, zorder=3))
    if show_rowlabels:
        for i, r in enumerate(rows):
            yy = y0 + (3 - i) * cellh + cellh / 2
            ax.text(x0 - 0.8, yy, r, fontsize=5.0, ha="right", va="center",
                    color=BASE_COLOR[r], fontweight="bold")
    return (x0 + n * cellw / 2, y0 + 4 * cellh)   # top-centre anchor


def image_box(ax, path, x0, y0, w, h, border=True):
    """Embed a PNG into a data-coord rectangle via an inset axes."""
    ins = ax.inset_axes([x0, y0, w, h], transform=ax.transData, zorder=4)
    ins.imshow(plt.imread(str(path)), aspect="auto", interpolation="lanczos")
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_visible(border)
        s.set_linewidth(0.5)
        s.set_edgecolor("#b0b0b0")
    return ins


# --------------------------------------------------------------------------- #
#  figure
# --------------------------------------------------------------------------- #
def build():
    figstyle.apply()
    W, H = 188.0, 108.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis("off")

    SEQ = "ATCGATAA"

    # ===================== Panel A — Data preprocessing ==================== #
    ax_, ay, aw, ah = 3.0, 61.0, 38.0, 46.0
    panel(ax, ax_, ay, aw, ah, "A", "Data preprocessing", TINT["A"])
    # input sequence as coloured letters (same bases as the grid below)
    ax.text(ax_ + aw / 2, ay + ah - 9.5, "Input sequence  (101 bp)",
            fontsize=6.4, ha="center", va="center", color=INK)
    sx = ax_ + aw / 2 - (len(SEQ) - 1) * 2.7 / 2
    for j, b in enumerate(SEQ):
        ax.text(sx + j * 2.7, ay + ah - 15, b, fontsize=6.4, ha="center",
                va="center", color=BASE_COLOR.get(b, INK), fontweight="bold")
    ax.text(sx + (len(SEQ) - 0.3) * 2.7, ay + ah - 15, "…", fontsize=6.4,
            ha="left", va="center", color="#777", fontweight="bold")
    # arrow down to grid
    arrow(ax, (ax_ + aw / 2, ay + ah - 17), (ax_ + aw / 2, ay + ah - 20.5))
    # one-hot grid
    g_cellw, g_cellh = 3.0, 2.6
    gx = ax_ + aw / 2 - len(SEQ) * g_cellw / 2 + 1.0
    top = onehot(ax, gx, ay + 5.5, g_cellw, g_cellh, SEQ)
    ax.text(ax_ + aw / 2, ay + 3.0, "one-hot encoding  (4 × L)",
            fontsize=6.0, ha="center", va="center", color=INK, style="italic")

    # ===================== Panel B — Individual expert ===================== #
    bx, by, bw, bh = 3.0, 3.0, 38.0, 55.0
    panel(ax, bx, by, bw, bh, "B", "Per-TF expert", TINT["B"])
    col = bx + 12.0           # spine x for the vertical stack
    y_in = by + bh - 11
    box(ax, col, y_in, 19, 5.2, "one-hot  (4 × L)", fc="white", fs=6.0)
    expert(ax, col, y_in - 10.5, 17, 9.0, "Expert\n(ConvNet)", EXPERT_COLORS["ConvNet"])
    y_hid = y_in - 21.5
    box(ax, col, y_hid, 19, 5.4, "Hidden  (E = 32)", fc="white", fs=6.0)
    y_fc = y_hid - 8.5
    box(ax, col, y_fc, 19, 5.4, "FC", fc=lighten(OI["yellow"], 0.35), fs=6.4, bold=True)
    y_out = y_fc - 8.5
    box(ax, col, y_out, 19, 5.4, "P(bound)", fc="white", fs=6.6, bold=True)
    for a, b in [(y_in - 2.6, y_in - 6.0), (y_in - 15.0, y_hid + 2.7),
                 (y_hid - 2.7, y_fc + 2.7), (y_fc - 2.7, y_out + 2.7)]:
        arrow(ax, (col, a), (col, b))
    # attached caption (heterogeneous backbones) — stacked in the right gutter
    cap_x = bx + bw - 2.2
    ax.text(cap_x, by + bh - 15,
            "trained once\nper TF\n\nbackbones\n· ConvNet\n· DeepSEA\n· DanQ\n· DNABERT-6",
            fontsize=4.8, ha="right", va="top", color="#6a6a6a", linespacing=1.32)

    # ===================== Panel C — Heterogeneous MoE ===================== #
    cx0, cy0, cw, ch = 47.0, 3.0, 76.0, 104.0
    panel(ax, cx0, cy0, cw, ch, "C", "Heterogeneous Mixture-of-Experts", TINT["C"],
          title_center=True)
    cmid = cx0 + cw / 2
    # shared input one-hot at top, fans out to the 4 experts. Label sits high
    # (just under the title); the grid is dropped lower so its A/T/C/G letters
    # never collide with the label.
    ax.text(cmid, cy0 + ch - 7.5, "shared input one-hot", fontsize=6.0,
            ha="center", va="center", color=INK, style="italic")
    sh_cellw, sh_cellh = 2.6, 2.3
    shx = cmid - len(SEQ) * sh_cellw / 2
    grid_y = cy0 + ch - 25
    onehot(ax, shx, grid_y, sh_cellw, sh_cellh, SEQ, show_rowlabels=False)
    sh_bottom = (cmid, grid_y)

    experts = list(EXPERT_COLORS.items())
    ex_y = cy0 + ch - 38
    xs = [cx0 + cw * f for f in (0.16, 0.39, 0.62, 0.85)]
    emb_y = ex_y - 13
    # "frozen experts (x7 TFs)" repetition plate behind the expert row
    ax.add_patch(FancyBboxPatch(
        (cx0 + 4.5, ex_y - 7.2), cw - 9, 14.6,
        boxstyle="round,pad=0,rounding_size=1.5", linewidth=0.7,
        edgecolor="#9cc4ad", facecolor="none", linestyle=(0, (4, 3)), zorder=2))
    for (name, fc), xc in zip(experts, xs):
        # fan-out arrow from shared input to each expert (drawn first, low z)
        arrow(ax, sh_bottom, (xc, ex_y + 5.4),
              connect="arc3,rad=0.0", color="#7a7a7a", lw=0.9, mut=6)
        expert(ax, xc, ex_y, 16.0, 9.0, name, fc, fs=5.8)
        box(ax, xc, emb_y, 16.0, 6.2, "Embedding\nE = 32", fc=lighten(fc, 0.62),
            ec=fc, fs=5.8)
        arrow(ax, (xc, ex_y - 4.5), (xc, emb_y + 3.2))
    # label drawn AFTER the arrows, with an opaque backing so the fan-out lines
    # pass cleanly behind it instead of striking through the text
    ax.text(cx0 + cw / 2, ex_y + 8.4, "frozen heterogeneous experts   (× 7 TFs)",
            fontsize=6.2, ha="center", va="center", color="#2f6f4e",
            fontweight="bold", zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=TINT["C"],
                      edgecolor="none"))

    # gating network (consumes all embeddings)
    gate_y = emb_y - 13
    gate = box(ax, cmid, gate_y, cw - 16, 7.0,
               "Gating network   (softmax over experts)",
               fc=lighten(OI["purple"], 0.45), ec=OI["purple"], fs=6.6, bold=True)
    for xc in xs:
        arrow(ax, (xc, emb_y - 3.2), (xc, gate_y + 3.6),
              color="#7a7a7a", lw=0.9, mut=6)
    # weighted combine + FC
    comb_y = gate_y - 12
    box(ax, cmid, comb_y, cw - 26, 6.6, "Weighted combine  +  FC",
        fc=lighten(OI["yellow"], 0.35), ec="#b9912a", fs=6.6, bold=True)
    arrow(ax, (cmid, gate_y - 3.6), (cmid, comb_y + 3.4))
    # P(bound)
    out_y = comb_y - 11
    box(ax, cmid, out_y, 26, 7.2, "P(bound)", fc="white", fs=8.0, bold=True,
        lw=1.1)
    arrow(ax, (cmid, comb_y - 3.4), (cmid, out_y + 3.7))

    # ===================== Panel D — Explainability ======================== #
    dx0, dy0, dw, dh = 129.0, 3.0, 56.0, 104.0
    panel(ax, dx0, dy0, dw, dh, "D", "Explainability: ShiftSmooth", TINT["D"],
          title_center=True)
    dmid = dx0 + dw / 2
    ix = dx0 + 3.5
    iw = dw - 7.0
    sh_h = 5.0               # input-logo height (mild stretch from 13.6:1)
    lg_h = 13.0             # attribution-logo height (the two key results)

    def attribution(label, expr, tag_text, tag_face, tag_edge, img, y_label):
        """One labelled attribution row: title (+ math) · tag · logo below."""
        ax.text(ix, y_label, label, fontsize=6.5, ha="left", va="center",
                color=INK, fontweight="bold")
        if expr:
            ax.text(ix + 0.3, y_label - 3.8, expr, fontsize=6.0, ha="left",
                    va="center", color="#555")
        box(ax, dx0 + dw - 8.0, y_label, 14.5, 4.8, tag_text, fc=tag_face,
            ec=tag_edge, fs=5.5, bold=True)
        image_box(ax, ASSETS / img, ix, y_label - 6.0 - lg_h, iw, lg_h)
        return y_label - 6.0 - lg_h          # bottom y of the logo

    # --- input sequence logo (motif sites boxed) ---
    ax.text(dmid, dy0 + dh - 11, "Input sequence",
            fontsize=6.4, ha="center", va="center", color=INK)
    in_top = dy0 + dh - 13.5
    image_box(ax, ASSETS / "sequence_both_highlighted.png",
              ix, in_top - sh_h, iw, sh_h)
    arrow(ax, (dmid, in_top - sh_h - 0.6), (dmid, in_top - sh_h - 4.0))

    # --- vanilla gradients ---
    vg_bottom = attribution(
        "Vanilla gradients", "∂ ŷ / ∂ x", "raw gradient",
        lighten(OI["vermillion"], 0.55), OI["vermillion"], "moe_vg_both.png",
        in_top - sh_h - 11.0)
    arrow(ax, (dmid, vg_bottom - 0.6), (dmid, vg_bottom - 5.2))

    # --- shift / average mini-pipeline ---
    pipe_y = vg_bottom - 10.0
    steps = ["shift ±N", "MoE", "shift back", "average"]
    pw = iw / len(steps)
    for k, s in enumerate(steps):
        c = ix + pw * (k + 0.5)
        box(ax, c, pipe_y, pw - 2.0, 5.4, s, fc="white", fs=5.2)
        if k:
            arrow(ax, (ix + pw * k + 1.0, pipe_y),
                  (ix + pw * (k + 0.5) - (pw - 2.0) / 2, pipe_y), lw=0.8, mut=4.5)
    arrow(ax, (dmid, pipe_y - 3.1), (dmid, pipe_y - 7.6))

    # --- shiftsmooth ---
    attribution(
        "ShiftSmooth", "average over ±N shifts", "smoothed",
        lighten(OI["green"], 0.5), OI["green"], "moe_shiftsmooth_both.png",
        pipe_y - 11.5)

    # ===================== cross-panel flow arrows ========================= #
    # A -> B  (encode, down the left column)
    arrow(ax, (ax_ + aw / 2, ay + 0.3), (bx + bw / 2, by + bh - 0.4),
          lw=1.4, color=INK, mut=8)
    ax.text(ax_ + aw / 2 + 1.6, (ay + by + bh) / 2, "encode", fontsize=5.8,
            ha="left", va="center", color="#555", style="italic")
    # B -> C  (the per-TF experts become the frozen experts in the MoE)
    arrow(ax, (bx + bw + 0.3, by + bh - 20), (cx0 - 0.3, by + bh - 20),
          lw=1.4, color=INK, mut=8)
    ax.text((bx + bw + cx0) / 2, by + bh - 11, "trained experts",
            fontsize=5.3, ha="center", va="center", color="#555",
            style="italic", rotation=90)
    # C -> D  (explain the trained model's predictions)
    arrow(ax, (cx0 + cw + 0.3, cy0 + ch / 2), (dx0 - 0.3, cy0 + ch / 2),
          lw=1.4, color=INK, mut=8)
    ax.text((cx0 + cw + dx0) / 2, cy0 + ch / 2 + 4.5, "attribute",
            fontsize=5.8, ha="center", va="center", color="#555",
            style="italic", rotation=90)

    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=str(ROOT / "paper"))
    ap.add_argument("--name", default="figure1_hetmoe")
    args = ap.parse_args()
    fig = build()
    os.makedirs(args.outdir, exist_ok=True)
    fig.savefig(os.path.join(args.outdir, args.name + ".pdf"))
    fig.savefig(os.path.join(args.outdir, args.name + ".png"), dpi=600)
    plt.close(fig)
    print("wrote", os.path.join(args.outdir, args.name + ".{pdf,png}"))


if __name__ == "__main__":
    main()
