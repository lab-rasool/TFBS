"""Nature-style matplotlib helpers: a colorblind-safe Okabe-Ito palette, a fixed
model->color map, journal rcParams (Helvetica/Arial, despined, thin axes), column
widths, and a PDF+PNG save helper. Call ``apply()`` once before plotting.

If Arial/Helvetica are not installed, matplotlib falls back to DejaVu Sans (Nature
wants Helvetica/Arial — install them for the final submission).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ---- Okabe-Ito colorblind-safe palette ----
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "sky": "#56B4E9",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#999999",
}

# ---- fixed model -> color map (consistent across every figure) ----
MODEL_COLORS = {
    "HetMoE": OKABE_ITO["blue"],
    "DNABERT": OKABE_ITO["orange"], "DNABERT-6": OKABE_ITO["orange"], "DNABERT6": OKABE_ITO["orange"],
    "DeepSEA": OKABE_ITO["green"],
    "DanQ": OKABE_ITO["vermillion"],
    "MoE": OKABE_ITO["sky"], "orig-MoE": OKABE_ITO["sky"], "moe": OKABE_ITO["sky"],
    "static_mean": OKABE_ITO["purple"], "static mean": OKABE_ITO["purple"],
    "best_single": OKABE_ITO["grey"], "best single": OKABE_ITO["grey"],
    "ConvNet": OKABE_ITO["sky"],
}

# ---- column widths (Nature: 89 mm single, 183 mm double, ~120 mm 1.5) ----
MM = 1.0 / 25.4
COL1 = 89 * MM
COL1_5 = 120 * MM
COL2 = 183 * MM

FIG_DIR = os.path.join("results", "figures", "nature")


def apply():
    """Set Nature-style rcParams globally."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.transparent": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "legend.frameon": False, "legend.handlelength": 1.4,
        "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.0, "lines.markersize": 4,
        "axes.labelpad": 2.0, "axes.titlepad": 4.0,
        "figure.constrained_layout.use": False,
    })


def color(model):
    return MODEL_COLORS.get(model, OKABE_ITO["black"])


def panel_label(ax, letter, x=-0.20, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="right")


def save(fig, name):
    """Write <name>.pdf (vector) and <name>.png (600 dpi) into results/figures/nature/."""
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name + ".pdf"))
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=600)
    plt.close(fig)
