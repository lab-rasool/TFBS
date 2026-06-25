"""Shared evaluation metrics (brier, ece, auc). Single source of truth; the
former hetmoe.py and evaluate.py each had their own copies.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= edges[b + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(y[m].mean() - p[m].mean())
    return float(e)


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")


def paired_bootstrap(preds, y, seed, n_boot=1000):
    """Paired percentile bootstrap of AUROC.

    ``preds`` maps model name -> probability array; one **common** resample index
    per replicate is shared across all models, so model-vs-model comparisons are
    paired. A degenerate (single-class) resample is redrawn once. Deterministic
    given ``seed``. Returns ``{name: np.ndarray(n_boot)}`` (insertion order of
    ``preds`` preserved).
    """
    y = np.asarray(y)
    n = len(y)
    rng = np.random.default_rng(seed)
    boot = {m: np.empty(n_boot) for m in preds}
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            idx = rng.integers(0, n, n)
        yb = y[idx]
        for m in preds:
            boot[m][b] = roc_auc_score(yb, preds[m][idx])
    return boot
