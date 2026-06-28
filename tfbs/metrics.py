"""Shared evaluation metrics (brier, ece, auc). Single source of truth; the
former hetmoe.py and evaluate.py each had their own copies.
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def brier(y, p):
    return float(np.mean((p - y) ** 2))


def auprc(y, p):
    """Area under the precision-recall curve (average precision). On a balanced
    500/500 set the no-skill baseline is 0.5; reported alongside AUROC because
    AUROC is prevalence-invariant and overstates usable precision when binding is
    rare (McDermott et al., NeurIPS 2024)."""
    return float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan")


def precision_at_recall(y, p, target_recall=0.5):
    """Precision at the operating point achieving >= ``target_recall`` (on the
    given test set's prevalence). Reports an actually-usable threshold metric."""
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    pos, neg = int(np.sum(y)), int(len(y) - np.sum(y))
    ok = tpr >= target_recall
    if not ok.any():
        return float("nan")
    i = np.argmax(ok)  # first threshold reaching the target recall
    tp, fp = tpr[i] * pos, fpr[i] * neg
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")


def precision_at_prevalence(y, p, target_recall=0.5, prevalence=1e-3):
    """Deployment-precision estimate: precision at ``target_recall`` re-weighted to
    a realistic genomic ``prevalence`` (binding sites are rare, ~0.1%). Uses
    precision = TPR*pi / (TPR*pi + FPR*(1-pi)) from the ROC operating point, so a
    high balanced-set AUROC does not hide poor real-world precision (the standard
    critique of AUROC/auPRC on balanced sets)."""
    if len(set(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    ok = tpr >= target_recall
    if not ok.any():
        return float("nan")
    i = np.argmax(ok)
    pi = prevalence
    num = tpr[i] * pi
    den = num + fpr[i] * (1 - pi)
    return float(num / den) if den > 0 else float("nan")


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
