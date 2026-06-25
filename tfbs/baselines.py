"""State-of-the-art TFBS baselines for comparison (Reviewer 2, Comment 2).

Trains DeepSEA, DanQ, and DNABERT on the three in-distribution ``_AC`` files (via
``ChipDataLoader`` with dinucleotide-shuffle negatives) and evaluates them with the
*identical rigorous protocol* used for the experts/MoE: the real 50/50 ``_B`` test
and OOD sets, deterministic inference (dropout off / eval mode), and a B=1000
percentile bootstrap for 95% CIs.  The resulting table is merged with the MoE/expert
numbers already in ``results/evaluation_summary.csv`` so every model is reported on
the same datasets, splits, and metric.

BPNet is intentionally *not* given an AUC row: it predicts base-resolution binding
profiles (a regression task), not a single binary call, so it is discussed as a
methodological caveat rather than forced into this binary-classification comparison.

Usage::

    python baselines.py                 # train + eval DeepSEA, DanQ, DNABERT
    python baselines.py --skip_dnabert  # CNN baselines only (fast)
    python baselines.py --dnabert_epochs 3
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tfbs.data import ChipDataLoader, chipseq_dataset
from tfbs.utils import get_tf_name, load_files_from_folder, set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = "./results/baselines"
IN_TFS = ["ARID3A", "FOXM1", "GATA3"]
OOD_TFS = ["BCLAF1", "CTCF", "POLR2A", "RBBP5", "SAP30", "STAT3"]


# ----------------------------------------------------------------------------
# Baseline architectures (adapted to the 4 x 147 one-hot input)
# ----------------------------------------------------------------------------
class DeepSEA(nn.Module):
    """DeepSEA-style 3-layer CNN classifier (Zhou & Troyanskaya, 2015).

    Adapted to the short 147 bp input with BatchNorm (training stability) and an
    adaptive max-pool so the classifier head stays small.
    """

    def __init__(self, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4, 320, 8), nn.BatchNorm1d(320), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(dropout),
            nn.Conv1d(320, 480, 8), nn.BatchNorm1d(480), nn.ReLU(), nn.MaxPool1d(4), nn.Dropout(dropout),
            nn.Conv1d(480, 960, 4), nn.BatchNorm1d(960), nn.ReLU(), nn.Dropout(0.5),
            nn.AdaptiveMaxPool1d(1),
        )
        self.head = nn.Sequential(nn.Linear(960, 925), nn.ReLU(), nn.Dropout(0.5), nn.Linear(925, 1))
        self.embedding_dim = 925  # head penultimate width (input to the final Linear)

    def forward(self, x, return_embedding=False):
        feat = self.conv(x).flatten(1)
        if return_embedding:
            # penultimate of the classifier head (ReLU(Linear(960,925))); dropout is
            # identity in eval, so this is the deterministic 925-d expert feature.
            return self.head[1](self.head[0](feat))
        return self.head(feat)


class DanQ(nn.Module):
    """DanQ CNN-BiLSTM classifier (Quang & Xie, 2016)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(4, 320, 13)
        self.pool = nn.MaxPool1d(4)
        self.drop1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(320, 160, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.5)
        self.head = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.Linear(256, 1))
        self.embedding_dim = 256  # head penultimate width (input to the final Linear)

    def forward(self, x, return_embedding=False):
        x = self.drop1(self.pool(F.relu(self.conv(x))))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        flat = self.drop2(x).flatten(1)
        if return_embedding:
            # penultimate of the classifier head (ReLU(LazyLinear(256))); the lazy
            # layer must already be materialised (train_cnn does one forward first).
            return self.head[1](self.head[0](flat))
        return self.head(flat)


# ----------------------------------------------------------------------------
# Training / evaluation for the CNN baselines
# ----------------------------------------------------------------------------
def train_cnn(model, train_file, seed, batch_size=96, max_epochs=60, patience=6, lr=1e-3):
    set_seed(seed)
    data = ChipDataLoader(train_file).load_data(shuffle=True)
    tr, va = train_test_split(data, test_size=0.2, random_state=seed)
    tl = DataLoader(chipseq_dataset(tr), batch_size=batch_size, shuffle=True)
    vl = DataLoader(chipseq_dataset(va), batch_size=batch_size, shuffle=False)
    model = model.to(device)
    # materialise LazyLinear before building the optimizer
    xb, _ = next(iter(tl))
    model(xb.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_auc, best_state, bad = -1, None, 0
    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device).float()
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for xb, yb in vl:
                ps.append(torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel())
                ts.append(yb.numpy().ravel())
        auc = roc_auc_score(np.concatenate(ts), np.concatenate(ps))
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_auc


def cnn_predict(model, path):
    data = ChipDataLoader(path).load_data(shuffle=False)  # real 50/50
    loader = DataLoader(chipseq_dataset(data), batch_size=len(data), shuffle=False)
    xb, yb = next(iter(loader))
    model.eval()
    with torch.no_grad():
        preds = torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel()
    return yb.numpy().ravel().astype(int), preds


# ----------------------------------------------------------------------------
# DNABERT (fine-tune zhihan1996/DNA_bert_6)
# ----------------------------------------------------------------------------
def _kmers(seq, k=6):
    return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))


def _read_seqs(path, with_negs):
    """Return (sequences, labels). with_negs adds a dinuc-shuffle negative per row."""
    cdl = ChipDataLoader(path)
    seqs, labels = [], []
    import gzip, csv
    with gzip.open(path, "rt") as fh:
        next(fh)
        for row in csv.reader(fh, delimiter="\t"):
            s, lab = row[2], int(row[3])
            if with_negs:
                seqs.append(cdl.dinucshuffle(s)); labels.append(0)
            seqs.append(s); labels.append(lab)
    return seqs, labels


def train_dnabert(train_file, seed, epochs=3, batch_size=32, lr=2e-5, max_len=110):
    from transformers import AutoTokenizer, BertForSequenceClassification

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
    model = BertForSequenceClassification.from_pretrained(
        "zhihan1996/DNA_bert_6", num_labels=2
    ).to(device)
    seqs, labels = _read_seqs(train_file, with_negs=True)
    tr_s, va_s, tr_y, va_y = train_test_split(seqs, labels, test_size=0.2, random_state=seed)

    def encode(slist):
        return tok([_kmers(s) for s in slist], truncation=True, padding="max_length",
                   max_length=max_len, return_tensors="pt")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    tr_enc = encode(tr_s)
    tr_y_t = torch.tensor(tr_y)
    n = len(tr_y)
    best_auc, best_state = -1, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            ids = tr_enc["input_ids"][idx].to(device)
            am = tr_enc["attention_mask"][idx].to(device)
            yb = tr_y_t[idx].to(device)
            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=am, labels=yb)
            out.loss.backward()
            opt.step()
        # validation AUC
        model.eval()
        va_enc = encode(va_s)
        ps = []
        with torch.no_grad():
            for i in range(0, len(va_s), 128):
                ids = va_enc["input_ids"][i:i + 128].to(device)
                am = va_enc["attention_mask"][i:i + 128].to(device)
                logit = model(input_ids=ids, attention_mask=am).logits
                ps.append(torch.softmax(logit, 1)[:, 1].cpu().numpy())
        auc = roc_auc_score(va_y, np.concatenate(ps))
        print(f"  [DNABERT {get_tf_name(train_file)}] epoch {ep} val AUC {auc:.4f}")
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return (model, tok), best_auc


def dnabert_predict(bundle, path, max_len=110):
    model, tok = bundle
    seqs, labels = _read_seqs(path, with_negs=False)  # real 50/50
    enc = tok([_kmers(s) for s in seqs], truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    model.eval()
    ps = []
    with torch.no_grad():
        for i in range(0, len(seqs), 128):
            ids = enc["input_ids"][i:i + 128].to(device)
            am = enc["attention_mask"][i:i + 128].to(device)
            logit = model(input_ids=ids, attention_mask=am).logits
            ps.append(torch.softmax(logit, 1)[:, 1].cpu().numpy())
    return np.array(labels).astype(int), np.concatenate(ps)


# ----------------------------------------------------------------------------
# Bootstrap evaluation (matches evaluate.py --protocol rigorous)
# ----------------------------------------------------------------------------
def bootstrap_ci(y, preds, di, n_boot=1000, base_seed=42):
    n = len(y)
    rng = np.random.default_rng(base_seed + di)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            idx = rng.integers(0, n, n)
        aucs[b] = roc_auc_score(y[idx], preds[idx])
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y, preds)), float(aucs.mean()), float(lo), float(hi)


def evaluate_baseline(model_name, predict_fn, test_files, ood_files, n_boot, base_seed):
    rows = []
    datasets = [("in_distribution", p) for p in test_files] + \
               [("out_of_distribution", p) for p in ood_files]
    for di, (data_type, path) in enumerate(datasets):
        tf = get_tf_name(path)
        y, preds = predict_fn(path)
        point, mean, lo, hi = bootstrap_ci(y, preds, di, n_boot, base_seed)
        rows.append({"model": model_name, "data_type": data_type, "dataset_tf": tf,
                     "point_auc": point, "mean_auc": mean, "ci95_low": lo, "ci95_high": hi})
        print(f"  [{model_name}] {data_type[:3]}:{tf:<8} AUC={mean:.4f} CI[{lo:.4f},{hi:.4f}]")
    return rows


def existing_model_rows():
    """Mean + percentile CI for the MoE/experts from the canonical rigorous summary."""
    df = pd.read_csv("./results/evaluation_summary.csv")
    rep = [c for c in df.columns if c.startswith("boot_") and c.endswith("_auc")]
    lbl = {"moe": "MoE", "expert_ARID3A": "ARID3A-expert",
           "expert_FOXM1": "FOXM1-expert", "expert_GATA3": "GATA3-expert"}
    rows = []
    for _, r in df.iterrows():
        arr = r[rep].values.astype(float)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        rows.append({"model": lbl[r["model"]], "data_type": r["data_type"],
                     "dataset_tf": r["dataset_tf"], "point_auc": r["point_auc"],
                     "mean_auc": float(arr.mean()), "ci95_low": float(lo), "ci95_high": float(hi)})
    return rows


def to_latex(pivot, caption, label):
    cols = list(pivot.columns)
    out = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\hline",
           "Model & " + " & ".join(cols) + " \\\\", "\\hline"]
    for model, row in pivot.iterrows():
        out.append(model + " & " + " & ".join(f"{row[c]:.3f}" for c in cols) + " \\\\")
    out += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--skip_dnabert", action="store_true")
    ap.add_argument("--dnabert_epochs", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    test_files = load_files_from_folder("./data/test")
    ood_files = load_files_from_folder("./data/ood")
    train_files = load_files_from_folder("./data/train")

    all_rows = existing_model_rows()

    # ---- CNN baselines: one model per in-dist TF, ensembled by mean over TFs ----
    for Model, mname in [(DeepSEA, "DeepSEA"), (DanQ, "DanQ")]:
        print(f"\n=== {mname} ===")
        per_tf_models = []
        for tf_i, tf_file in enumerate(train_files):
            set_seed(args.seed + tf_i)
            m, va = train_cnn(Model(), tf_file, seed=args.seed + tf_i)
            print(f"  trained {mname} on {get_tf_name(tf_file)} (val AUC {va:.4f})")
            per_tf_models.append(m)

        def predict_ensemble(path, _models=per_tf_models):
            ys, all_preds = None, []
            for m in _models:
                y, p = cnn_predict(m, path)
                ys = y
                all_preds.append(p)
            return ys, np.mean(all_preds, axis=0)

        all_rows += evaluate_baseline(mname, predict_ensemble, test_files, ood_files,
                                      args.n_boot, args.seed)

    # ---- DNABERT: one fine-tuned model per in-dist TF, ensembled ----
    if not args.skip_dnabert:
        print("\n=== DNABERT ===")
        bundles = []
        for tf_i, tf_file in enumerate(train_files):
            bundle, va = train_dnabert(tf_file, seed=args.seed + tf_i, epochs=args.dnabert_epochs)
            print(f"  trained DNABERT on {get_tf_name(tf_file)} (val AUC {va:.4f})")
            bundles.append(bundle)

        def predict_dnabert_ensemble(path, _bundles=bundles):
            ys, all_preds = None, []
            for b in _bundles:
                y, p = dnabert_predict(b, path)
                ys = y
                all_preds.append(p)
            return ys, np.mean(all_preds, axis=0)

        all_rows += evaluate_baseline("DNABERT", predict_dnabert_ensemble, test_files,
                                      ood_files, args.n_boot, args.seed)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "baseline_comparison_long.csv"), index=False)

    # Pivot to a mean-AUC comparison table (rows=models, cols=datasets)
    order = ["MoE", "ARID3A-expert", "FOXM1-expert", "GATA3-expert", "DeepSEA", "DanQ"]
    if not args.skip_dnabert:
        order.append("DNABERT")
    for dtype, tfs, tag in [("in_distribution", IN_TFS, "indist"),
                            ("out_of_distribution", OOD_TFS, "ood")]:
        sub = df[df.data_type == dtype]
        pivot = sub.pivot_table(index="model", columns="dataset_tf", values="mean_auc")
        pivot = pivot.reindex([m for m in order if m in pivot.index])[tfs]
        pivot["mean"] = pivot.mean(axis=1)
        pivot.to_csv(os.path.join(OUT_DIR, f"baseline_comparison_{tag}.csv"))
        cap = (f"Mean AUC (B={args.n_boot} bootstrap) on the {dtype.replace('_', '-')} 50/50 "
               "held-out sets. All models share the identical evaluation protocol.")
        open(os.path.join(OUT_DIR, f"baseline_comparison_{tag}.tex"), "w").write(
            to_latex(pivot, cap, f"tab:baselines_{tag}"))
        print(f"\n=== {dtype} mean AUC ===")
        print(pivot.round(4).to_string())

    print(f"\nWrote baseline tables to {OUT_DIR}/")


if __name__ == "__main__":
    main()
