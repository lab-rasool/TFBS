"""Ablation studies justifying the MoE design choices (Reviewer 2, Comment 7).

Three sweeps, each trained from scratch with a fixed seed and evaluated with the
canonical rigorous protocol (real 50/50 ``_B`` sets, inference dropout off, B=1000
percentile bootstrap):

  1. **Embedding size** E in {16, 32, 64, 128} -- retrain the 3 experts + MoE at each
     size; report in-distribution and OOD mean AUC.
  2. **Frozen vs. fine-tuned experts** -- the canonical MoE freezes the experts; the
     fine-tuned variant adds the expert parameters to the MoE optimizer and recomputes
     embeddings each step. Report AUC *and* training cost (wall-clock, trainable params).
  3. **Number of experts** N_e in {1, 2, 3} -- train the MoE on every subset of the
     experts of that size; report mean OOD AUC.

Outputs go to ``results/ablation/``.

Usage::

    python ablation.py                       # all three sweeps
    python ablation.py --sweeps embedding     # a subset
"""

import argparse
import itertools
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tfbs.data import ChipDataLoader, chipseq_dataset
from tfbs.metrics import paired_bootstrap
from tfbs.models import ConvNet, MixtureOfExperts
from tfbs.utils import EarlyStopping, get_tf_name, load_files_from_folder, set_seed

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
OUT_DIR = "./results/ablation"
TRAIN_DIR, TEST_DIR, OOD_DIR = "./data/train", "./data/test", "./data/ood"


def expert_config(hp, embedding_dim):
    return {"poolType": hp["poolType"], "sigmaConv": hp["sigmaConv"],
            "dropprob": hp["dropprob"], "learning_rate": hp["learning_rate"],
            "momentum_rate": hp["momentum_rate"], "embedding_dim": embedding_dim}


def loaders_for(train_file, seed, batch_size=96):
    data = ChipDataLoader(train_file).load_data(shuffle=True)
    tr, va = train_test_split(data, test_size=0.2, random_state=seed)
    return (DataLoader(chipseq_dataset(tr), batch_size=batch_size, shuffle=True),
            DataLoader(chipseq_dataset(va), batch_size=batch_size, shuffle=False))


def train_expert(config, train_loader, valid_loader, seed, max_epochs=80, patience=5):
    set_seed(seed)
    model = ConvNet(config).to(device)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=config["learning_rate"], momentum=config["momentum_rate"], nesterov=True)
    es = EarlyStopping(patience=patience)
    best_auc, best_state = -1, None
    for _ in range(max_epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x), y.float())
            loss.backward(); opt.step()
        model.eval()
        ps, ts = [], []
        with torch.no_grad():
            for x, y in valid_loader:
                ps.append(torch.sigmoid(model(x.to(device), training=False)).cpu().numpy().ravel())
                ts.append(y.numpy().ravel())
        auc = roc_auc_score(np.concatenate(ts), np.concatenate(ps))
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
        if es(auc):
            break
    model.load_state_dict(best_state)
    return model


def _emb(experts, x, training=False):
    return torch.cat([e(x, training=training, return_embedding=True) for e in experts], dim=1)


def train_moe(experts, train_loaders, valid_loaders, embedding_dim, seed,
              finetune=False, max_epochs=150, patience=10, lr=0.01, batch_size=256):
    set_seed(seed)
    moe = MixtureOfExperts(num_experts=len(experts), embedding_size=embedding_dim,
                           hidden_dim=embedding_dim).to(device)
    params = list(moe.parameters())
    if finetune:
        for e in experts:
            e.train()
            params += [p for p in e.parameters() if p.requires_grad]
    else:
        for e in experts:
            e.eval()
    opt = torch.optim.SGD(params, lr=lr, momentum=0.98, nesterov=True)
    es = EarlyStopping(patience=patience)

    # Precompute embeddings once for the frozen case (fast); recompute per step if fine-tuning.
    if not finetune:
        tr_emb, tr_y = _collect_embeddings(experts, train_loaders)
        va_emb, va_y = _collect_embeddings(experts, valid_loaders)
        n = tr_emb.shape[0]

    best_auc, best_state = -1, None
    for _ in range(max_epochs):
        moe.train()
        if finetune:
            for loader in train_loaders:
                for x, y in loader:
                    x, y = x.to(device), y.to(device)
                    opt.zero_grad()
                    out = moe(_emb(experts, x, training=True))
                    loss = F.binary_cross_entropy_with_logits(out, y.float())
                    loss.backward(); opt.step()
        else:
            perm = torch.randperm(n, device=device)
            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(moe(tr_emb[idx]), tr_y[idx].float())
                loss.backward(); opt.step()
        # validation
        moe.eval()
        with torch.no_grad():
            if finetune:
                for e in experts:
                    e.eval()
                va_emb2, va_y2 = _collect_embeddings(experts, valid_loaders)
                vp = torch.sigmoid(moe(va_emb2)).cpu().numpy().ravel(); vy = va_y2.cpu().numpy().ravel()
                for e in experts:
                    e.train()
            else:
                vp = torch.sigmoid(moe(va_emb)).cpu().numpy().ravel(); vy = va_y.cpu().numpy().ravel()
        auc = roc_auc_score(vy, vp)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.clone() for k, v in moe.state_dict().items()}
        if es(auc):
            break
    moe.load_state_dict(best_state)
    for e in experts:
        e.eval()
    return moe


def _collect_embeddings(experts, loaders):
    embs, ys = [], []
    with torch.no_grad():
        for loader in loaders:
            for x, y in loader:
                embs.append(_emb(experts, x.to(device), training=False))
                ys.append(y.to(device))
    return torch.cat(embs), torch.cat(ys)


def rigorous_eval(experts, moe, test_files, ood_files, n_boot=1000, base_seed=42):
    """Return {dataset_tf: moe_mean_auc} for the MoE on the real 50/50 sets."""
    out = {}
    datasets = [("in", p) for p in test_files] + [("ood", p) for p in ood_files]
    for di, (kind, path) in enumerate(datasets):
        data = ChipDataLoader(path).load_data(shuffle=False)
        loader = DataLoader(chipseq_dataset(data), batch_size=len(data), shuffle=False)
        x, y = next(iter(loader))
        x = x.to(device); y = y.numpy().ravel().astype(int)
        with torch.no_grad():
            preds = torch.sigmoid(moe(_emb(experts, x, training=False))).cpu().numpy().ravel()
        boot = paired_bootstrap({"moe": preds}, y, base_seed + di, n_boot)["moe"]
        out[(kind, get_tf_name(path))] = float(boot.mean())
    return out


def _summarise(evald):
    indist = np.mean([v for (k, _), v in evald.items() if k == "in"])
    ood = np.mean([v for (k, _), v in evald.items() if k == "ood"])
    return indist, ood


def load_hps(train_files, save_path="./models"):
    return [torch.load(f"{save_path}/hyperparams/{get_tf_name(f)}.pth") for f in train_files]


# ----------------------------------------------------------------------------
def sweep_embedding(train_files, test_files, ood_files, hps, seed, n_boot):
    rows = []
    for E in [16, 32, 64, 128]:
        experts = []
        for i, tf in enumerate(train_files):
            tl, vl = loaders_for(tf, seed)
            experts.append(train_expert(expert_config(hps[i], E), tl, vl, seed=seed + i))
        tr_loaders = [loaders_for(tf, seed)[0] for tf in train_files]
        va_loaders = [loaders_for(tf, seed)[1] for tf in train_files]
        moe = train_moe(experts, tr_loaders, va_loaders, embedding_dim=E, seed=seed)
        indist, ood = _summarise(rigorous_eval(experts, moe, test_files, ood_files, n_boot, seed))
        rows.append({"embedding_dim": E, "in_distribution_meanAUC": indist, "ood_meanAUC": ood})
        print(f"[embedding] E={E:>3}  in-dist={indist:.4f}  OOD={ood:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "ablation_embedding_size.csv"), index=False)
    return df


def sweep_frozen_vs_finetuned(train_files, test_files, ood_files, hps, seed, n_boot, E=32):
    rows = []
    for finetune in [False, True]:
        experts = []
        for i, tf in enumerate(train_files):
            tl, vl = loaders_for(tf, seed)
            experts.append(train_expert(expert_config(hps[i], E), tl, vl, seed=seed + i))
        tr_loaders = [loaders_for(tf, seed)[0] for tf in train_files]
        va_loaders = [loaders_for(tf, seed)[1] for tf in train_files]
        t0 = time.time()
        moe = train_moe(experts, tr_loaders, va_loaders, embedding_dim=E, seed=seed, finetune=finetune)
        cost = time.time() - t0
        trainable = sum(p.numel() for p in moe.parameters())
        if finetune:
            trainable += sum(p.numel() for e in experts for p in e.parameters())
        indist, ood = _summarise(rigorous_eval(experts, moe, test_files, ood_files, n_boot, seed))
        rows.append({"experts": "fine-tuned" if finetune else "frozen",
                     "in_distribution_meanAUC": indist, "ood_meanAUC": ood,
                     "moe_train_seconds": round(cost, 1), "trainable_params": trainable})
        print(f"[frozen/finetune] {'fine-tuned' if finetune else 'frozen':<10} "
              f"in-dist={indist:.4f} OOD={ood:.4f} cost={cost:.1f}s params={trainable}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "ablation_frozen_vs_finetuned.csv"), index=False)
    return df


def sweep_num_experts(train_files, test_files, ood_files, hps, seed, n_boot, E=32):
    # Train the 3 experts once, then build MoEs on every subset of each size.
    experts_all, tfs = [], [get_tf_name(f) for f in train_files]
    for i, tf in enumerate(train_files):
        tl, vl = loaders_for(tf, seed)
        experts_all.append(train_expert(expert_config(hps[i], E), tl, vl, seed=seed + i))
    rows = []
    for k in [1, 2, 3]:
        oods = []
        for combo in itertools.combinations(range(3), k):
            experts = [experts_all[j] for j in combo]
            tl = [loaders_for(train_files[j], seed)[0] for j in combo]
            vl = [loaders_for(train_files[j], seed)[1] for j in combo]
            moe = train_moe(experts, tl, vl, embedding_dim=E, seed=seed)
            _, ood = _summarise(rigorous_eval(experts, moe, test_files, ood_files, n_boot, seed))
            oods.append(ood)
            print(f"[num_experts] N={k} experts={'+'.join(tfs[j] for j in combo)} OOD={ood:.4f}")
        rows.append({"num_experts": k, "ood_meanAUC_mean": float(np.mean(oods)),
                     "ood_meanAUC_best": float(np.max(oods))})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "ablation_num_experts.csv"), index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--sweeps", nargs="+",
                    default=["embedding", "frozen", "num_experts"],
                    choices=["embedding", "frozen", "num_experts"])
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    train_files = load_files_from_folder(TRAIN_DIR)
    test_files = load_files_from_folder(TEST_DIR)
    ood_files = load_files_from_folder(OOD_DIR)
    hps = load_hps(train_files)

    if "embedding" in args.sweeps:
        print("\n=== Sweep 1: embedding size ===")
        print(sweep_embedding(train_files, test_files, ood_files, hps, args.seed, args.n_boot).to_string(index=False))
    if "frozen" in args.sweeps:
        print("\n=== Sweep 2: frozen vs fine-tuned ===")
        print(sweep_frozen_vs_finetuned(train_files, test_files, ood_files, hps, args.seed, args.n_boot).to_string(index=False))
    if "num_experts" in args.sweeps:
        print("\n=== Sweep 3: number of experts ===")
        print(sweep_num_experts(train_files, test_files, ood_files, hps, args.seed, args.n_boot).to_string(index=False))
    print(f"\nWrote ablation tables to {OUT_DIR}/")


if __name__ == "__main__":
    main()
