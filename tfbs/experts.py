"""Heterogeneous expert zoo: dataset assembly, frozen feature extractors,
per-TF linear probes, and the build/cache/load/subset of the zoo. Extracted
verbatim from the former hetmoe.py god-module.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from tfbs.constants import TRAIN_TFS, OOD_TFS
from tfbs.data import ChipDataLoader, revcomp
from tfbs.models import ConvNet, FeatureProbeExpert
from tfbs.utils import get_tf_name, load_files_from_folder, set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "./results/cache"


def _cdir(seed):
    """Per-seed cache subdirectory so multi-seed robustness runs don't clobber."""
    return os.path.join(CACHE_DIR, f"seed{seed}")


# ---------------------------------------------------------------------------
# File discovery (deterministic by TF name)
# ---------------------------------------------------------------------------
def _by_tf(folder):
    files = {get_tf_name(f): f for f in load_files_from_folder(folder)}
    return files


def train_files():
    f = _by_tf("./data/train")
    return [f[tf] for tf in TRAIN_TFS]


def eval_files():
    """(data_type, tf, path) for the 3 in-distribution + 6 OOD curated sets."""
    test = _by_tf("./data/test")
    ood = _by_tf("./data/ood")
    out = [("in_distribution", tf, test[tf]) for tf in TRAIN_TFS]
    out += [("out_of_distribution", tf, ood[tf]) for tf in OOD_TFS]
    return out


# ---------------------------------------------------------------------------
# Canonical aligned datasets: (one-hot 4x147, label, raw sequence)
# ---------------------------------------------------------------------------
def build_dataset(path, is_train, seed):
    """Return (X[n,4,147] float32 tensor, y[n] int, seqs[list[str]]).

    is_train -> _AC file with one dinucleotide-shuffle negative synthesised per row
    (seeded, generated ONCE so every expert embeds identical sequences).
    eval     -> _B curated 500/500 real-negative set (shuffle=False)."""
    set_seed(seed)
    rows = ChipDataLoader(path).load_data_with_seq(shuffle=is_train)
    X = torch.from_numpy(np.stack([r[0] for r in rows]).astype(np.float32))
    y = np.array([r[1][0] for r in rows], dtype=int)
    seqs = [r[2] for r in rows]
    return X, y, seqs


# ---------------------------------------------------------------------------
# Foundation-model feature extractor (frozen, pluggable)
# ---------------------------------------------------------------------------
_FM_REGISTRY = {
    "dnabert2": "zhihan1996/DNABERT-2-117M",
    "hyenadna": "LongSafari/hyenadna-tiny-1k-seqlen-hf",
    "nt": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
}


class FMExtractor:
    """Frozen genomic-FM mean-pooled sequence embeddings. Loaded lazily so a run that
    only uses CNN experts never pays the import/download cost."""

    def __init__(self, fm_name="dnabert2", batch_size=64, max_len=128):
        self.fm_name = fm_name
        self.repo = _FM_REGISTRY[fm_name]
        self.batch_size = batch_size
        self.max_len = max_len
        self.tok = None
        self.model = None
        self.dim = None

    def _load(self):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(self.repo, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.repo, trust_remote_code=True).to(device)
        self.model.eval()

    @torch.no_grad()
    def features(self, seqs):
        if self.model is None:
            self._load()
        out = []
        for i in range(0, len(seqs), self.batch_size):
            chunk = seqs[i : i + self.batch_size]
            enc = self.tok(chunk, return_tensors="pt", padding="max_length",
                           truncation=True, max_length=self.max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            res = self.model(**enc)
            hidden = res[0] if isinstance(res, (tuple, list)) else res.last_hidden_state
            mask = enc.get("attention_mask")
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                pooled = hidden.mean(1)
            out.append(pooled.float().cpu().numpy())
        feat = np.concatenate(out, axis=0)
        self.dim = feat.shape[1]
        return feat


# ---------------------------------------------------------------------------
# Per-backbone feature producers -> D-dim features for an arbitrary dataset's X/seqs
# ---------------------------------------------------------------------------
@torch.no_grad()
def _cnn_features(model, X):
    """Penultimate features of a trained DeepSEA/DanQ (return_embedding=True)."""
    model.eval()
    out = []
    for i in range(0, len(X), 512):
        xb = X[i : i + 512].to(device)
        out.append(model(xb, return_embedding=True).float().cpu().numpy())
    return np.concatenate(out, axis=0)


@torch.no_grad()
def _convnet_embed(expert, X):
    """ConvNet already emits a 32-d LayerNorm'd embedding + a scalar logit."""
    expert.eval()
    embs, preds = [], []
    for i in range(0, len(X), 512):
        xb = X[i : i + 512].to(device)
        embs.append(expert(xb, training=False, return_embedding=True).float().cpu().numpy())
        preds.append(torch.sigmoid(expert(xb, training=False)).float().cpu().numpy().ravel())
    return np.concatenate(embs, 0), np.concatenate(preds, 0)


# ---------------------------------------------------------------------------
# Probe head training (frozen-feature linear probe -> E-dim embedding + logit)
# ---------------------------------------------------------------------------
def train_probe(feat, y, embedding_dim, seed, epochs=200, patience=15, lr=1e-3):
    """Train a FeatureProbeExpert on cached D-dim features (the expert's own TF)."""
    set_seed(seed)
    in_dim = feat.shape[1]
    Xtr, Xva, ytr, yva = train_test_split(feat, y, test_size=0.2, random_state=seed,
                                          stratify=y if len(set(y)) > 1 else None)
    probe = FeatureProbeExpert(in_dim, embedding_dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    Xtr_t = torch.from_numpy(Xtr).float().to(device)
    ytr_t = torch.from_numpy(ytr.astype(np.float32)).to(device).unsqueeze(1)
    Xva_t = torch.from_numpy(Xva).float().to(device)
    best_auc, best_state, bad = -1.0, None, 0
    n = len(Xtr_t)
    bs = 256
    for ep in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(probe(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        probe.eval()
        with torch.no_grad():
            va = torch.sigmoid(probe(Xva_t)).cpu().numpy().ravel()
        auc = roc_auc_score(yva, va) if len(set(yva)) > 1 else 0.5
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in probe.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()
    return probe, best_auc


@torch.no_grad()
def _probe_embed(probe, feat):
    probe.eval()
    out_e, out_p = [], []
    for i in range(0, len(feat), 2048):
        fb = torch.from_numpy(feat[i : i + 2048]).float().to(device)
        out_e.append(probe(fb, return_embedding=True).float().cpu().numpy())
        out_p.append(torch.sigmoid(probe(fb)).float().cpu().numpy().ravel())
    return np.concatenate(out_e, 0), np.concatenate(out_p, 0)


# ---------------------------------------------------------------------------
# Build the full zoo: per (backbone, TF) expert -> embeddings + preds on all datasets
# ---------------------------------------------------------------------------
class DNABERT6Features:
    """A DNABERT-6 (zhihan1996/DNA_bert_6) classifier FINE-TUNED per TF (the model
    behind the 0.749 OOD baseline), used as a strong embedding extractor: mean-pooled
    last hidden state (768-d). Fine-tuning is what realises the per-TF OOD headroom the
    oracle-ceiling analysis assumes; a frozen base DNABERT-6 is far weaker."""

    def __init__(self, train_file, seed, epochs=3):
        from tfbs.baselines import _kmers, train_dnabert
        (self.model, self.tok), self.val_auc = train_dnabert(train_file, seed, epochs=epochs)
        self.model.eval()
        self._kmers = _kmers

    @torch.no_grad()
    def features(self, seqs, max_len=110, bs=128):
        out = []
        for i in range(0, len(seqs), bs):
            chunk = [self._kmers(s) for s in seqs[i : i + bs]]
            enc = self.tok(chunk, truncation=True, padding="max_length",
                           max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            hid = self.model.bert(input_ids=enc["input_ids"],
                                  attention_mask=enc["attention_mask"]).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hid * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def predict(self, seqs, max_len=110, bs=128):
        """Fine-tuned DNABERT-6 classifier P(bound) -- the per-TF DNABERT baseline whose
        3-TF mean is the published 0.749 OOD bar (matches baselines.dnabert_predict)."""
        out = []
        for i in range(0, len(seqs), bs):
            chunk = [self._kmers(s) for s in seqs[i : i + bs]]
            enc = self.tok(chunk, truncation=True, padding="max_length",
                           max_length=max_len, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logit = self.model(input_ids=enc["input_ids"],
                               attention_mask=enc["attention_mask"]).logits
            out.append(torch.softmax(logit, 1)[:, 1].float().cpu().numpy())
        return np.concatenate(out, axis=0)


def build_zoo(fm_name="dnabert2", embedding_dim=32, seed=42, backbones=None,
              save_cache=True, save_models=True, verbose=True, dnabert_epochs=3):
    """Returns:
      experts: ordered list of dicts {name, backbone, tf, val_auc}
      emb[dataset_key] : (n, num_experts*E) concatenated embeddings (expert order)
      pred[dataset_key]: (num_experts, n) per-expert scalar predictions
      y[dataset_key]   : (n,) labels
      meta             : dataset_key -> (data_type, tf)
    dataset_key is 'train::<TF>' for the 3 _AC sets and '<data_type>::<TF>' for eval.
    """
    if backbones is None:
        # DNABERT6 = the fine-tuned model behind the 0.749 OOD baseline (the strong
        # member that realises the oracle headroom); loads reliably on this stack
        # (DNABERT-2 does not). Swap in a frozen FM name (hyenadna/dnabert2/nt) to add it.
        backbones = ["ConvNet", "DeepSEA", "DanQ", "DNABERT6"]
    cdir = _cdir(seed)
    os.makedirs(cdir, exist_ok=True)
    if save_models:
        os.makedirs(f"./models/zoo/seed{seed}", exist_ok=True)

    tr_files = {tf: f for tf, f in zip(TRAIN_TFS, train_files())}

    # ---- assemble every dataset we need embeddings on ----
    datasets = {}  # key -> (X, y, seqs)
    for tf in TRAIN_TFS:
        datasets[f"train::{tf}"] = build_dataset(tr_files[tf], is_train=True, seed=seed)
    for data_type, tf, path in eval_files():
        datasets[f"{data_type}::{tf}"] = build_dataset(path, is_train=False, seed=seed)
    meta = {f"train::{tf}": ("train", tf) for tf in TRAIN_TFS}
    for data_type, tf, _ in eval_files():
        meta[f"{data_type}::{tf}"] = (data_type, tf)

    keys = list(datasets.keys())
    y = {k: datasets[k][1] for k in keys}

    # per-expert outputs
    experts = []
    per_expert_emb = {}   # expert_name -> {key: (n,E)}
    per_expert_pred = {}  # expert_name -> {key: (n,)}

    # A frozen pretrained FM (if present in the backbone list) produces TF-independent
    # features, so extract them ONCE for all datasets and reuse across the 3 probes.
    frozen_fm_feats = None
    frozen_fm_name = next((b for b in backbones if b in _FM_REGISTRY), None)
    if frozen_fm_name is not None:
        fm = FMExtractor(frozen_fm_name)
        if verbose:
            print(f"[zoo] extracting frozen {frozen_fm_name} features (shared)", flush=True)
        frozen_fm_feats = {k: fm.features(datasets[k][2]) for k in keys}

    for backbone in backbones:
        for tf in TRAIN_TFS:
            name = f"{backbone}::{tf}"
            if verbose:
                print(f"[zoo] building expert {name}", flush=True)

            if backbone == "ConvNet":
                cfg = torch.load(f"./models/hyperparams/{tf}.pth")  # trusted local checkpoint
                expert = ConvNet(cfg).to(device)
                expert.load_state_dict(torch.load(f"./models/experts/{tf}.pth", map_location=device))
                expert.eval()
                emb, pred = {}, {}
                for k in keys:
                    e, p = _convnet_embed(expert, datasets[k][0])
                    emb[k], pred[k] = e, p
                val_auc = float("nan")

            elif backbone in ("DeepSEA", "DanQ"):
                from tfbs.baselines import DanQ, DeepSEA, train_cnn
                Model = DeepSEA if backbone == "DeepSEA" else DanQ
                model, val_auc = train_cnn(Model(), tr_files[tf], seed=seed)
                if save_models:
                    torch.save(model.state_dict(), f"./models/zoo/seed{seed}/{backbone}_{tf}.pth")
                feats = {k: _cnn_features(model, datasets[k][0]) for k in keys}
                probe, _ = train_probe(feats[f"train::{tf}"], y[f"train::{tf}"], embedding_dim, seed)
                emb, pred = {}, {}
                for k in keys:
                    emb[k], pred[k] = _probe_embed(probe, feats[k])

            elif backbone == "DNABERT6":
                ext = DNABERT6Features(tr_files[tf], seed, epochs=dnabert_epochs)
                val_auc = ext.val_auc
                feats = {k: ext.features(datasets[k][2]) for k in keys}
                probe, _ = train_probe(feats[f"train::{tf}"], y[f"train::{tf}"], embedding_dim, seed)
                emb, pred = {}, {}
                for k in keys:
                    emb[k], pred[k] = _probe_embed(probe, feats[k])
                if save_cache:
                    # also cache the fine-tuned DNABERT-6 classifier preds: their 3-TF
                    # mean is the published 0.749 OOD baseline, used for the Phase-C
                    # paired bootstrap of HetMoE vs DNABERT.
                    for k in keys:
                        np.savez(os.path.join(cdir, f"dnabert6base_{tf}__{k.replace('::','_')}.npz"),
                                 pred=ext.predict(datasets[k][2]), y=y[k])
                del ext  # free the fine-tuned BERT before the next TF
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            else:  # frozen foundation model (shared features) + per-TF probe
                probe, val_auc = train_probe(frozen_fm_feats[f"train::{tf}"],
                                             y[f"train::{tf}"], embedding_dim, seed)
                emb, pred = {}, {}
                for k in keys:
                    emb[k], pred[k] = _probe_embed(probe, frozen_fm_feats[k])

            per_expert_emb[name] = emb
            per_expert_pred[name] = pred
            experts.append({"name": name, "backbone": backbone, "tf": tf,
                            "val_auc": float(val_auc) if val_auc == val_auc else None})
            if save_cache:
                for k in keys:
                    np.savez(os.path.join(cdir, f"{name.replace('::','_')}__{k.replace('::','_')}.npz"),
                             emb=emb[k], pred=pred[k], y=y[k])

    expert_order = [e["name"] for e in experts]
    emb_cat = {k: np.concatenate([per_expert_emb[n][k] for n in expert_order], axis=1)
               for k in keys}
    pred_stack = {k: np.stack([per_expert_pred[n][k] for n in expert_order], axis=0)
                  for k in keys}
    if save_cache:
        with open(os.path.join(cdir, "manifest.json"), "w") as f:
            json.dump({"expert_order": expert_order, "embedding_dim": embedding_dim,
                       "experts": experts, "keys": keys, "seed": seed}, f, indent=2)
    return {"experts": experts, "expert_order": expert_order, "emb": emb_cat,
            "pred": pred_stack, "y": y, "meta": meta, "embedding_dim": embedding_dim}


def load_zoo_cache(seed=42):
    """Reconstruct the build_zoo() return dict from results/cache/ (Phase B: train the
    gate over cached embeddings with NO GPU / no expert forward). Lets the cheap
    config sweep reuse the one expensive Phase-A expert build."""
    cdir = _cdir(seed)
    with open(os.path.join(cdir, "manifest.json")) as f:
        man = json.load(f)
    order, keys = man["expert_order"], man["keys"]
    per_emb, per_pred, Y = {n: {} for n in order}, {n: {} for n in order}, {}
    for n in order:
        for k in keys:
            d = np.load(os.path.join(cdir, f"{n.replace('::','_')}__{k.replace('::','_')}.npz"))
            per_emb[n][k], per_pred[n][k] = d["emb"], d["pred"]
            Y[k] = d["y"]
    meta = {}
    for k in keys:
        dtype, tf = k.split("::")
        meta[k] = ("train" if dtype == "train" else dtype, tf)
    emb_cat = {k: np.concatenate([per_emb[n][k] for n in order], axis=1) for k in keys}
    pred_stack = {k: np.stack([per_pred[n][k] for n in order], axis=0) for k in keys}
    return {"experts": man["experts"], "expert_order": order, "emb": emb_cat,
            "pred": pred_stack, "y": Y, "meta": meta, "embedding_dim": man["embedding_dim"]}


# ---------------------------------------------------------------------------
# Gate training on cached concatenated embeddings (unchanged MoE class)
# ---------------------------------------------------------------------------
def subset_zoo(zoo, backbones=None, group_by_backbone=False):
    """Return a new zoo with a subset of experts (for the N_e sweep) and/or a 4-way
    GROUPED gate that mean-pools each backbone family's per-TF experts into one E-dim
    block (the more-defensible-OOD variant). Operates purely on cached arrays."""
    order = zoo["expert_order"]
    E = zoo["embedding_dim"]
    keys = list(zoo["y"].keys())
    idx = {n: i for i, n in enumerate(order)}
    keep = [n for n in order if backbones is None or n.split("::")[0] in backbones]

    def block(k, n):
        i = idx[n]
        return zoo["emb"][k][:, i * E:(i + 1) * E]

    if group_by_backbone:
        from collections import OrderedDict
        groups = OrderedDict()
        for n in keep:
            groups.setdefault(n.split("::")[0], []).append(n)
        new_order = list(groups.keys())
        emb, pred = {}, {}
        for k in keys:
            blocks, preds = [], []
            for members in groups.values():
                blocks.append(np.mean([block(k, m) for m in members], axis=0))
                preds.append(zoo["pred"][k][[idx[m] for m in members]].mean(axis=0))
            emb[k] = np.concatenate(blocks, axis=1)
            pred[k] = np.stack(preds, axis=0)
        return {**zoo, "expert_order": new_order, "emb": emb, "pred": pred}

    keep_idx = [idx[n] for n in keep]
    emb = {k: np.concatenate([block(k, n) for n in keep], axis=1) for k in keys}
    pred = {k: zoo["pred"][k][keep_idx] for k in keys}
    return {**zoo, "expert_order": keep, "emb": emb, "pred": pred}


def _load_dnabert_baseline(keys, y, cdir):
    """3-TF mean of the fine-tuned DNABERT-6 classifier preds per dataset = the 0.749
    OOD baseline (returns None if a full-zoo cache with DNABERT6 wasn't built)."""
    base = {}
    for k in keys:
        ksan = k.replace("::", "_")
        per_tf = []
        for tf in TRAIN_TFS:
            fp = os.path.join(cdir, f"dnabert6base_{tf}__{ksan}.npz")
            if not os.path.exists(fp):
                return None
            per_tf.append(np.load(fp)["pred"])
        base[k] = np.mean(per_tf, axis=0)
    return base
