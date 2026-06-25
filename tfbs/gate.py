"""Gate training/prediction over cached concatenated expert embeddings. The
MoE class itself (tfbs.models.MixtureOfExperts) is used UNCHANGED.
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from tfbs.models import MixtureOfExperts
from tfbs.utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_gate(emb_tr, y_tr, emb_va, y_va, num_experts, embedding_dim, seed,
               l2norm=True, entropy_reg=1e-3, gate_temperature=1.0,
               hidden_dim=32, epochs=500, patience=20, lr=0.01):
    set_seed(seed)
    moe = MixtureOfExperts(num_experts, embedding_dim, hidden_dim,
                           l2norm=l2norm, entropy_reg=entropy_reg,
                           gate_temperature=gate_temperature).to(device)
    opt = torch.optim.SGD(moe.parameters(), lr=lr, momentum=0.98, nesterov=True)
    Xtr = torch.from_numpy(emb_tr).float().to(device)
    ytr = torch.from_numpy(y_tr.astype(np.float32)).to(device).unsqueeze(1)
    Xva = torch.from_numpy(emb_va).float().to(device)
    n, bs = len(Xtr), 256
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        moe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            logits, gw = moe(Xtr[idx], return_gate=True)
            loss = F.binary_cross_entropy_with_logits(logits, ytr[idx])
            if entropy_reg:
                # maximise gate entropy (anti-collapse) -> subtract from loss
                loss = loss - entropy_reg * MixtureOfExperts.gate_entropy(gw)
            loss.backward()
            opt.step()
        moe.eval()
        with torch.no_grad():
            va = torch.sigmoid(moe(Xva)).cpu().numpy().ravel()
        auc = roc_auc_score(y_va, va) if len(set(y_va)) > 1 else 0.5
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in moe.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        moe.load_state_dict(best_state)
    moe.eval()
    return moe, best_auc


@torch.no_grad()
def gate_predict(moe, emb):
    moe.eval()
    Xt = torch.from_numpy(emb).float().to(device)
    logits, gw = moe(Xt, return_gate=True)
    return torch.sigmoid(logits).cpu().numpy().ravel(), gw.cpu().numpy()
