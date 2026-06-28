"""Active model definitions for the TFBS Mixture-of-Experts pipeline.

Holds the three live modules: ``ConvNet`` (the DeepBIND-style expert),
``FeatureProbeExpert`` (linear probe over a frozen feature extractor, used by
the heterogeneous zoo), and ``MixtureOfExperts`` (the embedding-gated MoE).
DeepSEA/DanQ trunk definitions live in :mod:`tfbs.experts`.

NOTE (reproducibility): ``ConvNet.wRect`` (the conv bias) is a trained
``nn.Parameter`` -- it is saved in the checkpoint and optimised like any other
weight, so loading a checkpoint reproduces its predictions exactly. (It was
previously a non-saved random tensor, which made the ConvNet expert and the
gate built on it non-reproducible; see docs/reproduce.md.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bernoulli

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
        self.nummotif = 16
        self.motiflen = 24
        self.poolType = config["poolType"]
        self.sigmaConv = config["sigmaConv"]
        self.dropprob = config["dropprob"]
        self.learning_rate = config["learning_rate"]
        self.momentum_rate = config["momentum_rate"]
        # Embedding size (E). Defaults to 32 so existing checkpoints load
        # unchanged; the ablation study overrides it via config["embedding_dim"].
        self.embedding_dim = config.get("embedding_dim", 32)

        self.wConv = nn.Parameter(torch.randn(self.nummotif, 4, self.motiflen))
        torch.nn.init.normal_(self.wConv, mean=0, std=self.sigmaConv)
        # Conv bias: a trainable nn.Parameter so it is saved in the checkpoint and
        # optimised (was a non-saved random tensor -> the pipeline was not
        # reproducible). Init keeps the original negated-standard-normal convention.
        w = torch.empty(self.nummotif)
        torch.nn.init.normal_(w)
        self.wRect = nn.Parameter(-w)

        if self.poolType == "maxavg":
            self.adjust_dimensions = nn.Linear(2 * self.nummotif, self.embedding_dim)
        else:
            self.adjust_dimensions = nn.Linear(self.nummotif, self.embedding_dim)

        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.classifier = nn.Linear(self.embedding_dim, 1)

    def forward(self, x, training=True, return_embedding=False):
        x = x.float()
        conv = F.conv1d(x, self.wConv, bias=self.wRect, stride=1, padding=0)
        rect = F.relu(conv)

        if self.poolType == "maxavg":
            maxPool, _ = torch.max(rect, dim=2)
            avgPool = torch.mean(rect, dim=2)
            pool = torch.cat((maxPool, avgPool), 1)
        else:
            pool, _ = torch.max(rect, dim=2)

        adjusted_pool = self.adjust_dimensions(pool)
        adjusted_pool = self.layer_norm(adjusted_pool)

        if training:
            mask = bernoulli.rvs(self.dropprob, size=adjusted_pool.shape[1]).astype(
                float
            )
            mask = torch.from_numpy(mask).to(x.device).float()
            adjusted_pool *= mask

        if return_embedding:
            return adjusted_pool

        output = self.classifier(adjusted_pool)
        return output


class FeatureProbeExpert(nn.Module):
    """A linear probe over a FROZEN feature extractor's output.

    Produces a uniform ``embedding_dim`` (E) LayerNorm'd embedding -- the gate
    input the MoE expects -- plus a scalar logit, *without* touching the (frozen)
    trunk. This is how the heterogeneous zoo gives DeepSEA/DanQ trunks and the
    pretrained DNABERT-2 backbone a common E-dim embedding: the expensive trunk
    forward is run once and cached (see ``cache_embeddings.py``), then this cheap
    probe is trained per training TF on the cached D-dim features. Mirrors the
    ConvNet expert's ``Linear(->E) -> LayerNorm`` embedding so the gate is unchanged.

    ``forward`` matches the expert interface ``(x, training, return_embedding)``;
    here ``x`` is a cached feature vector and ``training`` is accepted for signature
    compatibility (dropout is the standard ``nn.Dropout``, gated by train()/eval()).
    """

    def __init__(self, in_dim, embedding_dim=32, dropout=0.0):
        super(FeatureProbeExpert, self).__init__()
        self.in_dim = in_dim
        self.embedding_dim = embedding_dim
        self.proj = nn.Linear(in_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, 1)

    def forward(self, x, training=True, return_embedding=False):
        x = x.float()
        emb = self.layer_norm(self.proj(x))
        if return_embedding:
            return emb
        return self.classifier(self.dropout(emb))


class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, embedding_size, hidden_dim=32,
                 l2norm=False, entropy_reg=0.0, gate_temperature=1.0):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.embedding_size = embedding_size
        # Hidden dimension (H). Defaults to 32 so existing checkpoints load
        # unchanged; ablations keep it equal to the expert embedding size.
        self.hidden_dim = hidden_dim
        # --- robustness knobs (additive; defaults reproduce the original forward
        # exactly, so models/moe/moe_model.pth still loads bit-identically) ---
        # l2norm: L2-normalize each per-expert E-dim block before the gate, so a
        #   large-norm backbone (e.g. DNABERT-2) cannot dominate the gate logits and
        #   collapse routing onto one expert.
        # gate_temperature: softmax temperature tau (>1 = softer routing).
        # entropy_reg: coefficient consumed by the het-MoE training loop (load-
        #   balance / anti-collapse); not used in forward.
        self.l2norm = l2norm
        self.entropy_reg = entropy_reg
        self.gate_temperature = gate_temperature
        self.gate = nn.Linear(embedding_size * num_experts, num_experts)
        self.experts = nn.ModuleList(
            [nn.Linear(embedding_size, hidden_dim) for _ in range(num_experts)]
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_gate=False):
        if self.l2norm:
            # L2-normalize each contiguous E-dim block independently
            blocks = x.view(x.shape[0], self.num_experts, self.embedding_size)
            blocks = F.normalize(blocks, p=2, dim=2)
            x = blocks.reshape(x.shape[0], self.num_experts * self.embedding_size)
        gating_weights = F.softmax(self.gate(x) / self.gate_temperature, dim=1)
        expert_outputs = [
            expert(x[:, i * self.embedding_size : (i + 1) * self.embedding_size])
            for i, expert in enumerate(self.experts)
        ]
        stacked_outputs = torch.stack(expert_outputs, dim=2)
        moe_output = torch.bmm(stacked_outputs, gating_weights.unsqueeze(2)).squeeze(2)
        output = self.classifier(moe_output)
        if return_gate:
            return output, gating_weights
        return output

    @staticmethod
    def gate_entropy(gating_weights):
        """Mean per-sample entropy of the gate distribution (nats). Higher = the gate
        spreads across experts; near 0 = collapse onto a single expert."""
        return -(gating_weights * (gating_weights + 1e-9).log()).sum(dim=1).mean()
