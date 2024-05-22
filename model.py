import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bernoulli
from transformers import AutoTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_6")
bert_model = BertModel.from_pretrained("zhihan1996/DNA_bert_6")


class DeepBIND(nn.Module):
    def __init__(self, config):
        super(DeepBIND, self).__init__()
        # Model configuration
        self.nummotif = config["nummotif"]
        self.motiflen = config["motiflen"]

        self.poolType = config["poolType"]
        self.neuType = config["neuType"]
        self.mode = "training"
        self.learning_rate = config["learning_rate"]
        self.momentum_rate = config["momentum_rate"]
        self.sigmaConv = config["sigmaConv"]
        self.wConv = nn.Parameter(torch.randn(self.nummotif, 4, self.motiflen))
        torch.nn.init.normal_(self.wConv, mean=0, std=self.sigmaConv)
        self.wConv.requires_grad = True
        self.wRect = torch.randn(self.nummotif).to(device)
        torch.nn.init.normal_(self.wRect)
        self.wRect = -self.wRect
        self.wRect.requires_grad = True
        self.dropprob = config["dropprob"]
        self.sigmaNeu = config["sigmaNeu"]
        self.wHidden = torch.randn(2 * self.nummotif, 32).to(device)

        self.beta1 = config["beta1"]
        self.beta2 = config["beta2"]
        self.beta3 = config["beta3"]
        # Initialize weights
        wRect_init = torch.randn(self.nummotif)
        nn.init.normal_(wRect_init)

        self.wHiddenBias = torch.randn(32).to(device)
        if self.neuType == "nohidden":
            if self.poolType == "maxavg":
                self.wNeu = nn.Parameter(torch.randn(2 * self.nummotif, 1))
            else:
                self.wNeu = nn.Parameter(torch.randn(self.nummotif, 1))
            self.wNeuBias = nn.Parameter(torch.randn(1))
            nn.init.normal_(self.wNeu, mean=0, std=self.sigmaNeu)
            nn.init.normal_(self.wNeuBias, mean=0, std=self.sigmaNeu)
        else:
            if self.poolType == "maxavg":
                self.wHidden = nn.Parameter(torch.randn(2 * self.nummotif, 32))
            else:
                self.wHidden = nn.Parameter(torch.randn(self.nummotif, 32))
            self.wNeu = nn.Parameter(torch.randn(32, 1))
            self.wNeuBias = nn.Parameter(torch.randn(1))
            nn.init.normal_(self.wHidden, mean=0, std=0.3)
            nn.init.normal_(self.wNeu, mean=0, std=self.sigmaNeu)
            nn.init.normal_(self.wNeuBias, mean=0, std=self.sigmaNeu)
            self.wHiddenBias = nn.Parameter(torch.randn(32))
            nn.init.normal_(self.wHiddenBias, mean=0, std=0.3)

    def forward(self, x, training=True, return_embeddings=False):
        x = x.float()  # Ensure input x is float
        conv = F.conv1d(x, self.wConv, bias=self.wRect, stride=1, padding=0)
        rect = F.relu(conv)

        if self.poolType == "maxavg":
            maxPool, _ = torch.max(rect, dim=2)
            avgPool = torch.mean(rect, dim=2)
            pool = torch.cat((maxPool, avgPool), 1)
        else:
            pool, _ = torch.max(rect, dim=2)

        if return_embeddings:
            return pool

        if self.neuType == "nohidden":
            if training:
                mask = bernoulli.rvs(self.dropprob, size=pool.shape[1]).astype(float)
                mask = torch.from_numpy(mask).to(x.device).float()
                pooldrop = pool * mask
                output = pooldrop.mm(self.wNeu) + self.wNeuBias
            else:
                output = pool.mm(self.wNeu) + self.wNeuBias
        else:
            hid = pool.mm(self.wHidden) + self.wHiddenBias
            hid = F.relu(hid)
            if training:
                mask = bernoulli.rvs(self.dropprob, size=hid.shape[1]).astype(float)
                mask = torch.from_numpy(mask).to(x.device).float()
                hiddrop = hid * mask
                output = hiddrop.mm(self.wNeu) + self.wNeuBias
            else:
                output = hid.mm(self.wNeu) + self.wNeuBias

        return output


class CNNEmbedding(nn.Module):
    def __init__(self, embedding_dim, dropout_rate=0.5):
        super(CNNEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 36, embedding_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Define the transformer model
class TransformerModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        dropout_rate=0.5,
        dim_feedforward=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super(TransformerModel, self).__init__()
        self.embedding = CNNEmbedding(embedding_dim, dropout_rate)
        self.transformer = nn.Transformer(
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_model=embedding_dim,
            dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer(src, src)
        output = self.fc(output)
        return output


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

        self.wConv = nn.Parameter(torch.randn(self.nummotif, 4, self.motiflen))
        torch.nn.init.normal_(self.wConv, mean=0, std=self.sigmaConv)
        self.wRect = torch.randn(self.nummotif).to(device)
        torch.nn.init.normal_(self.wRect)
        self.wRect = -self.wRect

        if self.poolType == "maxavg":
            self.adjust_dimensions = nn.Linear(2 * self.nummotif, 32)
        else:
            self.adjust_dimensions = nn.Linear(self.nummotif, 32)

        self.layer_norm = nn.LayerNorm(32)
        self.classifier = nn.Linear(32, 1)

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


# weight sum version
class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, embedding_size=32):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.embedding_size = embedding_size
        self.gate = nn.Linear(num_experts * embedding_size, num_experts)
        self.classifier = nn.Linear(embedding_size, 1)

    def forward(self, embeddings):
        gating_weights = F.softmax(self.gate(embeddings), dim=1)
        embeddings = embeddings.view(-1, self.num_experts, self.embedding_size)
        gating_weights = gating_weights.unsqueeze(-1)
        combined_embedding = torch.mean(gating_weights * embeddings, dim=1)
        return self.classifier(combined_embedding)
