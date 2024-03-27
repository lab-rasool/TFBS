import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import bernoulli

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvNet(nn.Module):
    def __init__(self, config):
        super(ConvNet, self).__init__()
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

    def forward(self, x, training=True):
        x = x.float()  # Ensure input x is float
        conv = F.conv1d(x, self.wConv, bias=self.wRect, stride=1, padding=0)
        rect = F.relu(conv)

        if self.poolType == "maxavg":
            maxPool, _ = torch.max(rect, dim=2)
            avgPool = torch.mean(rect, dim=2)
            pool = torch.cat((maxPool, avgPool), 1)
        else:
            pool, _ = torch.max(rect, dim=2)

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
