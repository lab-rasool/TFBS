import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from rich.console import Console
from scipy.stats import bernoulli
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import ChipDataLoader, chipseq_dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

console = Console()


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


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


def train(config, train_loader, valid_loader):
    console.print(config)
    model = ConvNet(config).to(device)
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=model.learning_rate,
        momentum=model.momentum_rate,
        nesterov=True,
    )
    early_stopping = EarlyStopping(patience=5)
    best_auc = 0
    for epoch in range(500):  # Number of epochs
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(torch.sigmoid(output), target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_auc = 0
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predictions = torch.sigmoid(output)
                valid_auc += roc_auc_score(target.cpu(), predictions.cpu())

        avg_auc = valid_auc / len(valid_loader)
        console.print(f"Epoch {epoch}: Validation AUC = {avg_auc}")
        if avg_auc > best_auc:
            best_auc = avg_auc
            torch.save(
                {
                    "conv": model.wConv,
                    "rect": model.wRect,
                    "wHidden": model.wHidden,
                    "wHiddenBias": model.wHiddenBias,
                    "wNeu": model.wNeu,
                    "wNeuBias": model.wNeuBias,
                },
                "./models/deepBIND/best_model.pth",
            )

        early_stopping(avg_auc)

        if early_stopping.early_stop:
            console.print("Early stopping")
            break

    console.print("Training complete. Best AUC:", best_auc)


def test(config, checkpoint, test_loader):
    checkpoint = torch.load(checkpoint)
    model = ConvNet(config).to(device)
    model.wConv = checkpoint["conv"]
    model.wRect = checkpoint["rect"]
    model.wHidden = checkpoint["wHidden"]
    model.wHiddenBias = checkpoint["wHiddenBias"]
    model.wNeu = checkpoint["wNeu"]
    model.wNeuBias = checkpoint["wNeuBias"]
    model.eval()

    # Evaluate the model
    total_preds, total_targets = [], []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_sig = torch.sigmoid(output)
            total_preds.extend(pred_sig.cpu().numpy())
            total_targets.extend(target.cpu().numpy())

    auc_score = roc_auc_score(total_targets, total_preds)
    print(f"AUC on test data = {auc_score}")

    # Plot ROC curve
    fpr, tpr, _ = metrics.roc_curve(total_targets, total_preds)
    plt.figure()
    plt.plot(fpr, tpr, label="DeepBIND (area = %0.2f)" % auc_score)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("./results/DeepBIND_ROC_curve.png")


def objective(trial, train_loader, valid_loader):
    config = {
        "nummotif": 16,
        "motiflen": 24,
        "poolType": trial.suggest_categorical("poolType", ["max", "maxavg"]),
        "neuType": trial.suggest_categorical("neuType", ["hidden", "nohidden"]),
        "dropprob": trial.suggest_float("dropprob", 0.5, 1.0, step=0.25),
        "sigmaConv": trial.suggest_float("sigmaConv", 1e-7, 1e-3, log=True),
        "sigmaNeu": trial.suggest_float("sigmaNeu", 1e-5, 1e-2, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.0005, 0.05, log=True),
        "momentum_rate": trial.suggest_float("momentum_rate", 0.95, 0.99),
        "beta1": trial.suggest_float("beta1", 1e-10, 1e-3, log=True),
        "beta2": trial.suggest_float("beta2", 1e-10, 1e-3, log=True),
        "beta3": trial.suggest_float("beta3", 1e-10, 1e-3, log=True),
    }

    model = ConvNet(config).to(device)

    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=model.learning_rate,
        momentum=model.momentum_rate,
        nesterov=True,
    )

    best_auc = 0
    for epoch in range(10):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy(torch.sigmoid(output), target)
            loss.backward()
            optimizer.step()

        model.eval()
        valid_auc = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predictions = torch.sigmoid(output)
                valid_auc += roc_auc_score(
                    target.cpu().numpy(), predictions.cpu().numpy()
                )

        avg_auc = valid_auc / len(valid_loader)
        if avg_auc > best_auc:
            best_auc = avg_auc

    return best_auc


def main():
    chipseq_train = "/mnt/f/Projects/GenomicAttributions/data/encode/GATA3_SH-SY5Y_GATA3_(SC-269)_USC_AC.seq.gz"
    chipseq_test = "/mnt/f/Projects/GenomicAttributions/data/encode/GATA3_SH-SY5Y_GATA3_(SC-269)_USC_B.seq.gz"

    # 0. load data
    alldataset = ChipDataLoader(chipseq_train).load_data()
    test_data = ChipDataLoader(chipseq_test).load_data()
    train_data, valid_data = train_test_split(alldataset, test_size=0.2)
    train_loader = DataLoader(
        dataset=chipseq_dataset(train_data),
        batch_size=64,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=chipseq_dataset(valid_data),
        batch_size=64,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=chipseq_dataset(test_data),
        batch_size=len(test_data),
        shuffle=False,
    )

    # 1. Hyperparameter optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, train_loader, valid_loader),
        n_trials=10,
        gc_after_trial=True,
    )

    # 2. Save best hyperparameters
    best_hyperparameters = study.best_trial.params
    torch.save(
        best_hyperparameters,
        "./models/deepBIND/best_hyperparameters.pth",
    )

    # 3. Train and test the model with the best hyperparameters
    config = {
        "nummotif": 16,
        "motiflen": 24,
        "poolType": best_hyperparameters["poolType"],
        "neuType": best_hyperparameters["neuType"],
        "dropprob": best_hyperparameters["dropprob"],
        "sigmaConv": best_hyperparameters["sigmaConv"],
        "sigmaNeu": best_hyperparameters["sigmaNeu"],
        "learning_rate": best_hyperparameters["learning_rate"],
        "momentum_rate": best_hyperparameters["momentum_rate"],
        "beta1": best_hyperparameters["beta1"],
        "beta2": best_hyperparameters["beta2"],
        "beta3": best_hyperparameters["beta3"],
    }

    train(
        config=config,
        train_loader=train_loader,
        valid_loader=valid_loader,
    )
    test(
        config=config,
        checkpoint="./models/deepBIND/best_model.pth",
        test_loader=test_loader,
    )


if __name__ == "__main__":
    main()
