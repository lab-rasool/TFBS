# --------------------------------------------
# TODO: Refactor
# TODO: Generate attribution maps of the CNN
# --------------------------------------------

import csv
import gzip

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from rich.progress import Progress
from rich.traceback import install
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

install()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNEmbedding(nn.Module):
    def __init__(self, embedding_dim, dropout_rate=0.5):
        super(CNNEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = nn.Conv1d(32, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 36, embedding_dim)

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


class ChipDataLoader:
    def __init__(self, filename, motiflen=24):
        self.filename = filename
        self.motiflen = motiflen

    def seqtopad(self, sequence, kind="DNA"):
        rows = len(sequence) + 2 * self.motiflen - 2
        S = np.zeros([rows, 4], dtype=np.float32)
        base = "ACGT" if kind == "DNA" else "ACGU"
        for i in range(rows):
            if i < self.motiflen - 1 or i >= len(sequence) + self.motiflen - 1:
                S[i] = 0.25
            elif sequence[i - self.motiflen + 1] in base:
                S[i, base.index(sequence[i - self.motiflen + 1])] = 1
        return S.T

    def dinucshuffle(self, sequence):
        b = [sequence[i : i + 2] for i in range(0, len(sequence), 2)]
        np.random.shuffle(b)
        return "".join(b)

    def load_data(self, shuffle=True, label_index=3):
        data_set = []
        with gzip.open(self.filename, "rt") as file:
            next(file)  # Skip header if necessary
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                sequence = row[2]
                label = int(row[label_index]) if label_index is not None else 0
                padded_sequence = self.seqtopad(sequence)
                if shuffle:
                    shuffled_sequence = self.seqtopad(self.dinucshuffle(sequence))
                    data_set.append([shuffled_sequence, [0]])
                data_set.append([padded_sequence, [label]])
        return data_set


class chipseq_dataset(Dataset):
    def __init__(self, xy=None):
        self.x_data = np.asarray([el[0] for el in xy], dtype=np.float32)
        self.y_data = np.asarray([el[1] for el in xy], dtype=np.float32)
        self.x_data = torch.from_numpy(self.x_data).to(device)
        self.y_data = torch.from_numpy(self.y_data).to(device)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def load_data(filename):
    chipseq = ChipDataLoader(filename)
    return chipseq.load_data()


def initialize_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)


# Training function
def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    criterion,
    num_epochs,
    save_model=False,
):
    model.apply(initialize_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_auc = 0
    with Progress() as progress:
        task = progress.add_task("[cyan]Training...", total=num_epochs)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for i, train_loader in enumerate(train_dataloader):
                batch_input, batch_target = next(iter(train_loader))
                batch_input, batch_target = (
                    batch_input.to(device),
                    batch_target.to(device),
                )
                optimizer.zero_grad()
                outputs = model(batch_input)
                loss = criterion(outputs.squeeze(), batch_target.squeeze())
                if torch.isnan(loss):
                    print("NaN detected in loss, stopping training")
                    return -1  # or handle appropriately
                loss.backward()

                # Implementing gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dataloader)

            model.eval()
            valid_loss = 0
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for i, valid_loader in enumerate(valid_dataloader):
                    batch_input, batch_target = next(iter(valid_loader))
                    batch_input, batch_target = (
                        batch_input.to(device),
                        batch_target.to(device),
                    )
                    outputs = model(batch_input)
                    loss = criterion(outputs.squeeze(), batch_target.squeeze())
                    valid_loss += loss.item()

                    pred = torch.sigmoid(outputs).squeeze().cpu().numpy()
                    all_preds.extend(pred)
                    all_targets.extend(batch_target.squeeze().cpu().numpy())

            valid_loss /= len(valid_dataloader)
            if np.isnan(all_preds).any() or np.isnan(all_targets).any():
                valid_auc = 0.0
            else:
                valid_auc = roc_auc_score(all_targets, all_preds)

            progress.update(task, advance=1)

            if valid_auc > best_auc:
                best_auc = valid_auc

    if save_model:
        torch.save(model.state_dict(), "transformer.pth")

    return best_auc


def test_model(model, test_loader, device):
    model.eval()
    auc = []
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred_sig = torch.sigmoid(output)
            pred = pred_sig.cpu().detach().numpy().reshape(output.shape[0])
            labels = target.cpu().numpy().reshape(output.shape[0])
            auc.append(metrics.roc_auc_score(labels, pred))

            print(labels, pred)

    print(f"Test AUC: {np.mean(auc):.4f}")
    return np.mean(auc)


def objective(trial):
    # Hyperparameter settings
    batch_size = trial.suggest_int("batch_size", 32, 128)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    embedding_dim = trial.suggest_categorical(
        "embedding_dim", [128, 256, 512, 768, 1024, 2048]
    )
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 6)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 2, 6)

    # Data loading
    chipseq = ChipDataLoader(
        "/mnt/f/Projects/GenomicAttributions/data/encode/GATA1_K562_GATA-1_USC_AC.seq.gz"
    )
    alldataset = chipseq.load_data()

    # Splitting data into train and validation sets
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    splits = kf.split(alldataset)

    train_dataloader = []
    valid_dataloader = []

    for train_index, valid_index in splits:
        train_dataset = [alldataset[i] for i in train_index]
        valid_dataset = [alldataset[i] for i in valid_index]

        train_loader = DataLoader(
            dataset=chipseq_dataset(train_dataset), batch_size=batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            dataset=chipseq_dataset(valid_dataset), batch_size=batch_size, shuffle=False
        )

        train_dataloader.append(train_loader)
        valid_dataloader.append(valid_loader)

    # Model and training
    model = TransformerModel(
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    return train_model(
        model, train_dataloader, valid_dataloader, optimizer, criterion, 100
    )


def main():
    # ==============================
    # Hyperparameter tuning
    # ==============================
    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///transformer.db",
        study_name="transformer",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)
    study = optuna.load_study(
        study_name="transformer",
        storage="sqlite:///transformer.db",
    )

    # ==============================
    # Training
    # ==============================
    chipseq = ChipDataLoader(
        "/mnt/f/Projects/GenomicAttributions/data/encode/GATA1_K562_GATA-1_USC_AC.seq.gz"
    )
    alldataset = chipseq.load_data()
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    splits = kf.split(alldataset)
    train_dataloader = []
    valid_dataloader = []
    batch_size = study.best_trial.params["batch_size"]
    for train_index, valid_index in splits:
        train_dataset = [alldataset[i] for i in train_index]
        valid_dataset = [alldataset[i] for i in valid_index]
        train_loader = DataLoader(
            dataset=chipseq_dataset(train_dataset), batch_size=batch_size, shuffle=True
        )
        valid_loader = DataLoader(
            dataset=chipseq_dataset(valid_dataset), batch_size=batch_size, shuffle=False
        )
        train_dataloader.append(train_loader)
        valid_dataloader.append(valid_loader)
    model = TransformerModel(
        embedding_dim=study.best_trial.params["embedding_dim"],
        dropout_rate=study.best_trial.params["dropout_rate"],
        num_encoder_layers=study.best_trial.params["num_encoder_layers"],
        num_decoder_layers=study.best_trial.params["num_decoder_layers"],
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=study.best_trial.params["lr"])
    criterion = nn.BCEWithLogitsLoss()
    train_model(
        model,
        train_dataloader,
        valid_dataloader,
        optimizer,
        criterion,
        250,
        save_model=True,
    )

    # ==============================
    # Testing
    # ==============================
    # "/mnt/f/Projects/GenomicAttributions/data/encode/GATA1_K562_GATA-1_USC_AC.seq.gz"

    chipseq_test = ChipDataLoader(
        "/mnt/f/Projects/GenomicAttributions/data/encode/GATA1_K562_GATA-1_USC_B.seq.gz"
    )
    testdataset = chipseq_test.load_data()
    test_dataset = chipseq_dataset(testdataset)
    test_batch_size = test_dataset.len
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=test_batch_size, shuffle=False
    )
    model.load_state_dict(torch.load("transformer.pth"))
    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()
