import csv
import gzip
import os

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

console = Console()
table = Table(title="Model Performance")
table.add_column("Epoch", justify="center")
table.add_column("Train Loss", justify="center")
table.add_column("Valid Loss", justify="center")
table.add_column("Valid AUC", justify="center")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
console.print(f"Device: {device}")


class EmbeddingGenerator:
    def __init__(self, model_path, tokenizer_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def tokenize_dna_sequence(self, sequence, pad_char="N", kmer_size=6):
        # chunk the dna sequence into 6-mers, padding the last one if necessary
        chunks = [
            sequence[i : i + kmer_size] for i in range(0, len(sequence), kmer_size)
        ]
        if len(chunks[-1]) < kmer_size:
            # Pad the last chunk if it is shorter than kmer_size
            chunks[-1] += pad_char * (kmer_size - len(chunks[-1]))

        # tokenize the chunks
        tokens = [self.tokenizer(chunk, add_special_tokens=False) for chunk in chunks]
        # flatten the tokens
        tokens = [token for chunk in tokens for token in chunk["input_ids"]]
        return tokens

    def generate_embeddings(self, sequence):
        inputs_dna = self.tokenize_dna_sequence(sequence)
        inputs_dna = torch.tensor(inputs_dna).unsqueeze(0)
        inputs_dna = inputs_dna.to(self.device)
        with torch.no_grad():
            hidden_states_dna = self.model(inputs_dna).last_hidden_state
        return hidden_states_dna.cpu().numpy()


def create_embeddings_file(data, embedding_generator, output_file):
    if data is None:
        print("No data to process.")
        return
    all_data = []
    for sequence, label in tqdm(data):
        embeddings = embedding_generator.generate_embeddings(sequence)
        embeddings_bytes = embeddings.tobytes()  # Convert array to bytes
        all_data.append([sequence, label, embeddings_bytes])

    # Create DataFrame
    df = pd.DataFrame(all_data, columns=["sequence", "label", "embedding"])

    # Ensure consistent data types
    df["label"] = df["label"].astype(int)

    # Writing to Parquet
    df.to_parquet(output_file)
    print(f"Data written to {output_file}.")


def read_embeddings_file(input_file):
    df = pd.read_parquet(input_file)
    # [1, 17, 768]
    df["embedding"] = df["embedding"].apply(
        lambda x: np.frombuffer(x, dtype=np.float32).reshape(1, 17, 768)
    )
    return df


class ChipDataLoader:
    def __init__(self, filename, motiflen=24):
        self.filename = filename
        self.motiflen = motiflen

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
                if shuffle:
                    shuffled_sequence = self.dinucshuffle(sequence)
                    data_set.append([shuffled_sequence, 0])
                data_set.append([sequence, label])
        return data_set


def generate_embeddings_file(path_to_data, output_file, shuffle=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding_generator = EmbeddingGenerator(
        "zhihan1996/DNA_bert_6", "zhihan1996/DNA_bert_6", device
    )

    data_loader = ChipDataLoader(path_to_data)
    data = data_loader.load_data(shuffle=True)
    create_embeddings_file(data, embedding_generator, output_file)


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


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        num_heads,
        num_layers,
        num_classes,
        dim_feedforward=2048,
        dropout=0.1,
        pooling="avg",
    ):
        super(SequenceClassifier, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.pooling = pooling
        if self.pooling == "max_avg":
            classifier_input_dim = 2 * input_dim
        else:
            classifier_input_dim = input_dim
        self.classifier = nn.Linear(classifier_input_dim, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        if self.pooling == "avg":
            x = x.mean(dim=1)
        elif self.pooling == "max":
            x, _ = x.max(dim=1)
        elif self.pooling == "max_avg":
            x_max, _ = x.max(dim=1)
            x_avg = x.mean(dim=1)
            x = torch.cat((x_max, x_avg), dim=1)
        out = self.classifier(x)
        return out


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = [torch.tensor(e.squeeze(0)) for e in embeddings]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def train(model, dataloader, optimizer, criterion, device, beta1=0, beta2=0):
    model.train()
    total_loss = 0
    for embeddings, labels in dataloader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(embeddings)
        ce_loss = criterion(outputs, labels)
        l1_loss = beta1 * sum(p.abs().sum() for p in model.transformer.parameters())
        l1_loss += beta2 * model.classifier.weight.abs().sum()
        loss = ce_loss + l1_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(
                outputs[:, 1].cpu().numpy()
            )  # Assumes binary classification
            all_labels.extend(labels.cpu().numpy())
    loss = total_loss / len(dataloader)
    auc_score = roc_auc_score(all_labels, all_preds)
    return loss, auc_score, all_labels, all_preds


def plot_roc_curve(all_labels, all_preds):
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    auc_score = roc_auc_score(all_labels, all_preds)
    plt.figure()
    plt.plot(fpr, tpr, label="Transformer-DNABERT (area = %0.2f)" % auc_score)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("./results/transformerDNABERT_ROC_curve.png")


def read_embeddings_file(input_file):
    df = pd.read_parquet(input_file)
    df["embedding"] = df["embedding"].apply(
        lambda x: np.frombuffer(x, dtype=np.float32).reshape(1, 17, 768)
    )
    return df


def objective(trial):
    # Hyperparameters
    num_layers = trial.suggest_int("num_layers", 1, 8)
    num_heads = trial.suggest_categorical("num_heads", [2, 4, 8, 16, 32, 64, 128])
    dim_feedforward = trial.suggest_int("dim_feedforward", 512, 2048, step=256)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-10, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512, 1024])
    pooling = trial.suggest_categorical("pooling", ["avg", "max", "max_avg"])
    beta1 = trial.suggest_float("beta1", 1e-10, 1e-3, log=True)
    beta2 = trial.suggest_float("beta2", 1e-10, 1e-3, log=True)

    model = SequenceClassifier(
        input_dim=768,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=2,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        pooling=pooling,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Read embeddings and labels
    embeddings_df = read_embeddings_file("./data/train.parquet")
    X = list(embeddings_df["embedding"])
    y = list(embeddings_df["label"])

    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Prepare datasets
    train_dataset = EmbeddingDataset(X_train, y_train)
    valid_dataset = EmbeddingDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    early_stopping = EarlyStopping(patience=5)

    for epoch in tqdm(range(10), desc="Training", leave=False):
        train_loss = train(
            model, train_loader, optimizer, criterion, device, beta1, beta2
        )
        valid_loss, valid_auc, _, _ = evaluate(model, valid_loader, criterion, device)

        if early_stopping(valid_loss):
            break

        trial.report(valid_auc, epoch)  # Report validation AUC instead of loss

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # empty the cache
    torch.cuda.empty_cache()

    return valid_auc  # Return validation AUC instead of loss


def main():
    torch.cuda.empty_cache()

    train_output_file = "./data/train.parquet"
    test_output_file = "./data/test.parquet"
    chipseq_train = "/mnt/f/Projects/GenomicAttributions/data/encode/train/GATA3_SH-SY5Y_GATA3_(SC-269)_USC_AC.seq.gz"
    chipseq_test = "/mnt/f/Projects/GenomicAttributions/data/encode/test/GATA3_SH-SY5Y_GATA3_(SC-269)_USC_B.seq.gz"

    # if train_output_file exists, skip generating embeddings
    if not os.path.exists(train_output_file):
        generate_embeddings_file(chipseq_train, train_output_file)
    if not os.path.exists(test_output_file):
        generate_embeddings_file(chipseq_test, test_output_file)

    try:
        console.print("Loading existing study")
        study = optuna.load_study(study_name="TFBS", storage="sqlite:///transformer.db")
        best_params = study.best_trial.params
    except Exception as e:
        console.print("Creating a new study")
        study = optuna.create_study(
            storage="sqlite:///transformer.db",
            study_name="TFBS",
            direction="maximize",
            load_if_exists=True,
        )
        study.optimize(
            objective, n_trials=10, gc_after_trial=True, show_progress_bar=True
        )
        best_params = study.best_trial.params

    batch_size = best_params["batch_size"]
    epochs = 500

    # Data preparation
    embeddings_df = read_embeddings_file("./data/train.parquet")
    X = list(embeddings_df["embedding"])
    y = list(embeddings_df["label"])

    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Load the test set
    test_df = read_embeddings_file("./data/test.parquet")
    X_test = list(test_df["embedding"])
    y_test = list(test_df["label"])

    # print the total number of samples in the training and test sets
    console.print(f"Train samples: {len(X_train)}")
    console.print(f"Test samples: {len(X_valid)}")
    console.print(f"Test samples: {len(X_test)}")

    console.print(f"Best trial score: {study.best_trial.value}")
    console.print("Best hyperparameters:", study.best_trial.params)

    # Prepare datasets
    train_dataset = EmbeddingDataset(X_train, y_train)
    valid_dataset = EmbeddingDataset(X_valid, y_valid)
    test_dataset = EmbeddingDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    final_model = SequenceClassifier(
        input_dim=768,
        num_heads=best_params["num_heads"],
        num_layers=best_params["num_layers"],
        num_classes=2,
        dim_feedforward=best_params["dim_feedforward"],
        dropout=best_params["dropout"],
        pooling=best_params["pooling"],
    ).to(device)

    optimizer = optim.AdamW(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss()

    early_stopping = EarlyStopping(patience=5, min_delta=1e-5)

    for epoch in range(epochs):
        train_loss = train(
            final_model,
            train_loader,
            optimizer,
            criterion,
            device,
            best_params["beta1"],
            best_params["beta2"],
        )
        valid_loss, valid_auc, _, _ = evaluate(
            final_model,
            valid_loader,
            criterion,
            device,
        )

        console.print(
            f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, Valid AUC: {valid_auc:.4f}"
        )
        table.add_row(
            str(epoch + 1), f"{train_loss:.4f}", f"{valid_loss:.4f}", f"{valid_auc:.4f}"
        )

        early_stopping(valid_loss)
        if early_stopping.early_stop:
            console.print("Early stopping triggered")
            break

    console.print(table)

    test_loss, test_auc, all_labels, all_preds = evaluate(
        final_model, test_loader, criterion, device
    )
    console.print(f"Test Loss: {test_loss:.4f}, Test AUC: {test_auc:.4f}")
    plot_roc_curve(all_labels, all_preds)

    # Save the model
    if not os.path.exists("./models/transformerDNABERT"):
        os.makedirs("./models/transformerDNABERT")
    torch.save(final_model.state_dict(), "./models/transformerDNABERT/model.pth")


if __name__ == "__main__":
    main()
