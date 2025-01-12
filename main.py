import argparse
import os
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import warnings

from data import ChipDataLoader, chipseq_dataset
from model import ConvNet, MixtureOfExperts
from utils import EarlyStopping, get_tf_name, load_files_from_folder

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
console = Console()
console.print(f"Using device: {device}")

def load_model(model_path, config):
    model = ConvNet(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def train_expert(config, train_loader, valid_loader, save_path, train_file):
    model = ConvNet(config).to(device)
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=config["learning_rate"],
        momentum=config["momentum_rate"],
        nesterov=True,
    )
    early_stopping = EarlyStopping(patience=5)
    best_auc = 0

    for epoch in range(500):
        model.train()
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        valid_auc = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predictions = torch.sigmoid(output)
                valid_auc += roc_auc_score(target.cpu(), predictions.cpu())

        avg_auc = valid_auc / len(valid_loader)
        avg_loss = running_loss / len(train_loader)
        console.print(
            f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Valid AUC: {avg_auc:.4f} | Train File: {train_file}"
        )

        if avg_auc > best_auc:
            best_auc = avg_auc
            torch.save(model.state_dict(), save_path)

        early_stopping(avg_auc)

        if early_stopping.early_stop:
            console.print("Early stopping")
            break

    console.print(f"Training complete for {train_file}. Best AUC: {best_auc:.4f}")


def generate_embeddings(data_loader, models):
    all_embeddings = []
    all_targets = []
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        embeddings = [model(data, return_embedding=True) for model in models]
        concatenated = torch.cat(embeddings, dim=1)
        all_embeddings.append(concatenated)
        all_targets.append(target)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_embeddings, all_targets


def train_moe(models, moe_model, train_loader, valid_loader, save_path, num_epochs=500, lr=0.001):
    optimizer = torch.optim.SGD(
        [param for param in moe_model.parameters() if param.requires_grad],
        lr=lr,
        momentum=0.98,
        nesterov=True,
    )
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    train_embeddings, train_targets = generate_embeddings(train_loader, models)
    valid_embeddings, valid_targets = generate_embeddings(valid_loader, models)

    for epoch in range(num_epochs):
        moe_model.train()
        embeddings, targets = train_embeddings.to(device), train_targets.to(device)
        optimizer.zero_grad()
        outputs = moe_model(embeddings)
        loss = nn.functional.binary_cross_entropy_with_logits(outputs, targets.float())
        loss.backward(retain_graph=True)
        optimizer.step()

        moe_model.eval()
        val_targets = []
        val_outputs = []
        with torch.no_grad():
            valid_embeddings, _ = embeddings.to(device), targets.to(device)
            outputs = moe_model(valid_embeddings)
            val_outputs.append(outputs.detach().cpu())
            val_targets.append(targets.detach().cpu())
        val_outputs = torch.cat(val_outputs)
        val_targets = torch.cat(val_targets)
        val_auc = roc_auc_score(val_targets, torch.sigmoid(val_outputs))
        console.print(
            f"Epoch {epoch} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}"
        )

        if early_stopping(val_auc):
            console.print("Early stopping triggered.")
            break

    torch.save(moe_model.state_dict(), save_path+"/moe/moe_model.pth")
    torch.save(
        {
            "num_experts": len(models),
            "embedding_size": 32,
            "state_dict": moe_model.state_dict(),
        },
        save_path+"/moe/moe_model_config.pth",
    )

    console.print(
        f"Training complete. Best validation AUC: {early_stopping.best_score:.4f}"
    )


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
        "dim_feedforward": trial.suggest_int("dim_feedforward", 32, 256, step=32),
        "num_heads": trial.suggest_categorical("num_heads", [1, 2, 4, 8, 16]),
        "encoder_dropout": trial.suggest_float("encoder_dropout", 0.1, 0.5),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
    }
    model = ConvNet(config).to(device)
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=config["learning_rate"],
        momentum=config["momentum_rate"],
        nesterov=True,
    )
    best_auc = 0
    for epoch in range(10):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target.float())
            loss.backward()
            optimizer.step()
        model.eval()
        valid_auc = 0
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                predictions = torch.sigmoid(output)
                valid_auc += roc_auc_score(target.cpu(), predictions.cpu())
        avg_auc = valid_auc / len(valid_loader)
        if avg_auc > best_auc:
            best_auc = avg_auc
    return best_auc


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ConvNet models for ChIP-seq data."
    )
    parser.add_argument(
        "--train_folder",
        default="./data/train",
        help="Path to folder containing training ChIP-seq files",
    )
    parser.add_argument(
        "--save_path",
        default="./models",
        help="Directory to save trained models and results",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=96, 
        help="Batch size for training",
    )

    args = parser.parse_args()
    train_folder = args.train_folder
    save_path = args.save_path
    batch_size = args.batch_size

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/hyperparams", exist_ok=True)
    os.makedirs(f"{save_path}/experts", exist_ok=True)
    os.makedirs(f"{save_path}/moe", exist_ok=True)

    # -------------------------------------------------------------------------------------
    # Find Best Individual Expert Model Hyperparameters
    # -------------------------------------------------------------------------------------
    # Optimize hyperparameters for each training file
    chiqseq_train_files = load_files_from_folder(train_folder)
    best_hyperparameters_list = []
    for i, train_file in enumerate(chiqseq_train_files):
        alldataset = ChipDataLoader(train_file).load_data()
        train_data, valid_data = train_test_split(alldataset, test_size=0.2, stratify=[label for _, label in alldataset])
        train_loader = DataLoader(
            dataset=chipseq_dataset(train_data),
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset=chipseq_dataset(valid_data),
            batch_size=batch_size,
            shuffle=False,
        )
        # Hyperparameter optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, train_loader, valid_loader),
            n_trials=10,
            gc_after_trial=True,
        )
        best_hyperparameters = study.best_trial.params
        torch.save(
            best_hyperparameters,
            f"{save_path}/hyperparams/{get_tf_name(train_file)}.pth",
        )
        best_hyperparameters_list.append(best_hyperparameters)
        console.print(best_hyperparameters)

    # -------------------------------------------------------------------------------------
    # Train Individual Expert Models using Best Hyperparameters
    # -------------------------------------------------------------------------------------
    for i, (train_file, best_hyperparameters) in enumerate(
        zip(chiqseq_train_files, best_hyperparameters_list)
    ):
        alldataset = ChipDataLoader(train_file).load_data()
        train_data, valid_data = train_test_split(alldataset, test_size=0.2)
        train_loader = DataLoader(
            dataset=chipseq_dataset(train_data),
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset=chipseq_dataset(valid_data),
            batch_size=batch_size,
            shuffle=False,
        )
        config = {
            "nummotif": 16,
            "motiflen": 24,
            "poolType": best_hyperparameters["poolType"],
            "sigmaConv": best_hyperparameters["sigmaConv"],
            "dropprob": best_hyperparameters["dropprob"],
            "learning_rate": best_hyperparameters["learning_rate"],
            "momentum_rate": best_hyperparameters["momentum_rate"],
            "num_features": 32,
            "d_model": 32,
            "num_heads": best_hyperparameters["num_heads"],
            "dim_feedforward": best_hyperparameters["dim_feedforward"],
            "encoder_dropout": best_hyperparameters["encoder_dropout"],
            "num_layers": best_hyperparameters["num_layers"],
        }
        # Train model with best hyperparameters
        train_expert(
            config=config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            save_path=f"{save_path}/experts/{get_tf_name(train_file)}.pth",
            train_file=train_file,
        )

    # -------------------------------------------------------------------------------------
    # MixtureOfExperts
    # -------------------------------------------------------------------------------------
    # Load and combine data from all training files
    combined_data = []
    for path in chiqseq_train_files:
        data_loader = ChipDataLoader(path)
        data = data_loader.load_data()
        combined_data.extend(data)

    # Split data into training and validation sets
    train_data, valid_data = train_test_split(
        combined_data, test_size=0.2, stratify=[label for _, label in combined_data]
    )
    train_loader = DataLoader(
        dataset=chipseq_dataset(train_data), batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        dataset=chipseq_dataset(valid_data), batch_size=batch_size, shuffle=False
    )

    # Load pre-trained models
    model_paths = [
        f"{save_path}/experts/{get_tf_name(train_file)}.pth"
        for train_file in chiqseq_train_files
    ]
    configs = [
        torch.load(f"{save_path}/hyperparams/{get_tf_name(train_file)}.pth")
        for train_file in chiqseq_train_files
    ]
    models = [load_model(path, config) for path, config in zip(model_paths, configs)]
    moe_model = MixtureOfExperts(num_experts=len(models), embedding_size=32).to(device)
    train_moe(
        models,
        moe_model,
        train_loader,
        valid_loader,
        save_path=save_path,
        num_epochs=500,
        lr=0.01,
    )

if __name__ == "__main__":
    main()