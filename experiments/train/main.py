import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import warnings

from tfbs.data import ChipDataLoader, chipseq_dataset
from tfbs.models import ConvNet, MixtureOfExperts
from tfbs.utils import EarlyStopping, get_tf_name, load_files_from_folder, order_files_by, set_seed
from tfbs.constants import TRAIN_TFS, train_dir_and_shuffle, TRAIN_NEG_MODE

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
console = Console()
console.print(f"Using device: {device}")

def load_model(model_path, config):
    model = ConvNet(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
                output = model(data, training=False)  # dropout off for clean validation
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


def generate_embeddings(data_loader, models, training=False, no_grad=True):
    """Concatenate the per-expert embeddings for every batch.

    ``training`` controls whether the experts' manual Bernoulli dropout is applied
    while producing embeddings. Defaults to ``False`` so the MoE is trained on the
    same (clean) embeddings it sees at inference, removing the train/inference
    dropout mismatch present in the original pipeline.

    ``no_grad`` (default True) detaches the experts from the autograd graph -- the
    standard frozen-expert MoE setup. The fine-tuned ablation passes ``no_grad=False``
    so gradients can flow back into the experts.
    """
    import contextlib

    all_embeddings = []
    all_targets = []
    ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
    with ctx:
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            embeddings = [model(data, training=training, return_embedding=True) for model in models]
            concatenated = torch.cat(embeddings, dim=1)
            all_embeddings.append(concatenated)
            all_targets.append(target)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    return all_embeddings, all_targets


def train_moe(models, moe_model, train_loader, valid_loader, save_path, num_epochs=500,
              lr=0.01, batch_size=256):
    """Train the gating MoE on precomputed (dropout-off) expert embeddings.

    Uses mini-batch SGD with Nesterov momentum (matching the manuscript's
    description), the genuine validation embeddings for early stopping, and keeps
    the best-validation weights. Experts stay frozen (only ``moe_model`` is in the
    optimizer).
    """
    optimizer = torch.optim.SGD(
        [param for param in moe_model.parameters() if param.requires_grad],
        lr=lr,
        momentum=0.98,
        nesterov=True,
    )
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)

    train_embeddings, train_targets = generate_embeddings(train_loader, models)
    valid_embeddings, valid_targets = generate_embeddings(valid_loader, models)
    train_embeddings = train_embeddings.to(device)
    train_targets = train_targets.to(device)
    valid_embeddings = valid_embeddings.to(device)
    valid_targets = valid_targets.to(device)
    n = train_embeddings.shape[0]

    best_auc = -1.0
    best_state = None
    for epoch in range(num_epochs):
        moe_model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            outputs = moe_model(train_embeddings[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs, train_targets[idx].float()
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        moe_model.eval()
        with torch.no_grad():
            val_logits = moe_model(valid_embeddings)
        val_auc = roc_auc_score(
            valid_targets.detach().cpu(), torch.sigmoid(val_logits.detach().cpu())
        )
        console.print(
            f"Epoch {epoch} | Loss: {epoch_loss / n:.4f} | Val AUC: {val_auc:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in moe_model.state_dict().items()}

        if early_stopping(val_auc):
            console.print("Early stopping triggered.")
            break

    if best_state is not None:
        moe_model.load_state_dict(best_state)  # restore best-validation weights

    torch.save(moe_model.state_dict(), save_path+"/moe/moe_model.pth")
    torch.save(
        {
            "num_experts": len(models),
            "embedding_size": moe_model.embedding_size,
            "hidden_dim": moe_model.hidden_dim,
            "state_dict": moe_model.state_dict(),
        },
        save_path+"/moe/moe_model_config.pth",
    )

    console.print(
        f"Training complete. Best validation AUC: {best_auc:.4f}"
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits, init, and dropout",
    )
    parser.add_argument(
        "--use_saved_hyperparams",
        action="store_true",
        help="Skip the Optuna search and reuse the documented hyperparameters in "
        "models/hyperparams/*.pth (deterministic, reproducible retraining).",
    )
    parser.add_argument(
        "--moe_only",
        action="store_true",
        help="Skip Optuna + expert training; retrain only the MoE from the saved experts.",
    )

    args = parser.parse_args()
    # Negative-mode switch: in genomic mode, default to the GC-matched-negative dir and
    # load with shuffle=False (real negatives already in the file). An explicit
    # --train_folder still wins.
    _mode_dir, _train_shuffle = train_dir_and_shuffle()
    train_folder = args.train_folder if args.train_folder != "./data/train" else _mode_dir
    print(f"[main] negative mode = {TRAIN_NEG_MODE}; train_folder = {train_folder}; "
          f"load_data shuffle = {_train_shuffle}")
    save_path = args.save_path
    batch_size = args.batch_size
    seed = args.seed
    set_seed(seed)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/hyperparams", exist_ok=True)
    os.makedirs(f"{save_path}/experts", exist_ok=True)
    os.makedirs(f"{save_path}/moe", exist_ok=True)

    # Canonical TRAIN_TFS order (not raw os.listdir) so the expert/MoE-embedding
    # concatenation order is machine-independent and matches evaluation.
    chiqseq_train_files = order_files_by(load_files_from_folder(train_folder), TRAIN_TFS)

    # -------------------------------------------------------------------------------------
    # Find Best Individual Expert Model Hyperparameters
    # -------------------------------------------------------------------------------------
    # Optimize hyperparameters for each training file (or reuse the documented ones)
    best_hyperparameters_list = []
    for i, train_file in enumerate(chiqseq_train_files):
        if args.moe_only:
            break  # skip Optuna + expert training; retrain only the MoE below
        tf_name = get_tf_name(train_file)
        if args.use_saved_hyperparams:
            best_hyperparameters = torch.load(f"{save_path}/hyperparams/{tf_name}.pth", map_location=device)
            console.print(f"[{tf_name}] reusing saved hyperparameters: {best_hyperparameters}")
            best_hyperparameters_list.append(best_hyperparameters)
            continue
        alldataset = ChipDataLoader(train_file).load_data(shuffle=_train_shuffle)
        train_data, valid_data = train_test_split(alldataset, test_size=0.2, stratify=[label for _, label in alldataset], random_state=seed)
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
        import optuna  # imported lazily so --use_saved_hyperparams needs no optuna install

        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: objective(trial, train_loader, valid_loader),
            n_trials=10,
            gc_after_trial=True,
        )
        best_hyperparameters = study.best_trial.params
        torch.save(
            best_hyperparameters,
            f"{save_path}/hyperparams/{tf_name}.pth",
        )
        best_hyperparameters_list.append(best_hyperparameters)
        console.print(best_hyperparameters)

    # -------------------------------------------------------------------------------------
    # Train Individual Expert Models using Best Hyperparameters
    # -------------------------------------------------------------------------------------
    for i, (train_file, best_hyperparameters) in enumerate(
        zip(chiqseq_train_files, best_hyperparameters_list)
    ):
        alldataset = ChipDataLoader(train_file).load_data(shuffle=_train_shuffle)
        train_data, valid_data = train_test_split(alldataset, test_size=0.2, random_state=seed)
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
        data = data_loader.load_data(shuffle=_train_shuffle)
        combined_data.extend(data)

    # Split data into training and validation sets
    train_data, valid_data = train_test_split(combined_data, test_size=0.2, random_state=seed)
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
        torch.load(f"{save_path}/hyperparams/{get_tf_name(train_file)}.pth", map_location=device)
        for train_file in chiqseq_train_files
    ]
    models = [load_model(path, config) for path, config in zip(model_paths, configs)]
    moe_model = MixtureOfExperts(num_experts=len(models), embedding_size=32).to(device)

    # Train MixtureOfExperts model.
    # lr=0.01 matches the value documented in the manuscript (Section 3.4.2/3.4.3).
    # NOTE (ACTION NEEDED): the originally released moe_model.pth was trained with
    # the previously hard-coded lr=0.1; retrain once with this documented setting
    # (and the fixed validation-based early stopping) for the camera-ready.
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