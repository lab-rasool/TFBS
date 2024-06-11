import argparse
import os

# import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from rich.console import Console
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from data import ChipDataLoader, chipseq_dataset
from model import ConvNet, MixtureOfExperts
from utils import EarlyStopping, get_tf_name, load_files_from_folder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
console = Console()
console.print(f"Using device: {device}")


def load_model(model_path, config):
    model = ConvNet(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def train(config, train_loader, valid_loader, save_path, train_file):
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
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target.float())
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
        console.print(
            f"Epoch {epoch} | Train Loss: {loss.item():.4f} | Valid AUC: {avg_auc:.4f} | Train File: {train_file}"
        )
        if avg_auc > best_auc:
            best_auc = avg_auc
            torch.save(model.state_dict(), save_path)

        early_stopping(avg_auc)

        if early_stopping.early_stop:
            console.print("Early stopping")
            break

    console.print(f"Training complete for {train_file}. Best AUC:", best_auc)


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


def train_moe(models, moe_model, train_loader, valid_loader, num_epochs=500, lr=0.001):
    optimizer = torch.optim.SGD(
        [param for param in moe_model.parameters() if param.requires_grad],
        lr=lr,
        momentum=0.9,
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

    console.print(
        f"Training complete. Best validation AUC: {early_stopping.best_score:.4f}"
    )


def evaluate_moe_on_all_tests(models, moe_model, test_loaders, test_files, save_path):
    results = {}
    for test_idx, (test_loader, test_file) in enumerate(zip(test_loaders, test_files)):
        total_preds, total_targets = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                embeddings = [model(data, return_embedding=True) for model in models]
                concatenated = torch.cat(embeddings, dim=1)
                output = moe_model(concatenated)
                pred_sig = torch.sigmoid(output)
                total_preds.extend(pred_sig.cpu().numpy())
                total_targets.extend(target.cpu().numpy())

        auc_score = roc_auc_score(total_targets, total_preds)
        results[f"Test_{test_idx}"] = auc_score

        # # Plot ROC curve for this particular model-test pair
        fpr, tpr, _ = metrics.roc_curve(total_targets, total_preds)

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # save to csv instead of plotting
        with open(f"{save_path}model_moe_test_{get_tf_name(test_file)}.csv", "w") as f:
            f.write("fpr,tpr\n")
            for i in range(len(fpr)):
                f.write(f"{fpr[i]},{tpr[i]}\n")

    return results


def evaluate_all_models_on_all_tests(
    model_paths, configs, test_loaders, save_path, test_files, train_files
):
    results = {}
    for model_idx, (model_path, config) in enumerate(zip(model_paths, configs)):
        checkpoint = torch.load(model_path)
        model = ConvNet(config).to(device)
        model.load_state_dict(checkpoint)

        train_file = train_files[model_idx]
        train_tf = get_tf_name(train_file)

        for test_idx, (test_loader, test_file) in enumerate(
            zip(test_loaders, test_files)
        ):
            test_tf = get_tf_name(test_file)
            total_preds, total_targets = [], []

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred_sig = torch.sigmoid(output)
                    total_preds.extend(pred_sig.cpu().numpy())
                    total_targets.extend(target.cpu().numpy())

            auc_score = roc_auc_score(total_targets, total_preds)
            results[f"Model_{train_tf}_Test_{test_tf}"] = auc_score

            fpr, tpr, _ = metrics.roc_curve(total_targets, total_preds)

            # instead of plotting, save it to csv
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            with open(f"{save_path}model_{train_tf}_test_{test_tf}.csv", "w") as f:
                f.write("fpr,tpr\n")
                for i in range(len(fpr)):
                    f.write(f"{fpr[i]},{tpr[i]}\n")

    return results


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
        required=True,
        help="Path to folder containing training ChIP-seq files",
    )
    parser.add_argument(
        "--test_folder",
        required=True,
        help="Path to folder containing testing ChIP-seq files",
    )
    parser.add_argument(
        "--save_path",
        required=True,
        help="Directory to save trained models and results",
    )
    parser.add_argument(
        "--ood_folder",
        default=None,
        help="Path to folder containing out-of-distribution ChIP-seq files",
    )

    # python main.py --train_folder "./data/train" --test_folder "./data/test" --save_path "./models/moe" --ood_folder "./data/ood"

    args = parser.parse_args()
    train_folder = args.train_folder
    test_folder = args.test_folder
    save_path = args.save_path
    ood_folder = args.ood_folder

    chiqseq_train_files = load_files_from_folder(train_folder)
    chiqseq_test_files = load_files_from_folder(test_folder)

    num_experts = len(chiqseq_train_files)

    # # --------------------------------------------------------------------------------
    # # Train Individual Expert Models
    # # --------------------------------------------------------------------------------
    # for i, train_file in enumerate(chiqseq_train_files):
    #     alldataset = ChipDataLoader(train_file).load_data()
    #     train_data, valid_data = train_test_split(alldataset, test_size=0.2)
    #     train_loader = DataLoader(
    #         dataset=chipseq_dataset(train_data),
    #         batch_size=128,
    #         shuffle=True,
    #     )
    #     valid_loader = DataLoader(
    #         dataset=chipseq_dataset(valid_data),
    #         batch_size=128,
    #         shuffle=False,
    #     )
    #     # 1. Hyperparameter optimization
    #     study = optuna.create_study(direction="maximize")
    #     study.optimize(
    #         lambda trial: objective(trial, train_loader, valid_loader),
    #         n_trials=10,
    #         gc_after_trial=True,
    #     )
    #     # 2. Save best hyperparameters
    #     best_hyperparameters = study.best_trial.params
    #     torch.save(
    #         best_hyperparameters,
    #         f"{save_path}/best_hyperparameters_{i}.pth",
    #     )
    #     best_hyperparameters = torch.load(f"{save_path}/best_hyperparameters_{i}.pth")
    #     console.print(best_hyperparameters)
    #     config = {
    #         "nummotif": 16,
    #         "motiflen": 24,
    #         "poolType": best_hyperparameters["poolType"],
    #         "sigmaConv": best_hyperparameters["sigmaConv"],
    #         "dropprob": best_hyperparameters["dropprob"],
    #         "learning_rate": best_hyperparameters["learning_rate"],
    #         "momentum_rate": best_hyperparameters["momentum_rate"],
    #         "num_features": 32,
    #         "d_model": 32,
    #         "num_heads": best_hyperparameters["num_heads"],
    #         "dim_feedforward": best_hyperparameters["dim_feedforward"],
    #         "encoder_dropout": best_hyperparameters["encoder_dropout"],
    #         "num_layers": best_hyperparameters["num_layers"],
    #     }
    #     train_tf = get_tf_name(train_file)
    #     # 3. Train model with best hyperparameters
    #     train(
    #         config=config,
    #         train_loader=train_loader,
    #         valid_loader=valid_loader,
    #         save_path=f"{save_path}/best_model_{train_tf}.pth",
    #         train_file=train_file,
    #     )

    # 4. Evaluate all models on all tests
    model_paths = [
        f"{save_path}/best_model_{get_tf_name(train_file)}.pth"
        for train_file in chiqseq_train_files
    ]
    configs = [
        torch.load(f"{save_path}/best_hyperparameters_{i}.pth")
        for i in range(num_experts)
    ]
    test_loaders = [
        DataLoader(
            dataset=chipseq_dataset(ChipDataLoader(path).load_data()),
            batch_size=len(ChipDataLoader(path).load_data()),
            shuffle=False,
        )
        for path in chiqseq_test_files
    ]
    results = evaluate_all_models_on_all_tests(
        model_paths=model_paths,
        configs=configs,
        test_loaders=test_loaders,
        save_path=f"{save_path}/results/",
        test_files=chiqseq_test_files,
        train_files=chiqseq_train_files,
    )
    console.print(results)

    # --------------------------------------------------------------------------------
    # MixtureOfExperts
    # --------------------------------------------------------------------------------

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
        dataset=chipseq_dataset(train_data), batch_size=128, shuffle=True
    )
    valid_loader = DataLoader(
        dataset=chipseq_dataset(valid_data), batch_size=128, shuffle=False
    )

    # Load pre-trained models
    configs = [
        torch.load(f"{save_path}/best_hyperparameters_{i}.pth")
        for i in range(num_experts)
    ]
    model_paths = [
        f"{save_path}/best_model_{get_tf_name(train_file)}.pth"
        for train_file in chiqseq_train_files
    ]
    models = [load_model(path, config) for path, config in zip(model_paths, configs)]
    moe_model = MixtureOfExperts(num_experts=len(models), embedding_size=32).to(device)
    train_moe(models, moe_model, train_loader, valid_loader, lr=0.01)
    results = evaluate_moe_on_all_tests(
        models,
        moe_model,
        test_loaders,
        chiqseq_test_files,
        f"{save_path}/results/",
    )
    console.print(results)

    # --------------------------------------------------------------------------------
    # Out-of-Distribution TF Evaluation for all models
    # --------------------------------------------------------------------------------
    if ood_folder is not None:
        chipseq_ood_files = load_files_from_folder(ood_folder)
        ood_loaders = [
            DataLoader(
                dataset=chipseq_dataset(ChipDataLoader(path).load_data()),
                batch_size=len(ChipDataLoader(path).load_data()),
                shuffle=False,
            )
            for path in chipseq_ood_files
        ]
        ood_results = evaluate_all_models_on_all_tests(
            model_paths=model_paths,
            configs=configs,
            test_loaders=ood_loaders,
            save_path=f"{save_path}/ood/",
            test_files=chipseq_ood_files,
            train_files=chiqseq_train_files,
        )
        console.print(ood_results)

        # Evaluate MoE on OOD data
        ood_results = evaluate_moe_on_all_tests(
            models,
            moe_model,
            ood_loaders,
            chipseq_ood_files,
            f"{save_path}/ood/",
        )
        console.print(ood_results)


def plot_results_roc(results_path):
    # Group ROC files by TF
    roc_files = [f for f in os.listdir(results_path) if f.endswith(".csv")]
    roc_tf_files = {}
    for roc_file in roc_files:
        model_name = roc_file.split("_")[1]
        test_tf = roc_file.split("_")[3]
        test_tf = test_tf.split(".")[0]
        if test_tf not in roc_tf_files:
            roc_tf_files[test_tf] = []
        roc_tf_files[test_tf].append(roc_file)

    # Create a figure with n subplots based on the number of TFs files
    # make it a square grid when possible
    n = len(roc_tf_files)
    cols = 2
    rows = n // cols + n % cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))

    # Plot ROC curves for each TF
    for idx, (tf, files) in enumerate(roc_tf_files.items()):
        ax = axes[idx // cols, idx % cols]
        for file in files:
            fpr, tpr = [], []
            with open(f"{results_path}/{file}", "r") as f:
                next(f)
                for line in f:
                    line = line.strip().split(",")
                    fpr.append(float(line[0]))
                    tpr.append(float(line[1]))
            auc = metrics.auc(fpr, tpr)
            model_name = file.split("_")[1]
            if "moe" in file:
                ax.plot(
                    fpr,
                    tpr,
                    label=f"MoE (AUC = {auc:.2f})",
                    linestyle="--",
                    linewidth=2,
                )
            else:
                ax.plot(
                    fpr,
                    tpr,
                    label=f"{model_name} (AUC = {auc:.2f})",
                    linewidth=2,
                )
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"{tf} Transcription Factor")
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{results_path}/roc_curves.png")


if __name__ == "__main__":
    # main()
    plot_results_roc("./models/moe/ood")
    plot_results_roc("./models/moe/results")
