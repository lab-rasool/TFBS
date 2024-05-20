import optuna
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
from utils import EarlyStopping

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
console = Console()
console.print(f"Using device: {device}")


def train(config, train_loader, valid_loader, save_path):
    model = ConvNet(config).to(device)

    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=model.learning_rate,
        momentum=model.momentum_rate,
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
            f"Epoch {epoch} | Train Loss: {loss.item():4f} | Valid AUC: {avg_auc:4f}"
        )
        if avg_auc > best_auc:
            best_auc = avg_auc
            torch.save(model.state_dict(), save_path)

        early_stopping(avg_auc)

        if early_stopping.early_stop:
            console.print("Early stopping")
            break

    console.print("Training complete. Best AUC:", best_auc)


def evaluate_all_models_on_all_tests(model_paths, configs, test_loaders, save_path):
    results = {}
    for model_idx, model_path in enumerate(model_paths):
        checkpoint = torch.load(model_path)
        configs[model_idx]["nummotif"] = 16
        configs[model_idx]["motiflen"] = 24
        model = ConvNet(configs[model_idx]).to(device)
        model.load_state_dict(checkpoint)

        for test_idx, test_loader in enumerate(test_loaders):
            total_preds, total_targets = [], []

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    pred_sig = torch.sigmoid(output)
                    total_preds.extend(pred_sig.cpu().numpy())
                    total_targets.extend(target.cpu().numpy())

            auc_score = roc_auc_score(total_targets, total_preds)
            results[f"Model_{model_idx}_Test_{test_idx}"] = auc_score

            # Plot ROC curve for this particular model-test pair
            fpr, tpr, _ = metrics.roc_curve(total_targets, total_preds)
            plt.figure()
            plt.plot(fpr, tpr, label=f"Model {model_idx} (AUC = {auc_score:.2f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve: Model {model_idx} on Test {test_idx}")
            plt.legend(loc="lower right")
            plt.savefig(f"{save_path}_model_{model_idx}_test_{test_idx}_roc.png")

    return results


def load_model(model_path, config):
    config["nummotif"] = 16
    config["motiflen"] = 24
    model = ConvNet(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


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
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

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


def evaluate_moe_on_all_tests(models, moe_model, test_loaders):
    results = {}
    for test_idx, test_loader in enumerate(test_loaders):
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

        # Plot ROC curve for this particular model-test pair
        fpr, tpr, _ = metrics.roc_curve(total_targets, total_preds)
        plt.figure()
        plt.plot(fpr, tpr, label=f"Test {test_idx} (AUC = {auc_score:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: MoE on Test {test_idx}")
        plt.legend(loc="lower right")
        plt.savefig(f"./models/moe/moe_test_{test_idx}_roc.png")

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
                valid_auc += roc_auc_score(
                    target.cpu().numpy(), predictions.cpu().numpy()
                )
        avg_auc = valid_auc / len(valid_loader)
        if avg_auc > best_auc:
            best_auc = avg_auc
    return best_auc


def main():
    chiqseq_train = [
        "/mnt/f/Projects/GenomicAttributions/data/encode/GATA3_SH-SY5Y_GATA3_(SC-269)_USC_AC.seq.gz",
        "/mnt/f/Projects/GenomicAttributions/data/encode/ARID3A_HepG2_ARID3A_(NB100-279)_Stanford_AC.seq.gz",
        "/mnt/f/Projects/GenomicAttributions/data/encode/FOXM1_GM12878_FOXM1_(SC-502)_HudsonAlpha_AC.seq.gz",
    ]
    chiqseq_test = [
        "/mnt/f/Projects/GenomicAttributions/data/encode/GATA3_SH-SY5Y_GATA3_(SC-269)_USC_B.seq.gz",
        "/mnt/f/Projects/GenomicAttributions/data/encode/ARID3A_HepG2_ARID3A_(NB100-279)_Stanford_B.seq.gz",
        "/mnt/f/Projects/GenomicAttributions/data/encode/FOXM1_GM12878_FOXM1_(SC-502)_HudsonAlpha_B.seq.gz",
        "/mnt/f/Projects/GenomicAttributions/data/encode/CTCF_NH-A_CTCF_Broad_B.seq.gz",
    ]

    num_experts = len(chiqseq_train)

    # --------------------------------------------------------------------------------
    # Train Individual Expert Models
    # --------------------------------------------------------------------------------
    for i in range(len(chiqseq_train)):
        alldataset = ChipDataLoader(chiqseq_train[i]).load_data()
        train_data, valid_data = train_test_split(alldataset, test_size=0.2)
        train_loader = DataLoader(
            dataset=chipseq_dataset(train_data),
            batch_size=128,
            shuffle=True,
        )
        valid_loader = DataLoader(
            dataset=chipseq_dataset(valid_data),
            batch_size=128,
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
            f"./models/moe/best_hyperparameters_{i}.pth",
        )
        best_hyperparameters = torch.load(f"./models/moe/best_hyperparameters_{i}.pth")
        console.print(best_hyperparameters)
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
        # 3. Train model with best hyperparameters
        train(
            config=config,
            train_loader=train_loader,
            valid_loader=valid_loader,
            save_path=f"./models/moe/best_model_{i}.pth",
        )

    # 4. Evaluate all models on all tests
    model_paths = [f"./models/moe/best_model_{i}.pth" for i in range(num_experts)]
    configs = [
        torch.load(f"./models/moe/best_hyperparameters_{i}.pth")
        for i in range(num_experts)
    ]
    test_loaders = [
        DataLoader(
            dataset=chipseq_dataset(ChipDataLoader(path).load_data()),
            batch_size=len(ChipDataLoader(path).load_data()),
            shuffle=False,
        )
        for path in chiqseq_test
    ]
    results = evaluate_all_models_on_all_tests(
        model_paths=model_paths,
        configs=configs,
        test_loaders=test_loaders,
        save_path="./models/moe/results",
    )
    console.print(results)

    # --------------------------------------------------------------------------------
    # MixtureOfExperts
    # --------------------------------------------------------------------------------

    # Load and combine data from all training files
    combined_data = []
    for path in chiqseq_train:
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
        torch.load(f"./models/moe/best_hyperparameters_{i}.pth")
        for i in range(num_experts)
    ]
    model_paths = [f"./models/moe/best_model_{i}.pth" for i in range(num_experts)]
    models = [load_model(path, config) for path, config in zip(model_paths, configs)]

    moe_model = MixtureOfExperts(num_experts=len(models), embedding_size=32).to(device)
    train_moe(models, moe_model, train_loader, valid_loader, lr=0.1)
    results = evaluate_moe_on_all_tests(models, moe_model, test_loaders)
    console.print(results)


if __name__ == "__main__":
    main()
