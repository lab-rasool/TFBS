import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from data import ChipDataLoader, chipseq_dataset
from model import ConvNet, MixtureOfExperts
from utils import get_tf_name, load_files_from_folder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
console = Console()
console.print(f"Using device: {device}")


def load_model(model_path, config):
    model = ConvNet(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_model(model, data_loader, models=None):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            if models:
                embeddings = [m(data, return_embedding=True) for m in models]
                data = torch.cat(embeddings, dim=1)
            output = model(data)
            preds = torch.sigmoid(output)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    return np.array(all_targets), np.array(all_preds)


def calculate_metrics(targets, preds):
    preds_bin = (preds > 0.5).astype(int)
    auc_score = roc_auc_score(targets, preds)
    precision = precision_score(targets, preds_bin, zero_division=0)
    recall = recall_score(targets, preds_bin, zero_division=0)
    f1 = f1_score(targets, preds_bin, zero_division=0)
    return auc_score, precision, recall, f1


def compare_models(expert_metrics, moe_metrics):
    metrics = ["AUC", "Precision", "Recall", "F1"]
    results = {}

    for i, metric in enumerate(metrics):
        expert_scores = expert_metrics[:, i]
        moe_scores = moe_metrics[:, i]

        t_stat, t_p_value = ttest_rel(expert_scores, moe_scores)
        wilcoxon_stat, wilcoxon_p_value = wilcoxon(expert_scores, moe_scores)

        results[metric] = {
            "Expert Mean": np.mean(expert_scores),
            "MoE Mean": np.mean(moe_scores),
            "T-test p-value": t_p_value,
            "Wilcoxon p-value": wilcoxon_p_value,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare ConvNet expert models and MoE model on ChIP-seq data."
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
        help="Directory to load trained models and results",
    )

    args = parser.parse_args()
    train_folder = args.train_folder
    test_folder = args.test_folder
    save_path = args.save_path

    chiqseq_train_files = load_files_from_folder(train_folder)
    chiqseq_test_files = load_files_from_folder(test_folder)
    num_experts = len(chiqseq_train_files)

    # Load test data
    test_loaders = [
        DataLoader(
            dataset=chipseq_dataset(ChipDataLoader(path).load_data()),
            batch_size=len(ChipDataLoader(path).load_data()),
            shuffle=False,
        )
        for path in chiqseq_test_files
    ]

    # Evaluate expert models
    expert_metrics = []
    embedding_size = 0
    for i, train_file in enumerate(chiqseq_train_files):
        config = torch.load(f"{save_path}/best_hyperparameters_{i}.pth")
        model_path = f"{save_path}/best_model_{get_tf_name(train_file)}.pth"
        model = load_model(model_path, config)

        # Assuming each expert model outputs embeddings of the same size
        with torch.no_grad():
            sample_data, _ = next(iter(test_loaders[0]))
            sample_data = sample_data.to(device)
            embedding_size = model(sample_data, return_embedding=True).shape[1]

        model_metrics = []
        for test_loader in test_loaders:
            targets, preds = evaluate_model(model, test_loader)
            auc, precision, recall, f1 = calculate_metrics(targets, preds)
            model_metrics.append([auc, precision, recall, f1])

        expert_metrics.append(model_metrics)

    expert_metrics = np.array(expert_metrics)

    # Load MoE model
    moe_model = MixtureOfExperts(
        num_experts=num_experts, embedding_size=embedding_size
    ).to(device)
    moe_model.load_state_dict(torch.load(f"{save_path}/moe_model.pth"))

    # Load expert models for generating embeddings
    expert_models = [
        load_model(
            f"{save_path}/best_model_{get_tf_name(train_file)}.pth",
            torch.load(f"{save_path}/best_hyperparameters_{i}.pth"),
        )
        for i, train_file in enumerate(chiqseq_train_files)
    ]

    # Evaluate MoE model
    moe_metrics = []
    for test_loader in test_loaders:
        targets, preds = evaluate_model(moe_model, test_loader, models=expert_models)
        auc, precision, recall, f1 = calculate_metrics(targets, preds)
        moe_metrics.append([auc, precision, recall, f1])

    moe_metrics = np.array(moe_metrics)

    # Compare models
    comparison_results = compare_models(expert_metrics.mean(axis=0), moe_metrics)
    console.print(comparison_results)


if __name__ == "__main__":
    main()
