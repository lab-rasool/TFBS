"""Canonical evaluation of the ConvNet experts + MoE on the real 50/50 held-out
sets (``_B`` files, ``shuffle=False``), with inference dropout disabled and a paired
percentile bootstrap. Writes ``evaluation_summary.csv``, ``evaluation_results.json``,
``bootstrap_paired.csv`` and ROC curves to ``results/``.

The non-canonical legacy MC-dropout protocol (25/75 synthetic negatives, dropout on)
was removed in the cleanup; recover it from git history if needed.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rich.console import Console
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from tfbs.data import ChipDataLoader, chipseq_dataset
from tfbs.metrics import paired_bootstrap
from tfbs.models import ConvNet, MixtureOfExperts
from tfbs.utils import get_tf_name, load_files_from_folder, set_seed

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
console = Console()


def load_model(model_path, config):
    """Load a trained ConvNet expert (dropout off via eval / training=False)."""
    model = ConvNet(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def get_tf_from_filename(filename):
    return os.path.basename(filename).split("_")[0]


def plot_roc_curves(results, dataset_type, save_dir="./results"):
    """
    Plot ROC curves with confidence intervals for each TF in subplots.
    """
    # Get unique TFs across all datasets
    all_tfs = set()
    for model_results in results.values():
        datasets = model_results[dataset_type]
        for tf_results in datasets:
            if tf_results:  # Skip empty results
                all_tfs.add(tf_results[0]['dataset_tf'])

    # Calculate number of rows and columns for subplots
    n_tfs = len(all_tfs)
    n_cols = min(3, n_tfs)  # Maximum 3 columns
    n_rows = (n_tfs + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Color scheme for different models
    unique_models = [model_name for model_name in results.keys() if not model_name.startswith('expert_')]
    unique_models.extend(sorted([model_name for model_name in results.keys() if model_name.startswith('expert_')]))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_models)))
    color_dict = dict(zip(unique_models, colors))

    # Plot for each TF
    for idx, tf in enumerate(sorted(all_tfs)):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        for model_name, model_results in results.items():
            datasets = model_results[dataset_type]

            # Find the results for this TF
            tf_results = None
            for dataset in datasets:
                if dataset and dataset[0]['dataset_tf'] == tf:
                    tf_results = dataset
                    break

            if tf_results:
                all_tprs = []

                # Collect ROC curves for all trials
                for trial in tf_results:
                    fpr, tpr, _ = roc_curve(trial['targets'], trial['predictions'])
                    interp_tpr = np.interp(np.linspace(0, 1, 100), fpr, tpr)
                    all_tprs.append(interp_tpr)

                # Calculate mean and std of TPR
                mean_tpr = np.mean(all_tprs, axis=0)
                std_tpr = np.std(all_tprs, axis=0)
                fpr_grid = np.linspace(0, 1, 100)
                mean_auc = np.mean([trial['auc'] for trial in tf_results])
                std_auc = np.std([trial['auc'] for trial in tf_results])

                # Set line style based on model type
                linestyle = '--' if model_name == 'moe' else '-'
                color = color_dict[model_name]

                # Plot mean ROC curve
                label = f"{model_name} ({mean_auc:.3f} ± {std_auc:.3f})"
                ax.plot(fpr_grid, mean_tpr, color=color, linestyle=linestyle, label=label)

                # Plot confidence interval
                ax.fill_between(
                    fpr_grid,
                    mean_tpr - std_tpr,
                    mean_tpr + std_tpr,
                    color=color,
                    alpha=0.1
                )

        # Add random baseline
        ax.plot([0, 1], [0, 1], 'k:', label='Random' if idx == 0 else None)

        # Customize subplot
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        if col == 0:
            ax.set_ylabel('True Positive Rate')
        if row == n_rows - 1:
            ax.set_xlabel('False Positive Rate')
        ax.set_title(f'{tf}')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize='small')

    # Remove empty subplots if any
    for idx in range(n_tfs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])

    # Save plot
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"roc_curves_{dataset_type}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def run_rigorous_evaluation(expert_models, expert_tfs, moe_model, test_files, ood_files,
                            n_boot=1000, base_seed=42, save_dir="./results"):
    """Rigorous, reproducible evaluation on the real 50/50 held-out sets.

    Loads each ``_B`` file with ``shuffle=False`` (the curated 500 positive / 500
    negative set, with NO synthetic dinucleotide-shuffle negatives added), runs
    every model with **inference dropout disabled** (deterministic predictions),
    and quantifies uncertainty with a paired bootstrap: B resamples of the test
    set using a *common* index across all four models per replicate, so the
    MoE-vs-expert comparisons are paired.

    Writes ``evaluation_summary.csv`` (per-model bootstrap AUCs + 95% percentile
    CIs), ``evaluation_results.json`` (deterministic predictions/targets), and
    ``bootstrap_paired.csv`` (paired MoE-minus-expert differences + CIs).
    """
    os.makedirs(save_dir, exist_ok=True)
    datasets = [("in_distribution", p) for p in test_files] + \
               [("out_of_distribution", p) for p in ood_files]
    model_names = ["moe"] + [f"expert_{tf}" for tf in expert_tfs]

    results = {m: {"in_distribution": [], "out_of_distribution": []} for m in model_names}
    summary_rows, paired_rows = [], []

    for di, (data_type, path) in enumerate(datasets):
        tf = get_tf_from_filename(path)
        data = ChipDataLoader(path).load_data(shuffle=False)  # real 50/50, no synthetic negs
        loader = DataLoader(chipseq_dataset(data), batch_size=len(data), shuffle=False)
        xb, yb = next(iter(loader))
        xb = xb.to(device)
        y = yb.cpu().numpy().ravel().astype(int)

        with torch.no_grad():
            embs = [e(xb, training=False, return_embedding=True) for e in expert_models]
            cat = torch.cat(embs, dim=1)
            preds = {"moe": torch.sigmoid(moe_model(cat)).cpu().numpy().ravel()}
            for tf_name, e in zip(expert_tfs, expert_models):
                preds[f"expert_{tf_name}"] = torch.sigmoid(e(xb, training=False)).cpu().numpy().ravel()

        for m in model_names:
            results[m][data_type].append([{
                "dataset_tf": tf, "auc": float(roc_auc_score(y, preds[m])),
                "predictions": preds[m].tolist(), "targets": y.tolist(),
            }])

        boot = paired_bootstrap(preds, y, base_seed + di, n_boot)

        for m in model_names:
            arr = boot[m]
            lo, hi = np.percentile(arr, [2.5, 97.5])
            row = {"model": m, "data_type": data_type, "dataset_tf": tf,
                   "point_auc": float(roc_auc_score(y, preds[m])),
                   "mean_auc": float(arr.mean()), "std_auc": float(arr.std(ddof=1)),
                   "sem_auc": float(arr.std(ddof=1) / np.sqrt(n_boot)),
                   "ci95_low": float(lo), "ci95_high": float(hi)}
            for b in range(n_boot):
                row[f"boot_{b + 1}_auc"] = float(arr[b])
            summary_rows.append(row)

        for tf_name in expert_tfs:
            d = boot["moe"] - boot[f"expert_{tf_name}"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
            paired_rows.append({"data_type": data_type, "dataset_tf": tf,
                                "comparison": f"MoE - {tf_name}", "mean_diff": float(d.mean()),
                                "ci95_low": float(lo), "ci95_high": float(hi),
                                "p_value": p, "significant": bool(lo > 0 or hi < 0)})

        means = {m: float(boot[m].mean()) for m in model_names}
        best = max(means, key=means.get)
        console.print(f"[rigorous] {data_type:>20} {tf:<8} best={best:<14} "
                      f"MoE={means['moe']:.4f} CI[{summary_rows[-4]['ci95_low']:.4f},"
                      f"{summary_rows[-4]['ci95_high']:.4f}]")

    # Save predictions (deterministic) as JSON
    with open(os.path.join(save_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f)
    # Save summary CSV with bootstrap columns
    summary_df = pd.DataFrame(summary_rows)
    base_cols = ["model", "data_type", "dataset_tf", "point_auc", "mean_auc",
                 "std_auc", "sem_auc", "ci95_low", "ci95_high"]
    boot_cols = sorted([c for c in summary_df.columns if c.startswith("boot_")],
                       key=lambda x: int(x.split("_")[1]))
    summary_df = summary_df[base_cols + boot_cols]
    summary_df.to_csv(os.path.join(save_dir, "evaluation_summary.csv"), index=False)
    pd.DataFrame(paired_rows).to_csv(os.path.join(save_dir, "bootstrap_paired.csv"), index=False)
    # ROC curves (deterministic single curve per model/dataset)
    plot_roc_curves(results, "in_distribution", save_dir)
    plot_roc_curves(results, "out_of_distribution", save_dir)
    console.print(f"[rigorous] wrote evaluation_summary.csv, evaluation_results.json, "
                  f"bootstrap_paired.csv to {save_dir}")
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate expert + MoE models on the rigorous 50/50 paired-bootstrap protocol."
    )
    parser.add_argument("--test_folder", default="./data/test")
    parser.add_argument("--ood_folder", default="./data/ood")
    parser.add_argument("--save_path", default="./models")
    parser.add_argument("--protocol", choices=["rigorous"], default="rigorous",
                        help="rigorous = real 50/50 sets, dropout off, paired bootstrap (canonical).")
    parser.add_argument("--n_boot", type=int, default=1000, help="bootstrap replicates")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Per-dataset bootstrap seed is base_seed + dataset_idx.")
    args = parser.parse_args()

    set_seed(args.base_seed)
    test_files = load_files_from_folder(args.test_folder)
    ood_files = load_files_from_folder(args.ood_folder)

    # Load models (expert order follows os.listdir of the test folder)
    expert_tfs = [get_tf_name(f) for f in test_files]
    configs = [torch.load(f"{args.save_path}/hyperparams/{tf}.pth", map_location=device)
               for tf in expert_tfs]
    model_paths = [f"{args.save_path}/experts/{tf}.pth" for tf in expert_tfs]
    expert_models = [load_model(p, c) for p, c in zip(model_paths, configs)]

    moe_model = MixtureOfExperts(num_experts=len(expert_models), embedding_size=32).to(device)
    moe_model.load_state_dict(torch.load(f"{args.save_path}/moe/moe_model.pth", map_location=device))
    moe_model.eval()

    run_rigorous_evaluation(
        expert_models, expert_tfs, moe_model, test_files, ood_files,
        n_boot=args.n_boot, base_seed=args.base_seed, save_dir="./results",
    )


if __name__ == "__main__":
    main()
