# import os

# import numpy as np
# import pandas as pd
# import torch
# from rich.console import Console
# from rich.table import Table
# from scipy.stats import f_oneway, ttest_rel
# from sklearn import metrics
# from sklearn.metrics import roc_auc_score
# from torch.utils.data import DataLoader

# from data import ChipDataLoader, chipseq_dataset
# from model import ConvNet, MixtureOfExperts
# from utils import get_tf_name, load_files_from_folder

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# console = Console()


# def load_model(model_path, config):
#     model = ConvNet(config).to(device)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model


# def evaluate_model(model, data_loader):
#     total_preds, total_targets = [], []
#     with torch.no_grad():
#         for data, target in data_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             pred_sig = torch.sigmoid(output)
#             total_preds.extend(pred_sig.cpu().numpy())
#             total_targets.extend(target.cpu().numpy())
#     auc_score = roc_auc_score(total_targets, total_preds)
#     return auc_score, total_preds, total_targets


# def bootstrap_auc(predictions, targets, n_bootstraps=1000):
#     rng = np.random.RandomState(42)
#     bootstrapped_scores = []
#     for _ in range(n_bootstraps):
#         indices = rng.randint(0, len(predictions), len(predictions))
#         if len(np.unique(targets[indices])) < 2:
#             continue
#         score = roc_auc_score(targets[indices], predictions[indices])
#         bootstrapped_scores.append(score)
#     return np.array(bootstrapped_scores)


# def evaluate_all_models_on_all_tests(
#     model_paths, configs, test_loaders, save_path, test_files, train_files
# ):
#     results = {}
#     num_folds = 5

#     for model_idx, model_path in enumerate(model_paths):
#         config = configs[model_idx]
#         checkpoint = torch.load(model_path)
#         model = ConvNet(config).to(device)
#         model.load_state_dict(checkpoint)

#         # Determine the TF and fold
#         train_file_idx = model_idx // num_folds
#         fold_idx = model_idx % num_folds
#         train_file = train_files[train_file_idx]
#         train_tf = get_tf_name(train_file)

#         for test_idx, (test_loader, test_file) in enumerate(
#             zip(test_loaders, test_files)
#         ):
#             test_tf = get_tf_name(test_file)
#             total_preds, total_targets = [], []

#             with torch.no_grad():
#                 for data, target in test_loader:
#                     data, target = data.to(device), target.to(device)
#                     output = model(data)
#                     pred_sig = torch.sigmoid(output)
#                     total_preds.extend(pred_sig.cpu().numpy())
#                     total_targets.extend(target.cpu().numpy())

#             auc_score = roc_auc_score(total_targets, total_preds)
#             key = f"Model_{train_tf}_Fold_{fold_idx}_Test_{test_tf}"
#             results[key] = auc_score

#             fpr, tpr, _ = metrics.roc_curve(total_targets, total_preds)

#             if not os.path.exists(save_path):
#                 os.makedirs(save_path, exist_ok=True)

#             with open(
#                 f"{save_path}/model_{train_tf}_fold_{fold_idx}_test_{test_tf}.csv", "w"
#             ) as f:
#                 f.write("fpr,tpr\n")
#                 for i in range(len(fpr)):
#                     f.write(f"{fpr[i]},{tpr[i]}\n")

#     # Aggregating results across folds
#     aggregated_results = {}
#     for key in results:
#         model_tf, _, test_tf = (
#             key.split("_")[1],
#             key.split("_")[3],
#             key.split("_")[5],
#         )
#         agg_key = f"Model_{model_tf}_Test_{test_tf}"
#         if agg_key not in aggregated_results:
#             aggregated_results[agg_key] = []
#         aggregated_results[agg_key].append(results[key])

#     averaged_results = {
#         key: np.mean(value) for key, value in aggregated_results.items()
#     }
#     return averaged_results


# def evaluate_experts_and_moe(models, moe_model, test_loader, save_path, n_trials=10):
#     expert_aucs = {i: [] for i in range(len(models))}
#     moe_aucs = []
#     for _ in range(n_trials):
#         for i, model in enumerate(models):
#             auc, _, _ = evaluate_model(model, test_loader)
#             expert_aucs[i].append(auc)
#         results = evaluate_moe_on_all_tests(
#             models, moe_model, [test_loader], ["temp"], save_path
#         )
#         moe_auc = results["Test_0"]
#         moe_aucs.append(moe_auc)
#     return expert_aucs, moe_aucs


# def evaluate_moe_on_all_tests(models, moe_model, test_loaders, test_files, save_path):
#     results = {}
#     for test_idx, (test_loader, test_file) in enumerate(zip(test_loaders, test_files)):
#         total_preds, total_targets = [], []
#         with torch.no_grad():
#             for data, target in test_loader:
#                 data, target = data.to(device), target.to(device)
#                 embeddings = [model(data, return_embedding=True) for model in models]
#                 concatenated = torch.cat(embeddings, dim=1)
#                 output = moe_model(concatenated)
#                 pred_sig = torch.sigmoid(output)
#                 total_preds.extend(pred_sig.cpu().numpy())
#                 total_targets.extend(target.cpu().numpy())

#         auc_score = roc_auc_score(total_targets, total_preds)
#         results[f"Test_{test_idx}"] = auc_score

#         fpr, tpr, _ = metrics.roc_curve(total_targets, total_preds)

#         if save_path and not os.path.exists(save_path):
#             os.makedirs(save_path, exist_ok=True)

#         with open(f"{save_path}/model_moe_test_{get_tf_name(test_file)}.csv", "w") as f:
#             f.write("fpr,tpr\n")
#             for i in range(len(fpr)):
#                 f.write(f"{fpr[i]},{tpr[i]}\n")

#     return results


# def analyze_performance(expert_aucs, moe_aucs, tf_names):
#     results = {}
#     moe_mean_auc = np.mean(moe_aucs)
#     moe_std_auc = np.std(moe_aucs)

#     for i, (aucs, tf_name) in enumerate(zip(expert_aucs.values(), tf_names)):
#         mean_auc = np.mean(aucs)
#         std_auc = np.std(aucs)

#         results[f"Expert_{tf_name}"] = {
#             "mean_auc": mean_auc,
#             "std_auc": std_auc,
#         }

#     results["MoE"] = {
#         "mean_auc": moe_mean_auc,
#         "std_auc": moe_std_auc,
#     }

#     return results


# def paired_t_test(expert_aucs, moe_aucs):
#     results = {}
#     for i, aucs in expert_aucs.items():
#         t_stat, p_value = ttest_rel(aucs, moe_aucs)
#         results[f"Expert_{i} vs MoE"] = {"t_stat": t_stat, "p_value": p_value}
#     return results


# def anova_test(expert_aucs, moe_aucs):
#     all_aucs = list(expert_aucs.values()) + [moe_aucs]
#     f_stat, p_value = f_oneway(*all_aucs)
#     return f_stat, p_value


# def main():
#     train_folder = "./data/train"
#     save_path = "./models/moe"
#     ood_folder = "./data/ood"
#     n_trials = 10

#     # DATA
#     chiqseq_train_files = load_files_from_folder(train_folder)
#     chiqseq_ood_files = load_files_from_folder(ood_folder)
#     tf_names = [get_tf_name(train_file) for train_file in chiqseq_train_files]
#     ood_loaders = [
#         DataLoader(
#             dataset=chipseq_dataset(ChipDataLoader(path).load_data()),
#             batch_size=len(ChipDataLoader(path).load_data()),
#             shuffle=False,
#         )
#         for path in chiqseq_ood_files
#     ]

#     model_paths = [f"{save_path}/best_model_{tf_name}.pth" for tf_name in tf_names]
#     configs = [
#         torch.load(f"{save_path}/best_config_{tf_name}.pth") for tf_name in tf_names
#     ]

#     models = [load_model(path, config) for path, config in zip(model_paths, configs)]
#     moe_model = MixtureOfExperts(num_experts=len(models), embedding_size=32).to(device)
#     moe_model.load_state_dict(torch.load(f"{save_path}/moe_model.pth"))
#     moe_model.eval()

#     console.print("Evaluating models and MoE on Out-of-Distribution data...")

#     for idx, ood_loader in enumerate(ood_loaders):
#         expert_aucs, moe_aucs = evaluate_experts_and_moe(
#             models, moe_model, ood_loader, save_path, n_trials=n_trials
#         )
#         results = analyze_performance(expert_aucs, moe_aucs, tf_names)

#         table = Table(
#             title=f"\n{get_tf_name(chiqseq_ood_files[idx])} [{idx+1}/{len(ood_loaders)}]"
#         )
#         table.add_column("Model", justify="left")
#         table.add_column("Mean AUC", justify="right")
#         table.add_column("Std AUC", justify="right")

#         max_auc = max(value["mean_auc"] for value in results.values())

#         for key, value in results.items():
#             mean_auc_str = f"{value['mean_auc']:.4f}"
#             std_auc_str = f"{value['std_auc']:.4e}"
#             if value["mean_auc"] == max_auc:
#                 table.add_row(key, mean_auc_str, std_auc_str, style="bold green")
#             else:
#                 table.add_row(key, mean_auc_str, std_auc_str)

#         console.print(table)

#         # Perform statistical tests
#         t_test_results = paired_t_test(expert_aucs, moe_aucs)
#         f_stat, p_value = anova_test(expert_aucs, moe_aucs)

#         # Display statistical test results
#         console.print("\nPaired t-test results:")
#         for comparison, stats in t_test_results.items():
#             console.print(
#                 f"{comparison}: t-statistic = {stats['t_stat']}, p-value = {stats['p_value']}"
#             )

#         console.print(f"\nANOVA test: F-statistic = {f_stat}, p-value = {p_value}")

#         ood_file_name = get_tf_name(chiqseq_ood_files[idx])
#         results_path = f"{save_path}/ood/results_{ood_file_name}.csv"
#         if not os.path.exists(f"{save_path}/ood"):
#             os.makedirs(f"{save_path}/ood", exist_ok=True)
#         with open(results_path, "w") as f:
#             f.write("Model,Mean_AUC,Std_AUC\n")
#             for key, value in results.items():
#                 f.write(f"{key},{value['mean_auc']:.4f},{value['std_auc']:.4f}\n")


# if __name__ == "__main__":
#     main()

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from scipy.stats import f_oneway, ttest_rel
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import ChipDataLoader, chipseq_dataset
from model import ConvNet, MixtureOfExperts
from utils import get_tf_name, load_files_from_folder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
console = Console()


def load_model(model_path, config):
    model = ConvNet(config).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def evaluate_model(model, data_loader):
    total_preds, total_targets = [], []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred_sig = torch.sigmoid(output)
            total_preds.extend(pred_sig.cpu().numpy())
            total_targets.extend(target.cpu().numpy())
    auc_score = roc_auc_score(total_targets, total_preds)
    return auc_score, total_preds, total_targets


def bootstrap_auc(predictions, targets, n_bootstraps=1000):
    rng = np.random.RandomState(42)
    bootstrapped_scores = []
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(predictions), len(predictions))
        if len(np.unique(targets[indices])) < 2:
            continue
        score = roc_auc_score(targets[indices], predictions[indices])
        bootstrapped_scores.append(score)
    return np.array(bootstrapped_scores)


def evaluate_all_models_on_all_tests(
    model_paths, configs, test_loaders, save_path, test_files, train_files
):
    results = {}
    num_folds = 5

    for model_idx, model_path in enumerate(model_paths):
        config = configs[model_idx]
        checkpoint = torch.load(model_path)
        model = ConvNet(config).to(device)
        model.load_state_dict(checkpoint)

        # Determine the TF and fold
        train_file_idx = model_idx // num_folds
        fold_idx = model_idx % num_folds
        train_file = train_files[train_file_idx]
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
            key = f"Model_{train_tf}_Fold_{fold_idx}_Test_{test_tf}"
            results[key] = auc_score

            fpr, tpr, _ = metrics.roc_curve(total_targets, total_preds)

            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            with open(
                f"{save_path}/model_{train_tf}_fold_{fold_idx}_test_{test_tf}.csv", "w"
            ) as f:
                f.write("fpr,tpr\n")
                for i in range(len(fpr)):
                    f.write(f"{fpr[i]},{tpr[i]}\n")

    # Aggregating results across folds
    aggregated_results = {}
    for key in results:
        model_tf, _, test_tf = (
            key.split("_")[1],
            key.split("_")[3],
            key.split("_")[5],
        )
        agg_key = f"Model_{model_tf}_Test_{test_tf}"
        if agg_key not in aggregated_results:
            aggregated_results[agg_key] = []
        aggregated_results[agg_key].append(results[key])

    averaged_results = {
        key: np.mean(value) for key, value in aggregated_results.items()
    }
    return averaged_results


def evaluate_experts_and_moe(models, moe_model, test_loader, save_path, n_trials=10):
    expert_aucs = {i: [] for i in range(len(models))}
    moe_aucs = []
    for _ in range(n_trials):
        for i, model in enumerate(models):
            auc, _, _ = evaluate_model(model, test_loader)
            expert_aucs[i].append(auc)
        results = evaluate_moe_on_all_tests(
            models, moe_model, [test_loader], ["temp"], save_path
        )
        moe_auc = results["Test_0"]
        moe_aucs.append(moe_auc)
    return expert_aucs, moe_aucs


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

        fpr, tpr, _ = metrics.roc_curve(total_targets, total_preds)

        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        with open(f"{save_path}/model_moe_test_{get_tf_name(test_file)}.csv", "w") as f:
            f.write("fpr,tpr\n")
            for i in range(len(fpr)):
                f.write(f"{fpr[i]},{tpr[i]}\n")

    return results


def analyze_performance(expert_aucs, moe_aucs, tf_names):
    results = {}
    moe_mean_auc = np.mean(moe_aucs)
    moe_std_auc = np.std(moe_aucs)

    for i, (aucs, tf_name) in enumerate(zip(expert_aucs.values(), tf_names)):
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        results[f"Expert_{tf_name}"] = {
            "mean_auc": mean_auc,
            "std_auc": std_auc,
        }

    results["MoE"] = {
        "mean_auc": moe_mean_auc,
        "std_auc": moe_std_auc,
    }

    return results


def paired_t_test(expert_aucs, moe_aucs):
    results = {}
    for i, aucs in expert_aucs.items():
        t_stat, p_value = ttest_rel(aucs, moe_aucs)
        results[f"Expert_{i} vs MoE"] = {"t_stat": t_stat, "p_value": p_value}
    return results


def anova_test(expert_aucs, moe_aucs):
    all_aucs = list(expert_aucs.values()) + [moe_aucs]
    f_stat, p_value = f_oneway(*all_aucs)
    return f_stat, p_value


def plot_results(all_results, title):
    fig, ax = plt.subplots(figsize=(12, 6))

    models = list(next(iter(all_results.values())).keys())
    # colors = plt.cm.tab20(np.linspace(0, 1, len(models)))

    # make the first 3 models shades of blue
    # make the last model red
    colors = ["blue", "dodgerblue", "deepskyblue", "red"]

    bar_width = 0.2
    bar_positions = np.arange(len(all_results))

    for i, model in enumerate(models):
        mean_aucs = [all_results[dataset][model]["mean_auc"] for dataset in all_results]
        std_aucs = [all_results[dataset][model]["std_auc"] for dataset in all_results]

        ax.bar(
            bar_positions + i * bar_width,
            mean_aucs,
            yerr=std_aucs,
            width=bar_width,
            label=model,
            capsize=3,
            color=colors[i % len(colors)],
        )

    ax.set_xticks(bar_positions + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(all_results.keys(), rotation=45, ha="right", fontsize=14)
    ax.set_ylabel("AUC", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(loc="upper left", title="Model")

    plt.tight_layout()
    # plt.show()
    plt.savefig("results.png")


def main():
    train_folder = "./data/train"
    save_path = "./models/moe"
    ood_folder = "./data/ood"
    n_trials = 10

    # DATA
    chiqseq_train_files = load_files_from_folder(train_folder)
    chiqseq_ood_files = load_files_from_folder(ood_folder)
    tf_names = [get_tf_name(train_file) for train_file in chiqseq_train_files]
    ood_loaders = [
        DataLoader(
            dataset=chipseq_dataset(ChipDataLoader(path).load_data()),
            batch_size=len(ChipDataLoader(path).load_data()),
            shuffle=False,
        )
        for path in chiqseq_ood_files
    ]

    model_paths = [f"{save_path}/best_model_{tf_name}.pth" for tf_name in tf_names]
    configs = [
        torch.load(f"{save_path}/best_config_{tf_name}.pth") for tf_name in tf_names
    ]

    models = [load_model(path, config) for path, config in zip(model_paths, configs)]
    moe_model = MixtureOfExperts(num_experts=len(models), embedding_size=32).to(device)
    moe_model.load_state_dict(torch.load(f"{save_path}/moe_model.pth"))
    moe_model.eval()

    console.print("Evaluating models and MoE on Out-of-Distribution data...")

    all_results = {}

    for idx, ood_loader in enumerate(ood_loaders):
        expert_aucs, moe_aucs = evaluate_experts_and_moe(
            models, moe_model, ood_loader, save_path, n_trials=n_trials
        )
        results = analyze_performance(expert_aucs, moe_aucs, tf_names)
        dataset_name = get_tf_name(chiqseq_ood_files[idx])
        all_results[dataset_name] = results

        table = Table(title=f"\n{dataset_name} [{idx+1}/{len(ood_loaders)}]")
        table.add_column("Model", justify="left")
        table.add_column("Mean AUC", justify="right")
        table.add_column("Std AUC", justify="right")

        max_auc = max(value["mean_auc"] for value in results.values())

        for key, value in results.items():
            mean_auc_str = f"{value['mean_auc']:.4f}"
            std_auc_str = f"{value['std_auc']:.4e}"
            if value["mean_auc"] == max_auc:
                table.add_row(key, mean_auc_str, std_auc_str, style="bold green")
            else:
                table.add_row(key, mean_auc_str, std_auc_str)

        console.print(table)

        # Perform statistical tests
        t_test_results = paired_t_test(expert_aucs, moe_aucs)
        f_stat, p_value = anova_test(expert_aucs, moe_aucs)

        # Display statistical test results
        console.print("\nPaired t-test results:")
        for comparison, stats in t_test_results.items():
            console.print(
                f"{comparison}: t-statistic = {stats['t_stat']}, p-value = {stats['p_value']}"
            )

        console.print(f"\nANOVA test: F-statistic = {f_stat}, p-value = {p_value}")

        ood_file_name = get_tf_name(chiqseq_ood_files[idx])
        results_path = f"{save_path}/ood/results_{ood_file_name}.csv"
        if not os.path.exists(f"{save_path}/ood"):
            os.makedirs(f"{save_path}/ood", exist_ok=True)
        with open(results_path, "w") as f:
            f.write("Model,Mean_AUC,Std_AUC\n")
            for key, value in results.items():
                f.write(f"{key},{value['mean_auc']:.4f},{value['std_auc']:.4f}\n")

    # Plot results
    plot_results(all_results, "Performance of Models on OOD Data")


if __name__ == "__main__":
    main()
