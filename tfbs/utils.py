import os
import random

import numpy as np
import torch


def set_seed(seed):
    """Seed all RNGs for reproducible training/evaluation.

    Seeds Python ``random``, NumPy, and PyTorch (CPU + CUDA) and forces
    cuDNN into deterministic mode. Call this at the start of any entry
    point (and per-trial with ``base_seed + trial_idx``) so that data
    splits, weight initialisation, and the manual Bernoulli dropout in
    ``model.ConvNet`` are reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, auc_score):
        if self.best_score is None or auc_score > self.best_score + self.min_delta:
            self.best_score = auc_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False


def load_files_from_folder(folder_path):
    return [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
    ]


def get_tf_name(file_path):
    return os.path.basename(file_path).split("_")[0]
