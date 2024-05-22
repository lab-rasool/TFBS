import os


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
