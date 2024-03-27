import csv
import gzip

import numpy as np
import torch
from torch.utils.data import Dataset


class ChipDataLoader:
    def __init__(self, filename, motiflen=24):
        self.filename = filename
        self.motiflen = motiflen

    def seqtopad(self, sequence, kind="DNA"):
        rows = len(sequence) + 2 * self.motiflen - 2
        S = np.zeros([rows, 4], dtype=np.float32)
        base = "ACGT" if kind == "DNA" else "ACGU"
        for i in range(rows):
            if i < self.motiflen - 1 or i >= len(sequence) + self.motiflen - 1:
                S[i] = 0.25
            elif sequence[i - self.motiflen + 1] in base:
                S[i, base.index(sequence[i - self.motiflen + 1])] = 1
        return S.T

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
                padded_sequence = self.seqtopad(sequence)
                if shuffle:
                    shuffled_sequence = self.seqtopad(self.dinucshuffle(sequence))
                    data_set.append([shuffled_sequence, [0]])
                data_set.append([padded_sequence, [label]])
        return data_set


class chipseq_dataset(Dataset):
    def __init__(self, xy=None):
        self.x_data = np.asarray([el[0] for el in xy], dtype=np.float32)
        self.y_data = np.asarray([el[1] for el in xy], dtype=np.float32)
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
