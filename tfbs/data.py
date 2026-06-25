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

    def load_data_with_seq(self, shuffle=True, label_index=3):
        """Like ``load_data`` but each element is ``[padded_onehot, [label], raw_seq]``.

        Row order is identical to ``load_data`` (synthetic dinucleotide-shuffle
        negative -- when ``shuffle=True`` -- immediately before its positive). The raw
        string lets a *heterogeneous* expert zoo embed the exact same sequences the
        one-hot CNN experts see: BPE/k-mer foundation models (DNABERT-2) tokenize
        ``raw_seq`` while the CNNs consume ``padded_onehot``, kept aligned by index.
        Call ``utils.set_seed`` first for a reproducible synthetic-negative set.
        """
        data_set = []
        with gzip.open(self.filename, "rt") as file:
            next(file)  # Skip header if necessary
            reader = csv.reader(file, delimiter="\t")
            for row in reader:
                sequence = row[2]
                label = int(row[label_index]) if label_index is not None else 0
                if shuffle:
                    shuffled_sequence = self.dinucshuffle(sequence)
                    data_set.append([self.seqtopad(shuffled_sequence), [0], shuffled_sequence])
                data_set.append([self.seqtopad(sequence), [label], sequence])
        return data_set


def revcomp(seq):
    """Reverse-complement of an ACGT string (non-ACGT chars pass through unchanged).
    Used for reverse-complement test-time averaging of expert embeddings (a near-free
    OOD gain in TFBS) and train-only RC augmentation."""
    comp = {"A": "T", "C": "G", "G": "C", "T": "A",
            "a": "t", "c": "g", "g": "c", "t": "a", "N": "N", "n": "n"}
    return "".join(comp.get(b, b) for b in reversed(seq))


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
