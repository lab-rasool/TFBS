import csv
import gzip
import random

import numpy as np
import requests
from Bio import SeqIO
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def fetch_bed(accession):
    url = f"{ENCODE_BASE_URL}/files/{accession}/?format=json"
    print(f"Requesting URL: {url}")
    response = requests.get(url)

    if response.status_code == 200:
        metadata = response.json()
        download_url = f"{ENCODE_BASE_URL}{metadata['@id']}@@download/{metadata['href'].split('/')[-1]}"
        print(f"Downloading BED file from: {download_url}")

        file_response = requests.get(download_url)
        if file_response.status_code == 200:
            bed_filename = f"{accession}.bed"
            with open(bed_filename, "wb") as f:
                f.write(gzip.decompress(file_response.content))
            print(f"File downloaded and decompressed: {bed_filename}")
            return bed_filename
        else:
            print("Error downloading file")
    else:
        print(f"Error fetching BED file: {response.status_code} - {response.text}")
    return None


def extract_sequences(
    bed_file_path, fasta_file_path, output_file_path, fixed_length=101
):
    # Load the entire FASTA file into memory
    fasta_sequences = {
        record.id: record.seq for record in SeqIO.parse(fasta_file_path, "fasta")
    }

    with open(bed_file_path, "r") as bed_file, open(
        output_file_path, "w"
    ) as output_file:
        output_file.write("FoldID\tEventID\tseq\tBound\n")
        seq_counter = 1

        for line in tqdm(bed_file, leave=False):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])

            if chrom in fasta_sequences:
                sequence = fasta_sequences[chrom][start:end]
                sequence_length = end - start

                # If sequence is shorter than desired, pad it
                if sequence_length < fixed_length:
                    padding_length = fixed_length - sequence_length
                    sequence = (
                        sequence + "N" * padding_length
                    )  # Pad with 'N' or any character you prefer
                elif sequence_length > fixed_length:
                    sequence = sequence[
                        :fixed_length
                    ]  # Trim the sequence to the fixed length

                event_id = f"seq_{seq_counter:05d}_peak"
                output_file.write(f"A\t{event_id}\t{str(sequence)}\t1\n")
                seq_counter += 1


def seqtopad(sequence, motlen, kind="DNA"):
    rows = len(sequence) + 2 * motlen - 2
    S = np.empty([rows, 4])
    bases = "ACGT"
    basesRNA = "ACGU"
    base = bases if kind == "DNA" else basesRNA
    for i in range(rows):
        for j in range(4):
            if (
                i - motlen + 1 < len(sequence)
                and sequence[i - motlen + 1] == "N"
                or i < motlen - 1
                or i > len(sequence) + motlen - 2
            ):
                S[i, j] = np.float32(0.25)
            elif sequence[i - motlen + 1] == base[j]:
                S[i, j] = np.float32(1)
            else:
                S[i, j] = np.float32(0)
    return np.transpose(S)


def dinucshuffle(sequence):
    b = [sequence[i : i + 2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d = "".join([str(x) for x in b])
    return d


def complement(seq):
    complement = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    complseq = [complement[base] for base in seq]
    return complseq


def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return "".join(complement(seq))


class Chip:
    def __init__(self, filename, motiflen=24, reverse_complemet_mode=False):
        self.file = filename
        self.motiflen = motiflen
        self.reverse_complemet_mode = reverse_complemet_mode

    def openFile(self):
        train_dataset = []

        # with gzip.open(self.file, "rt") as data:
        with open(self.file, "r") as data:
            next(data)
            reader = csv.reader(data, delimiter="\t")
            if not self.reverse_complemet_mode:
                for row in reader:
                    train_dataset.append([seqtopad(row[2], self.motiflen), [1]])
                    train_dataset.append(
                        [seqtopad(dinucshuffle(row[2]), self.motiflen), [0]]
                    )
            else:
                for row in reader:
                    train_dataset.append([seqtopad(row[2], self.motiflen), [1]])
                    train_dataset.append(
                        [seqtopad(reverse_complement(row[2]), self.motiflen), [1]]
                    )
                    train_dataset.append(
                        [seqtopad(dinucshuffle(row[2]), self.motiflen), [0]]
                    )
                    train_dataset.append(
                        [
                            seqtopad(
                                dinucshuffle(reverse_complement(row[2])), self.motiflen
                            ),
                            [0],
                        ]
                    )

        return train_dataset


class chipseq_dataset(Dataset):
    def __init__(self, sequences, labels):
        self.x_data = sequences
        self.y_data = labels
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def pad_sequence(seq, max_len):
    # Assuming seq is a numpy array
    padded_seq = np.zeros((max_len, seq.shape[1]))  # pad with zeros
    padded_seq[: seq.shape[0], : seq.shape[1]] = (
        seq  # place the original sequence in the padded sequence
    )
    return padded_seq


def load_data(data_to_load=None, batch_size=32):
    if data_to_load is None:
        raise ValueError("data_to_load is None")

    chipseq = Chip(data_to_load)
    sequences = chipseq.openFile()

    # Determine the maximum length
    max_len = max(len(seq[0]) for seq in sequences)

    # Pad sequences to have the same length
    padded_sequences = [pad_sequence(seq[0], max_len) for seq in sequences]
    labels = [seq[1] for seq in sequences]

    # Convert lists to numpy arrays
    padded_sequences = np.array(padded_sequences, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)

    dataset = chipseq_dataset(padded_sequences, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


if __name__ == "__main__":
    ENCODE_BASE_URL = "https://www.encodeproject.org"
    # BED_FILE_ACCESSION = "ENCFF891OQP"
    BED_FILE_ACCESSION = "ENCFF509ZLE"
    FASTA_FILE_PATH = "./data/GRCh38.fna"
    OUTPUT_FILE_PATH = f"./data/encode/{BED_FILE_ACCESSION}.fasta"

    # # Chip-seq Data
    # nummotif = 16  # number of motifs to discover
    # # dictionary to implement reverse-complement mode
    # dictReverse = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    # reverse_mode = False

    bed_file_path = fetch_bed(BED_FILE_ACCESSION)
    if bed_file_path:
        extract_sequences(bed_file_path, FASTA_FILE_PATH, OUTPUT_FILE_PATH, 101)
        print(f"Sequences extracted to {OUTPUT_FILE_PATH}")

    dataloader = load_data(OUTPUT_FILE_PATH)
    for x, y in dataloader:
        print(x.shape, y.shape)
        break
