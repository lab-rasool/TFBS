import csv
import gzip

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingGenerator:
    def __init__(self, model_path, tokenizer_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def tokenize_dna_sequence(self, sequence, pad_char="N", kmer_size=6):
        # chunk the dna sequence into 6-mers, padding the last one if necessary
        chunks = [
            sequence[i : i + kmer_size] for i in range(0, len(sequence), kmer_size)
        ]
        if len(chunks[-1]) < kmer_size:
            # Pad the last chunk if it is shorter than kmer_size
            chunks[-1] += pad_char * (kmer_size - len(chunks[-1]))

        # tokenize the chunks
        tokens = [self.tokenizer(chunk, add_special_tokens=False) for chunk in chunks]
        # flatten the tokens
        tokens = [token for chunk in tokens for token in chunk["input_ids"]]
        return tokens

    def generate_embeddings(self, sequence):
        inputs_dna = self.tokenize_dna_sequence(sequence)
        inputs_dna = torch.tensor(inputs_dna).unsqueeze(0)
        inputs_dna = inputs_dna.to(self.device)
        with torch.no_grad():
            hidden_states_dna = self.model(inputs_dna).last_hidden_state
        return hidden_states_dna.cpu().numpy()


def create_embeddings_file(data, embedding_generator, output_file):
    if data is None:
        print("No data to process.")
        return
    all_data = []
    for sequence, label in tqdm(data):
        embeddings = embedding_generator.generate_embeddings(sequence)
        embeddings_bytes = embeddings.tobytes()  # Convert array to bytes
        all_data.append([sequence, label, embeddings_bytes])

    # Create DataFrame
    df = pd.DataFrame(all_data, columns=["sequence", "label", "embedding"])

    # Ensure consistent data types
    df["label"] = df["label"].astype(int)

    # Writing to Parquet
    df.to_parquet(output_file)
    print(f"Data written to {output_file}.")


def read_embeddings_file(input_file):
    df = pd.read_parquet(input_file)
    # [1, 17, 768]
    df["embedding"] = df["embedding"].apply(
        lambda x: np.frombuffer(x, dtype=np.float32).reshape(1, 17, 768)
    )
    return df


class ChipDataLoader:
    def __init__(self, filename, motiflen=24):
        self.filename = filename
        self.motiflen = motiflen

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
                if shuffle:
                    shuffled_sequence = self.dinucshuffle(sequence)
                    data_set.append([shuffled_sequence, 0])
                data_set.append([sequence, label])
        return data_set


def generate_embeddings_file(path_to_data, output_file, shuffle=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding_generator = EmbeddingGenerator(
        "zhihan1996/DNA_bert_6", "zhihan1996/DNA_bert_6", device
    )

    data_loader = ChipDataLoader(path_to_data)
    data = data_loader.load_data(shuffle=True)
    create_embeddings_file(data, embedding_generator, output_file)


if __name__ == "__main__":
    train_output_file = "./data/train.parquet"
    test_output_file = "./data/test.parquet"

    chipseq_train = "/mnt/f/Projects/GenomicAttributions/data/encode/GATA3_SH-SY5Y_GATA3_(SC-269)_USC_AC.seq.gz"
    chipseq_test = "/mnt/f/Projects/GenomicAttributions/data/encode/GATA3_SH-SY5Y_GATA3_(SC-269)_USC_B.seq.gz"

    generate_embeddings_file(chipseq_train, train_output_file)
    generate_embeddings_file(chipseq_test, test_output_file)
