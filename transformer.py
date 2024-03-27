# --------------------------------------------
# TODO: Refactor
# TODO: Metrics like ROC-AUC
# TODO: Generate attribution maps of the CNN
# --------------------------------------------

import csv
import gzip

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from rich.progress import Progress
from rich.traceback import install
from torch.utils.data import DataLoader, Dataset

install()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the CNN model
class CNNEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(CNNEmbedding, self).__init__()
        self.conv1 = nn.Conv1d(4, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc = nn.Linear(128 * 36, embedding_dim)

    def forward(self, x):
        # print("---------------- CNN Embedding -----------------")
        # print("1", x.shape)
        x = self.conv1(x)
        # print("2", x.shape)
        x = self.relu(x)
        # print("3", x.shape)
        x = self.maxpool(x)
        # print("4", x.shape)
        x = self.conv2(x)
        # print("5", x.shape)
        x = self.relu(x)
        # print("6", x.shape)
        x = self.maxpool(x)
        # print("7", x.shape)
        x = x.view(x.size(0), -1)
        # print("8", x.shape)
        x = self.fc(x)
        # print("9", x.shape)
        # print("---------------- CNN Embedding -----------------")
        return x


# Define the transformer model
class TransformerModel(nn.Module):
    def __init__(self, embedding_dim):
        super(TransformerModel, self).__init__()
        self.embedding = CNNEmbedding(embedding_dim)
        self.transformer = nn.Transformer(
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            d_model=embedding_dim,
            dim_feedforward=512,
            batch_first=True,
        )
        self.fc = nn.Linear(embedding_dim, 1)  # Adjusted input size

    def forward(self, src):
        # print("---------------- Transformer -----------------")
        # print("0", src.shape)
        src = self.embedding(src)
        # print("10", src.shape)
        output = self.transformer(src, src)
        # print("11", output.shape)
        output = self.fc(output)
        # print("12", output.shape)
        # print("---------------- Transformer -----------------")
        return output


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
        self.x_data = torch.from_numpy(self.x_data).to(device)
        self.y_data = torch.from_numpy(self.y_data).to(device)
        self.len = len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


chipseq = ChipDataLoader(
    "/mnt/f/Projects/GenomicAttributions/data/encode/GATA1_K562_GATA-1_USC_AC.seq.gz"
)
alldataset = chipseq.load_data()
train_dataset_pad = alldataset
size = int(len(train_dataset_pad) / 3)
firstvalid = train_dataset_pad[:size]
secondvalid = train_dataset_pad[size : size + size]
thirdvalid = train_dataset_pad[size + size :]
firsttrain = secondvalid + thirdvalid
secondtrain = firstvalid + thirdvalid
thirdtrain = firstvalid + secondvalid

train1_dataset = chipseq_dataset(firsttrain)
train2_dataset = chipseq_dataset(secondtrain)
train3_dataset = chipseq_dataset(thirdtrain)
valid1_dataset = chipseq_dataset(firstvalid)
valid2_dataset = chipseq_dataset(secondvalid)
valid3_dataset = chipseq_dataset(thirdvalid)

batch_size = 64
train_loader1 = DataLoader(dataset=train1_dataset, batch_size=batch_size, shuffle=True)
train_loader2 = DataLoader(dataset=train2_dataset, batch_size=batch_size, shuffle=True)
train_loader3 = DataLoader(dataset=train3_dataset, batch_size=batch_size, shuffle=True)
valid1_loader = DataLoader(dataset=valid1_dataset, batch_size=batch_size, shuffle=False)
valid2_loader = DataLoader(dataset=valid2_dataset, batch_size=batch_size, shuffle=False)
valid3_loader = DataLoader(dataset=valid3_dataset, batch_size=batch_size, shuffle=False)
train_dataloader = [train_loader1, train_loader2, train_loader3]
valid_dataloader = [valid1_loader, valid2_loader, valid3_loader]

embedding_dim = 768
model = TransformerModel(embedding_dim).to(device)

print(model)

criterion = nn.MSELoss()
# criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop with rich progress tracking
num_epochs = 10

with Progress() as progress:
    task = progress.add_task("[cyan]Training...", total=num_epochs)

    for epoch in range(num_epochs):
        for i, train_loader in enumerate(train_dataloader):
            # INPUT: [batch, 4, 147]
            # TARGET: [batch, 1]
            # MODEL: [batch, 1]

            batch_input, batch_target = next(iter(train_loader))
            batch_input, batch_target = batch_input.to(device), batch_target.to(device)

            # Forward pass
            outputs = model(batch_input)
            loss = criterion(outputs, batch_target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress.update(task, advance=1)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("Training finished!")

    for i, valid_loader in enumerate(valid_dataloader):
        with torch.no_grad():
            for batch_input, batch_target in valid_loader:
                batch_input, batch_target = (
                    batch_input.to(device),
                    batch_target.to(device),
                )
                outputs = model(batch_input)
                loss = criterion(outputs, batch_target)

            print(f"Validation Loss: {loss.item():.4f}")

    print("Validation finished!")


# test on all data
test_dataset = chipseq_dataset(alldataset)
test_batch_size = test_dataset.len
test_loader = DataLoader(
    dataset=test_dataset, batch_size=test_batch_size, shuffle=False
)

with torch.no_grad():
    losses = []
    for batch_input, batch_target in test_loader:
        batch_input, batch_target = batch_input.to(device), batch_target.to(device)
        outputs = model(batch_input)
        loss = criterion(outputs, batch_target)
        losses.append(loss.item())

    print(f"Test Loss: {np.mean(losses):.4f}")

torch.cuda.empty_cache()
