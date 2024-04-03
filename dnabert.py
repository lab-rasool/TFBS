import random

import torch
from transformers import AutoTokenizer, BertModel

tokenizer = AutoTokenizer.from_pretrained(
    "zhihan1996/DNA_bert_6", trust_remote_code=True
)
model = BertModel.from_pretrained("zhihan1996/DNA_bert_6")


def random_dna(length):
    return "".join([random.choice("ACGT") for _ in range(length)])


dna = random_dna(147)
print(len(dna))  # expect to be 60
inputs = tokenizer(dna, return_tensors="pt")["input_ids"]
hidden_states = model(inputs)[0]

print(hidden_states.shape)  # expect to be [1, 3, 768]
print(hidden_states)
# embedding with mean pooling
embedding_mean = torch.mean(hidden_states[0], dim=0)
print(embedding_mean.shape)  # expect to be 768

# embedding with max pooling
embedding_max = torch.max(hidden_states[0], dim=0)[0]
print(embedding_max.shape)  # expect to be 768
