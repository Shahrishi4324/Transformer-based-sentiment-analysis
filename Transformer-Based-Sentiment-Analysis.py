import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import spacy
import nltk
from nltk.corpus import imdb
from collections import Counter
import numpy as np

# Download the IMDb dataset
nltk.download('imdb')
nltk.download('punkt')

# Load SpaCy for tokenization
nlp = spacy.load('en_core_web_sm')

# Preprocessing function
def preprocess_text(text):
    tokens = [token.text.lower() for token in nlp(text)]
    return tokens

# IMDb dataset class
class IMDbDataset(Dataset):
    def __init__(self, data, labels, vocab=None, max_len=512):
        self.data = data
        self.labels = labels
        self.max_len = max_len
        self.vocab = vocab or self.build_vocab(self.data)
        self.pad_token = self.vocab.get('<PAD>', 0)

    def build_vocab(self, texts, min_freq=2):
        counter = Counter()
        for text in texts:
            counter.update(text)
        vocab = {word: i+2 for i, (word, freq) in enumerate(counter.items()) if freq >= min_freq}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab

    def encode_text(self, text):
        return [self.vocab.get(token, self.vocab['<UNK>']) for token in text[:self.max_len]]

    def pad_sequence(self, seq):
        return seq + [self.pad_token] * (self.max_len - len(seq))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.encode_text(self.data[idx])
        text = self.pad_sequence(text)
        label = self.labels[idx]
        return torch.tensor(text), torch.tensor(label)
    
# Load and preprocess IMDb data
train_data, test_data = imdb.load_data(train=True), imdb.load_data(train=False)
train_texts, train_labels = zip(*[(preprocess_text(review), label) for review, label in train_data])
test_texts, test_labels = zip(*[(preprocess_text(review), label) for review, label in test_data])

# Create datasets and dataloaders
train_dataset = IMDbDataset(train_texts, train_labels)
test_dataset = IMDbDataset(test_texts, test_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import math

# Embedding layer with positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1), :]