import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import spacy
import nltk
from nltk.corpus import imdb
from collections import Counter
import numpy as np
import math

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
    
# Multi-head attention mechanism
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        return x.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.depth)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        attention = torch.matmul(attention_weights, v)

        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(attention)

        return output, attention_weights


# Feed-forward network
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.nn.functional.relu(self.fc1(x)))
        return self.fc2(x)
    
# Transformer encoder layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, mask)
        out1 = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1 + self.dropout(ffn_output))
        return out2


# Transformer encoder
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_len=512, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
# Sentiment analysis model with transformer encoder
class SentimentTransformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, num_classes, max_len=512, dropout=0.1):
        super(SentimentTransformer, self).__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_len, dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        enc_output = self.encoder(x, mask)
        enc_output = enc_output.mean(dim=1)  # Global average pooling
        logits = self.fc(enc_output)
        return logits

# Hyperparameters
num_layers = 4
d_model = 128
num_heads = 8
d_ff = 512
num_classes = 2
input_vocab_size = len(train_dataset.vocab)
learning_rate = 1e-4
num_epochs = 10

# Initialize the model, loss function, and optimizer
model = SentimentTransformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

# Evaluate on the test set
test_accuracy = evaluate_model(model, test_loader)
print(f"Test Accuracy: {test_accuracy:.4f}")