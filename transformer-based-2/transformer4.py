import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MusicTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4*d_model,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)

    def forward(self, x):
        # x: (B, T)
        mask = self.generate_square_subsequent_mask(x.size(1)).to(x.device)  # (T, T)
        x = self.embedding(x)  # (B, T, D)
        x = self.positional_encoding(x)  # (B, T, D)
        x = self.transformer(x, mask=mask)  # (B, T, D)
        logits = self.fc_out(x)  # (B, T, vocab_size)
        return logits
