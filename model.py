import torch
import torch.nn as nn

class MusicLSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_size=512, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        """
        x: Tensor [batch_size, seq_len]
        hidden: (h_0, c_0), optional
        returns: logits [batch_size, seq_len, vocab_size]
        """
        embedded = self.embedding(x)  # → [batch, seq, embed_dim]
        output, hidden = self.lstm(embedded, hidden)  # → [batch, seq, hidden_size]
        logits = self.fc(output)  # → [batch, seq, vocab_size]
        return logits, hidden
