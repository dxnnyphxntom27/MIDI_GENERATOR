import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, ff_dim=2048, dropout=0.1):
        super().__init__()
        # Wektory osadzeń dla tokenów
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout)

        # Pojedyncza warstwa dekodera z batch_first=True
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True  # <-- ważne
        )

        # Dekoder z wieloma warstwami
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Wyjście — logits dla każdego tokenu w sekwencji
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def _generate_square_subsequent_mask(self, sz):
        """Maska, która blokuje podgląd przyszłych tokenów."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len] — token IDs
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        seq_len = x.size(1)
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)

        embedded = self.token_embedding(x) * math.sqrt(self.token_embedding.embedding_dim)
        embedded = self.positional_encoding(embedded)  # [batch, seq_len, embed_dim]

        decoded = self.transformer_decoder(
            tgt=embedded,
            memory=torch.zeros(x.size(0), 0, embedded.size(2), device=x.device),  # brak encoder memory
            tgt_mask=mask
        )

        logits = self.fc_out(decoded)  # [batch, seq_len, vocab_size]
        return logits


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Tworzymy macierz [max_len, embed_dim] z pozycjami
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # kanały parzyste
        pe[:, 1::2] = torch.cos(position * div_term)  # kanały nieparzyste
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


if __name__ == "__main__":
    # Test działania modelu
    vocab_size = 1000
    model = TransformerModel(vocab_size)
    x = torch.randint(0, vocab_size, (2, 512))  # batch=2, seq_len=512
    out = model(x)
    print("Wyjście:", out.shape)  # spodziewane: [2, 512, vocab_size]
