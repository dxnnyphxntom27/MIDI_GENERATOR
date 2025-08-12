import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pickle
import json
import os
import sys
from datetime import datetime
from transformer4 import MusicTransformerModel

# === ŚCIEŻKI ===
DATASET_DIR = "E:/MIDI_GENERATOR/transformer-based-2"
DEVICE = "cuda"
LOG_PATH = os.path.join(DATASET_DIR, "training_log.txt")

# === PARAMETRY TRENINGU ===
BATCH_SIZE = 16
SEQ_LEN = 128
EPOCHS = 6
LR = 1e-4
MODEL_SAVE_PATH = os.path.join(DATASET_DIR, "transformer_pretrained.pt")
MAX_SAMPLES_TRAIN = 100000    # 1150000 ~2h treningu
MAX_SAMPLES_VAL = 50000

# === REDIREKCJA PRINTA DO KONSOLI + PLIKU ===
class Logger(object):
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.log = open(logfile_path, "a", encoding="utf-8")
        self.start_time = datetime.now()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(LOG_PATH)

print(f"\n🕒 Start treningu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# === DANE ===
with open(os.path.join(DATASET_DIR, "vocab.json"), "r") as f:
    vocab = json.load(f)
vocab_size = len(vocab)

with open(os.path.join(DATASET_DIR, "train_dataset.pkl"), "rb") as f:
    train_data = pickle.load(f)
with open(os.path.join(DATASET_DIR, "val_dataset.pkl"), "rb") as f:
    val_data = pickle.load(f)

# Po załadowaniu train_data
print(f"📊 Liczba utworów w train_data: {len(train_data)}")
print(f"📊 Przykładowe ID-y z pierwszego utworu: {train_data[0][:20]}")

with open(os.path.join(DATASET_DIR, "vocab_reverse.json"), "r") as f:
    vocab_reverse = json.load(f)

# Wyświetlenie pierwszych tokenów pierwszego utworu (czyli z jakich tokenów uczymy)
tokens = [vocab_reverse[str(id)] for id in train_data[0][:20]]
print(f"📝 Odpowiadające tokeny: {tokens}")

# === DATASET ===
class MIDIDataset(Dataset):
    def __init__(self, sequences, seq_len, max_samples=None):
        self.seq_len = seq_len
        self.sequences = sequences
        self.index_map = []

        for song_idx, seq in enumerate(sequences):
            max_start = len(seq) - seq_len - 1
            if max_start > 0:
                for start in range(max_start):
                    self.index_map.append((song_idx, start))
                    if max_samples and len(self.index_map) >= max_samples:
                        break
            if max_samples and len(self.index_map) >= max_samples:
                break

        print(f"📚 Dataset zawiera {len(self.index_map)} próbek")
        print(f"🧮 Szacowana liczba kroków na epokę: {len(self.index_map) // BATCH_SIZE}")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        song_idx, start_idx = self.index_map[idx]
        segment = self.sequences[song_idx][start_idx : start_idx + self.seq_len + 1]
        x = torch.tensor(segment[:-1], dtype=torch.long)
        y = torch.tensor(segment[1:], dtype=torch.long)
        return x, y

train_dataset = MIDIDataset(train_data, SEQ_LEN, max_samples=MAX_SAMPLES_TRAIN)
val_dataset = MIDIDataset(val_data, SEQ_LEN, max_samples=MAX_SAMPLES_VAL)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === MODEL ===
model = MusicTransformerModel(vocab_size=vocab_size).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# === TRENING ===
best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 100 == 0:
            print(f"[train] epoch {epoch} step {i} loss {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)

    # === EWALUACJA ===
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            output = model(x)
            loss = criterion(output.view(-1, vocab_size), y.view(-1))
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"🔁 Epoch {epoch}/{EPOCHS} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

    # === ZAPISZ NAJLEPSZY MODEL ===
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"💾 Model zapisany: {MODEL_SAVE_PATH}")

print("\n✅ Trening zakończony.")
print(f"🕒 Koniec: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
