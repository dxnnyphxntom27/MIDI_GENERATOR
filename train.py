import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from model import MusicLSTMModel
from dataset import MIDIDataset
import json
import os

# 🔧 PARAMETRY
SEQ_LEN = 128
BATCH_SIZE = 32
EPOCHS = 80
EMBED_DIM = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 2
DROPOUT = 0.2
LR = 0.001
PATIENCE = 5
CHECKPOINT_PATH = "best_model.pt"

# 📚 Wczytaj słownik
with open("vocab.json") as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# 🎼 Dataset + podział train/val
full_dataset = MIDIDataset(tensor_dir="TensorizedWithChords", seq_len=SEQ_LEN)
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# 🧠 Model + optymalizator + loss
device = torch.device("cuda")
model = MusicLSTMModel(vocab_size, EMBED_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# 📉 EarlyStopping i Checkpoint
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(X)
        logits = logits.view(-1, vocab_size)
        y = y.view(-1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if i % 50 == 0:
            print(f"Epoch {epoch+1} Step {i}/{len(train_loader)} Loss: {loss.item():.4f}")

    avg_train_loss = total_loss / len(train_loader)
    
    # 🔍 Walidacja
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(device), y_val.to(device)
            logits, _ = model(X_val)
            logits = logits.view(-1, vocab_size)
            y_val = y_val.view(-1)
            loss = criterion(logits, y_val)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f" - Epoch {epoch+1}/{EPOCHS} | Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

    # 🛑 Early stopping + checkpoint
    if avg_val_loss < best_val_loss - 1e-4:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), CHECKPOINT_PATH)
        print(f" - New best model saved at {CHECKPOINT_PATH}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f" - No improvement for {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= PATIENCE:
            print(f" - Early stopping triggered after {epoch+1} epochs.")
            break

print("Training finished.")
