# train.py (z Early Stopping oraz logowaniem do training_logs.txt)
import json
import time
from pathlib import Path
import random
import numpy as np
import shutil
import logging
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MusicDataset
from model import TransformerModel

# === ŚCIEŻKI ===
BASE_DIR = Path("E:/MIDI_GENERATOR/transformer-best")
VOCAB_PATH = BASE_DIR / "vocab.json"
DATASET_DIR = BASE_DIR / "dataset_ids"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = CHECKPOINT_DIR / "training_logs.txt"

# === PARAMETRY TRENINGU ===
SEQ_LEN = 512
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_INTERVAL = 50        # co ile batchy logować stratę
NUM_WORKERS = 4          # DataLoader workers
GRAD_CLIP = 1.0          # wartość clip (None żeby wyłączyć)
SEED = 42

# === EARLY STOPPING ===
PATIENCE = 2             # jeżeli val_loss nie poprawi się przez PATIENCE epok -> stop
MIN_DELTA = 1e-6         # minimalna różnica, aby uznać poprawę

# === LOGGING SETUP ===
logger = logging.getLogger("train_logger")
logger.setLevel(logging.INFO)
logger.handlers.clear()

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
# File handler (append)
fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
fh.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

logger.addHandler(ch)
logger.addHandler(fh)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_vocab_size(vocab_path: Path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return len(vocab), vocab


def save_checkpoint(state, path: Path):
    torch.save(state, path)
    logger.info(f"💾 Zapisano checkpoint: {path}")


def main():
    logger.info("STARTING TRAINING")
    logger.info(f"Base dir: {BASE_DIR}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch size: {BATCH_SIZE}, Seq len: {SEQ_LEN}, Epochs: {EPOCHS}")

    set_seed(SEED)

    # 1) Wczytanie vocab i rozmiaru
    vocab_size, vocab_map = load_vocab_size(VOCAB_PATH)
    logger.info(f"📦 Vocab size: {vocab_size}")

    # 2) Dataset i DataLoader (plik .npy ma shape (num_seq, seq_len))
    train_path = DATASET_DIR / "train.npy"
    val_path = DATASET_DIR / "val.npy"

    if not train_path.exists() or not val_path.exists():
        logger.error("Nie znaleziono train.npy lub val.npy w dataset_ids. Upewnij się, że pipeline datasetów został uruchomiony.")
        raise FileNotFoundError("Nie znaleziono train.npy lub val.npy w dataset_ids.")

    train_dataset = MusicDataset(train_path)  # MusicDataset zwraca x,y i ma ignore_index
    val_dataset = MusicDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    logger.info(f"📊 Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")
    logger.info(f"📊 Batches train: {len(train_loader)}, val: {len(val_loader)}")

    # 3) Model
    model = TransformerModel(vocab_size=vocab_size).to(DEVICE)

    # 4) Loss i optimizer (ignore_index pobieramy z datasetu)
    ignore_index = train_dataset.ignore_index if hasattr(train_dataset, "ignore_index") else -100
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Early stopping trackers
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_ckpt_path = CHECKPOINT_DIR / "best_model.pt"

    try:
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.time()

            # ----- TRAIN -----
            model.train()
            running_loss = 0.0
            for batch_idx, (x, y) in enumerate(train_loader, start=1):
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                logits = model(x)  # [batch, seq_len, vocab_size]

                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()

                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

                optimizer.step()
                running_loss += loss.item()

                if batch_idx % LOG_INTERVAL == 0 or batch_idx == 1:
                    logger.info(f"[Epoka {epoch}] Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

            avg_train_loss = running_loss / len(train_loader)

            # ----- VALIDATION -----
            model.eval()
            val_running = 0.0
            with torch.no_grad():
                for x_val, y_val in val_loader:
                    x_val = x_val.to(DEVICE, non_blocking=True)
                    y_val = y_val.to(DEVICE, non_blocking=True)

                    logits_val = model(x_val)
                    loss_val = criterion(logits_val.view(-1, logits_val.size(-1)), y_val.view(-1))
                    val_running += loss_val.item()

            avg_val_loss = val_running / len(val_loader)
            epoch_time = time.time() - epoch_start

            logger.info(f"📅 Epoch {epoch}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s")

            # ----- CHECK EARLY STOPPING & SAVE -----
            improved = (avg_val_loss + MIN_DELTA) < best_val_loss
            if improved:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                epochs_no_improve = 0

                # Zapis najlepszej wagi (ostatnia przed ewentualnym overfittingiem)
                torch.save(model.state_dict(), best_ckpt_path)
                logger.info(f"🏆 Poprawa: zapisano nowy najlepszy model (epoch {epoch}) -> {best_ckpt_path}")
            else:
                epochs_no_improve += 1
                logger.info(f"ℹ️ Brak poprawy przez {epochs_no_improve} epok(y) (patience={PATIENCE})")

            # Zapis checkpointu każdej epoki (opcja)
            ckpt_path = CHECKPOINT_DIR / f"model_epoch_{epoch}.pt"
            save_checkpoint({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "vocab_size": vocab_size
            }, ckpt_path)

            # Czy przerwać trening?
            if epochs_no_improve > PATIENCE:
                logger.info(f"⛔ Early stopping: brak poprawy przez {epochs_no_improve} epok (> {PATIENCE}). Kończę trening.")
                # kopiujemy najlepszy model na wyraźny plik early_stopped_checkpoint (opcjonalnie)
                early_path = CHECKPOINT_DIR / "early_stopped_checkpoint.pt"
                # Zapis stanu + metadanych
                save_checkpoint({
                    "stopped_epoch": epoch,
                    "best_epoch": best_epoch,
                    "best_val_loss": float(best_val_loss),
                    "model_state_dict": torch.load(best_ckpt_path, map_location="cpu"),
                    "vocab_size": vocab_size
                }, early_path)
                logger.info(f"✅ Najlepszy model przed przeuczeniem znajduje się w: {best_ckpt_path}")
                break

    except KeyboardInterrupt:
        logger.warning("⛔ Przerwano trening (KeyboardInterrupt). Zapisuję aktualny stan...")
        save_checkpoint({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, CHECKPOINT_DIR / "interrupt_last.pt")

    except Exception as e:
        logger.exception(f"Nieoczekiwany błąd podczas treningu: {e}")
        # optionally save state
        save_checkpoint({
            "error": str(e),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, CHECKPOINT_DIR / "error_checkpoint.pt")
        raise

    logger.info("Koniec treningu.")


if __name__ == "__main__":
    main()
