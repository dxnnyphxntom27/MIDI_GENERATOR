import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random

# === ŚCIEŻKI ===
TOKENIZED_DIR = Path("E:/MIDI_GENERATOR/transformer-final/TokenizedWithChords")
VOCAB_PATH = Path("E:/MIDI_GENERATOR/transformer-final/vocab.json")
OUTPUT_DIR = Path("E:/MIDI_GENERATOR/transformer-final/dataset_ids")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === PARAMETRY ===
SEQ_LEN = 512
TRAIN_SPLIT = 0.9

def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)

def tokens_to_ids(token_files, token_to_id):
    """Zamienia listę plików z tokenami na listę ID."""
    all_ids = []
    for file_path in tqdm(token_files, desc="Konwersja tokenów na ID"):
        with open(file_path, "r", encoding="utf-8") as f:
            tokens = f.read().splitlines()
            ids = [token_to_id[token] for token in tokens if token in token_to_id]
            all_ids.extend(ids)
    return all_ids

def split_and_save(all_ids, seq_len, train_split):
    """Dzieli dane na sekwencje i zapisuje do train/val .npy"""
    # Przycinamy do pełnych sekwencji
    total_len = len(all_ids) // seq_len * seq_len
    all_ids = np.array(all_ids[:total_len], dtype=np.int32)
    
    # Losowe przetasowanie
    np.random.shuffle(all_ids.reshape(-1, seq_len))

    # Podział train / val
    num_train = int(len(all_ids) * train_split)
    train_ids = all_ids[:num_train]
    val_ids = all_ids[num_train:]

    # Zapis
    np.save(OUTPUT_DIR / "train.npy", train_ids)
    np.save(OUTPUT_DIR / "val.npy", val_ids)

    print(f"✅ Zapisano train.npy ({train_ids.shape[0]} tokenów) i val.npy ({val_ids.shape[0]} tokenów)")

def main():
    # Wczytanie słownika
    token_to_id = load_vocab(VOCAB_PATH)
    print(f"📦 Wczytano słownik o {len(token_to_id)} tokenach")

    # Lista plików
    token_files = list(TOKENIZED_DIR.glob("**/*.txt"))
    print(f"🎵 Znaleziono {len(token_files)} plików z tokenami")

    # Losowe wymieszanie plików
    random.shuffle(token_files)

    # Konwersja tokenów na ID
    all_ids = tokens_to_ids(token_files, token_to_id)
    print(f"🔢 Łączna liczba tokenów po konwersji: {len(all_ids)}")

    # Podział i zapis
    split_and_save(all_ids, SEQ_LEN, TRAIN_SPLIT)

if __name__ == "__main__":
    main()
