import os
import json
import random
import pickle
from pathlib import Path

TOKENIZED_DIR = Path("E:/MIDI_GENERATOR/transformer-based-2/TokenizedWithChords")
VOCAB_PATH = Path("E:/MIDI_GENERATOR/transformer-based-2/vocab.json")
OUTPUT_DIR = Path("E:/MIDI_GENERATOR/transformer-based-2/")
TRAIN_OUT = OUTPUT_DIR / "train_dataset.pkl"
VAL_OUT = OUTPUT_DIR / "val_dataset.pkl"

VAL_SPLIT = 0.1
SEED = 42
MIN_LENGTH = 32

def load_vocab():
    with open(VOCAB_PATH, "r") as f:
        return json.load(f)

def load_token_file(path, vocab):
    with open(path, "r") as f:
        tokens = f.read().splitlines()
    try:
        token_ids = [vocab[token] for token in tokens if token in vocab]
    except KeyError as e:
        print(f"❌ Błąd w pliku {path.name}: {e}")
        return []
    return token_ids

def tokenize_all():
    vocab = load_vocab()
    all_sequences = []

    artist_dirs = [d for d in TOKENIZED_DIR.iterdir() if d.is_dir()]
    print(f"🔍 Przeszukuję {len(artist_dirs)} artystów...\n")

    for artist_dir in artist_dirs:
        for token_file in artist_dir.glob("*.txt"):
            token_ids = load_token_file(token_file, vocab)
            if len(token_ids) >= MIN_LENGTH:
                all_sequences.append(token_ids)

    print(f"\n📦 Łącznie sekwencji: {len(all_sequences)}")

    random.seed(SEED)
    random.shuffle(all_sequences)

    split_idx = int(len(all_sequences) * (1 - VAL_SPLIT))
    train_data = all_sequences[:split_idx]
    val_data = all_sequences[split_idx:]

    with open(TRAIN_OUT, "wb") as f:
        pickle.dump(train_data, f)

    with open(VAL_OUT, "wb") as f:
        pickle.dump(val_data, f)

    print(f"✅ Zapisano:")
    print(f"   • train_dataset.pkl ({len(train_data)} sekwencji)")
    print(f"   • val_dataset.pkl ({len(val_data)} sekwencji)")

if __name__ == "__main__":
    tokenize_all()
