import os
from pathlib import Path
import json

TOKENIZED_DIR = Path("E:/MIDI_GENERATOR/transformer-based-2/TokenizedWithChords")
VOCAB_PATH = Path("E:/MIDI_GENERATOR/transformer-based-2/vocab.json")
VOCAB_REVERSE_PATH = Path("E:/MIDI_GENERATOR/transformer-based-2/vocab_reverse.json")

def build_token_vocab(token_dir):
    vocab = set()

    artist_dirs = [d for d in token_dir.iterdir() if d.is_dir()]
    print(f"🎶 Znaleziono {len(artist_dirs)} artystów.\n")

    for artist_dir in artist_dirs:
        print(f"🎼 Artysta: {artist_dir.name}")
        token_files = list(artist_dir.glob("*.txt"))
        print(f"   🧾 Liczba plików: {len(token_files)}")

        for token_file in token_files:
            with open(token_file, "r") as f:
                tokens = f.read().splitlines()
                vocab.update(tokens)

    print(f"\n📦 Łącznie unikalnych tokenów: {len(vocab)}")

    sorted_vocab = sorted(vocab)
    token_to_id = {token: i for i, token in enumerate(sorted_vocab)}
    id_to_token = {i: token for token, i in token_to_id.items()}

    with open(VOCAB_PATH, "w") as f:
        json.dump(token_to_id, f, indent=2)

    with open(VOCAB_REVERSE_PATH, "w") as f:
        json.dump(id_to_token, f, indent=2)

    print("✅ Słowniki zapisane jako:")
    print(f"   • {VOCAB_PATH.name}")
    print(f"   • {VOCAB_REVERSE_PATH.name}")

if __name__ == "__main__":
    build_token_vocab(TOKENIZED_DIR)
