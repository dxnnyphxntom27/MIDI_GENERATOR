import os
from pathlib import Path
import json

TOKENIZED_DIR = Path("E:/MIDI_GENERATOR/TokenizedWithChords")
VOCAB_PATH = Path("E:/MIDI_GENERATOR/vocab.json")
VOCAB_REVERSE_PATH = Path("E:/MIDI_GENERATOR/vocab_reverse.json")

# Dozwolone prefiksy tokenów
ALLOWED_PREFIXES = ("note_on_", "duration_", "wait_", "instrument_")

def build_token_vocab(token_dir):
    vocab = set()
    total_files = 0

    artist_dirs = [d for d in token_dir.iterdir() if d.is_dir()]
    print(f"🎶 Znaleziono {len(artist_dirs)} artystów.\n")

    for artist_dir in artist_dirs:
        print(f"🎼 Przetwarzam artystę: {artist_dir.name}")
        token_files = list(artist_dir.glob("*.txt"))
        print(f"   🧾 Liczba plików: {len(token_files)}")

        for i, token_file in enumerate(token_files, start=1):
            try:
                with open(token_file, "r") as f:
                    tokens = f.read().split()
                    for token in tokens:
                        if token.startswith(ALLOWED_PREFIXES):
                            vocab.add(token)
                total_files += 1
                if i % 10 == 0 or i == len(token_files):
                    print(f"     ➜ Przetworzono {i}/{len(token_files)} plików...")
            except Exception as e:
                print(f"   ❌ Błąd przy {token_file.name}: {e}")

        print(f"   ✅ Skończono {artist_dir.name} — łącznie tokenów: {len(vocab)}\n")

    sorted_vocab = sorted(list(vocab))
    token_to_id = {token: idx for idx, token in enumerate(sorted_vocab)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    print(f"🏁 Zakończono! Łącznie plików: {total_files}, unikalnych tokenów: {len(token_to_id)}")
    return token_to_id, id_to_token

if __name__ == "__main__":
    token_to_id, id_to_token = build_token_vocab(TOKENIZED_DIR)

    with open(VOCAB_PATH, "w") as f:
        json.dump(token_to_id, f, indent=2)

    with open(VOCAB_REVERSE_PATH, "w") as f:
        json.dump(id_to_token, f, indent=2)

    print(f"\n💾 Zapisano słowniki tokenów do:\n- {VOCAB_PATH}\n- {VOCAB_REVERSE_PATH}")
