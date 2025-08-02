import os
from pathlib import Path
import json
import torch

# Ścieżki do katalogów i słownika
TOKENIZED_DIR = Path("E:/MIDI_GENERATOR/TokenizedWithChords")
TENSORIZED_DIR = Path("E:/MIDI_GENERATOR/TensorizedWithChords")
VOCAB_PATH = Path("E:/MIDI_GENERATOR/vocab.json")

# Wczytanie słownika tokenów
with open(VOCAB_PATH, "r") as f:
    token_to_id = json.load(f)

# Konwersja pojedynczego pliku .txt → tensor
def convert_token_file_to_tensor(txt_path):
    with open(txt_path, "r") as f:
        tokens = f.read().split()

    ids = [token_to_id[token] for token in tokens if token in token_to_id]
    return torch.tensor(ids, dtype=torch.long)

# Przetwarzanie wszystkich plików w folderze
def process_all_tokens(token_dir, tensor_dir):
    tensor_dir.mkdir(parents=True, exist_ok=True)

    artist_dirs = [d for d in token_dir.iterdir() if d.is_dir()]
    print(f"🎶 Znaleziono {len(artist_dirs)} artystów do przetworzenia.\n")

    for artist_dir in artist_dirs:
        tensor_artist_dir = tensor_dir / artist_dir.name
        tensor_artist_dir.mkdir(parents=True, exist_ok=True)

        token_files = list(artist_dir.glob("*.txt"))
        print(f"🎼 Przetwarzam artystę: {artist_dir.name} ({len(token_files)} plików)")

        for i, token_file in enumerate(token_files, start=1):
            try:
                tensor = convert_token_file_to_tensor(token_file)
                out_path = tensor_artist_dir / f"{token_file.stem}.pt"
                torch.save(tensor, out_path)
                if i % 10 == 0 or i == len(token_files):
                    print(f"   ➜ {i}/{len(token_files)} OK: {token_file.name}")
            except Exception as e:
                print(f"   ❌ Błąd przy {token_file.name}: {e}")

        print(f"   ✅ Zakończono {artist_dir.name}\n")

    print("🏁 Gotowe! Wszystkie pliki zostały przetworzone.")

if __name__ == "__main__":
    process_all_tokens(TOKENIZED_DIR, TENSORIZED_DIR)
