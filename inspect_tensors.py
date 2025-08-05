import torch
import json
from pathlib import Path

# 📂 Ścieżki
TENSOR_DIR = Path("E:/MIDI_GENERATOR/TensorizedWithChords")
VOCAB_PATH = Path("E:/MIDI_GENERATOR/vocab.json")

# 📖 Załaduj słownik
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# 🔍 Analiza
max_global = -1
files_with_errors = []

pt_files = list(TENSOR_DIR.rglob("*.pt"))
print(f"🔎 Sprawdzam {len(pt_files)} plików...")

for pt_path in pt_files:
    try:
        ids = torch.load(pt_path)
        max_id = ids.max().item()
        if max_id >= vocab_size:
            files_with_errors.append((pt_path.name, max_id))
        if max_id > max_global:
            max_global = max_id
    except Exception as e:
        print(f"❌ Błąd przy {pt_path.name}: {e}")

# 📊 Raport
print("\n📌 Maksymalny ID w całym zbiorze:", max_global)
print("📏 Rozmiar słownika:", vocab_size)

if files_with_errors:
    print(f"\n❗ Znaleziono {len(files_with_errors)} plików z ID >= vocab_size:")
    for name, max_id in files_with_errors:
        print(f"  {name:30} | Max ID: {max_id}")
else:
    print("✅ Wszystkie pliki zgodne ze słownikiem.")
