from pathlib import Path
from collections import Counter

# 🔧 Ścieżka do pliku tokenów
TOKEN_FILE = Path('E:/MIDI_GENERATOR/TokenizedWithChords/2_Brothers_on_the_4th_Floor/Come_Take_My_Hand.txt')  # <- podmień na własną ścieżkę

def inspect_token_file(path):
    with open(path, "r") as f:
        tokens = f.read().strip().split()

    print(f"📦 Liczba tokenów: {len(tokens)}")
    print(f"🔍 Pierwsze 30 tokenów:\n{' '.join(tokens[:30])}\n")

    # Analiza typów tokenów
    prefix_counts = Counter(token.split("_")[0] for token in tokens)
    print("📊 Tokeny wg typu:")
    for prefix, count in prefix_counts.items():
        print(f"  {prefix:12}: {count}")

    # Dodatkowo: top 10 akordów
    chord_tokens = [t for t in tokens if t.startswith("chord_")]
    top_chords = Counter(chord_tokens).most_common(10)
    print("\n🎶 Top 10 najczęściej występujących akordów:")
    for chord, count in top_chords:
        print(f"  {chord:10}: {count}")

if __name__ == "__main__":
    inspect_token_file(TOKEN_FILE)
