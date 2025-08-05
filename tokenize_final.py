import pickle
import os
from pathlib import Path
from collections import defaultdict, Counter
import json

# 📂 Ścieżki
PKL_DIR = Path("E:/MIDI_GENERATOR/PreprocessedWithDrums")
TOKEN_DIR = Path("E:/MIDI_GENERATOR/TokenizedWithChords")
CHORD_VOCAB_PATH = Path("E:/MIDI_GENERATOR/chord_vocab.json")

WAIT_RESOLUTION = 0.05  # 50 ms
MAX_SHIFT_STEPS = 100
TOP_CHORDS = 500        # <--- tu możesz ustawić np. 1000, jeśli chcesz więcej

# 📘 Globalny słownik tymczasowy
chord_counter = Counter()
chord_vocab = {}
next_chord_id = 0
allowed_chords = set()

def quantize_time(t):
    return round(t / WAIT_RESOLUTION) * WAIT_RESOLUTION

def count_chords_in_track(track):
    notes_by_start = defaultdict(list)
    for note in track['notes']:
        start = quantize_time(note['start'])
        notes_by_start[start].append(note['pitch'])

    for pitches in notes_by_start.values():
        chord_key = tuple(sorted(pitches))
        chord_counter[chord_key] += 1

def build_chord_vocab():
    global chord_vocab, allowed_chords, next_chord_id
    most_common = chord_counter.most_common(TOP_CHORDS)
    allowed_chords = set([chord for chord, _ in most_common])
    for chord_key in allowed_chords:
        chord_vocab[chord_key] = next_chord_id
        next_chord_id += 1

def get_chord_token(pitches):
    chord_key = tuple(sorted(pitches))
    if chord_key not in chord_vocab:
        return None  # zignorujemy ten akord
    return f"chord_{chord_vocab[chord_key]}"

def track_to_tokens(track):
    tokens = []
    notes_by_start = defaultdict(list)
    for note in track['notes']:
        start = quantize_time(note['start'])
        notes_by_start[start].append(note['pitch'])

    sorted_times = sorted(notes_by_start.keys())
    prev_time = 0.0

    for start_time in sorted_times:
        delta = start_time - prev_time
        if delta < 0:
            continue
        shift_steps = min(int(delta / WAIT_RESOLUTION), MAX_SHIFT_STEPS)
        if shift_steps > 0:
            tokens.append(f"time_shift_{shift_steps}")

        pitches = notes_by_start[start_time]
        chord_token = get_chord_token(pitches)
        if chord_token:
            tokens.append(chord_token)

        prev_time = start_time

    return tokens

def scan_all_chords(pkl_dir):
    for artist_dir in pkl_dir.iterdir():
        if not artist_dir.is_dir():
            continue
        for pkl_file in artist_dir.glob("*.pkl"):
            try:
                with open(pkl_file, "rb") as f:
                    tracks = pickle.load(f)
                for track in tracks:
                    count_chords_in_track(track)
            except Exception as e:
                print(f"❌ Błąd przy {pkl_file.name}: {e}")

def tokenize_file(pkl_path):
    with open(pkl_path, "rb") as f:
        tracks = pickle.load(f)

    all_tokens = []
    for track in tracks:
        tokens = track_to_tokens(track)
        all_tokens.extend(tokens)
    return all_tokens

def process_all_pkl(pkl_dir, token_dir):
    token_dir.mkdir(parents=True, exist_ok=True)

    for artist_dir in pkl_dir.iterdir():
        if not artist_dir.is_dir():
            continue

        token_artist_dir = token_dir / artist_dir.name
        token_artist_dir.mkdir(parents=True, exist_ok=True)

        print(f"🎶 Tokenizuję: {artist_dir.name}")
        for pkl_file in artist_dir.glob("*.pkl"):
            try:
                tokens = tokenize_file(pkl_file)
                out_path = token_artist_dir / f"{pkl_file.stem}.txt"
                with open(out_path, "w") as f:
                    f.write(" ".join(tokens))
                print(f"   ✅ {pkl_file.name} ({len(tokens)} tokenów)")
            except Exception as e:
                print(f"   ❌ Błąd przy {pkl_file.name}: {e}")

    # Zapisz słownik akordów
    with open(CHORD_VOCAB_PATH, "w") as f:
        chord_json = {str(k): v for k, v in chord_vocab.items()}
        json.dump(chord_json, f, indent=2)

    print(f"💾 Zapisano chord_vocab.json ({len(chord_vocab)} akordów)")

if __name__ == "__main__":
    print("📊 Skanowanie akordów...")
    scan_all_chords(PKL_DIR)
    print(f"🔢 Znaleziono {len(chord_counter)} unikalnych akordów, wybieram top {TOP_CHORDS}")
    build_chord_vocab()
    process_all_pkl(PKL_DIR, TOKEN_DIR)
