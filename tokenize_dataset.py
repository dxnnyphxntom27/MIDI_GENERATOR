import pickle
import os
from pathlib import Path
from collections import defaultdict

# Nowa lokalizacja zapisu
TOKEN_DIR = Path("E:/MIDI_GENERATOR/TokenizedWithChords")
PKL_DIR = Path("E:/MIDI_GENERATOR/PreprocessedWithDrums")

WAIT_RESOLUTION = 0.05  # 50 ms jednostki wait


def quantize_time(time):
    return round(time / WAIT_RESOLUTION) * WAIT_RESOLUTION


def track_to_tokens(track):
    tokens = []
    program = track.get('program', None)
    is_drum = track.get('is_drum', False)

    if is_drum:
        tokens.append("instrument_drum")
    else:
        tokens.append(f"instrument_{program}")

    # Grupowanie nut wg czasu rozpoczęcia
    notes_by_start = defaultdict(list)
    for note in track['notes']:
        start = quantize_time(note['start'])
        end = quantize_time(note['end'])
        notes_by_start[start].append((note['pitch'], end))

    # Sortowanie po czasie
    sorted_starts = sorted(notes_by_start.keys())
    prev_time = 0.0

    for start_time in sorted_starts:
        wait_time = start_time - prev_time
        if wait_time > 0:
            wait_steps = int(wait_time / WAIT_RESOLUTION)
            tokens.append(f"wait_{wait_steps}")

        notes = notes_by_start[start_time]
        for pitch, _ in notes:
            tokens.append(f"note_on_{pitch}")

        # Ustalenie najkrótszego wspólnego trwania
        shortest_duration = min(
            quantize_time(end - start_time) for _, end in notes)
        if shortest_duration > 0:
            duration_steps = int(shortest_duration / WAIT_RESOLUTION)
            tokens.append(f"wait_{duration_steps}")

        for pitch, _ in notes:
            tokens.append(f"note_off_{pitch}")

        prev_time = start_time + shortest_duration

    return tokens


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


if __name__ == "__main__":
    process_all_pkl(PKL_DIR, TOKEN_DIR)
