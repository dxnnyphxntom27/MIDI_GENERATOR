import os
from pathlib import Path
import pretty_midi
import json
from collections import defaultdict

# === ŚCIEŻKI DOMYŚLNE ===
CHORD_VOCAB_PATH = Path("E:/MIDI_GENERATOR/app/chord_vocab.json")
DEFAULT_OUTPUT_DIR = Path("E:/MIDI_GENERATOR/app/USER_TOKENIZED")
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === PARAMETRY ===
TIME_RESOLUTION = 24
MAX_NOTES_IN_CHORD = 6
UNKNOWN_TOKEN_ID = 9999999
MAX_TIME_SHIFT = 192
DEFAULT_KEY = "key_unknown"

def round_ticks(ticks):
    return int(round(ticks / TIME_RESOLUTION) * TIME_RESOLUTION)

def load_chord_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chord_vocab = {}
    for k, v in raw.items():
        if k.startswith("chord_") and isinstance(v, list):
            chord_id = int(k.split("_", 1)[1])
            chord_vocab[tuple(v)] = chord_id
    # odwrotne mapowanie (tu nie używane, ale zostawiamy)
    rev_vocab = {v: k for k, v in chord_vocab.items()}
    return chord_vocab, rev_vocab

def tokenize_midi_file(
    midi_path,
    chord_vocab,
    max_notes_in_chord=MAX_NOTES_IN_CHORD,
    max_time_shift=MAX_TIME_SHIFT
):
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception as e:
        print(f"❌ Błąd podczas ładowania pliku {midi_path}: {e}")
        return None

    tokens = [DEFAULT_KEY]
    tempo = int(pm.get_tempo_changes()[1][0]) if pm.get_tempo_changes()[1].size > 0 else 120
    tokens.append(f"tempo_{tempo}")

    track_metadata = []
    track_chords = {}
    for track_index, instrument in enumerate(pm.instruments):
        if instrument.is_drum or not instrument.notes:
            continue
        track_metadata.append((track_index, instrument.program))
        notes = sorted(instrument.notes, key=lambda n: n.start)
        grouped = defaultdict(list)
        for note in notes:
            tick = round_ticks(pm.time_to_tick(note.start))
            grouped[tick].append(note)
        track_chords[track_index] = grouped

    if not track_chords:
        print(f"⚠️ Brak ścieżek instrumentalnych w {Path(midi_path).name}, pomijam")
        return None

    best_track_id = max(
        track_chords.items(),
        key=lambda kv: sum(1 for notes in kv[1].values() if len(notes) > 1)
    )[0]
    program = next(prog for idx, prog in track_metadata if idx == best_track_id)
    tokens.append("track_0")
    tokens.append(f"instrument_{program}")

    best_chords = track_chords[best_track_id]
    sorted_ticks = sorted(best_chords.keys())
    last_tick = 0

    for tick in sorted_ticks:
        delta = tick - last_tick
        if delta > 0:
            while delta > max_time_shift:
                tokens.append(f"time_shift_{max_time_shift}")
                delta -= max_time_shift
            if delta > 0:
                tokens.append(f"time_shift_{delta}")
        notes = best_chords[tick]
        chord_tuple = tuple(sorted(min(n.pitch, 127) for n in notes))[:max_notes_in_chord]
        chord_id = chord_vocab.get(chord_tuple, UNKNOWN_TOKEN_ID)
        tokens.append(f"chord_{chord_id}")
        avg_duration = sum(n.end - n.start for n in notes) / len(notes)
        duration_ticks = round_ticks(pm.time_to_tick(avg_duration))
        tokens.append(f"duration_{duration_ticks}")
        last_tick = tick

    # Filtrowanie tokenów
    filtered_tokens = []
    skip_next_duration = False
    for tok in tokens:
        if tok == f"time_shift_{max_time_shift}":
            continue
        if tok == f"chord_{UNKNOWN_TOKEN_ID}":
            skip_next_duration = True
            continue
        if skip_next_duration and tok.startswith("duration_"):
            skip_next_duration = False
            continue
        filtered_tokens.append(tok)
    tokens = filtered_tokens

    return tokens

def tokenize_files(
    midi_files,
    chord_vocab_path=CHORD_VOCAB_PATH,
    output_dir=DEFAULT_OUTPUT_DIR
):
    """
    Tokenizuje listę plików midi_files i zapisuje .txt do output_dir.
    """
    chord_vocab, rev_vocab = load_chord_vocab(chord_vocab_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_tokenized = 0
    for midi_path in midi_files:
        midi_path = Path(midi_path)
        tokens = tokenize_midi_file(midi_path, chord_vocab)
        if tokens is None:
            continue
        output_txt = output_dir / (midi_path.stem + ".txt")
        with open(output_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(tokens))
        num_tokenized += 1
        print(f"✅ Zapisano tokeny do: {output_txt}")
    print(f"Tokenizacja zakończona. Plików ztokenizowanych: {num_tokenized}/{len(midi_files)}.")

# Przykład użycia:
# from tokenize_user_input import tokenize_files
# tokenize_files(["file1.mid", "file2.mid"], output_dir="app/USER_TOKENIZED")
