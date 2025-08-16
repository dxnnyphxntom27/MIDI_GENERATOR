import os
from pathlib import Path
import pretty_midi
import music21
import json
from collections import Counter, defaultdict
import tempfile
import concurrent.futures

# === ŚCIEŻKI ===
MIDI_DIR = Path("E:/MIDI_GENERATOR/Dataset")
OUTPUT_DIR = Path("E:/MIDI_GENERATOR/transformer-final/TokenizedWithChords")
VOCAB_PATH = Path("E:/MIDI_GENERATOR/transformer-final/chord_vocab.json")
VOCAB_REV_PATH = Path("E:/MIDI_GENERATOR/transformer-final/chord_vocab_reverse.json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
# === PARAMETRY ===
TIME_RESOLUTION = 24
MAX_NOTES_IN_CHORD = 6
TOP_K_CHORDS = 5000
UNKNOWN_TOKEN_ID = TOP_K_CHORDS
MAX_TIME_SHIFT = 192                 # maksymalny pojedynczy time_shift
MAX_CONSECUTIVE_SHIFTS = 3            # ile maksymalnie time_shift_MAX_TIME_SHIFT pod rząd
USE_KEY_DETECTION = False              # czy analizować tonację
DEFAULT_KEY = "key_unknown"           # jeśli USE_KEY_DETECTION=False

def round_ticks(ticks):
    return int(round(ticks / TIME_RESOLUTION) * TIME_RESOLUTION)

def detect_key(pm):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            pm.write(tmp.name)
            tmp_path = tmp.name
        stream = music21.converter.parse(tmp_path)
        key = stream.analyze("key")
        return f"key_{key.tonic.name}_{key.mode}"
    except Exception as e:
        print("Exception in detect_key:", e)
        return "key_unknown"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

def extract_chord_counts():
    counter = Counter()
    print("📊 Zliczanie występowania akordów we wszystkich plikach...\n")
    total_files = sum(1 for artist in MIDI_DIR.iterdir() if artist.is_dir() for _ in artist.glob("*.mid"))
    processed = 0
    for artist_dir in MIDI_DIR.iterdir():
        if not artist_dir.is_dir():
            continue
        print(f"🎤 Artysta: {artist_dir.name}")
        for midi_file in artist_dir.glob("*.mid"):
            processed += 1
            print(f"   [{processed}/{total_files}] 🎵 {midi_file.name}")
            try:
                pm = pretty_midi.PrettyMIDI(str(midi_file))
            except Exception as e:
                print(f"   ❌ Pomiń: {e}")
                continue
            for instrument in pm.instruments:
                if instrument.is_drum:
                    continue
                notes = sorted(instrument.notes, key=lambda n: n.start)
                last_time = 0.0
                current_chord = []
                for note in notes:
                    delta_ticks = round_ticks(pm.time_to_tick(note.start - last_time))
                    if delta_ticks > 0:
                        if current_chord:
                            chord = tuple(sorted(min(n.pitch, 127) for n in current_chord))[:MAX_NOTES_IN_CHORD]
                            counter[chord] += 1
                            current_chord.clear()
                        last_time = note.start
                    current_chord.append(note)
                if current_chord:
                    chord = tuple(sorted(min(n.pitch, 127) for n in current_chord))[:MAX_NOTES_IN_CHORD]
                    counter[chord] += 1
    print(f"\n📦 Zliczono {len(counter)} unikalnych akordów.\n")
    return counter

def process_tokenize_file(args):
    midi_file, chord_vocab, output_artist_dir = args
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_file))
    except Exception as e:
        print(f"❌ {midi_file.name}: {e}")
        return

    # Tonacja
    if USE_KEY_DETECTION:
        tokens = [detect_key(pm)]
    else:
        tokens = [DEFAULT_KEY]

    # Tempo
    tempo = int(pm.get_tempo_changes()[1][0]) if pm.get_tempo_changes()[1].size > 0 else 120
    tokens.append(f"tempo_{tempo}")

    # Metadane i nuty dla każdej ścieżki
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

    # Jeśli brak ścieżek instrumentalnych → pomijamy
    if not track_chords:
        print(f"⚠️ Brak ścieżek instrumentalnych w {midi_file.name}, pomijam")
        return

    # Wybór najlepszej ścieżki po liczbie akordów
    best_track_id = max(
        track_chords.items(),
        key=lambda kv: sum(1 for notes in kv[1].values() if len(notes) > 1)
    )[0]
    program = next(prog for idx, prog in track_metadata if idx == best_track_id)

    # Dodajemy track_0 i instrument
    tokens.append("track_0")
    tokens.append(f"instrument_{program}")

    # Generowanie tokenów nut
    best_chords = track_chords[best_track_id]
    sorted_ticks = sorted(best_chords.keys())
    last_tick = 0

    for tick in sorted_ticks:
        delta = tick - last_tick
        if delta > 0:
            while delta > MAX_TIME_SHIFT:
                tokens.append(f"time_shift_{MAX_TIME_SHIFT}")
                delta -= MAX_TIME_SHIFT
            if delta > 0:
                tokens.append(f"time_shift_{delta}")
        notes = best_chords[tick]
        chord = tuple(sorted(min(n.pitch, 127) for n in notes))[:MAX_NOTES_IN_CHORD]
        chord_id = chord_vocab.get(chord, UNKNOWN_TOKEN_ID)
        tokens.append(f"chord_{chord_id}")
        avg_duration = sum(n.end - n.start for n in notes) / len(notes)
        duration_ticks = round_ticks(pm.time_to_tick(avg_duration))
        tokens.append(f"duration_{duration_ticks}")
        last_tick = tick

    # Filtr skracający zbyt długie pauzy
    filtered_tokens = []
    skip_next_duration = False
    for tok in tokens:
        if tok == f"time_shift_{MAX_TIME_SHIFT}":
            continue
        if tok == f"chord_{UNKNOWN_TOKEN_ID}":

            skip_next_duration = True
            continue
        if skip_next_duration and tok.startswith("duration_"):
            skip_next_duration = False
            continue
        filtered_tokens.append(tok)
    tokens = filtered_tokens


    # Zapis do pliku
    with open(output_artist_dir / (midi_file.stem + ".txt"), "w") as f:
        f.write("\n".join(tokens))

def tokenize(chord_vocab):
    tasks = []
    for artist_dir in MIDI_DIR.iterdir():
        if not artist_dir.is_dir():
            continue
        output_artist_dir = OUTPUT_DIR / artist_dir.name
        output_artist_dir.mkdir(parents=True, exist_ok=True)
        for midi_file in artist_dir.glob("*.mid"):
            tasks.append((midi_file, chord_vocab, output_artist_dir))
    print(f"Tokenizing {len(tasks)} files in parallel...")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        list(executor.map(process_tokenize_file, tasks))

def main():
    chord_counts = extract_chord_counts()
    most_common = dict(chord_counts.most_common(TOP_K_CHORDS))
    chord_vocab = {chord: idx for idx, chord in enumerate(most_common)}
    chord_vocab["<UNKNOWN>"] = UNKNOWN_TOKEN_ID

    # Poprawny zapis JSON
    with open(VOCAB_PATH, "w") as f:
        json.dump(
            {f"chord_{v}": list(k) if isinstance(k, tuple) else k for k, v in chord_vocab.items()},
            f,
            indent=2
        )
    with open(VOCAB_REV_PATH, "w") as f:
        json.dump(
            {str(v): list(k) if isinstance(k, tuple) else k for k, v in chord_vocab.items()},
            f,
            indent=2
        )

    print(f"✅ Zapisano słowniki: {VOCAB_PATH.name} i {VOCAB_REV_PATH.name} ({TOP_K_CHORDS} najczęstszych + UNKNOWN)\n")
    tokenize(chord_vocab)
    print("\n🎯 Tokenizacja zakończona.")

if __name__ == "__main__":
    main()
