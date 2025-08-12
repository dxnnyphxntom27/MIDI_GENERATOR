import os
from pathlib import Path
import pretty_midi
import music21
import json
from collections import Counter
import tempfile
import concurrent.futures

# === ŚCIEŻKI ===
MIDI_DIR = Path("E:/MIDI_GENERATOR/Dataset")
OUTPUT_DIR = Path("E:/MIDI_GENERATOR/transformer-based-2/TokenizedWithChords")
VOCAB_PATH = Path("E:/MIDI_GENERATOR/transformer-based-2/chord_vocab.json")
VOCAB_REV_PATH = Path("E:/MIDI_GENERATOR/transformer-based-2/chord_vocab_reverse.json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === PARAMETRY ===
TIME_RESOLUTION = 24
MAX_NOTES_IN_CHORD = 6
TOP_K_CHORDS = 5000
UNKNOWN_TOKEN_ID = TOP_K_CHORDS

def round_ticks(ticks):
    return int(round(ticks / TIME_RESOLUTION) * TIME_RESOLUTION)

def detect_key(pm):
    try:
        print("Saving MIDI to temporary file...")
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            pm.write(tmp.name)
            tmp_path = tmp.name
        print(f"Parsing MIDI file {tmp_path} with music21...")
        stream = music21.converter.parse(tmp_path)
        print("Analyzing key...")
        key = stream.analyze("key")
        print(f"Key analysis result: tonic={key.tonic.name}, mode={key.mode}")
        os.remove(tmp_path)
        return f"key_{key.tonic.name}_{key.mode}"
    except Exception as e:
        print("Exception in detect_key:", e)
        return "key_unknown"

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
    tokens = [detect_key(pm)]
    tempo = int(pm.get_tempo_changes()[1][0]) if pm.get_tempo_changes()[1].size > 0 else 120
    tokens.append(f"tempo_{tempo}")
    for track_index, instrument in enumerate(pm.instruments):
        tokens.append(f"track_{track_index}")
        tokens.append(f"instrument_{instrument.program}")
        notes = sorted(instrument.notes, key=lambda n: n.start)
        last_time = 0.0
        current_chord = []
        for note in notes:
            delta_ticks = round_ticks(pm.time_to_tick(note.start - last_time))
            if delta_ticks > 0:
                if current_chord:
                    chord = tuple(sorted(min(n.pitch, 127) for n in current_chord))[:MAX_NOTES_IN_CHORD]
                    chord_id = chord_vocab.get(chord, UNKNOWN_TOKEN_ID)
                    tokens.append(f"chord_{chord_id}")
                    current_chord.clear()
                tokens.append(f"time_shift_{delta_ticks}")
                last_time = note.start
            current_chord.append(note)
        if current_chord:
            chord = tuple(sorted(min(n.pitch, 127) for n in current_chord))[:MAX_NOTES_IN_CHORD]
            chord_id = chord_vocab.get(chord, UNKNOWN_TOKEN_ID)
            tokens.append(f"chord_{chord_id}")
    with open(output_artist_dir / (midi_file.stem + ".txt"), "w") as f:
        f.write("\n".join(tokens))

def tokenize(chord_vocab):
    reverse_vocab = {v: k for k, v in chord_vocab.items()}
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
    with open(VOCAB_PATH, "w") as f:
        json.dump({f"chord_{v}": list(k) if k != "<UNKNOWN>" else "<UNKNOWN>" for k, v in chord_vocab.items()}, f, indent=2)
    with open(VOCAB_REV_PATH, "w") as f:
        json.dump({str(v): list(k) if k != "<UNKNOWN>" else "<UNKNOWN>" for k, v in chord_vocab.items()}, f, indent=2)
    print(f"✅ Zapisano słowniki: {VOCAB_PATH.name} i {VOCAB_REV_PATH.name} ({TOP_K_CHORDS} najczęstszych + UNKNOWN)\n")
    tokenize(chord_vocab)
    print("\n🎯 Tokenizacja zakończona.")

if __name__ == "__main__":
    main()
