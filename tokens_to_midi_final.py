import mido
from mido import MidiFile, MidiTrack, Message
import json

# 📂 Ścieżki
INPUT_TOKENS_PATH = "generated_tokens.txt"
CHORD_VOCAB_PATH = "E:/MIDI_GENERATOR/chord_vocab.json"
OUTPUT_MIDI_PATH = "generated_song.mid"

# ⏱️ Konfiguracja
TICKS_PER_BEAT = 480
WAIT_RESOLUTION = 24  # time_shift_1 = 24 ticków (50ms przy 120BPM)
NOTE_DURATION = 480   # długość każdej nuty = ćwierćnuta

# 📘 Wczytaj chord_vocab.json
with open(CHORD_VOCAB_PATH, "r") as f:
    chord_vocab_raw = json.load(f)

# Zamień stringi tuple na listy pitchów
chord_map = {}
for k, v in chord_vocab_raw.items():
    try:
        pitches = [int(p) for p in k.strip("()").split(",") if p.strip().isdigit()]
        if pitches:
            chord_map[f"chord_{v}"] = pitches
    except Exception as e:
        print(f"⚠️ Błąd przy akordzie {k}: {e}")

        chord_map[f"chord_{v}"] = pitches

# 📖 Wczytaj tokeny
with open(INPUT_TOKENS_PATH, "r") as f:
    tokens = f.read().strip().split()

# 🛠️ Utwórz MIDI
mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
track = MidiTrack()
mid.tracks.append(track)

# 🎹 Ustaw domyślny instrument
track.append(Message("program_change", program=0, time=0))

current_time = 0

for token in tokens:
    if token.startswith("time_shift_"):
        shift = int(token.split("_")[2])
        current_time += shift * WAIT_RESOLUTION

    elif token.startswith("chord_"):
        pitches = chord_map.get(token)
        if not pitches:
            continue  # nieznany akord

        # Dodaj note_on dla każdego pitcha
        for i, pitch in enumerate(pitches):
            track.append(Message("note_on", note=pitch, velocity=64, time=current_time if i == 0 else 0))

        # Dodaj note_off po NOTE_DURATION tickach
        for i, pitch in enumerate(pitches):
            track.append(Message("note_off", note=pitch, velocity=64, time=NOTE_DURATION if i == 0 else 0))

        current_time = 0  # resetujemy czas po zagraniu akordu

# 💾 Zapisz wynik
mid.save(OUTPUT_MIDI_PATH)
print(f"✅ Zapisano plik MIDI jako: {OUTPUT_MIDI_PATH}")
