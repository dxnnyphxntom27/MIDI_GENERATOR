import mido
from mido import MidiFile, MidiTrack, Message
import os

# 📥 Plik z tokenami
INPUT_TOKENS_PATH = "generated_tokens.txt"
OUTPUT_MIDI_PATH = "generated_song.mid"

# ⏱️ Konfiguracja czasu
TICKS_PER_BEAT = 480
WAIT_RESOLUTION = 24  # wait_1 = 24 ticków = 50ms przy 120BPM

# 🛠️ Inicjalizacja MIDI
mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
track = MidiTrack()
mid.tracks.append(track)

current_program = 0
current_time = 0
pending_notes = []  # nuty oczekujące na duration

# 📖 Wczytaj tokeny
with open(INPUT_TOKENS_PATH, "r") as f:
    content = f.read().strip()
    tokens = content.split()

print(f"📦 Liczba tokenów: {len(tokens)}")
print(f"📊 Przykładowe tokeny: {tokens[:20]}")
print(f"📌 Unikalne typy: {set(t.split('_')[0] for t in tokens)}")
print(f"📦 Liczba tokenów: {len(tokens)}")
print(f"📊 Przykładowe tokeny: {tokens[:20]}")
print(f"📌 Unikalne typy: {set(t.split('_')[0] for t in tokens)}")


for token in tokens:
    if token.startswith("instrument_"):
        current_program = int(token.split("_")[1])
        track.append(Message("program_change", program=current_program, time=current_time))
        current_time = 0

    elif token.startswith("wait_"):
        wait = int(token.split("_")[1])
        current_time += wait * WAIT_RESOLUTION

    elif token.startswith("note_on_"):
        note = int(token.split("_")[2])
        pending_notes.append(note)

    elif token.startswith("duration_"):
        duration = int(token.split("_")[1]) * WAIT_RESOLUTION

        for i, note in enumerate(pending_notes):
            # pierwsza nuta dostaje accumulated `current_time`, reszta 0
            track.append(Message("note_on", note=note, velocity=64, time=current_time if i == 0 else 0))
            track.append(Message("note_off", note=note, velocity=64, time=duration))
        current_time = 0
        pending_notes = []

# 📝 Zapisz plik MIDI
mid.save(OUTPUT_MIDI_PATH)
print(f"✅ MIDI zapisane jako {OUTPUT_MIDI_PATH}")
