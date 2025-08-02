import mido
from mido import MidiFile, MidiTrack, Message
import os

# 📥 Plik z tokenami (wygenerowany wcześniej)
INPUT_TOKENS_PATH = "generated_tokens.txt"
OUTPUT_MIDI_PATH = "generated_song.mid"

# ⏱️ Ustal czas dla wait_X
TICKS_PER_BEAT = 480
WAIT_RESOLUTION = 20  # czyli wait_1 = 24 ticków (120 BPM = 0.5 sek/beat)

# 🛠️ Utwórz plik MIDI
mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
track = MidiTrack()
mid.tracks.append(track)

current_program = 0
current_time = 0

with open(INPUT_TOKENS_PATH, "r") as f:
    tokens = [line.strip() for line in f if line.strip()]

for token in tokens:
    if token.startswith("instrument_"):
        current_program = int(token.split("_")[1])
        track.append(Message("program_change", program=current_program, time=current_time))
        current_time = 0

    elif token.startswith("note_on_"):
        note = int(token.split("_")[2])
        track.append(Message("note_on", note=note, velocity=64, time=current_time))
        current_time = 0

    elif token.startswith("note_off_"):
        note = int(token.split("_")[2])
        track.append(Message("note_off", note=note, velocity=64, time=current_time))
        current_time = 0

    elif token.startswith("wait_"):
        wait = int(token.split("_")[1])
        current_time += wait * WAIT_RESOLUTION  # dodaj do bieżącego czasu

# 📝 Zapisz MIDI
mid.save(OUTPUT_MIDI_PATH)
print(f"✅ MIDI zapisane jako {OUTPUT_MIDI_PATH}")
