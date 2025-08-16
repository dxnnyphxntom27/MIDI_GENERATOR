import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import json

# === ŚCIEŻKI ===
INPUT_TOKENS_PATH = "E:/MIDI_GENERATOR/transformer-based-2/generated_tokens.txt"
CHORD_VOCAB_PATH = "E:/MIDI_GENERATOR/transformer-based-2/chord_vocab.json"
OUTPUT_MIDI_PATH = "E:/MIDI_GENERATOR/transformer-based-2/generated_song.mid"

# === KONFIGURACJA ===
TICKS_PER_BEAT = 480

print(f"📄 Wczytywanie tokenów z: {INPUT_TOKENS_PATH}")
with open(INPUT_TOKENS_PATH, "r") as f:
    tokens = []
    for line in f:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        tokens.extend(parts)
print(f"🔢 Liczba tokenów: {len(tokens)}")

print(f"📄 Wczytywanie słownika akordów z: {CHORD_VOCAB_PATH}")
with open(CHORD_VOCAB_PATH, "r") as f:
    chord_vocab = json.load(f)
print(f"🔢 Liczba zdefiniowanych akordów: {len(chord_vocab)}")

# === UTWORZONY PLIK MIDI ===
mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
tracks = [MidiTrack()]
mid.tracks.append(tracks[0])

# === ZMIENNE STANU ===
track_times = {0: 0}
track_channels = {0: 0}
channel_counter = 1
current_track = 0
instrument = 0
tempo_set = False

found_chords = 0
not_found_chords = 0

print("🚦 Start przetwarzania tokenów...\n")

for idx, token in enumerate(tokens):
    print(f"[{idx}] ➡️ Token: {token}")

    if token.startswith("tempo_"):
        bpm_str = token[len("tempo_"):]
        try:
            bpm = int(bpm_str)
            if not tempo_set:
                tempo_msg = MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0)
                tracks[0].append(tempo_msg)
                print(f"🎵 Tempo ustawione: {bpm} BPM")
                tempo_set = True
        except ValueError:
            print(f"⚠️ Błędne tempo: {bpm_str}")

    elif token.startswith("track_"):
        current_track = int(token[len("track_"):])
        while len(tracks) <= current_track:
            new_track = MidiTrack()
            tracks.append(new_track)
            mid.tracks.append(new_track)
            print(f"➕ Dodano nowy track: {len(tracks)-1}")
        if current_track not in track_times:
            track_times[current_track] = 0
        if current_track not in track_channels:
            if channel_counter == 9:  # pomiń perkusyjny
                channel_counter += 1
            track_channels[current_track] = channel_counter
            print(f"🎚️ Track {current_track} przypisany do kanału {channel_counter}")
            channel_counter += 1

    elif token.startswith("instrument_"):
        instrument_str = token[len("instrument_"):]
        try:
            instrument = int(instrument_str)
            msg = Message("program_change", program=instrument, channel=track_channels[current_track], time=0)
            tracks[current_track].append(msg)
            print(f"🎻 Ustawiono instrument {instrument} na tracku {current_track}")
        except ValueError:
            print(f"⚠️ Błędny instrument: {instrument_str}")

    elif token.startswith("time_shift_"):
        shift_str = token[len("time_shift_"):]
        try:
            shift = int(shift_str)
            track_times[current_track] += shift
            print(f"⏱️ Track {current_track}: zwiększono czas o {shift} (total: {track_times[current_track]})")
        except ValueError:
            print(f"⚠️ Błędny time_shift: {shift_str}")

    elif token.startswith("chord_"):
        notes = chord_vocab.get(token)
        if notes is None:
            if token != "chord_5000":
                print(f"⚠️ Akord {token} nie znaleziony w słowniku.")
            not_found_chords += 1
            continue
        if not notes:
            print(f"⚠️ Akord {token} ma pustą listę nut.")
            not_found_chords += 1
            continue

        found_chords += 1

        time = track_times[current_track]
        print(f"🎶 Track {current_track} | Akord {token} | Nuty: {notes} | Opóźnienie: {time}")

        for i, note in enumerate(notes):
            note_time = time if i == 0 else 0
            msg_on = Message("note_on", note=note, velocity=64, time=note_time, channel=track_channels[current_track])
            msg_off = Message("note_off", note=note, velocity=64, time=TICKS_PER_BEAT // 2, channel=track_channels[current_track])
            tracks[current_track].append(msg_on)
            tracks[current_track].append(msg_off)
            print(f"   🎹 Nutka: {note} (ON z opóźnieniem {note_time}, OFF za {TICKS_PER_BEAT // 4})")

        track_times[current_track] = 0

    else:
        print(f"❓ Nierozpoznany token: {token}")

# === ZAPISZ WYNIK ===
mid.save(OUTPUT_MIDI_PATH)
print(f"\n💾 Plik zapisany: {OUTPUT_MIDI_PATH}")

print(f"\n✅ Akordy znalezione w słowniku: {found_chords}")
print(f"❌ Akordy NIE znalezione w słowniku: {not_found_chords}")
