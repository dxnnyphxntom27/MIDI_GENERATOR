import os
from pathlib import Path
import pretty_midi
import music21
import json
from collections import Counter
import tempfile

# === ŚCIEŻKI ===
MIDI_DIR = Path("E:/MIDI_GENERATOR/Dataset/Elvis_Presley")

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
        return "niema"

if __name__ == "__main__":
    midi_dir = Path(MIDI_DIR)
    midi_files = list(midi_dir.glob("*.mid"))
    print(f"Found {len(midi_files)} MIDI files in {midi_dir}")
    for midi_file in midi_files:
        print(f"\nProcessing: {midi_file}")
        try:
            pm = pretty_midi.PrettyMIDI(str(midi_file))
            key = detect_key(pm)
            print(f"Detected key: {key}")
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")