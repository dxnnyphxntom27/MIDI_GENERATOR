import os
import pretty_midi
import pickle
from pathlib import Path

# === ŚCIEŻKI ===
DATASET_DIR = Path("E:/MIDI_GENERATOR/Dataset")
OUTPUT_DIR = Path("E:/MIDI_GENERATOR/PreprocessedWithDrums")

def midi_to_multitrack_note_sequence(midi_path):
    """Zwraca listę tracków: {program, notes[], is_drum?}"""
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    track_list = []

    for instrument in midi_data.instruments:
        if not instrument.notes:
            continue  # pomiń puste

        notes = []
        for note in instrument.notes:
            notes.append({
                'pitch': note.pitch,
                'start': note.start,
                'end': note.end,
                'velocity': note.velocity
            })

        notes.sort(key=lambda x: x['start'])

        track = {
            'program': instrument.program,
            'notes': notes
        }

        if instrument.is_drum:
            track['is_drum'] = True

        track_list.append(track)

    return track_list if track_list else None


def process_dataset(dataset_dir, output_dir):
    """Przetwarza pliki MIDI i zapisuje każdy jako osobny plik .pkl"""
    for artist_dir in dataset_dir.iterdir():
        if not artist_dir.is_dir():
            continue

        artist_name = artist_dir.name
        output_artist_dir = output_dir / artist_name
        output_artist_dir.mkdir(parents=True, exist_ok=True)

        print(f"🥁 Przetwarzam artystę: {artist_name}")

        for midi_file in artist_dir.glob("*.mid"):
            try:
                track_list = midi_to_multitrack_note_sequence(str(midi_file))
                if not track_list:
                    continue

                output_file = output_artist_dir / f"{midi_file.stem}.pkl"
                with open(output_file, "wb") as f:
                    pickle.dump(track_list, f)

                print(f"   ✅ {midi_file.name} zapisany ({len(track_list)} tracków)")

            except Exception as e:
                print(f"   ❌ Błąd przy {midi_file.name}: {e}")


if __name__ == "__main__":
    print("⏳ Rozpoczynam przetwarzanie z perkusją...")
    process_dataset(DATASET_DIR, OUTPUT_DIR)
    print("✅ Gotowe — wersja z perkusją zapisana.")
