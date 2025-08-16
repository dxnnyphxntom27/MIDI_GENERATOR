import json
from pathlib import Path

CHORD_VOCAB_PATH = Path("E:/MIDI_GENERATOR/transformer-best/chord_vocab.json")
OUT_PATH = Path("E:/MIDI_GENERATOR/transformer-best/multiple_notes_chord_list.txt")

def main():
    with open(CHORD_VOCAB_PATH, "r", encoding="utf-8") as f:
        chord_vocab = json.load(f)

    multi_note_chords = [
        chord for chord, notes in chord_vocab.items()
        if isinstance(notes, list) and len(notes) > 1
    ]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(multi_note_chords))

    print(f"✅ Zapisano {len(multi_note_chords)} akordów wielonutowych do {OUT_PATH}")

if __name__ == "__main__":
    main()
