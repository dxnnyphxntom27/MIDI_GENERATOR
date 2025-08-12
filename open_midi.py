import mido

# Ścieżka do pliku MIDI i pliku wynikowego
midi_path = 'E:/MIDI_GENERATOR/generated_song.mid'
output_path = 'text_midi_output.txt'

# Wczytaj plik MIDI
mid = mido.MidiFile(midi_path)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write(f'Format: {mid.type}, Ścieżki: {len(mid.tracks)}, Czas: {mid.length:.2f} sekund\n\n')

    for i, track in enumerate(mid.tracks):
        f.write(f'🎼 Track {i}: {track.name}\n')
        for msg in track:
            f.write(f'  {msg}\n')

print(f'Zapisano dane MIDI do pliku: {output_path}')