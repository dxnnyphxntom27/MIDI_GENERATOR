import pickle

pkl_file = 'E:\MIDI_GENERATOR\Processed\Toto\Africa.pkl'

with open(pkl_file, 'rb') as file:
    data = pickle.load(file)

for i, track in enumerate(data):
    print(f"Track {i + 1}:")
    print(f"  Program: {track['program']}")
    print(f"  Notes Count: {len(track['notes'])}")
    for note in track['notes']:
        print(f"    Pitch: {note['pitch']}, Start: {note['start']}, End: {note['end']}, Velocity: {note['velocity']}")