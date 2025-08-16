# PRZYSPIESZENIE SKRYPTU + filtrowanie gównianych akordów
import json
from pathlib import Path
import torch
import torch.nn.functional as F
import pretty_midi
from model import TransformerModel

# === ŚCIEŻKI ===
BASE = Path("E:/MIDI_GENERATOR/transformer-best")
VOCAB_PATH = BASE / "vocab.json"
VOCAB_REV_PATH = BASE / "vocab_reverse.json"
CHORD_VOCAB_PATH = BASE / "chord_vocab.json"
MULTI_NOTE_CHORDS_PATH = BASE / "multiple_notes_chord_list.txt"
CHECKPOINT_PATH = BASE / "checkpoints/best_model.pt"
OUTPUT_MIDI = BASE / "generated.mid"

# === PARAMETRY GENERACJI ===
MAX_TOKENS = 100
START_TOKENS = ["key_unknown", "tempo_135", "track_0", "instrument_0"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
TIME_RESOLUTION = 24  # ticks per beat
TEMPERATURE = 1
TOP_K = 50
STOP_TOKEN = "<SONG_END>"
MAX_TRIES = 10

# --- Pomocnicze ---
def load_vocab():
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    for k, v in list(token_to_id.items()):
        if isinstance(v, str) and v.isdigit():
            token_to_id[k] = int(v)
    if VOCAB_REV_PATH.exists():
        with open(VOCAB_REV_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        id_to_token = {int(k): v for k, v in raw.items()}
    else:
        id_to_token = {int(v): k for k, v in token_to_id.items()}
    chord_id_to_notes = {}
    if CHORD_VOCAB_PATH.exists():
        with open(CHORD_VOCAB_PATH, "r", encoding="utf-8") as f:
            chord_vocab = json.load(f)
        for tok, notes in chord_vocab.items():
            if not tok.startswith("chord_"):
                continue
            try:
                cid = int(tok.split("_", 1)[1])
            except Exception:
                continue
            if isinstance(notes, list):
                chord_id_to_notes[cid] = list(map(int, notes))
    return token_to_id, id_to_token, chord_id_to_notes

def load_multi_note_chords(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def has_enough_harmonic_content(tokens, multi_note_chords, min_unique=2):
    chords_in_gen = {tok for tok in tokens if tok in multi_note_chords}
    return len(chords_in_gen) >= min_unique

def process_durations_and_time_shifts(tokens, min_tick=1):
    result = []
    for token in tokens:
        if isinstance(token, str) and (token.startswith("duration_") or token.startswith("time_shift_")):
            prefix, val = token.rsplit("_", 1)
            try:
                new_val = max(int(int(val) / 8), min_tick)
                result.append(f"{prefix}_{new_val}")
            except Exception:
                result.append(token)
        else:
            result.append(token)
    return result

def sample_from_logits(logits, temperature=1.0, top_k=None):
    if temperature == 0:
        return int(torch.argmax(logits).item())
    logits = logits / (temperature if temperature > 0 else 1.0)
    if top_k is not None and top_k > 0:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        probs = F.softmax(topk_vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1).item()
        return int(topk_idx[choice].item())
    else:
        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

def tokens_to_midi(tokens, chord_id_to_notes, output_path,
                   time_resolution=24,
                   default_tempo=120,
                   min_tempo=20,
                   max_tempo=300,
                   max_beats=64,
                   verbose=True):
    import statistics
    print("\n--- DEBUG: INICJALIZACJA PrettyMIDI ---")
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(default_tempo))
    print(f"Utworzono PrettyMIDI z initial_tempo (argument do konstruktora) = {default_tempo}")
    pm.resolution = time_resolution
    current_tempo = default_tempo
    print(f"DEBUG: Startowy current_tempo: {current_tempo}")
    current_tick = 0
    current_track_idx = None
    instruments = {}
    durations = {}
    cnt_time_shift = 0
    cnt_duration = 0
    cnt_chord = 0
    duration_vals = []
    N = len(tokens)
    def secs_per_tick(bpm):
        return 60.0 / (bpm * time_resolution)
    i = 0
    while i < N:
        token = tokens[i]
        if token is None:
            i += 1
            continue
        # tempo
        if isinstance(token, str) and token.startswith("tempo_"):
            try:
                bpm = float(token.split("_", 1)[1])
            except Exception:
                bpm = current_tempo
            clamped = False
            if bpm < min_tempo or bpm > max_tempo:
                bpm = max(min(bpm, max_tempo), min_tempo)
                clamped = True
            current_tempo = bpm
            pm._tick_scales.append((current_tick, 60.0 / current_tempo / time_resolution))
            i += 1
            continue
        if isinstance(token, str) and token.startswith("track_"):
            try:
                current_track_idx = int(token.split("_", 1)[1])
            except Exception:
                current_track_idx = None
            if current_track_idx is not None and current_track_idx not in instruments:
                instruments[current_track_idx] = pretty_midi.Instrument(program=0)
                durations[current_track_idx] = time_resolution
            i += 1
            continue
        if isinstance(token, str) and token.startswith("instrument_") and current_track_idx is not None:
            try:
                program_num = int(token.split("_", 1)[1])
                instruments[current_track_idx].program = program_num
            except Exception:
                pass
            i += 1
            continue
        if isinstance(token, str) and token.startswith("time_shift_"):
            try:
                shift = int(token.rsplit("_", 1)[1])
            except Exception:
                shift = 0
            current_tick += shift
            cnt_time_shift += 1
            i += 1
            continue
        if isinstance(token, str) and token.startswith("duration_") and current_track_idx is not None:
            try:
                dur = int(token.split("_", 1)[1])
            except Exception:
                dur = time_resolution
            max_ticks = int(time_resolution * max_beats)
            if dur <= 0:
                dur = time_resolution
            if dur > max_ticks:
                dur = max_ticks
            durations[current_track_idx] = dur
            cnt_duration += 1
            duration_vals.append(dur)
            i += 1
            continue
        if isinstance(token, str) and token.startswith("chord_") and current_track_idx is not None:
            try:
                chord_id = int(token.split("_", 1)[1])
            except Exception:
                chord_id = None
            if chord_id is not None and chord_id in chord_id_to_notes:
                notes = chord_id_to_notes[chord_id]
                start_seconds = current_tick * secs_per_tick(current_tempo)
                dur_ticks = durations.get(current_track_idx, time_resolution)
                end_seconds = (current_tick + dur_ticks) * secs_per_tick(current_tempo)
                for pitch in notes:
                    p = int(pitch)
                    if 0 <= p <= 127:
                        note = pretty_midi.Note(velocity=80, pitch=p, start=start_seconds, end=end_seconds)
                        instruments[current_track_idx].notes.append(note)
                cnt_chord += 1
            next_token = tokens[i+1] if (i+1) < N else None
            if not (isinstance(next_token, str) and next_token.startswith("time_shift_")):
                dur_ticks = durations.get(current_track_idx, time_resolution)
                current_tick += dur_ticks
            i += 1
            continue
        if token == "<SONG_END>":
            current_tick += time_resolution
            i += 1
            continue
        i += 1
    for idx in sorted(instruments.keys()):
        pm.instruments.append(instruments[idx])
    if verbose:
        print(f"Format: 1, Ścieżki: {len(instruments)}, Czas (s): {sum(len(inst.notes) for inst in pm.instruments) and (max((n.end for inst in pm.instruments for n in inst.notes), default=0)):.2f}")
        print(f"🎚 time_shift count: {cnt_time_shift}, duration tokens: {cnt_duration}, chord tokens placed: {cnt_chord}")
        if duration_vals:
            import statistics
            print(f"Durations (ticks) — min:{min(duration_vals)}, max:{max(duration_vals)}, mean:{statistics.mean(duration_vals):.1f}")
    pm.write(str(output_path))
    print(f"DEBUG: Plik MIDI zapisany do: {output_path}")
    return output_path

def generate_tokens(model, token_to_id, id_to_token, max_tokens=500, start_tokens=None,
                    temperature=1.0, top_k=None, stop_token=STOP_TOKEN):
    model.eval()
    start_tokens = start_tokens or []
    unk_id = token_to_id.get("<UNKNOWN>", None)
    input_tokens = []
    for t in start_tokens:
        if t in token_to_id:
            input_tokens.append(t)
        else:
            if stop_token in token_to_id:
                input_tokens.append(stop_token)
            elif unk_id is not None:
                input_tokens.append("<UNKNOWN>")
            else:
                input_tokens.append(list(token_to_id.keys())[0])
    input_ids = [token_to_id.get(t, token_to_id.get("<UNKNOWN>", 0)) for t in input_tokens]
    generated_tokens = input_tokens.copy()
    for step in range(max_tokens - len(input_ids)):
        tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            last_logits = logits[0, -1]
            next_id = sample_from_logits(last_logits.cpu(), temperature=temperature, top_k=top_k)
            next_token = id_to_token.get(int(next_id), None)
        if next_token is None:
            break
        generated_tokens.append(next_token)
        input_ids.append(next_id)
        if next_token == stop_token:
            break
    return generated_tokens

def main():
    token_to_id, id_to_token, chord_id_to_notes = load_vocab()
    multi_note_chords = load_multi_note_chords(MULTI_NOTE_CHORDS_PATH)
    model = TransformerModel(
        vocab_size=len(token_to_id),
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1
    ).to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE)
    print(f"✅ Wczytano model z {CHECKPOINT_PATH}")
    for attempt in range(1, MAX_TRIES + 1):
        print(f"\n🎲 Próba generacji #{attempt}...")
        tokens = generate_tokens(
            model=model,
            token_to_id=token_to_id,
            id_to_token=id_to_token,
            max_tokens=MAX_TOKENS,
            start_tokens=START_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            stop_token=STOP_TOKEN
        )
        if has_enough_harmonic_content(tokens, multi_note_chords):
            print(f"✅ Sukces: Znaleziono {len(tokens)} tokenów z wystarczającą ilością akordów")
            break
        else:
            print("⚠️ Niewystarczająca liczba różnych wielonutowych akordów — ponawiam próbę.")
    else:
        print("❌ Nie udało się wygenerować muzyki z wystarczającą ilością akordów.")
        return
    tokens = process_durations_and_time_shifts(tokens)
    midi_path = tokens_to_midi(tokens, chord_id_to_notes, OUTPUT_MIDI)
    print(f"💾 Zapisano plik MIDI: {midi_path}")

if __name__ == "__main__":
    main()
