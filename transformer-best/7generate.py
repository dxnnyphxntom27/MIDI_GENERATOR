# generate.py (z samplingiem, obsługą <SONG_END> i poprawnym tick->seconds)
import json
import torch
import torch.nn.functional as F
import pretty_midi
from pathlib import Path
from model import TransformerModel

# === ŚCIEŻKI ===
BASE = Path("E:/MIDI_GENERATOR/transformer-best")
VOCAB_PATH = BASE / "vocab.json"
VOCAB_REV_PATH = BASE / "vocab_reverse.json"   # if you have this; otherwise we'll build from vocab.json
CHORD_VOCAB_PATH = BASE / "chord_vocab.json"
CHECKPOINT_PATH = BASE / "checkpoints/best_model.pt"
OUTPUT_MIDI = BASE / "generated.mid"

# === PARAMETRY GENERACJI ===
MAX_TOKENS = 500
START_TOKENS = ["key_unknown", "tempo_120", "track_0", "instrument_0"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIME_RESOLUTION = 24  # ticks per beat (used as ticks_per_beat)
TEMPERATURE = 1.0     # 0 -> argmax, >0 -> sampling
TOP_K = 50            # None or int
STOP_TOKEN = "<SONG_END>"

# --- pomocnicze ---
def load_vocab():
    # token -> id
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    # ensure ints
    for k, v in list(token_to_id.items()):
        if isinstance(v, str) and v.isdigit():
            token_to_id[k] = int(v)

    # id -> token (try to load reverse file if present)
    if VOCAB_REV_PATH.exists():
        with open(VOCAB_REV_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # raw keys might be strings
        id_to_token = {int(k): v for k, v in raw.items()}
    else:
        id_to_token = {int(v): k for k, v in token_to_id.items()}

    # chord id -> list of midi pitches
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
            # else: "<UNKNOWN>" or similar -> skip
    return token_to_id, id_to_token, chord_id_to_notes

def sample_from_logits(logits, temperature=1.0, top_k=None):
    """
    logits: 1D tensor of shape (vocab_size,)
    returns: sampled token id (int)
    """
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
    """
    Bezpieczna konwersja tokenów -> PrettyMIDI, z clampowaniem tempa i duration.

    - time_resolution: ticks per beat (u Ciebie 24)
    - default_tempo: BPM, jeśli nie ma tokenu tempo_...
    - min_tempo / max_tempo: dopuszczalny zakres BPM
    - max_beats: maksymalna liczba taktów dozwolona dla duration (safety clamp)
    """

    import statistics
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(default_tempo))
    pm.resolution = time_resolution

    current_tempo = default_tempo
    current_tick = 0
    current_track_idx = None
    instruments = {}
    durations = {}  # per-track last seen duration in ticks

    # diagnostics
    cnt_time_shift = 0
    cnt_duration = 0
    cnt_chord = 0
    duration_vals = []

    N = len(tokens)

    # helper: seconds per tick given BPM
    def secs_per_tick(bpm):
        return 60.0 / (bpm * time_resolution)

    i = 0
    while i < N:
        token = tokens[i]
        if token is None:
            i += 1
            continue

        # tempo_X token -> parse BPM, clamp, and set PrettyMIDI initial tempo (first tempo encountered)
        if isinstance(token, str) and token.startswith("tempo_"):
            try:
                bpm = float(token.split("_", 1)[1])
            except Exception:
                bpm = current_tempo
            # clamp BPM to rozsądny zakres
            if bpm < min_tempo or bpm > max_tempo:
                if verbose:
                    print(f"⚠️ Tempo {bpm} poza zakresem [{min_tempo},{max_tempo}] — clampuję.")
                bpm = max(min(bpm, max_tempo), min_tempo)
            current_tempo = bpm
            # update PrettyMIDI initial tempo for writing out correct meta (sets base tempo)
            # pretty_midi accepts initial_tempo via constructor; to update, set attribute:
            pm.initial_tempo = float(current_tempo)
            i += 1
            continue

        # track_X
        if isinstance(token, str) and token.startswith("track_"):
            try:
                current_track_idx = int(token.split("_", 1)[1])
            except Exception:
                current_track_idx = None
            if current_track_idx is not None and current_track_idx not in instruments:
                instruments[current_track_idx] = pretty_midi.Instrument(program=0)
                durations[current_track_idx] = time_resolution  # default = 1 beat
            i += 1
            continue

        # instrument_X
        if isinstance(token, str) and token.startswith("instrument_") and current_track_idx is not None:
            try:
                program_num = int(token.split("_", 1)[1])
                instruments[current_track_idx].program = program_num
            except Exception:
                pass
            i += 1
            continue

        # time_shift_X -> move current_tick forward
        if isinstance(token, str) and token.startswith("time_shift_"):
            try:
                shift = int(token.split("_", 1)[1])
            except Exception:
                shift = 0
            current_tick += shift
            cnt_time_shift += 1
            i += 1
            continue

        # duration_X -> store last seen duration for current track
        if isinstance(token, str) and token.startswith("duration_") and current_track_idx is not None:
            try:
                dur = int(token.split("_", 1)[1])
            except Exception:
                dur = time_resolution
            # clamp duration to reasonable max (in ticks)
            max_ticks = int(time_resolution * max_beats)
            if dur <= 0:
                dur = time_resolution
            if dur > max_ticks:
                if verbose:
                    print(f"⚠️ duration {dur} too large -> clamp to {max_ticks} (max_beats={max_beats})")
                dur = max_ticks
            durations[current_track_idx] = dur
            cnt_duration += 1
            duration_vals.append(dur)
            i += 1
            continue

        # chord_X -> place chord at current_tick for current_track_idx
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
            # LOOKAHEAD: jeśli następny token nie jest time_shift, przesuwamy current_tick o duration
            next_token = tokens[i+1] if (i+1) < N else None
            if not (isinstance(next_token, str) and next_token.startswith("time_shift_")):
                # advance by duration as default gap
                dur_ticks = durations.get(current_track_idx, time_resolution)
                current_tick += dur_ticks
            i += 1
            continue

        # EOS token handling
        if token == "<SONG_END>":
            # optionally, advance to next track or reset tick? We'll increment slightly.
            current_tick += time_resolution
            i += 1
            continue

        # otherwise skip unknown token gracefully
        i += 1

    # append instruments
    for idx in sorted(instruments.keys()):
        pm.instruments.append(instruments[idx])

    # diagnostics print
    if verbose:
        print(f"Format: 1, Ścieżki: {len(instruments)}, Czas (s): {sum(len(inst.notes) for inst in pm.instruments) and (max((n.end for inst in pm.instruments for n in inst.notes), default=0)):.2f}")
        print()
        print(f"🎚 time_shift count: {cnt_time_shift}, duration tokens: {cnt_duration}, chord tokens placed: {cnt_chord}")
        if duration_vals:
            print(f"Durations (ticks) — min:{min(duration_vals)}, max:{max(duration_vals)}, mean:{statistics.mean(duration_vals):.1f}")

    # write midi
    pm.write(str(output_path))
    return output_path



def generate_tokens(model, token_to_id, id_to_token, max_tokens=500, start_tokens=None,
                    temperature=1.0, top_k=None, stop_token=STOP_TOKEN):
    model.eval()
    start_tokens = start_tokens or []
    # ensure token ids exist
    unk_id = token_to_id.get("<UNKNOWN>", None)
    eos_id = token_to_id.get(stop_token, None)

    input_tokens = []
    for t in start_tokens:
        if t in token_to_id:
            input_tokens.append(t)
        else:
            # fallback: try SONG_END or skip
            if stop_token in token_to_id:
                input_tokens.append(stop_token)
            elif unk_id is not None:
                # find any token id name mapping back to a token (convert id->token)
                input_tokens.append("<UNKNOWN>")
            else:
                input_tokens.append(list(token_to_id.keys())[0])  # pick arbitrary

    # convert to ids
    input_ids = [token_to_id.get(t, token_to_id.get("<UNKNOWN>", 0)) for t in input_tokens]
    generated_tokens = input_tokens.copy()

    # iterative generation
    for step in range(max_tokens - len(input_ids)):
        tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)  # [1, seq_len]
        with torch.no_grad():
            logits = model(tensor)               # [1, seq_len, vocab]
            last_logits = logits[0, -1]         # [vocab]
            next_id = sample_from_logits(last_logits.cpu(), temperature=temperature, top_k=top_k)
            next_token = id_to_token.get(int(next_id), None)
        if next_token is None:
            # unknown id -> stop
            break

        generated_tokens.append(next_token)
        input_ids.append(next_id)

        # stop if EOS seen
        if next_token == stop_token:
            break

        # keep input_ids length reasonable: don't grow arbitrarily (but model can accept full context)
        # we keep full growing context since model doesn't implement kv-cache (ok for modest lengths)

    return generated_tokens

def main():
    token_to_id, id_to_token, chord_id_to_notes = load_vocab()

    # model init
    model = TransformerModel(
        vocab_size=len(token_to_id),
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1
    ).to(DEVICE)

    # load weights
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE)
    print(f"✅ Wczytano model z {CHECKPOINT_PATH}")

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
    print(f"🎵 Wygenerowano {len(tokens)} tokenów")

    midi_path = tokens_to_midi(tokens, chord_id_to_notes, OUTPUT_MIDI)
    print(f"💾 Zapisano plik MIDI: {midi_path}")

if __name__ == "__main__":
    main()
