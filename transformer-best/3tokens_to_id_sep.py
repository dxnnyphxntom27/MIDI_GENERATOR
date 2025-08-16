# build_sequences_from_tokens.py
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

# === Ustawienia ===
BASE = Path("E:/MIDI_GENERATOR/transformer-best")
TOKEN_DIR = BASE / "TokenizedWithChords"
VOCAB_PATH = BASE / "vocab.json"
OUT_DIR = BASE / "dataset_ids"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 512
STRIDE = 256        # overlapping stride; ustaw = SEQ_LEN żeby nie było overlapu
TRAIN_SPLIT = 0.9   # split po plikach
SEED = 42
PAD_WITH_EOS = True  # jeśli False, obcina resztki
SONG_END = "<SONG_END>"

random.seed(SEED)
np.random.seed(SEED)

def load_vocab(p):
    with open(p, "r", encoding="utf-8") as f:
        tok2id = json.load(f)
    # ensure ids are ints
    for k,v in list(tok2id.items()):
        if isinstance(v, str) and v.isdigit():
            tok2id[k] = int(v)
    return tok2id

def ensure_eos(tok2id):
    if SONG_END not in tok2id:
        next_id = max(int(v) for v in tok2id.values()) + 1
        tok2id[SONG_END] = next_id
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(tok2id, f, ensure_ascii=False, indent=2)
        print(f"🔤 Dodano {SONG_END} -> {next_id} do vocab i zapisano.")
    return tok2id[SONG_END], tok2id

def list_token_files(root: Path):
    files = sorted(root.glob("**/*.txt"))
    return [p for p in files if p.is_file()]

def file_to_ids(path: Path, tok2id: dict, eos_id: int):
    toks = path.read_text(encoding="utf-8").splitlines()
    ids = [tok2id[t] for t in toks if t in tok2id]
    # zawsze doklejamy EOS na końcu pliku (ważne dla dopełniania/padding/pewności)
    ids.append(eos_id)
    return ids

def build_sequences_from_stream(ids_stream, seq_len, stride, pad_with_eos, eos_id):
    seqs = []
    if pad_with_eos:
        # pad so last window available
        if len(ids_stream) < seq_len:
            pad_needed = seq_len - len(ids_stream)
            ids_stream = ids_stream + [eos_id] * pad_needed
        # sliding windows
        for start in range(0, len(ids_stream) - seq_len + 1, stride):
            seqs.append(ids_stream[start:start+seq_len])
        # maybe pad the tail if not divisible and we want the final partial window
        last_start = ( (len(ids_stream) - seq_len) // stride ) * stride
        tail_start = last_start + stride
        if tail_start < len(ids_stream):
            tail = ids_stream[tail_start:]
            tail = tail + [eos_id] * (seq_len - len(tail))
            seqs.append(tail)
    else:
        total_full = (len(ids_stream) // seq_len)
        for i in range(total_full):
            start = i * seq_len
            seqs.append(ids_stream[start:start+seq_len])
    return seqs

def process_split(file_list, tok2id, eos_id, seq_len, stride, pad_with_eos):
    all_seqs = []
    unknown_tokens = 0
    for f in tqdm(file_list, desc="Pliki -> ID"):
        ids = file_to_ids(f, tok2id, eos_id)
        # If some tokens were not in vocab, file_to_ids silently skipped; optionally count
        # build windows
        seqs = build_sequences_from_stream(ids, seq_len, stride, pad_with_eos, eos_id)
        all_seqs.extend(seqs)
    arr = np.array(all_seqs, dtype=np.int32)
    return arr

def main():
    tok2id = load_vocab(VOCAB_PATH)
    eos_id, tok2id = ensure_eos(tok2id)

    files = list_token_files(TOKEN_DIR)
    if not files:
        raise SystemExit("Nie znaleziono token files.")

    random.shuffle(files)
    n_train = max(1, int(len(files) * TRAIN_SPLIT))
    train_files = files[:n_train]
    val_files = files[n_train:]

    print(f"Plików: {len(files)} → train: {len(train_files)}, val: {len(val_files)}")
    train_array = process_split(train_files, tok2id, eos_id, SEQ_LEN, STRIDE, PAD_WITH_EOS)
    val_array   = process_split(val_files,   tok2id, eos_id, SEQ_LEN, STRIDE, PAD_WITH_EOS)

    print("Zapisuję numpy arrays...")
    np.save(OUT_DIR / "train.npy", train_array)
    np.save(OUT_DIR / "val.npy",   val_array)

    meta = {
        "seq_len": SEQ_LEN,
        "stride": STRIDE,
        "pad_with_eos": PAD_WITH_EOS,
        "train_seqs": int(train_array.shape[0]),
        "val_seqs": int(val_array.shape[0]),
        "vocab_size": max(int(v) for v in tok2id.values()) + 1,
        "eos_id": int(eos_id)
    }
    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Gotowe. train: {train_array.shape}, val: {val_array.shape}")

if __name__ == "__main__":
    main()
