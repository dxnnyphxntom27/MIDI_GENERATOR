from pathlib import Path
import random

TOKENS_AFTER_META = 16
MAX_PROMPT_TOKENS = 32

def load_tokens(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def extract_meta_and_body(tokens):
    meta = tokens[:4]
    body = tokens[4:]
    return meta, body

def get_tempo_from_meta(meta_tokens):
    for t in meta_tokens:
        if t.startswith("tempo_"):
            try:
                return int(t.split("_", 1)[1])
            except Exception:
                pass
    return None

def get_instrument_from_meta(meta_tokens):
    for t in meta_tokens:
        if t.startswith("instrument_"):
            try:
                return int(t.split("_", 1)[1])
            except Exception:
                pass
    return None

def generate_prompt(
    token_dir,
    prompt_out,
    tokens_after_meta=TOKENS_AFTER_META,
    max_tokens=MAX_PROMPT_TOKENS,
    random_instrument=False,
    mode="multi"
):
    """
    token_dir: katalog z plikami .txt (tokeny ztokenizowane)
    prompt_out: plik wyjściowy (np. "app/user_prompt.txt")
    """
    token_files = sorted(Path(token_dir).glob("*.txt"))
    if not token_files:
        print(f"❌ Nie znaleziono plików .txt w katalogu: {token_dir}")
        return

    all_tokens = []
    tempos = []
    instruments = []

    for txt_path in token_files:
        tokens = load_tokens(txt_path)
        if len(tokens) < 5:
            print(f"⚠️ Za mało tokenów w pliku {txt_path}, pomijam")
            continue
        meta, body = extract_meta_and_body(tokens)
        tempo = get_tempo_from_meta(meta)
        instrument = get_instrument_from_meta(meta)
        if tempo is not None:
            tempos.append(tempo)
        if instrument is not None:
            instruments.append(instrument)
        body_part = body[:tokens_after_meta]
        if mode == "multi":
            all_tokens.append((meta, body_part))
        elif mode == "single":
            all_tokens.append((meta, body_part))
            break

    # --- Ustal meta promptu ---
    meta_prompt = ["key_unknown"]
    tempo_final = int(round(sum(tempos) / len(tempos))) if tempos else 120
    tempo_quantized = int(round(tempo_final / 5.0) * 5)  # <-- kwantyzacja do najbliższej liczby podzielnej przez 5
    meta_prompt.append(f"tempo_{tempo_quantized}")
    meta_prompt.append("track_0")
    if instruments:
        instr_final = random.choice(instruments) if random_instrument else instruments[0]
        meta_prompt.append(f"instrument_{instr_final}")
    else:
        meta_prompt.append("instrument_0")

    # --- Buduj prompt ---
    prompt = meta_prompt.copy()
    for meta, body_part in all_tokens:
        prompt.extend(body_part)
        if len(prompt) >= max_tokens:
            break

    # (opcjonalnie) usuń duration_0
    prompt = [t for t in prompt if not t.startswith("duration_0")]
    prompt = prompt[:max_tokens]

    # --- Zapisz prompt ---
    with open(prompt_out, "w", encoding="utf-8") as f:
        for t in prompt:
            f.write(t + "\n")

    print(f"✅ Prompt zapisany do {prompt_out}: {len(prompt)} tokenów")
    print("Prompt preview:", prompt[:10], "..." if len(prompt) > 10 else "")

