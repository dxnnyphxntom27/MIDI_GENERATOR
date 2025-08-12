import torch
import json
from transformer4 import MusicTransformerModel

# === ŚCIEŻKI ===
DATASET_DIR = "E:/MIDI_GENERATOR/transformer-based-2"
MODEL_PATH = f"{DATASET_DIR}/transformer_pretrained.pt"
VOCAB_PATH = f"{DATASET_DIR}/vocab.json"
VOCAB_REV_PATH = f"{DATASET_DIR}/vocab_reverse.json"
OUTPUT_PATH = f"{DATASET_DIR}/generated_tokens.txt"

# === PARAMETRY GENERACJI ===
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKENS = 1024  # ile tokenów wygenerować
TEMPERATURE = 1.0  # kreatywność, niższa = bardziej przewidywalne

# === PROMPT ===
prompt_tokens = [
    "key_E_minor",
    "tempo_90",
    "track_0",
    "instrument_25"
]

# === ŁADUJ SŁOWNIKI ===
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)
with open(VOCAB_REV_PATH, "r") as f:
    vocab_reverse = {int(k): v for k, v in json.load(f).items()}

vocab_size = len(vocab)
prompt_ids = [vocab[token] for token in prompt_tokens if token in vocab]

# === MODEL ===
model = MusicTransformerModel(vocab_size=vocab_size).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# === GENERUJ ===
generated = prompt_ids.copy()
input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(DEVICE)

print("🎹 Generowanie...")

with torch.no_grad():
    for _ in range(MAX_TOKENS):
        if input_ids.shape[1] > 2048:  # ogranicz długość wejścia (wymuszenie max_seq_len)
            input_ids = input_ids[:, -2048:]

        output = model(input_ids)
        logits = output[0, -1, :] / TEMPERATURE
        probs = torch.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_token_id)
        input_ids = torch.tensor(generated, dtype=torch.long).unsqueeze(0).to(DEVICE)

# === KONWERSJA DO TOKENÓW
generated_tokens = [vocab_reverse[token_id] for token_id in generated]

# === ZAPISZ DO .TXT
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for token in generated_tokens:
        f.write(token + "\n")

print(f"✅ Zapisano {len(generated_tokens)} tokenów do: {OUTPUT_PATH}")
