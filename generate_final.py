import torch
import json
import random
from model import MusicLSTMModel
from dataset import MIDIDataset

# 📂 Ścieżki
MODEL_PATH = "best_model.pt"
VOCAB_PATH = "vocab.json"
TENSOR_DIR = "TensorizedWithChords"
OUTPUT_PATH = "generated_tokens.txt"

# 🔧 Parametry
SEQ_LEN = 512
SEED_LEN = 32
TOP_K = 20
TEMPERATURE = 1.0

# 📖 Załaduj słownik
with open(VOCAB_PATH) as f:
    vocab = json.load(f)
id_to_token = {v: k for k, v in vocab.items()}
token_to_id = vocab

# 🎯 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicLSTMModel(vocab_size=len(vocab))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 🎼 Losowy seed
dataset = MIDIDataset(tensor_dir=TENSOR_DIR, seq_len=SEED_LEN + 1)
seed_x, _ = dataset[random.randint(0, len(dataset) - 1)]
input_ids = seed_x.tolist()
hidden = None

# 🔁 Generacja
for _ in range(SEQ_LEN):
    x = torch.tensor(input_ids[-1:], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, hidden = model(x, hidden)
    logits = logits[:, -1, :] / TEMPERATURE
    probs = torch.softmax(logits, dim=-1).squeeze()

    topk_probs, topk_indices = torch.topk(probs, TOP_K)
    next_id = topk_indices[torch.multinomial(topk_probs, 1).item()].item()
    input_ids.append(next_id)

# 💾 Zapis tokenów
tokens = [id_to_token[i] for i in input_ids if id_to_token[i].startswith(("chord_", "time_shift_"))]

with open(OUTPUT_PATH, "w") as f:
    f.write(" ".join(tokens))

print(f"✅ Wygenerowano {len(tokens)} tokenów → zapisano do {OUTPUT_PATH}")
