import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

class MIDIDataset(Dataset):
    def __init__(self, tensor_dir: str, seq_len: int = 128):
        self.paths = list(Path(tensor_dir).rglob("*.pt"))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        data = torch.load(path)  # [token_1, token_2, ..., token_N]

        if len(data) <= self.seq_len:
            # jeśli piosenka za krótka → pętla
            data = data.repeat((self.seq_len // len(data)) + 1)

        start = random.randint(0, len(data) - self.seq_len - 1)
        x = data[start : start + self.seq_len]
        y = data[start + 1 : start + self.seq_len + 1]
        return x, y
