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
        data = torch.load(path)  # [token_1, ..., token_N]

        if len(data) <= self.seq_len:
            repeats = (self.seq_len + 1) // len(data) + 1
            data = data.tolist() * repeats
            data = torch.tensor(data, dtype=torch.long)

        start = random.randint(0, len(data) - self.seq_len - 1)
        x = data[start : start + self.seq_len]
        y = data[start + 1 : start + self.seq_len + 1]

        assert x.shape == (self.seq_len,)
        assert y.shape == (self.seq_len,)
        assert x.max() < 600 and y.max() < 600, f"❌ ID > vocab! x.max={x.max()}, y.max={y.max()}"

        return x, y

