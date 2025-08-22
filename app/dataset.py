import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MusicDataset(Dataset):
    def __init__(self, npy_path, seq_len=None, ignore_index=-100, mask_last=True):
        self.path = Path(npy_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Plik .npy nie istnieje: {self.path}")

        arr = np.load(self.path, mmap_mode="r")
        self.ignore_index = ignore_index
        self.mask_last = mask_last

        if arr.ndim == 2:
            # oczekiwany, preferowany format
            self.data = arr
            self.num_sequences, self.seq_len = self.data.shape
        elif arr.ndim == 1:
            if seq_len is None:
                raise ValueError(
                    "1d, "
                    "2d."
                )
            total = len(arr)
            num_full = total // seq_len
            if num_full == 0:
                raise ValueError("1D too short.")
            new_len = num_full * seq_len
            flat = np.array(arr[:new_len], dtype=np.int32)
            self.data = flat.reshape(num_full, seq_len)
            self.num_sequences, self.seq_len = self.data.shape
        else:
            raise ValueError(f"Nieobsługiwany wymiar {arr.ndim} w pliku {self.path}")

    def __len__(self):
        return int(self.num_sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx], dtype=torch.long)

        x = seq.clone()
        y = seq.clone()

        # target to next-token prediction
        y[:-1] = seq[1:]

        # maskujemy ostatni token (jeśli wybrano)
        if self.mask_last:
            y[-1] = self.ignore_index

        return x, y


# ------------------ przykład użycia ------------------
if __name__ == "__main__":
    from pathlib import Path

    p = Path("E:/MIDI_GENERATOR/transformer-final/dataset_ids/train.npy")

    # Jeżeli plik jest 2D wystarczy:
    ds = MusicDataset(p)
    print(f"Liczba sekwencji: {len(ds)}, seq_len: {ds.seq_len}")

    # Jeżeli plik jest 1D, podaj seq_len:
    # ds1 = MusicDataset(p, seq_len=512)

    x, y = ds[0]
    print("X[:10]", x[:10])
    print("Y[:10]", y[:10])
