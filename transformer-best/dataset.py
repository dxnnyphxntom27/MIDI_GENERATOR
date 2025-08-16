import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class MusicDataset(Dataset):
    """
    Dataset dla Twoich plików .npy z tokenami.

    Obsługuje dwa formaty wejściowe:
      - 2D: (num_sequences, seq_len) -- DOMYŚLNY i PREFEROWANY
      - 1D: (total_tokens,) -- zostanie pocięty na sekwencje o długości seq_len

    Konstruktor:
        MusicDataset(npy_path, seq_len=None, ignore_index=-100, mask_last=True)

    Args:
        npy_path (str/Path): ścieżka do pliku .npy (train.npy / val.npy)
        seq_len (int|None): wymagane tylko gdy wejście jest 1D (do pocięcia)
        ignore_index (int): wartość, którą ustawiamy w y[-1] jeśli mask_last=True
        mask_last (bool): czy ustawić y[-1] = ignore_index (domyślnie True)

    Zwraca w __getitem__:
        x: LongTensor [seq_len]
        y: LongTensor [seq_len] (y[:-1] = x[1:], y[-1] = ignore_index jeśli mask_last)
    """

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
                    "Plik .npy jest 1D — podaj seq_len, aby pociąć go na sekwencje, "
                    "lub uruchom build_sequences_from_tokens.py żeby wygenerować 2D .npy."
                )
            total = len(arr)
            num_full = total // seq_len
            if num_full == 0:
                raise ValueError("Plik 1D jest za krótki względem podanego seq_len.")
            new_len = num_full * seq_len
            if new_len != total:
                # krótkie ostrzeżenie — przycinamy tail
                print(f"⚠️ Przycinam 1D {self.path} z {total} -> {new_len} tokenów (usunę {total - new_len}).")
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
