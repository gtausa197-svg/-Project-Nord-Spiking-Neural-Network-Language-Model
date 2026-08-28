from __future__ import annotations

import numpy as np
import torch


class TokenBin:
    def __init__(self, path: str):
        self.path = path
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) < 2:
            raise ValueError(f"{path} is too small")

    def __len__(self) -> int:
        return len(self.data)

    def get_batch(self, batch_size: int, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = len(self.data) - seq_len - 1
        if max_start <= 0:
            raise ValueError(f"dataset has {len(self.data)} tokens but seq_len={seq_len}")
        starts = np.random.randint(0, max_start, size=batch_size)
        x = np.stack([np.asarray(self.data[s : s + seq_len], dtype=np.int64) for s in starts])
        y = np.stack([np.asarray(self.data[s + 1 : s + seq_len + 1], dtype=np.int64) for s in starts])
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y
