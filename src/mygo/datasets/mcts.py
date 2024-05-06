from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import Dataset


class MCTSDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transform=None,
        target_transform=None,
    ) -> None:
        xs_path = root / Path("mcts_boards.npy")
        ys_path = root / Path("mcts_moves.npy")
        xs, ys = np.load(xs_path), np.load(ys_path)
        train_size = int(0.9 * xs.shape[0])

        self.transform, self.target_transform = transform, target_transform
        if train:
            self.xs, self.ys = xs[:train_size], ys[:train_size]
        else:
            self.xs, self.ys = xs[train_size:], ys[train_size:]

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, key: int) -> tuple[Any, Any]:
        x, y = self.xs[key], self.ys[key]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y
