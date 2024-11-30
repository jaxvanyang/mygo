from pathlib import Path
from shutil import copyfileobj
from typing import Any
from urllib.request import urlopen

import numpy as np
import torch
from torch.utils.data import Dataset


class MCTSDataset(Dataset):
    url_prefix = "https://github.com/maxpumperla/deep_learning_and_the_game_of_go/raw/master/code/generated_games"  # noqa: E501
    features_file = "features-40k.npy"
    labels_file = "labels-40k.npy"

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        download: bool = True,
        transform=None,
        target_transform=None,
    ) -> None:
        self.root = root if isinstance(root, Path) else Path(root)
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self._download()

        self.features = torch.from_numpy(np.load(self.root / self.features_file))
        self.labels = torch.from_numpy(np.load(self.root / self.labels_file).argmax(1))

        train_size = int(0.9 * len(self.features))
        segment_slice = slice(train_size) if train else slice(train_size, None)
        self.features = self.features[segment_slice].type(
            torch.float32
        )  # the model is using float32
        self.labels = self.labels[segment_slice]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, key: int) -> tuple[Any, Any]:
        feature, label = self.features[key], self.labels[key]
        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)

        return feature, label

    def _download(self) -> None:
        """Download feature and label files."""

        if not self.root.is_dir():
            self.root.mkdir(parents=True)

        for file in (self.features_file, self.labels_file):
            url = f"{self.url_prefix}/{file}"
            path = self.root / file

            if path.is_file():
                continue

            print(f"Downloading {url} -> {path}")
            with urlopen(url) as response, open(path, "wb") as f:
                copyfileobj(response, f)
