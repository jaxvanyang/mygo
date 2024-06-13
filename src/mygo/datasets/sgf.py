import itertools
import tarfile
from pathlib import Path
from shutil import copyfileobj
from typing import Any, Generator
from urllib.request import urlopen

import numpy as np
from torch.utils.data import Dataset, IterableDataset

from mygo.encoder.base import Encoder
from mygo.encoder.oneplane import OnePlaneEncoder
from mygo.game.game import Game
from mygo.game.move import PlayMove, from_pysgf_move
from mygo.pysgf import SGF


class KGSMixin:
    """Metadata and helper functions of KGS archives."""

    url_prefix = "https://dl.u-go.net/gamerecords"
    train_archives = (
        "KGS-2019_04-19-1255-.tar.bz2",
        "KGS-2019_02-19-1412-.tar.bz2",
        "KGS-2018_12-19-1992-.tar.bz2",
        "KGS-2018_10-19-1209-.tar.bz2",
        "KGS-2018_08-19-1447-.tar.bz2",
        "KGS-2018_06-19-1002-.tar.bz2",
        "KGS-2018_04-19-1612-.tar.bz2",
        "KGS-2018_02-19-1167-.tar.bz2",
        "KGS-2017_12-19-1488-.tar.bz2",
        "KGS-2017_10-19-1351-.tar.bz2",
        "KGS-2017_08-19-2205-.tar.bz2",
        "KGS-2017_06-19-910-.tar.bz2",
        "KGS-2017_04-19-913-.tar.bz2",
        "KGS-2017_02-19-525-.tar.bz2",
        "KGS-2016_12-19-1208-.tar.bz2",
        "KGS-2016_10-19-925-.tar.bz2",
        "KGS-2016_08-19-1374-.tar.bz2",
        "KGS-2016_06-19-1540-.tar.bz2",
        "KGS-2016_04-19-1081-.tar.bz2",
        "KGS-2016_02-19-577-.tar.bz2",
        "KGS-2015-19-8133-.tar.bz2",
        "KGS-2013-19-13783-.tar.bz2",
        "KGS-2011-19-19099-.tar.bz2",
        "KGS-2009-19-18837-.tar.bz2",
        "KGS-2007-19-11644-.tar.bz2",
        "KGS-2005-19-13941-.tar.bz2",
        "KGS-2003-19-7582-.tar.bz2",
        "KGS-2001-19-2298-.tar.bz2",
    )
    test_archives = (
        "KGS-2019_03-19-1478-.tar.bz2",
        "KGS-2019_01-19-2095-.tar.bz2",
        "KGS-2018_11-19-1879-.tar.bz2",
        "KGS-2018_09-19-1587-.tar.bz2",
        "KGS-2018_07-19-949-.tar.bz2",
        "KGS-2018_05-19-1590-.tar.bz2",
        "KGS-2018_03-19-833-.tar.bz2",
        "KGS-2018_01-19-1526-.tar.bz2",
        "KGS-2017_11-19-945-.tar.bz2",
        "KGS-2017_09-19-1353-.tar.bz2",
        "KGS-2017_07-19-1191-.tar.bz2",
        "KGS-2017_05-19-847-.tar.bz2",
        "KGS-2017_03-19-717-.tar.bz2",
        "KGS-2017_01-19-733-.tar.bz2",
        "KGS-2016_11-19-980-.tar.bz2",
        "KGS-2016_09-19-1170-.tar.bz2",
        "KGS-2016_07-19-1432-.tar.bz2",
        "KGS-2016_05-19-1011-.tar.bz2",
        "KGS-2016_03-19-895-.tar.bz2",
        "KGS-2016_01-19-756-.tar.bz2",
        "KGS-2014-19-13029-.tar.bz2",
        "KGS-2012-19-13665-.tar.bz2",
        "KGS-2010-19-17536-.tar.bz2",
        "KGS-2008-19-14002-.tar.bz2",
        "KGS-2006-19-10388-.tar.bz2",
        "KGS-2004-19-12106-.tar.bz2",
        "KGS-2002-19-3646-.tar.bz2",
    )

    @classmethod
    def download_and_extract_archives(cls, root: Path, train: bool = True) -> None:
        """Download and extract SGF archives."""

        if not root.is_dir():
            root.mkdir(parents=True)

        archives = cls.train_archives if train else cls.test_archives
        for archive in archives:
            url = f"{cls.url_prefix}/{archive}"
            path = root / archive

            if path.is_file():
                continue

            print(f"Downloading {url} -> {path}")
            with urlopen(url) as response, open(path, "wb") as f:
                copyfileobj(response, f)

            # only extract after downloading because checking if they are extracted is
            # too expensive
            with tarfile.open(path) as tar:
                print(f"Extracting {path}")
                tar.extractall(path=root)

    @classmethod
    def get_sgf_paths(
        cls, root: Path, train: bool = True
    ) -> Generator[str, None, None]:
        """Return a generator of extracted SGF files' relative paths to root."""

        for archive in cls.train_archives if train else cls.test_archives:
            with tarfile.open(root / archive) as tar:
                # ignore 1st name, because it's the parent directory
                yield from tar.getnames()[1:]


class KGSDataset(KGSMixin, Dataset):
    """Dataset of KGS game records."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        download: bool = True,
        game_count: int = 100,
        encoder: Encoder = OnePlaneEncoder(),
        transform=None,
        target_transform=None,
    ) -> None:
        self.archives = self.train_archives if train else self.test_archives
        assert game_count <= sum(int(a.split("-")[-2]) for a in self.archives)

        self.root = root if isinstance(root, Path) else Path(root)
        self.game_count = game_count
        self.encoder = encoder
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self.features, self.labels = [], []

        if download:
            self.download_and_extract_archives(self.root, train=train)

        paths = itertools.islice(self.get_sgf_paths(self.root, train=train), game_count)
        for path in paths:
            sgf_root = SGF.parse_file(self.root / path)
            game = Game.from_pysgf(sgf_root)
            node = sgf_root

            while node.children:
                # only select the first child for simplicity
                node = node.children[0]
                move = from_pysgf_move(node.move)

                if isinstance(move, PlayMove):
                    self.features.append(encoder.encode(game))
                    self.labels.append(move.point.encode(game.board_size))

                game.apply_move(move)

        self.features = np.array(self.features)
        self.labels = np.array(self.labels)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, key: int) -> tuple[Any, Any]:
        x, y = self.features[key], self.labels[key]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return x, y


class KGSIterableDataset(KGSMixin, IterableDataset):
    """Iterable dataset of KGS game records."""

    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        download: bool = True,
        game_count: int = 100,
        encoder: Encoder = OnePlaneEncoder(),
        transform=None,
        target_transform=None,
    ) -> None:
        self.archives = self.train_archives if train else self.test_archives
        assert game_count <= sum(int(a.split("-")[-2]) for a in self.archives)

        self.root = root if isinstance(root, Path) else Path(root)
        self.game_count = game_count
        self.encoder = encoder
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download_and_extract_archives(self.root, train=train)

        self.sgf_paths = list(
            itertools.islice(self.get_sgf_paths(self.root, train=train), game_count)
        )

    def __iter__(self) -> Generator[tuple[Any, Any], None, None]:
        for path in self.sgf_paths:
            sgf_root = SGF.parse_file(self.root / path)
            game = Game.from_pysgf(sgf_root)
            node = sgf_root

            while node.children:
                # only select the first child for simplicity
                node = node.children[0]
                move = from_pysgf_move(node.move)

                if isinstance(move, PlayMove):
                    feature = self.encoder.encode(game)
                    label = move.point.encode(game.board_size)

                    if self.transform:
                        feature = self.transform(feature)
                    if self.target_transform:
                        label = self.target_transform(label)

                    yield feature, label

                game.apply_move(move)
