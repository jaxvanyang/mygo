import itertools
import tarfile
from pathlib import Path
from shutil import copyfileobj
from typing import Any
from urllib.request import urlopen

import numpy as np
from pysgf import SGF
from torch.utils.data import Dataset

from mygo.encoder.base import Encoder
from mygo.encoder.oneplane import OnePlaneEncoder
from mygo.game.types import Game, Move, Point


class KGSDataset(Dataset):
    """KGS SGF game records."""

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
            self._download_and_extract_archives()

        def get_tarfile_names(archive):
            with tarfile.open(self.root / archive) as tar:
                return tar.getnames()[1:]  # ignore 1st name, because it's a directory

        names = itertools.chain(*(get_tarfile_names(a) for a in self.archives))
        names = itertools.islice(names, game_count)
        for name in names:
            sgf_root = SGF.parse_file(self.root / name)
            game = Game.from_sgf_root(sgf_root)
            node = sgf_root

            while node.children:
                # only select the first child for simplicity
                node = node.children[0]
                move = node.move

                if move.is_pass:
                    game.apply_move(Move.pass_())
                else:
                    col, row = move.coords
                    move_idx = row * game.size + col
                    self.features.append(encoder.encode(game))
                    self.labels.append(move_idx)
                    game.apply_move(Move.play(Point(row + 1, col + 1)))

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

    def _download_and_extract_archives(self) -> None:
        """Download and extract SGF archives."""

        if not self.root.is_dir():
            self.root.mkdir(parents=True)

        for archive in self.archives:
            url = f"{self.url_prefix}/{archive}"
            path = self.root / archive

            if path.is_file():
                continue

            print(f"Downloading {url} -> {path}")
            with urlopen(url) as response, open(path, "wb") as f:
                copyfileobj(response, f)

            # only extract after downloading
            # because checking if it's extracted is too expensive
            with tarfile.open(path) as tar:
                print(f"Extracting {path}")
                tar.extractall(path=self.root)
