#!/usr/bin/env python3

import os
import re
import tarfile
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from pysgf import SGF

from mygo.encoder import OnePlaneEncoder
from mygo.game.types import Game, Move, Point


def download_archive(root: Path, url: str) -> list[Path]:
    path = root / os.path.basename(url)

    if path.is_file():
        print(f"file {path} already exists")
    else:
        print(f"download {url} to {path}...")
        with urlopen(url) as resp, open(path, "wb") as f:
            f.write(resp.read())
        print(f"{url} downloaded to {path}")

    with tarfile.open(path, "r:bz2") as tar:
        member_paths = list(root / x for x in tar.getnames())[1:]

        if all(x.is_file() for x in member_paths):
            print(f"archive {path} already extracted")
        else:
            print(f"extracting {path}...")
            tar.extractall(path=root)
            print(f"extracted {path}")

    # strip the fist directory element
    return member_paths


def download_sgfs(root: Path, index: str, count: int) -> list[Path]:
    with urlopen(index) as resp:
        index_html = resp.read()

    index_html = index_html.decode("utf-8")
    archives = re.findall(r"https://.*\.tar\.bz2", index_html)[:count]
    sgf_paths = []

    with Pool(processes=cpu_count() - 1) as pool:
        tasks = []
        for archive in archives:
            tasks.append(pool.apply_async(download_archive, (root, archive)))

        for paths in (task.get() for task in tasks):
            sgf_paths.extend(paths)

    return sgf_paths


def parse_sgf(path: Path) -> tuple[list, list]:
    features, labels = [], []

    root = SGF.parse_file(path)
    game = Game.from_sgf_root(root)

    node = root
    while node.children:
        # only select the first child for simplicity
        node = node.children[0]
        move = node.move

        if move.is_pass:
            game.apply_move(Move.pass_())
        else:
            col, row = move.coords
            move_idx = row * game.size + col
            features.append(OnePlaneEncoder.encode(game))
            labels.append(move_idx)
            game.apply_move(Move.play(Point(row + 1, col + 1)))

    return features, labels


def main() -> None:
    index = "https://u-go.net/gamerecords"

    features_fmt = "kgs_features_{}.npy"
    labels_fmt = "kgs_labels_{}.npy"

    parser = ArgumentParser(
        prog="parse_kgs_sgfs",
        description="Download and parse KGS SGF archives to Numpy arrays for training.",
    )
    parser.add_argument(
        "--root",
        metavar="dir",
        default=Path("data/kgs_sgfs"),
        type=Path,
        help="root directory of generated files.",
    )
    parser.add_argument(
        "--archives",
        metavar="N",
        default=1,
        type=int,
        help="number of archives to use.",
    )

    args = parser.parse_args()
    root = args.root
    num_archives = args.archives

    root.mkdir(parents=True, exist_ok=True)
    sgf_paths = download_sgfs(root, index, num_archives)
    chunk_size = 1024

    with Pool(processes=cpu_count() - 1) as pool:
        tasks = [pool.apply_async(parse_sgf, (path,)) for path in sgf_paths]

        features, labels = [], []
        chunk = 0
        for task in tasks:
            new_features, new_labels = task.get()
            features.extend(new_features)
            labels.extend(new_labels)

            while len(features) > chunk_size:
                fs, features = np.array(features[:chunk_size]), features[chunk_size:]
                ls, labels = np.array(labels[:chunk_size]), labels[chunk_size:]
                print(f"saving chunk {chunk}...")
                np.save(root / features_fmt.format(chunk), fs)
                np.save(root / labels_fmt.format(chunk), ls)
                chunk += 1

    # remaining features and labels are ignored for simplicity


if __name__ == "__main__":
    main()
