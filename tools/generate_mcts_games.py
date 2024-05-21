#!/usr/bin/env python3

import logging
import random
import time
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from mygo.agent.naive import MCTSBot
from mygo.encoder import OnePlaneEncoder
from mygo.game.types import Game
from mygo.helper.log import logger


def generate_game(size: int, rounds: int, temp: float, seed: int) -> tuple:
    game = Game.new_game(size)
    bot = MCTSBot(rounds, temp)
    encoder = OnePlaneEncoder(size)
    random.seed(seed)
    boards, moves = [], []

    while not game.is_over:
        move = bot.select_move(game)
        if move.is_play:
            row, col = move.point
            move_idx = (row - 1) * size + col - 1
            boards.append(encoder.encode(game))
            moves.append(move_idx)

        game.apply_move(move)

    return np.array(boards), np.array(moves)


def main():
    parser = ArgumentParser(
        prog="generate_mcts_games",
        description="Generate Monte Carlo tree search games for training.",
    )
    parser.add_argument("--games", default=2, type=int, help="number of games.")
    parser.add_argument("--size", default=5, type=int, help="board size.")
    parser.add_argument(
        "--rounds", default=20, type=int, help="number of rounds of one single search."
    )
    parser.add_argument(
        "--temp", default=0.8, type=float, help="temperature of the UCT formula."
    )
    parser.add_argument(
        "--root",
        default=Path("data"),
        type=Path,
        help="root directory of generated files.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        default=cpu_count(),
        type=int,
        help="number of jobs to run simultaneously.",
    )

    args = parser.parse_args()
    games = args.games
    size = args.size
    rounds = args.rounds
    temp = args.temp
    root = args.root
    jobs = args.jobs
    feature_file = root / "mcts_boards.npy"
    label_file = root / "mcts_moves.npy"

    logger.setLevel(logging.INFO)

    xs, ys = [], []

    t0 = time.perf_counter()
    with Pool(processes=jobs) as pool:
        res_list = []

        for i in range(games):
            res_list.append(
                pool.apply_async(generate_game, (size, rounds, temp, 25565 + i))
            )

        for x, y in (res.get() for res in res_list):
            xs.append(x)
            ys.append(y)
    t = time.perf_counter() - t0
    print(f"time: {t:.3f}")

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    root.mkdir(exist_ok=True)
    np.save(feature_file, xs)
    np.save(label_file, ys)


if __name__ == "__main__":
    main()
