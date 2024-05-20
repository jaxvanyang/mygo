import itertools

import numpy as np
from numpy import ndarray

from mygo.encoder.base import Encoder
from mygo.game.types import Game, Move, Point


class SevenPlaneEncoder(Encoder):
    def __init__(self, board_size: int = 19) -> None:
        super().__init__(7, board_size)

    def encode(self, game: Game) -> ndarray:
        board = np.zeros(self.shape, dtype=np.float32)
        base_plane = {
            game.next_player: 0,
            -game.next_player: 3,
        }
        idx_range = range(game.size)

        for i, j in itertools.product(idx_range, idx_range):
            row, col = i + 1, j + 1
            go_string = game.board[row, col]

            if go_string is None:
                # TBD: should only check the ko rule to improve performance
                if not game.is_valid_move(Move.play(Point(row, col))):
                    board[6, i, j] = 1.0
                continue

            liberty_plane = (
                min(3, go_string.num_liberties) - 1 + base_plane[go_string.player]
            )
            board[liberty_plane, i, j] = 1.0

        return board
