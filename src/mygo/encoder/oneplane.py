import itertools

import numpy as np
from numpy import ndarray

from mygo.encoder.base import Encoder
from mygo.game.types import Game


class OnePlaneEncoder(Encoder):
    @staticmethod
    def encode(game: Game) -> ndarray:
        board = np.zeros(game.shape, dtype=np.float32)
        idx_range = range(game.size)

        for i, j in itertools.product(idx_range, idx_range):
            player = game.board.get_player((i + 1, j + 1))
            if player is None:
                continue
            board[i, j] = 1 if player == game.next_player else -1

        return board
