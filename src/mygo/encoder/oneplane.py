import itertools

import numpy as np
from numpy import ndarray

from mygo.encoder.base import Encoder
from mygo.game.types import Game


class OnePlaneEncoder(Encoder):
    def __init__(self, board_size: int = 19) -> None:
        super().__init__(1, board_size)

    def encode(self, game: Game) -> ndarray:
        assert self.size == game.size

        board = np.zeros(self.shape, dtype=np.float32)
        idx_range = range(self.size)

        for i, j in itertools.product(idx_range, idx_range):
            player = game.board.get_player((i + 1, j + 1))
            if player is None:
                continue
            board[0, i, j] = 1 if player == game.next_player else -1

        return board
