import itertools

import numpy as np
from numpy import ndarray

from mygo.encoder.base import Encoder
from mygo.game.game import Game


class OnePlaneEncoder(Encoder):
    def __init__(self, board_size: int = 19) -> None:
        super().__init__(1, board_size)

    def encode(self, game: Game) -> ndarray:
        assert self.size == game.board_size

        board = np.zeros(self.shape, dtype=np.float32)
        index_range = range(self.size)

        for x, y in itertools.product(index_range, index_range):
            player = game.last_board.get_player((x, y))
            if player is None:
                continue
            board[0, x, y] = 1 if player == game.next_player else -1

        return board
