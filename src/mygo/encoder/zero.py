from itertools import product as prod

import numpy as np

from mygo.game import Game
from mygo.game.basic import Player

from .base import Encoder


class ZeroEncoder(Encoder):
    def __init__(self, board_size: int = 19):
        super().__init__(17, board_size)

    def encode(self, game: Game) -> np.ndarray:
        assert self.size == game.board_size

        out = np.zeros(self.shape, dtype=np.float32)
        boards = game.boards
        next_player = game.next_player
        opponent = -next_player

        for i in range(8):
            if len(boards) <= i:
                break

            board = boards[-1 - i]

            for x, y in prod(range(self.size), repeat=2):
                if (player := board.get_player((x, y))) == next_player:
                    out[i * 2, x, y] = 1
                elif player == opponent:
                    out[i * 2 + 1, x, y] = 1

        out[16] = int(next_player == Player.black)

        return out
