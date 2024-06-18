import itertools

import numpy as np
from numpy import ndarray

from mygo.encoder.base import Encoder
from mygo.game.basic import Point
from mygo.game.game import Game
from mygo.game.move import PlayMove


class SevenPlaneEncoder(Encoder):
    def __init__(self, board_size: int = 19) -> None:
        super().__init__(7, board_size)

    def encode(self, game: Game) -> ndarray:
        board = np.zeros(self.shape, dtype=np.float32)
        base_plane = {
            game.next_player: 0,
            -game.next_player: 3,
        }

        index_range = range(game.board_size)
        for x, y in itertools.product(index_range, index_range):
            go_string = game.last_board[x, y]

            if go_string is None:
                # TODO: should only check the ko rule to improve performance
                if not game.is_valid_move(PlayMove(game.next_player, Point(x, y))):
                    board[6, x, y] = 1.0
                continue

            liberty_plane = (
                min(3, go_string.liberty_count) - 1 + base_plane[go_string.player]
            )
            board[liberty_plane, x, y] = 1.0

        return board
