from abc import ABC, abstractmethod

from numpy import ndarray

from mygo.game.types import Game, Point


class Encoder(ABC):
    def __init__(self, board_size: int = 19) -> None:
        super().__init__()
        self.size = board_size

    # pytype: disable=bad-return-type
    @staticmethod
    @abstractmethod
    def encode(game: Game) -> ndarray:
        """Return encoded board of game."""

    # pytype: enable=bad-return-type

    def decode_point_index(self, idx: int) -> Point:
        idx = int(idx)
        assert 0 <= idx < self.size**2

        row = idx // self.size + 1
        col = idx % self.size + 1
        return Point(row, col)
