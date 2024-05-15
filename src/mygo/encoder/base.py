from abc import ABC, abstractmethod

from numpy import ndarray

from mygo.game.types import Game, Point


class Encoder(ABC):
    def __init__(self, plane_count: int, board_size: int = 19) -> None:
        super().__init__()
        self.plane_count = plane_count
        self.size = board_size

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.plane_count, self.size, self.size

    # pytype: disable=bad-return-type
    @abstractmethod
    def encode(self, game: Game) -> ndarray:
        """Return encoded board of game."""

    # pytype: enable=bad-return-type

    def decode_point_index(self, idx: int) -> Point:
        idx = int(idx)
        assert 0 <= idx < self.size**2

        row = idx // self.size + 1
        col = idx % self.size + 1
        return Point(row, col)
