from abc import ABC, abstractmethod

from numpy import ndarray

from mygo.game import Game, Move, PassMove, Player, PlayMove, Point


class Encoder(ABC):
    def __init__(self, plane_count: int, board_size: int = 19) -> None:
        super().__init__()
        self.plane_count = plane_count
        # TODO: rename to board_size
        self.size = board_size

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.plane_count, self.size, self.size

    # pytype: disable=bad-return-type
    @abstractmethod
    def encode(self, game: Game) -> ndarray:
        """Return encoded board of game."""

    # pytype: enable=bad-return-type

    def decode_point(self, code: int) -> Point:
        """Return a Point instance based on the point code.

        Args:
            code: An integer code representing a point on the Go board.
        """
        if not isinstance(code, int):
            code = int(code)
        assert 0 <= code < self.size**2

        return Point(code // self.size, code % self.size)
