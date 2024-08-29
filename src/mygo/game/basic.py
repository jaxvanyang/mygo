"""Basic types of the Go game."""

import re
import string
from enum import IntEnum
from typing import Generator, NamedTuple


class Player(IntEnum):
    """The player of Go, may be called color sometimes."""

    black = 1
    white = -1

    def __str__(self) -> str:
        """Return the color of the player."""
        return "black" if self == Player.black else "white"

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        return f"{self.__class__.__name__}.{self}"

    def __neg__(self):
        """Return the opponent player."""
        return self.__class__(-self.value)

    @classmethod
    def from_sgf(cls, sgf: str) -> "Player":
        """Return a Player instance based on the SGF string.

        Args:
            sgf: The SGF string representing a player.
        """
        if sgf == "B":
            return cls.black
        if sgf == "W":
            return cls.white
        raise ValueError("unknown player string")

    @property
    def opponent(self) -> "Player":
        """The opponent player."""
        return -self

    @property
    def sgf(self) -> str:
        """The SGF representation of the player."""
        return "B" if self == Player.black else "W"


class Point(NamedTuple):
    """The position on the Go board.

    Coordinates are encoded using GTP coordinate system.

    Attributes:
        x: A zero-based int index number from left to right. Default is 0.
        y: A zero-based int index number from bottom to top. Default is 0.
    """

    gtp_coordinates = "ABCDEFGHJKLMNOPQRSTUVWXYZ"  # only support size 25
    sgf_coordinates = string.ascii_letters  # support size 52

    x: int = 0
    y: int = 0

    def __str__(self) -> str:
        """Return the GTP representation of the point."""
        assert 0 <= self.x <= len(self.gtp_coordinates)
        assert 0 <= self.y <= len(self.gtp_coordinates)

        return f"{self.gtp_coordinates[self.x]}{self.y + 1}"

    @classmethod
    def from_gtp(cls, gtp: str) -> "Point":
        """Create a new Point instance from the GTP vertex string.

        Args:
            gtp: A GTP vertex string representing a point.

        Raise:
            ValueError: Given string is not a valid GTP point.
        """

        gtp = gtp.upper()
        if match := re.match(f"([{Point.gtp_coordinates}])([\\d]+)", gtp):
            return Point(Point.gtp_coordinates.index(match[1]), int(match[2]) - 1)

        raise ValueError("given string is not a valid GTP point")

    @property
    def gtp(self) -> str:
        """The GTP representation of the point."""
        return str(self)

    def sgf(self, board_size: int = 19) -> str:
        """Return the SGF representation of the point.

        Args:
            board_size: The size of the Go board. Default is 19.
        """
        assert 0 <= self.x < board_size
        assert 0 <= self.y < board_size

        row = board_size - 1 - self.y
        return f"{self.sgf_coordinates[self.x]}{self.sgf_coordinates[row]}"

    def neighbors(self, board_size: int = 19) -> Generator["Point", None, None]:
        """Yield neighbor points of the point.

        Args:
            board_size: The size of the Go board. Default is 19.
        """
        assert 0 <= self.x < board_size
        assert 0 <= self.y < board_size

        if self.x + 1 < board_size:
            yield Point(self.x + 1, self.y)
        if self.x > 0:
            yield Point(self.x - 1, self.y)
        if self.y + 1 < board_size:
            yield Point(self.x, self.y + 1)
        if self.y > 0:
            yield Point(self.x, self.y - 1)

    def encode(self, board_size: int = 19) -> int:
        """Return an integer code representing the point on the board.

        Args:
            board_size: The size of the Go board. Default is 19.
        """
        return self.x * board_size + self.y
