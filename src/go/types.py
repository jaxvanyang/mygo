from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Color(Enum):
    black = 1
    white = 2

    @property
    def opposite(self):
        return Color.black if self == Color.white else Color.white


@dataclass(frozen=True)
class Point:
    row: int = 0
    col: int = 0

    def neighbors(self, size) -> list:
        ns = []
        if self.row < size:
            ns.append(Point(self.row + 1, self.col))
        if self.row > 1:
            ns.append(Point(self.row - 1, self.col))
        if self.col < size:
            ns.append(Point(self.row, self.col + 1))
        if self.col > 1:
            ns.append(Point(self.row, self.col - 1))

        return ns


class MoveType(Enum):
    play = 1
    pass_ = 2
    resign = 3


class Move:
    _COLS = "ABCDEFGHIJKLMNOPQRST"

    def __init__(self, move_type: MoveType, point: Point | None = None) -> None:
        # is_play no_point good
        # 0 0 0
        # 0 1 1
        # 1 0 1
        # 1 1 0
        assert (move_type == MoveType.play) ^ (point is None)
        self.move_type = move_type
        self.point = point

    def __repr__(self) -> str:
        return f"Move({self.move_type}, {self.point})"

    def __str__(self) -> str:
        if self.is_pass:
            return "pass"
        if self.is_resign:
            return "resign"
        return f"{self._COLS[self.point.col - 1]}{self.point.row}"

    @classmethod
    def play(cls, point):
        return cls(MoveType.play, point)

    @classmethod
    def pass_(cls):
        return cls(MoveType.pass_)

    @classmethod
    def resign(cls):
        return cls(MoveType.resign)

    @property
    def is_play(self) -> bool:
        return self.move_type == MoveType.play

    @property
    def is_pass(self) -> bool:
        return self.move_type == MoveType.pass_

    @property
    def is_resign(self) -> bool:
        return self.move_type == MoveType.resign


Points = Sequence[Point]


class GoString:
    def __init__(self, color: Color, stones: Points, liberties: Points) -> None:
        self.color = color
        self.stones = set(stones)
        self.liberties = set(liberties)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GoString):
            return False
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        return f"GoString({self.color}, {self.stones}, {self.liberties})"

    def __ior__(self, other):
        assert self.color == other.color
        self.stones |= other.stones
        self.liberties = (self.liberties | other.liberties) - self.stones
        return self

    def __deepcopy__(self, memo: dict):
        """Return a deep copy without copying points for efficiency."""
        cls = self.__class__
        new_string = cls.__new__(cls)
        memo[id(self)] = new_string
        new_string.color = self.color
        new_string.stones = copy(self.stones)
        new_string.liberties = copy(self.liberties)
        return new_string

    @property
    def num_liberties(self) -> int:
        return len(self.liberties)

    def remove_liberty(self, point: Point) -> None:
        assert point in self.liberties
        self.liberties.remove(point)

    def add_liberty(self, point: Point) -> None:
        self.liberties.add(point)
