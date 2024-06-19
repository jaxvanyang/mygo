"""Go string type."""

from copy import copy
from typing import Any, Collection

from mygo.game.basic import Player, Point

Points = Collection[Point]


class GoString:
    """Represent a Go string on the board.

    Attributes:
        player: The player of the Go string.
        stones: A set of the stone points of the Go string.
        liberties: A set of the liberty points of the Go string.
    """

    def __init__(self, player: Player, stones: Points, liberties: Points) -> None:
        """Initialize a go string.

        Args:
            player: The player of the Go string.
            stones: A collection of the stone points of the Go string.
            liberties: A collection of the liberty points of the Go string.
        """
        self.player = player
        self.stones = set(stones)
        self.liberties = set(liberties)

    def __eq__(self, other: Any) -> bool:
        """Return if this is equivalent to the other.

        Args:
            other: The other object to be compared with.
        """
        if not isinstance(other, GoString):
            return False
        return (
            self.player == other.player
            and self.stones == other.stones
            and self.liberties == other.liberties
        )

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        return f"GoString({self.player!r}, {self.stones!r}, {self.liberties!r})"

    def __ior__(self, other: "GoString") -> "GoString":
        """Merge a Go string.

        Args:
            other: The Go string to be merged, which must belong to the same player.

        Return:
            Merged go string, i.e., self.
        """
        if self.player != other.player:
            raise ValueError("cannot merge Go string of different player")

        self.stones |= other.stones
        self.liberties = (self.liberties | other.liberties) - self.stones

        return self

    def __deepcopy__(self, memo: dict) -> "GoString":
        """Return a deep copy without copying points for efficiency."""

        cls = self.__class__
        string = cls.__new__(cls)
        memo[id(self)] = string
        string.player = self.player
        string.stones = copy(self.stones)
        string.liberties = copy(self.liberties)

        return string

    @property
    def liberty_count(self) -> int:
        """The number of liberties."""
        return len(self.liberties)

    def add_liberty(self, point: Point) -> None:
        """Add a liberty from the Go string.

        Args:
            point: The liberty point to be add, it must not exist in the liberties."""
        if point in self.liberties:
            raise ValueError("point already in the liberties")

        self.liberties.add(point)

    def remove_liberty(self, point: Point) -> None:
        """Remove a liberty from the Go string.

        Args:
            point: The liberty point to be removed, it must exist in the liberties.
        """
        if point not in self.liberties:
            raise ValueError("point not in the liberties")

        self.liberties.remove(point)
