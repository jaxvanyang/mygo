"""Move types of the Go game."""

from abc import ABC, abstractmethod

from mygo import pysgf
from mygo.game.basic import Player, Point


class Move(ABC):
    """The move of a game round.

    Attributes:
        player: The player of the move.
    """

    def __init__(self, player: Player) -> None:
        """Initialize the move of the player.

        Args:
            player: The player of the move.
        """
        super().__init__()
        self.player = player

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        return f"{self.__class__.__name__}({self.player!r})"

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-friendly string representation of the move."""

    @property
    def is_pass(self) -> bool:
        """Is this a pass move."""
        return self.__class__.__name__ == "PassMove"

    @property
    def is_resign(self) -> bool:
        """Is this a resign move."""
        return self.__class__.__name__ == "ResignMove"

    @property
    def is_play(self) -> bool:
        """Is this a play move."""
        return self.__class__.__name__ == "PlayMove"

    @abstractmethod
    def to_pysgf(self) -> pysgf.Move:
        """Return an equivalent pysgf.Move instance.

        Raise:
            TypeError: The move is a resign move, which pysgf doesn't support.
        """

    def sgf(self, board_size: int = 19) -> str:
        """Return a string of the move's coordinates.

        Args:
            board_size: The size of the game board. Default is 19.
        """
        return self.to_pysgf().sgf((board_size, board_size))


class PassMove(Move):
    """A pass move of a game round."""

    def __str__(self) -> str:
        return "pass"

    def to_pysgf(self) -> pysgf.Move:
        return pysgf.Move(player=self.player.sgf)


class ResignMove(Move):
    """A resign move of a game round."""

    def __str__(self) -> str:
        return "resign"

    def to_pysgf(self) -> pysgf.Move:
        raise TypeError("pysgf.Move doesn't support resign move")


class PlayMove(Move):
    """A play move of a game round.

    Attributes:
        player: The player of the move.
        point: The stone position of the move.
    """

    def __init__(self, player: Player, point: Point) -> None:
        """Initialize the play move of the player.

        Args:
            player: The player of the move.
            point: The stone position of the move.
        """
        super().__init__(player)
        self.point = point

    def __repr__(self) -> str:
        return f"PlayMove({self.player!r}, {self.point!r})"

    def __str__(self) -> str:
        return self.point.gtp

    def to_pysgf(self) -> pysgf.Move:
        return pysgf.Move((self.point.x, self.point.y), self.player.sgf)


def from_gtp_move(move: str, player: Player) -> Move:
    """Return a Move instance from the GTP vertex string.

    Args:
        move: A GTP vertex string representing a move.
        player: The player of the move.
    """
    move = move.upper()

    if move == "PASS":
        return PassMove(player)
    if move == "RESIGN":
        return ResignMove(player)

    return PlayMove(player, Point.from_gtp(move))


def from_pysgf_move(move: pysgf.Move) -> Move:
    """Return a Move instance from the pysgf.Move.

    Args:
        move: A pysgf.Move object.
    """
    if move.is_pass:
        return PassMove(Player.from_sgf(move.player))

    assert isinstance(move.coords, tuple)
    return PlayMove(Player.from_sgf(move.player), Point(*move.coords))
