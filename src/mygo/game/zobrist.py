"""Zobrist hash helper of the Go board."""

import random

from mygo.game.basic import Player, Point
from mygo.game.move import PlayMove


class Zobrist:
    """Generate hashes of Go play moves. Only support board size <= 19."""

    EMPTY = 0

    random.seed(67731329655)  # make generated hashes reproducible
    _table = tuple(
        tuple(tuple(random.randrange(2**63) for _ in range(19)) for _ in range(19))
        for _ in range(2)
    )
    random.seed()  # reset seed

    @classmethod
    def hash(cls, player: Player, point: Point) -> int:
        """Return the hash based on the player and point of the move.

        Args:
            player: The player of the move.
            point: The point of the move.
        """
        i = 0 if player == Player.white else 1
        return cls._table[i][point.x][point.y]

    @classmethod
    def hash_move(cls, move: PlayMove) -> int:
        """Return the hash of the play move.

        Args:
            move: The play move.
        """
        return cls.hash(move.player, move.point)
