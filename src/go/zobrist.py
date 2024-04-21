import random
from itertools import product

from go.types import Color, Point


class Zobrist:
    """Zobrist hash table of moves."""

    _LIMIT = 2**63
    _table = {}

    EMPTY = 0

    random.seed(67731329655)
    for color, i, j in product(Color, range(1, 20), range(1, 20)):
        _table[color, Point(i, j)] = random.randrange(_LIMIT)

    @classmethod
    def hash(cls, color: Color, point: Point) -> int:
        return cls._table[color, point]
