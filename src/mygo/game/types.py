import itertools
import random
from collections import deque
from collections.abc import Sequence
from contextlib import contextmanager
from copy import copy
from enum import Enum, IntEnum
from typing import Any, NamedTuple

from mygo import pysgf
from mygo.pysgf import SGFNode


class Player(IntEnum):
    black = 1
    white = -1

    def __str__(self) -> str:
        return "black" if self == Player.black else "white"

    def __neg__(self):
        return Player.black if self == Player.white else Player.white

    @property
    def sgf(self) -> str:
        """Return SGF representation of the player."""
        return "B" if self == Player.black else "W"


class Point(NamedTuple):
    row: int = 0
    col: int = 0

    def neighbors(self, board_size: int = 19) -> list:
        ns = []
        if self.row < board_size:
            ns.append(Point(self.row + 1, self.col))
        if self.row > 1:
            ns.append(Point(self.row - 1, self.col))
        if self.col < board_size:
            ns.append(Point(self.row, self.col + 1))
        if self.col > 1:
            ns.append(Point(self.row, self.col - 1))

        return ns

    def sgf(self, board_size: int = 19) -> str:
        """Return SGF coordinates of the point."""
        col, row = self.col - 1, board_size - self.row
        return f"{pysgf.Move.SGF_COORD[col]}{pysgf.Move.SGF_COORD[row]}"


class Zobrist:
    """Zobrist hash table of moves."""

    _LIMIT = 2**63
    _table = {}

    EMPTY = 0

    random.seed(67731329655)
    for player, i, j in itertools.product(Player, range(1, 20), range(1, 20)):
        _table[player, Point(i, j)] = random.randrange(_LIMIT)

    @classmethod
    def hash(cls, player: Player, point: Point) -> int:
        return cls._table[player, point]


class MoveType(Enum):
    play = 1
    pass_ = 2
    resign = 3


class Move:
    _COLS = "ABCDEFGHJKLMNOPQRST"

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
        return f"Move({self.move_type}, {self.point!r})"

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

    def sgf(self, board_size: int = 19) -> str:
        """Return SGF coordinates of the move."""
        if not self.is_play:
            return ""
        return self.point.sgf(board_size)


Points = Sequence[Point]


class GoString:
    def __init__(self, player: Player, stones: Points, liberties: Points) -> None:
        self.player = player
        self.stones = set(stones)
        self.liberties = set(liberties)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GoString):
            return False
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        return f"GoString({self.player!r}, {self.stones!r}, {self.liberties!r})"

    def __ior__(self, other):
        assert self.player == other.player
        self.stones |= other.stones
        self.liberties = (self.liberties | other.liberties) - self.stones
        return self

    def __deepcopy__(self, memo: dict):
        """Return a deep copy without copying points for efficiency."""
        cls = self.__class__
        new_string = cls.__new__(cls)
        memo[id(self)] = new_string
        new_string.player = self.player
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
        assert point not in self.liberties
        self.liberties.add(point)


class StringBoard:
    def __init__(self, size: int) -> None:
        assert 1 <= size <= 19
        self.size = size
        self._grid = [[None for _ in range(size)] for _ in range(size)]
        self._hash = Zobrist.EMPTY

    def __repr__(self) -> str:
        return f"StringBoard({self.size!r})"

    def __str__(self) -> str:
        """Return ASCII representation."""

        last_line = f"   {' '.join(Move._COLS[:self.size])}"
        first_line = f"{last_line}\n"
        point_matrix = [["." for _ in range(self.size)] for _ in range(self.size)]

        if self.size < 3:
            star_offset = 0
        elif self.size == 3:
            star_offset = 1
        elif self.size <= 5:
            star_offset = 2
        elif self.size <= 11:
            star_offset = 3
        else:
            star_offset = 4

        # place '+' at star points
        if star_offset:
            star_indices = (star_offset - 1, self.size - star_offset)
            for i, j in itertools.product(star_indices, star_indices):
                point_matrix[i][j] = "+"

            if self.size % 2:
                point_matrix[self.size // 2][self.size // 2] = "+"

        s = first_line
        for i in range(self.size, 0, -1):
            for j in range(1, self.size + 1):
                if go_string := self[i, j]:
                    point_matrix[i - 1][j - 1] = (
                        "X" if go_string.player == Player.black else "O"
                    )
            s += f"{i:2} {' '.join(point_matrix[i - 1])} {i:2}\n"
        s += last_line

        return s

    def __getitem__(self, key: tuple[int, int]) -> GoString | None:
        """Return the Go string at point."""
        row, col = key
        return self._grid[row - 1][col - 1]

    def __setitem__(self, key: tuple[int, int], value: GoString | None) -> None:
        """Set the Go string at point."""
        row, col = key
        self._grid[row - 1][col - 1] = value

    def _remove_string(self, string: GoString) -> None:
        for point in string.stones:
            self[point] = None
            self._hash ^= Zobrist.hash(string.player, point)
            neighbor_strings = (
                self[p]
                for p in point.neighbors(self.size)
                if self[p] is not None and self[p] is not string
            )
            for s in neighbor_strings:
                if point in s.liberties:  # pytype: disable=attribute-error
                    continue
                s.add_liberty(point)  # pytype: disable=attribute-error

    def _add_string(self, string: GoString) -> None:
        """Revert _remove_string()."""
        for point in string.stones:
            neighbor_strings = (
                self[p]
                for p in point.neighbors(self.size)
                if self[p] is not None and self[p] is not string
            )
            for s in neighbor_strings:
                if point not in s.liberties:  # pytype: disable=attribute-error
                    continue
                s.remove_liberty(point)  # pytype: disable=attribute-error
            self._hash ^= Zobrist.hash(string.player, point)
            self[point] = string

    @property
    def zobrist_hash(self) -> int:
        return self._hash

    @property
    def empty_points(self) -> tuple[Point, ...]:
        return tuple(
            Point(i, j)
            for i in range(1, self.size + 1)
            for j in range(1, self.size + 1)
            if self[i, j] is None
        )

    def get_player(self, point: tuple[int, int]) -> Player | None:
        """
        Get the player of the stone at the point.

        Return None if no stone at the position.
        """
        string = self[point]
        return None if string is None else string.player

    def is_on_grid(self, point: Point) -> bool:
        return 1 <= point.row <= self.size and 1 <= point.col <= self.size

    def is_placeable(self, point: Point) -> bool:
        return self.is_on_grid(point) and self[point] is None

    def is_point_an_eye(self, point: Point, player: Player) -> bool:
        if self[point] is not None:
            return False
        for p in point.neighbors(self.size):
            if self.get_player(p) != player:
                return False

        friendly_corners = 0
        off_board_corners = 0
        corners = (
            Point(point.row - 1, point.col - 1),
            Point(point.row - 1, point.col + 1),
            Point(point.row + 1, point.col - 1),
            Point(point.row + 1, point.col + 1),
        )
        for p in corners:
            if not self.is_on_grid(p):
                off_board_corners += 1
                continue
            if self.get_player(p) == player:
                friendly_corners += 1

        if off_board_corners > 0:
            return off_board_corners + friendly_corners == 4
        return friendly_corners >= 3

    def place_stone(
        self, player: Player, point: Point
    ) -> tuple[list[GoString], list[GoString]]:
        """Place stone of player at point.

        Return a tuple of two GoString list. The first contains old adjacent same player
        Go strings. The second contains modified adjacent opposite Go strings.
        """
        assert self.is_placeable(point)

        adj_same = []
        adj_opposite = []
        liberties = []

        for p in point.neighbors(self.size):
            neighbor_string = self[p]
            if neighbor_string is None:
                liberties.append(p)
            elif neighbor_string.player == player and neighbor_string not in adj_same:
                adj_same.append(neighbor_string)
            elif (
                neighbor_string.player != player and neighbor_string not in adj_opposite
            ):
                adj_opposite.append(neighbor_string)

        # merge same player strings
        new_string = GoString(player, [point], liberties)
        for s in adj_same:
            new_string |= s
        for p in new_string.stones:
            self[p] = new_string

        self._hash ^= Zobrist.hash(player, point)

        # decrease opposite strings' liberties
        for s in adj_opposite:
            s.remove_liberty(point)
            if s.num_liberties == 0:
                self._remove_string(s)

        return adj_same, adj_opposite

    @contextmanager
    def place_stone_ctx(self, player: Player, point: Point):
        adj_same, adj_opposite = self.place_stone(player, point)
        try:
            yield self
        finally:
            for s in adj_opposite:
                if s.num_liberties == 0:
                    self._add_string(s)
                s.add_liberty(point)

            self._hash ^= Zobrist.hash(player, point)

            self[point] = None
            for string in adj_same:
                for p in string.stones:
                    self[p] = string


class Game:
    def __init__(
        self, board: StringBoard, next_player: Player, move: Move | None = None
    ) -> None:
        self.board = board
        self.move = move
        self.next_player = next_player
        self._history_situations = set()
        self._prev_is_pass = False

    def __repr__(self) -> str:
        return (
            f"Game(StringBoard({self.board.size!r}), {self.next_player}, {self.move!r})"
        )

    def __str__(self) -> str:
        """Return ASCII representation."""

        board_str = str(self.board)
        if self.move is None or not self.move.is_play:
            return board_str

        assert isinstance(self.move.point, Point)

        # add parentheses to last move
        lines = board_str.split("\n")
        size = self.board.size
        row, col = self.move.point
        line_idx = 1 + size - row
        ch_idx = 2 * col
        line = lines[line_idx]
        lines[line_idx] = f"{line[:ch_idx]}({line[ch_idx + 1]}){line[ch_idx + 3:]}"

        return "\n".join(lines)

    @classmethod
    def new_game(cls, size: int = 19):
        return cls(StringBoard(size), Player.black)

    @classmethod
    def from_sgf_root(cls, root: SGFNode):
        """Create a new game from a SGF root node."""

        assert root.move is None

        rows, cols = root.board_size
        assert rows == cols

        board = StringBoard(rows)
        for move in root.placements:
            player = Player.black if move.player == "B" else Player.white
            col, row = move.coords
            board.place_stone(player, Point(row + 1, col + 1))

        return cls(board, Player.black if root.next_player == "B" else Player.white)

    @property
    def size(self) -> int:
        return self.board.size

    @property
    def shape(self) -> tuple[int, int]:
        return self.size, self.size

    @property
    def is_over(self) -> bool:
        match self.move:
            case Move(move_type=MoveType.resign):
                return True
            case Move(move_type=MoveType.pass_):
                return self._prev_is_pass
            case _:
                return False

    @property
    def situation(self) -> tuple[Player, int]:
        return (self.next_player, self.board.zobrist_hash)

    @property
    def winner(self) -> Player:
        """Return the winner. The game must be over, or result will be wrong."""
        assert isinstance(self.move, Move)

        if self.move.is_resign:
            return self.next_player

        board = self.board
        size = board.size
        black_set = set()
        for i, j in itertools.product(range(1, size + 1), range(1, size + 1)):
            if board.get_player((i, j)) == Player.black:
                black_set.add(Point(i, j))

        black_queue = deque(black_set)
        while black_queue:
            point = black_queue.pop()
            for p in point.neighbors(size):
                if p in black_set:
                    continue
                if board.get_player(p) is None:
                    black_set.add(p)
                    black_queue.append(p)

        count = len(black_set)
        return Player.black if count > size * size - count else Player.white

    @property
    def valid_plays(self) -> list[Move]:
        """Return valid play moves."""
        return [
            m
            for m in (Move.play(p) for p in self.board.empty_points)
            if self.is_valid_move(m)
        ]

    @property
    def good_moves(self) -> list[Move]:
        """Return good moves in random order.

        Good moves are valid play moves except moves which place stone in self's eye.
        """
        moves = [
            move
            for move in self.valid_plays
            if not self.board.is_point_an_eye(move.point, self.next_player)
        ]
        random.shuffle(moves)
        return moves

    @property
    def score(self) -> int:
        """Return score of current player."""
        board = self.board
        limit = board.size + 1
        black_count, white_count = 0, 0
        for player in (
            board.get_player((i, j)) for i in range(1, limit) for j in range(1, limit)
        ):
            if player == Player.black:
                black_count += 1
            elif player == Player.white:
                white_count += 1

        diff = black_count - white_count
        return diff if self.next_player == Player.white else -diff

    def is_valid_move(self, move: Move) -> bool:
        if self.is_over:
            return False
        if not move.is_play:
            return True
        if not self.board.is_placeable(move.point):
            return False

        with self.board.place_stone_ctx(self.next_player, move.point) as board:
            # check if move is self capture
            if board[move.point].num_liberties == 0:  # pytype: disable=attribute-error
                return False

            # check if move is self capture
            if board[move.point].num_liberties == 0:  # pytype: disable=attribute-error
                return False

            # check if move violates the ko rule
            situation = (-self.next_player, board.zobrist_hash)
            if situation in self._history_situations:
                return False

        return True

    def apply_move(self, move: Move) -> int:
        """Apply the move. Return the number of captured stones."""

        old_next_player = self.next_player

        self._history_situations.add(self.situation)
        self._prev_is_pass = False if self.move is None else self.move.is_pass
        self.move = move
        self.next_player = -old_next_player
        if move.is_play:
            _, adj_opposite = self.board.place_stone(old_next_player, move.point)
            return sum(len(s.stones) for s in adj_opposite if s.num_liberties == 0)

        return 0

    @contextmanager
    def apply_move_ctx(self, move: Move):
        new_situation = self.situation
        old_prev_is_pass = self._prev_is_pass
        old_move = self.move
        old_next_player = self.next_player

        self._history_situations.add(self.situation)
        self._prev_is_pass = False if self.move is None else self.move.is_pass
        self.move = move
        self.next_player = -old_next_player

        try:
            if move.is_play:
                with self.board.place_stone_ctx(old_next_player, move.point):
                    yield self
            else:
                yield self
        finally:
            self.next_player = old_next_player
            self.move = old_move
            self._prev_is_pass = old_prev_is_pass
            self._history_situations.remove(new_situation)
