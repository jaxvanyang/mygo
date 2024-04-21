from copy import deepcopy
from typing import overload

from go.types import Color, GoString, Move, MoveType, Point
from go.zobrist import Zobrist


class StringBoard:
    def __init__(self, size: int) -> None:
        assert 1 <= size <= 19
        self.size = size
        self._grid = [[None for _ in range(size)] for _ in range(size)]
        self._hash = Zobrist.EMPTY

    def __repr__(self) -> str:
        return f"StringBoard({self.size})"

    def __str__(self) -> str:
        s = f"   {' '.join(Move._COLS[:self.size])}\n"
        for i in range(self.size, 0, -1):
            s += f"{i:2} "
            for j in range(1, self.size + 1):
                s += f"{self._get_point_repr(i, j)} "
            s += f"{i:2}\n"
        s += f"   {' '.join(Move._COLS[:self.size])}"

        return s

    @overload
    def __getitem__(self, key: tuple[int, int]) -> GoString | None: ...

    @overload
    def __getitem__(self, key: Point) -> GoString | None: ...

    def __getitem__(self, key):
        """Return the Go string at point."""
        match key:
            case (row, col):
                return self._grid[row - 1][col - 1]
            case Point(row, col):
                return self._grid[row - 1][col - 1]
            case _:
                raise TypeError

    @overload
    def __setitem__(self, key: tuple[int, int], value: GoString | None) -> None: ...

    @overload
    def __setitem__(self, key: Point, value: GoString | None) -> None: ...

    def __setitem__(self, key, value):
        """Set the Go string at point."""
        match key:
            case (row, col):
                self._grid[row - 1][col - 1] = value
            case Point(row, col):
                self._grid[row - 1][col - 1] = value
            case _:
                raise TypeError

    def _get_point_repr(self, row: int, col: int) -> str:
        string = self[Point(row, col)]
        if string is None:
            return "."
        return "X" if string.color == Color.black else "O"

    def _remove_string(self, string: GoString) -> None:
        for point in string.stones:
            self[point] = None
            self._hash ^= Zobrist.hash(string.color, point)
            for neighbor_string in [self[p] for p in point.neighbors(self.size)]:
                if neighbor_string is not None and neighbor_string is not string:
                    neighbor_string.add_liberty(point)

    @property
    def zobrist_hash(self) -> int:
        return self._hash

    def get_point_color(self, point: Point) -> Color | None:
        """Return the color of stone at point. Return False if no stone at point."""
        string = self[point]
        return None if string is None else string.color

    def is_on_grid(self, point: Point) -> bool:
        return 1 <= point.row <= self.size and 1 <= point.col <= self.size

    def is_placeable(self, point: Point) -> bool:
        return self.is_on_grid(point) and self[point] is None

    def is_point_an_eye(self, point: Point, color: Color) -> bool:
        if self[point] is not None:
            return False
        for p in point.neighbors(self.size):
            if self.get_point_color(p) != color:
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
            if self.get_point_color(p) == color:
                friendly_corners += 1

        if off_board_corners > 0:
            return off_board_corners + friendly_corners == 4
        return friendly_corners >= 3

    def place_stone(self, color: Color, point: Point) -> None:
        assert self.is_placeable(point)

        adj_same = []
        adj_opposite = []
        liberties = []

        for p in point.neighbors(self.size):
            neighbor_string = self[p]
            if neighbor_string is None:
                liberties.append(p)
            elif neighbor_string.color == color and neighbor_string not in adj_same:
                adj_same.append(neighbor_string)
            elif neighbor_string.color != color and neighbor_string not in adj_opposite:
                adj_opposite.append(neighbor_string)

        # merge same color strings
        new_string = GoString(color, [point], liberties)
        for s in adj_same:
            new_string |= s
        for p in new_string.stones:
            self[p] = new_string

        self._hash ^= Zobrist.hash(color, point)

        # decrease opposite strings' liberties
        for s in adj_opposite:
            s.remove_liberty(point)
            if s.num_liberties == 0:
                self._remove_string(s)


class Game:
    def __init__(
        self, board: StringBoard, next_color: Color, move: Move | None = None
    ) -> None:
        self.board = board
        self.move = move
        self.next_color = next_color
        self._history_situations = set()
        self._prev_is_pass = False

    def __repr__(self) -> str:
        return f"Game(StringBoard({self.board.size}), {self.next_color}, {self.move})"

    @classmethod
    def new_game(cls, size: int):
        return cls(StringBoard(size), Color.black)

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
    def situation(self) -> tuple[Color, int]:
        return (self.next_color, self.board.zobrist_hash)

    def is_valid_move(self, move: Move) -> bool:
        if self.is_over:
            return False
        if not move.is_play:
            return True
        if not self.board.is_placeable(move.point):
            return False

        board = deepcopy(self.board)
        board.place_stone(self.next_color, move.point)

        # check if move is self capture
        if board[move.point].num_liberties == 0:  # pytype: disable=attribute-error
            return False

        # check if move violates the ko rule
        situation = (self.next_color.opposite, board)
        if situation in self._history_situations:
            return False

        return True

    def apply_move(self, move: Move) -> None:
        self._history_situations |= {self.situation}
        if move.is_play:
            self.board.place_stone(self.next_color, move.point)
        self._prev_is_pass = False if self.move is None else self.move.is_pass
        self.move = move
        self.next_color = self.next_color.opposite
