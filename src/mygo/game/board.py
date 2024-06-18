"""Go board type."""

import itertools
from contextlib import contextmanager
from typing import Generator

from mygo.game.basic import Player, Point
from mygo.game.gostring import GoString
from mygo.game.move import Move, PlayMove
from mygo.game.zobrist import Zobrist


class StringBoard:
    """The board of the Go game, consists of Go strings.

    Attributes:
        board_size: The size of the board.
        hash: The Zobrist hash of the board.
    """

    def __init__(self, board_size: int) -> None:
        """Initialize the board based on the board size.

        Args:
            board_size: The size of the board, ranging from 1 to 19. Default is 19.
        """
        if board_size < 1 or board_size > 19:
            raise ValueError("only support board size between 1 and 19")

        self.board_size = board_size
        self.hash = Zobrist.EMPTY
        self._grid = [[None for _ in range(board_size)] for _ in range(board_size)]

    def __repr__(self) -> str:
        """Return the canonical string representation of the object."""
        return f"StringBoard({self.board_size!r})"

    def __str__(self) -> str:
        """Return the ASCII representation of the board."""

        horizontal_axis = f"   {' '.join(Point.gtp_coordinates[:self.board_size])}"
        point_matrix = [
            ["." for _ in range(self.board_size)] for _ in range(self.board_size)
        ]

        if self.board_size < 3:
            edge_distance = -1
        elif self.board_size == 3:
            edge_distance = 0
        elif self.board_size <= 5:
            edge_distance = 1
        elif self.board_size <= 11:
            edge_distance = 2
        else:
            edge_distance = 3

        # draw '+' at star points
        if edge_distance >= 0:
            star_indices = [edge_distance, self.board_size - 1 - edge_distance]
            if self.board_size % 2:
                center = self.board_size // 2
                point_matrix[center][center] = "+"
                if self.board_size >= 13:
                    star_indices.append(center)

            for i, j in itertools.product(star_indices, star_indices):
                point_matrix[i][j] = "+"

        # draw the board
        s = f"{horizontal_axis}\n"
        for row in range(self.board_size - 1, -1, -1):
            for col in range(self.board_size):
                if go_string := self[col, row]:
                    point_matrix[row][col] = (
                        "X" if go_string.player == Player.black else "O"
                    )
            s += f"{row+1:2} {' '.join(point_matrix[row])} {row+1:2}\n"
        s += horizontal_axis

        return s

    def __getitem__(self, point: tuple[int, int]) -> GoString | None:
        """Return the Go string at the point.

        Args:
            point: A tuple of two integers.

        Return:
            The Go string at the point if it exists, None otherwise.
        """
        return self._grid[point[0]][point[1]]

    def __setitem__(self, point: tuple[int, int], value: GoString | None) -> None:
        """Set the value at the point.

        Args:
            point: A tuple of two integers.
            value: A Go string or None.
        """
        self._grid[point[0]][point[1]] = value

    def _remove_string(self, string: GoString) -> None:
        for stone in string.stones:
            self[stone] = None
            self.hash ^= Zobrist.hash(string.player, stone)

            for point in stone.neighbors(self.board_size):
                if (neighbor := self[point]) not in (None, string):
                    # each neighbor string may be checked more than once,
                    # so only add the stone to liberties if it exists
                    neighbor.liberties.add(stone)

    def _add_string(self, string: GoString) -> None:
        """Revert _remove_string()."""
        for stone in string.stones:
            self[stone] = string
            self.hash ^= Zobrist.hash(string.player, stone)

            for point in stone.neighbors(self.board_size):
                if (neighbor := self[point]) not in (None, string):
                    # each neighbor string may be checked more than once,
                    # so only remove the stone from liberties if it exists
                    neighbor.liberties.discard(stone)

    @property
    def empties(self) -> Generator[Point, None, None]:
        """An generator of empty points on the board."""
        for x, y in itertools.product(range(self.board_size), range(self.board_size)):
            if self[point := Point(x, y)] is None:
                yield point

    @property
    def blacks(self) -> Generator[Point, None, None]:
        """An generator of black points on the board."""
        for x, y in itertools.product(range(self.board_size), range(self.board_size)):
            if (string := self[point := Point(x, y)]) is None:
                continue
            if string.player == Player.black:
                yield point

    @property
    def whites(self) -> Generator[Point, None, None]:
        """An generator of white points on the board."""
        for x, y in itertools.product(range(self.board_size), range(self.board_size)):
            if (string := self[point := Point(x, y)]) is None:
                continue
            if string.player == Player.white:
                yield point

    @property
    def area_diff(self) -> int:
        """The difference of black's area and white's area."""
        return self.count_area(Player.black) - self.count_area(Player.white)

    def get_player(self, point: tuple[int, int]) -> Player | None:
        """Return the player of the stone at the point.

        Args:
            point: A tuple of two integers.

        Return:
            The player of the stone if it exists, None otherwise.
        """
        return None if (string := self[point]) is None else string.player

    def is_on_grid(self, point: tuple[int, int]) -> bool:
        """Return if the point on the board.

        Args:
            point: A tuple of two integers.
        """
        return 0 <= point[0] < self.board_size and 0 <= point[1] < self.board_size

    def is_placeable(self, point: tuple[int, int]) -> bool:
        """Return if the point can be placed on the board.

        Args:
            point: A tuple of two integers.
        """
        return self.is_on_grid(point) and self[point] is None

    def is_eye(self, player: Player, point: tuple[int, int]) -> bool:
        """Return if the point is an eye.

        Args:
            player: The player who want to place a stone at the point.
            point: A tuple of two integers.
        """
        if not self.is_placeable(point):
            return False

        if not isinstance(point, Point):
            point = Point(*point)

        for neighbor in point.neighbors(self.board_size):
            if self.get_player(neighbor) != player:
                return False

        friendly_corners = 0
        off_board_corners = 0
        corners = (
            Point(point.x + 1, point.y + 1),
            Point(point.x + 1, point.y - 1),
            Point(point.x - 1, point.y + 1),
            Point(point.x - 1, point.y - 1),
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
        self, player: Player, stone: tuple[int, int]
    ) -> tuple[list[GoString], list[GoString]]:
        """Place the stone of the player.

        Args:
            player: The player of the stone.
            stone: A tuple of two integers.

        Return:
            A tuple of two lists of Go strings. The first contains removed Go strings of
            the same player. The second contains modified Go strings of the opponent
            player.
        """
        if not self.is_placeable(stone):
            raise ValueError("stone not placeable")

        if not isinstance(stone, Point):
            stone = Point(*stone)

        friends, opponents, liberties = [], [], []

        for point in stone.neighbors(self.board_size):
            if (neighbor := self[point]) is None:
                liberties.append(point)
            elif neighbor.player == player and neighbor not in friends:
                friends.append(neighbor)
            elif neighbor.player != player and neighbor not in opponents:
                opponents.append(neighbor)

        # merge same player strings
        new_string = GoString(player, (stone,), liberties)
        self.hash ^= Zobrist.hash(player, stone)
        for friend in friends:
            new_string |= friend
        for point in new_string.stones:
            self[point] = new_string

        # decrease opposite strings' liberties
        for opponent in opponents:
            opponent.remove_liberty(stone)
            if opponent.liberty_count == 0:
                self._remove_string(opponent)

        return friends, opponents

    @contextmanager
    def place_stone_ctx(
        self, player: Player, stone: tuple[int, int]
    ) -> Generator["StringBoard", None, None]:
        """Make a context that place the stone of the player.

        After the context, the board state will be restored.

        Args:
            player: The player of the stone.
            stone: A tuple of two integers.

        Yield:
            Self.
        """
        if not isinstance(stone, Point):
            stone = Point(*stone)

        friends, opponents = self.place_stone(player, stone)
        try:
            yield self
        finally:
            for opponent in opponents:
                if opponent.liberty_count == 0:
                    self._add_string(opponent)
                opponent.add_liberty(stone)
            self.hash ^= Zobrist.hash(player, stone)

            self[stone] = None
            for friend in friends:
                for point in friend.stones:
                    self[point] = friend

    def apply_move(self, move: Move) -> int:
        """Apply the move.

        Args:
            move: A Move instance.

        Return:
            The number of captured stones.
        """
        if not isinstance(move, PlayMove):
            return 0

        _, opponents = self.place_stone(move.player, move.point)
        return sum(len(s.stones) for s in opponents if s.liberty_count == 0)

    def count_area(self, player: Player) -> int:
        """Return the number of points the player's stones coocupy and surround.

        Note that dead stones are also counted, so this result may be incorrect.

        Args:
            player: The player of the area to be counted.
        """

        # max area includes undetermined area
        # count opponent's max area, the complement of that is our area
        player = -player
        opponents = list(self.blacks if player == Player.black else self.whites)
        area = set(opponents)

        while opponents:
            opponent = opponents.pop()
            for neighbor in opponent.neighbors(self.board_size):
                if neighbor not in area and self.get_player(neighbor) == player:
                    area.add(neighbor)
                    opponents.append(neighbor)

        return self.board_size * self.board_size - len(area)
