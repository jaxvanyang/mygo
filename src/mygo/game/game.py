"""Game types of the game of Go."""

from contextlib import contextmanager
from copy import deepcopy
from typing import Generator

from mygo.game.basic import Player
from mygo.game.board import StringBoard
from mygo.game.move import Move, PlayMove, from_pysgf_move
from mygo.pysgf import SGF, SGFNode


class Game:
    """The game of Go.

    Only support linear game history.

    Attributes:
        boards: A list of history boards.
        moves: A list of history moves.
        next_player: The next player of the game.
        komi: A float number, the komi of the game.
        situations: A set of history situations, a situation is a tuple of next player
          and the Zobrist hash of board. Current situation is not included.
    """

    def __init__(
        self,
        boards: list[StringBoard],
        moves: list[Move],
        next_player: Player = Player.black,
        komi: float = 5.5,
    ) -> None:
        """Initialize the game of Go.

        Args:
            boards: A list of history boards.
            moves: A list of history moves. It should has exactly one element less than
              boards, because each move represents the move between two boards.
            next_player: The next player of the game. Default is black.
            komi: A float number, the komi of the game. Default is 5.5.
        """

        if len(moves) + 1 != len(boards):
            raise ValueError("boards should have exactly one element more than moves")

        self.boards = boards
        self.moves = moves
        self.next_player = next_player
        self.komi = komi
        self.situations = set()

        for move, board in zip(self.moves, self.boards):
            self.situations.add((move.player, board.hash))

    def __str__(self) -> str:
        """Return the ASCII representation of the game."""

        board_ascii = str(self.last_board)
        if (last_move := self.last_move) is None or not isinstance(last_move, PlayMove):
            return board_ascii

        # add parentheses to the last move point
        lines = board_ascii.split("\n")
        row = self.board_size - last_move.point.y
        col = 2 * last_move.point.x + 2
        line = lines[row]
        lines[row] = f"{line[:col]}({line[col + 1]}){line[col + 3:]}"

        return "\n".join(lines)

    @classmethod
    def new(cls, board_size: int = 19, komi: float = 5.5) -> "Game":
        """Return a new game of Go.

        Args:
            board_size: The size of the board. Default is 19.
            komi: A float number, the komi of the game. Default is 5.5.
        """
        return cls([StringBoard(board_size)], [], komi=komi)

    @classmethod
    def from_pysgf(cls, node: SGFNode | SGF) -> "Game":
        """Return a new game of Go from a pysgf SGFNode.

        Only placements of the root node are parsed, and noly support square board game.

        Args:
            node: A pysgf SGFNode or a SGF instance.

        Raise:
            ValueError: Input SGF has a non-square board.
        """
        root = node.root

        row_size, col_size = root.board_size
        if row_size != col_size:
            raise ValueError("SGF's row size and column size not equal")

        board = StringBoard(row_size)
        for move in root.move_with_placements:
            board.apply_move(from_pysgf_move(move))

        # consider moves of root node as placements
        return cls([board], [], next_player=Player.from_sgf(root.next_player))

    @property
    def last_board(self) -> StringBoard:
        """The last history board of the game."""
        assert len(self.boards) > 0
        return self.boards[-1]

    @property
    def last_move(self) -> Move | None:
        """The last move of the game."""
        return self.moves[-1] if self.moves else None

    @property
    def board_size(self) -> int:
        """The size of the Go board."""
        return self.last_board.board_size

    @property
    def shape(self) -> tuple[int, int]:
        """A tuple of two integers, the shape of the Go board."""
        return self.board_size, self.board_size

    @property
    def is_over(self) -> bool:
        """If the game is over."""

        if (last_move := self.last_move) is None:
            return False
        if last_move.is_resign:
            return True
        if last_move.is_play:
            return False

        try:
            return self.moves[-2].is_pass
        except IndexError:
            return False

    @property
    def diff(self) -> float:
        """The difference of black's area and white's area minus komi."""
        return self.last_board.area_diff - self.komi

    @property
    def score(self) -> float:
        """The score of current player."""
        diff = self.diff
        return diff if self.next_player == Player.white else -diff

    @property
    def result(self) -> str:
        """A string represents the result in SGF format."""

        if not self.is_over:
            return "?"

        assert isinstance((last_move := self.last_move), Move)
        if last_move.is_resign:
            return f"{last_move.player.opponent.sgf}+Resign"

        if (diff := self.diff) == 0.0:
            return "0"

        return f"B+{diff}" if diff > 0.0 else f"W+{-diff}"

    @property
    def winner(self) -> Player | None:
        """The winner of the game if it's over, None otherwise."""

        if not self.is_over:
            return None

        assert isinstance((last_move := self.last_move), Move)
        if last_move.is_resign:
            return -last_move.player

        if (diff := self.diff) == 0.0:
            return None

        return Player.black if diff > 0.0 else Player.white

    @property
    def valid_plays(self) -> Generator[PlayMove, None, None]:
        """A generator of valid play moves."""
        for point in self.last_board.empties:
            if self.is_valid_move(move := PlayMove(self.next_player, point)):
                yield move

    @property
    def good_moves(self) -> Generator[PlayMove, None, None]:
        """A generator of good moves. Good moves are valid play moves except moves that
        place stone in self's eye.
        """
        for move in self.valid_plays:
            if not self.last_board.is_eye(move.player, move.point):
                yield move

    def reset(self, board_size: int = 19) -> None:
        """Reset the game with the board size.

        Args:
            board_size: The new size of the game board.
        """
        self.boards.clear()
        self.boards.append(StringBoard(board_size))
        self.moves.clear()
        self.next_player = Player.black
        self.situations.clear()

    def is_valid_move(self, move: Move) -> bool:
        """Return if the move is valid.

        Args:
            move: A move instance.
        """
        if self.is_over:
            return False
        if not isinstance(move, PlayMove):
            return True
        if not (last_board := self.last_board).is_placeable(move.point):
            return False

        with last_board.place_stone_ctx(self.next_player, move.point) as board:
            # check if move is self capture
            assert (string := board[move.point]) is not None
            if string.liberty_count == 0:
                return False

            # check if move violates the ko rule
            situation = (-move.player, board.hash)
            if situation in self.situations:
                return False

        return True

    def apply_move(self, move: Move) -> int:
        """Apply the move.

        Args:
            move: A Move instance.

        Return:
            The number of captured stones.
        """

        self.moves.append(move)
        self.boards.append(board := deepcopy(self.last_board))
        self.situations.add((move.player, board.hash))
        self.next_player = -move.player

        return board.apply_move(move)

    @contextmanager
    def apply_move_ctx(self, move: Move) -> Generator["Game", None, None]:
        """Make a context that played the move.

        After the context, the move will be reverted.

        Args:
            move: A Move instance.

        Yield:
            Self.
        """

        next_player = self.next_player
        situation = (move.player, self.last_board.hash)
        self.apply_move(move)

        try:
            yield self
        finally:
            self.next_player = next_player
            self.situations.remove(situation)
            self.boards.pop()
            self.moves.pop()
