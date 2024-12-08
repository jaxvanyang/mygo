from abc import ABC, abstractmethod

from mygo.game.basic import Player
from mygo.game.game import Game
from mygo.game.move import Move


class Agent(ABC):
    @staticmethod
    def rate_to_value(rate: float) -> float:
        """Convert winning rate to the approximate optimal value."""

        assert 0 <= rate <= 1
        return -1 + rate * 2.0

    @staticmethod
    def value_to_rate(value: float) -> float:
        """Convert the approximate optimal value to winning rate."""

        assert -1 <= value <= 1
        return (value + 1) / 2.0

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def select_move(self, game: Game, player: Player | None = None) -> Move:
        """Select a move for the next round.

        Args:
            game: The game of the move.
            player: The player of the move. Default is the default next player
              of the game.
        """
