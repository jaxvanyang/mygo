from abc import ABC, abstractmethod

from mygo.game.basic import Player
from mygo.game.game import Game
from mygo.game.move import Move


class Agent(ABC):
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
