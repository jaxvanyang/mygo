import abc

from mygo.game.game import Game
from mygo.game.move import Move


class Agent(abc.ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def select_move(self, game: Game) -> Move:
        """Return selected next move."""
