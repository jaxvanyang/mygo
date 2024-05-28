import abc

from mygo.game.types import Game, Move


class Agent(abc.ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abc.abstractmethod
    def select_move(self, game: Game) -> Move:
        """Return selected next move."""
