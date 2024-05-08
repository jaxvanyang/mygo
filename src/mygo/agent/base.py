import abc

from mygo.game.types import Game, Move


class Agent(abc.ABC):
    @abc.abstractmethod
    def select_move(self, game: Game) -> Move:
        """Return selected next move."""
