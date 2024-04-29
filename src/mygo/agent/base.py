import abc

from mygo.game.types import Move


class Agent(abc.ABC):
    @abc.abstractmethod
    def select_move(self, game) -> Move:
        """Return selected next move."""
