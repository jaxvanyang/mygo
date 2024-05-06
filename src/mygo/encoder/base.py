from abc import ABC, abstractmethod

from numpy import ndarray

from mygo.game.types import Game


class Encoder(ABC):
    @staticmethod
    @abstractmethod
    def encode(game: Game) -> ndarray:
        """Return encoded board of game."""
