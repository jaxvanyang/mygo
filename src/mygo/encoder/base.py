from abc import ABC, abstractmethod

from numpy import ndarray

from mygo.game.types import Game


class Encoder(ABC):
    # pytype: disable=bad-return-type
    @staticmethod
    @abstractmethod
    def encode(game: Game) -> ndarray:
        """Return encoded board of game."""

    # pytype: enable=bad-return-type
