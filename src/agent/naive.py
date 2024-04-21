import itertools
import random

from agent.base import Agent
from go.game import Game
from go.types import Move, Point


class RandomBot(Agent):
    def select_move(self, game: Game) -> Move:
        board = game.board
        size = board.size
        candidates = []
        for i, j in itertools.product(range(1, size + 1), range(1, size + 1)):
            candidate = Move.play(Point(i, j))
            if game.is_valid_move(candidate) and not board.is_point_an_eye(
                candidate.point, game.next_color
            ):
                candidates.append(candidate)

        if not candidates:
            return Move.pass_()

        return random.choice(candidates)
