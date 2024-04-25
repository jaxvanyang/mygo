"""Naive bots without machine learning."""

import math
import random

from agent.base import Agent
from go.game import Game
from go.types import Move


class RandomBot(Agent):
    def select_move(self, game: Game) -> Move:
        candidates = game.good_moves

        if not candidates:
            return Move.pass_()

        return random.choice(candidates)


class TreeSearchBot(Agent):
    """Search according to minimax algorithm."""

    def __init__(self, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth

    @classmethod
    def calc_move_score(
        cls,
        game: Game,
        move: Move,
        depth: int = 3,
        alpha: float = -math.inf,
        beta: float = math.inf,
    ) -> float:
        """
        Calculate the score of the move. Use alpha-beta pruning.

        The input game won't be changed. The input move must be a valid move.

        alpha records the best we can make so far, bigger is better (for ourselves).
        beta records the best the opponent can make so far, smaller is better (for the
        opponent).
        """

        assert depth >= 0

        player = game.next_color
        with game.apply_move_ctx(move) as game:

            if game.is_over:
                return math.inf if game.winner == player else -math.inf

            if depth == 0:
                return game.score

            for op_move in game.good_moves:
                # our score is negative to opponent's
                score = -cls.calc_move_score(game, op_move, depth - 1, -beta, -alpha)
                beta = min(beta, score)  # update opponent's best score
                if alpha >= beta:
                    # in this branch, the opponent can make a score smaller than the
                    # best we can make in other branch, so we can stop here
                    break

        # the final score is made by the opponent, alpha is just for pruning
        return beta

    def select_move(self, game: Game) -> Move:
        best_move_score = -math.inf
        best_move = Move.pass_()

        for move in game.good_moves:
            move_score = self.calc_move_score(game, move, self.depth)

            if move_score > best_move_score:
                best_move_score = move_score
                best_move = move

        return best_move
