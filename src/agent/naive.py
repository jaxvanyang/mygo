"""Naive bots without machine learning."""

import math
import random
from copy import deepcopy

from agent.base import Agent
from go.game import Game
from go.types import Color, Move


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


class MCTSNode:
    """Monte Carlo tree node."""

    def __init__(self, game: Game, parent=None) -> None:
        self.game = game
        self.parent = parent
        self.children = []
        self.black_wins = 0
        self.white_wins = 0
        good_moves = game.good_moves
        self.unvisited_moves = good_moves if good_moves else [Move.pass_()]

    def __repr__(self) -> str:
        return f"MCTSNode({self.game!r}, {self.parent!r})"

    @property
    def count(self) -> int:
        return self.black_wins + self.white_wins

    @property
    def win_rate(self) -> float:
        assert self.count >= 0.0
        if self.game.next_color == Color.black:
            return self.white_wins / self.count
        else:
            return self.black_wins / self.count

    def uct_score(self, child, total_count: int, temp: float) -> float:
        """Upper Confidence bound for Trees formula."""
        w = child.win_rate
        n = child.count

        return w + temp * math.sqrt(math.log2(total_count) / n)

    def select_child(self, temp: float):
        if self.unvisited_moves:
            game = deepcopy(self.game)
            game.apply_move(self.unvisited_moves.pop())
            child = MCTSNode(game, self)
            self.children.append(child)
            return child

        total_count = sum(child.count for child in self.children)
        best_score = -1.0
        best_child = None

        for child in self.children:
            score = self.uct_score(child, total_count, temp)
            if score > best_score:
                best_score = score
                best_child = child

        assert isinstance(best_child, MCTSNode)
        return best_child

    def update(self, winner: Color) -> None:
        """Update winner counts."""
        if winner == Color.black:
            self.black_wins += 1
        elif winner == Color.white:
            self.white_wins += 1

    def simulate(self, temp: float) -> None:
        """Simulate a random game."""

        if self.game.is_over:
            winner = self.game.winner
            node = self
            while node is not None:
                node.update(winner)
                node = node.parent

            return

        child = self.select_child(temp)
        child.simulate(temp)

    def select_move(self, resign_rate: float = -1.0) -> Move:
        """
        Select best move.

        Return resign move if win rate is lower than resign_rate.
        """
        best_rate, best_move = resign_rate, Move.resign()

        for child in self.children:
            win_rate = child.win_rate
            if win_rate > best_rate:
                best_rate, best_move = win_rate, child.game.move

        return best_move


class MCTSBot(Agent):
    """Monte Carlo tree search bot."""

    def __init__(
        self, rounds: int = 100, temp: float = 1.5, resign_rate: float = 0.1
    ) -> None:
        super().__init__()
        self.rounds = rounds
        self.temp = temp
        self.resign_rate = 0.1

    def select_move(self, game: Game) -> Move:
        root = MCTSNode(game)

        for _ in range(self.rounds):
            root.simulate(self.temp)

        return root.select_move(self.resign_rate)
