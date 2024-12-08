import time
from copy import deepcopy

import torch
import torch.nn.functional as F

from mygo.encoder import Encoder, ZeroEncoder
from mygo.game import Game, Move, Player, ResignMove
from mygo.helper.log import logger
from mygo.model import ZeroModel

from .base import Agent


class Action:
    """Store action stats."""

    def __init__(self, prob: float) -> None:
        self.prob = prob  # P(s,a)
        self.visit_count = 0  # N(s,a)
        self.total_value = 0  # W(s,a)

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count  # Q(s,a)


class Node:

    @torch.no_grad()
    def __init__(
        self, game: Game, encoder: Encoder, model: ZeroModel, parent=None
    ) -> None:
        self.game = game
        self.parent = parent
        self.edges = {}
        self.children = {}

        model.eval()
        x = encoder.encode(game)
        x = model.transform(x).unsqueeze(0)
        # TODO: use symmetry eval
        p, v = model(x)
        probs = F.softmax(p[0], 0).tolist()
        self.value = v.item()

        for i, p in enumerate(probs):
            move = encoder.decode_move_index(i, game.next_player)
            if not game.is_valid_move(move):
                continue
            self.edges[move] = Action(p)

    @property
    def visit_count(self):
        return sum(e.visit_count for e in self.edges.values())

    @property
    def last_move(self):
        return self.game.last_move

    def has_child(self, move: Move) -> bool:
        return move in self.children

    def add_child(self, move: Move, encoder: Encoder, model: ZeroModel) -> "Node":
        """Add a child node of the move.

        Return:
            The new added child node.
        """
        assert move not in self.children

        game = deepcopy(self.game)
        game.apply_move(move)
        node = Node(game, encoder, model, self)
        self.children[move] = node

        return node

    def select_node(self, encoder: Encoder, model: ZeroModel, temp: float) -> "Node":
        """Select and create a leaf node then return it."""

        if not self.edges:  # = game is over
            # TBD: the value may be updated to the game result
            return self

        move, action = self.select_edge(temp)
        if not self.has_child(move):
            return self.add_child(move, encoder, model)

        return self.children[move].select_node(encoder, model, temp)

    def select_edge(self, temp: float) -> tuple[Move, Action]:
        return max(self.edges.items(), key=lambda edge: self.action_score(edge, temp))

    def action_score(self, edge: tuple[Move, Action], temp: float) -> float:
        """Return the score of the edge's action.

        Score = Q(s,a) + U(s,a)
        """

        action = edge[1]
        u = temp * action.prob * (self.visit_count**0.5) / (1 + action.visit_count)

        return action.value + u

    def update(self) -> None:
        """Update ancestors' values and visit counts."""

        move = self.last_move
        value = -self.value
        node = self.parent

        while node is not None:
            node.edges[move].visit_count += 1
            node.edges[move].total_value += value

            move = node.last_move
            value = -value
            node = node.parent

    def select_move(self, resign_threshold: float) -> Move:
        move, action = max(self.edges.items(), key=lambda edge: edge[1].visit_count)
        win_rate = Agent.value_to_rate(action.value)
        logger.info(f"best move: {move}, winning rate: {win_rate:.1%}")
        logger.debug(f"visit count: {action.visit_count:,d}")
        if self.value < resign_threshold and action.value < resign_threshold:
            return ResignMove(self.game.next_player)

        return move


class ZeroAgent(Agent):

    def __init__(
        self,
        encoder: Encoder | None = None,
        model: ZeroModel | None = None,
        rounds: int = 1600,
        time: float = 0.0,
        temp: float = 1.5,
        resign_rate: float = 0.1,
    ) -> None:
        super().__init__("MyGo Zero")

        self.encoder = encoder or ZeroEncoder()
        assert isinstance(encoder, Encoder)
        self.model = model or ZeroModel(encoder.plane_count, board_size=encoder.size)
        self.root = None
        self.rounds = rounds
        self.time = time
        self.temp = temp
        self.resign_rate = resign_rate
        self.resign_threshold = self.rate_to_value(resign_rate)

    def select_move(self, game: Game, player: Player | None = None) -> Move:
        game = deepcopy(game)
        if player is not None:
            game.next_player = player

        # TODO: retain selected subtree
        root = Node(game, self.encoder, self.model)

        t0 = time.perf_counter()
        for i in range(self.rounds):
            node = root.select_node(self.encoder, self.model, self.temp)
            node.update()

            if self.time > 0 and (dt := time.perf_counter() - t0) >= self.time:
                logger.debug(
                    f"[{i}/{self.rounds}] computation time exceeded: {dt:.3f}s"
                )
                break

        return root.select_move(self.resign_threshold)
