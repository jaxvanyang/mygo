import numpy as np
import torch
from torch.nn import Module

from mygo.agent.base import Agent
from mygo.encoder.base import Encoder
from mygo.game.game import Game
from mygo.game.move import Move, PassMove, PlayMove


class MLBot(Agent):
    """Bot which uses a simple machine learning model."""

    def __init__(self, model: Module, encoder: Encoder) -> None:
        super().__init__(f"{model.__class__.__name__} Bot")
        self.model = model
        self.encoder = encoder

    def select_move(self, game: Game) -> Move:
        x = torch.from_numpy(self.encoder.encode(game))
        pred = self.model(x.unsqueeze(0))[0]

        # sample moves, only choose half
        eps = 1e-6
        pred = (pred**3).clamp(eps, 1 - eps)
        pred = pred / pred.sum()
        ranked_indices = np.random.choice(
            np.arange(pred.numel()),
            size=pred.numel() // 2,
            replace=False,
            p=pred.detach().numpy(),
        )

        for idx in ranked_indices:
            move = PlayMove(game.next_player, self.encoder.decode_point(idx))
            if game.is_valid_move(move):
                return move

        return PassMove(game.next_player)
