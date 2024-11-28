import numpy as np
import torch
from torch.nn import Module

from mygo.agent.base import Agent
from mygo.encoder.base import Encoder
from mygo.game.basic import Player
from mygo.game.game import Game
from mygo.game.move import Move, PassMove, PlayMove


class MLBot(Agent):
    """Bot which uses a simple machine learning model."""

    def __init__(self, model: Module, encoder: Encoder) -> None:
        super().__init__(f"{model.__class__.__name__} Bot")
        self.model = model
        self.encoder = encoder

    @torch.no_grad()
    def select_move(self, game: Game, player: Player | None = None) -> Move:
        self.model.eval()

        if player is None:
            player = game.next_player

        device = self.model.parameters().__next__().device
        x = torch.from_numpy(self.encoder.encode(game)).to(device)
        pred = self.model(x.unsqueeze(0))[0]

        # sample moves, only choose half for efficiency
        # TODO: use multinomial sampling
        eps = 1e-6
        pred = (pred**3).clamp(eps, 1 - eps)
        pred = pred / pred.sum()
        ranked_indices = np.random.choice(
            np.arange(pred.numel()),
            size=pred.numel() // 2,
            replace=False,
            p=pred.cpu().numpy(),
        )

        for idx in ranked_indices:
            move = PlayMove(player, self.encoder.decode_point(idx))
            if game.is_valid_move(move):
                return move

        return PassMove(player)
