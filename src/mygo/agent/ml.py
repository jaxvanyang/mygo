import torch
from torch.nn import Module

from mygo.agent.base import Agent
from mygo.encoder.base import Encoder
from mygo.game.types import Game, Move


class MLBot(Agent):
    """Bot which uses a simple machine learning model."""

    def __init__(self, model: Module, encoder: Encoder) -> None:
        super().__init__()
        self.model = model
        self.encoder = encoder

    def select_move(self, game: Game) -> Move:
        x = torch.from_numpy(self.encoder.encode(game))
        pred = self.model(x.unsqueeze(0))[0]

        # TODO: sample moves
        ranked_indices = pred.argsort(descending=True)

        for idx in ranked_indices:
            move = Move.play(self.encoder.decode_point_index(idx))
            if game.is_valid_move(move):
                return move

        return Move.pass_()
