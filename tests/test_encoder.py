import numpy as np
import pytest

from mygo.encoder import ZeroEncoder
from mygo.game import Game, PlayMove, Point


class TestZeroEncoder:

    @pytest.fixture
    def encoder(self):
        return ZeroEncoder()

    def test_encode(self, encoder):
        game = Game.new()
        prev_code = encoder.encode(game)

        assert (prev_code[:-1] == 0).all()
        assert (prev_code[-1] == 1).all()

        for i in range(game.board_size):
            game.apply_move(PlayMove(game.next_player, Point(i, i)))
            code = encoder.encode(game)

            move_code = np.zeros(encoder.shape[1:])
            move_code[i, i] = 1

            assert (code[0] - code[2] == 0).all()
            assert (code[1] - code[3] == move_code).all()
            assert (code[2:-1:2] == prev_code[1:-3:2]).all()
            assert (code[3:-1:2] == prev_code[0:-3:2]).all()
            assert (code[-1] == (1 - prev_code[-1])).all()

            prev_code = code
