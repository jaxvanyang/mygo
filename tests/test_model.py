import pytest
import torch

from mygo.encoder import OnePlaneEncoder
from mygo.model import SmallModel, TinyModel, ZeroModel


@pytest.fixture
def encoder():
    return OnePlaneEncoder()


@pytest.fixture
def x(encoder):
    return torch.randn(1, encoder.plane_count, encoder.size, encoder.size)


class TestTinyModel:
    def test_eval(self, encoder, x):
        model = TinyModel(encoder.size, encoder.plane_count)
        y = model(x)

        assert tuple(y.shape) == (1, encoder.size**2)


class TestSmallModel:

    def test_eval(self, encoder, x):
        model = SmallModel(encoder.size, encoder.plane_count)
        y = model(x)

        assert tuple(y.shape) == (1, encoder.size**2)


class TestZeroModel:

    def test_eval(self, encoder, x):
        model = ZeroModel(encoder.plane_count, board_size=encoder.size)
        p, v = model(x)

        assert tuple(p.shape) == (1, encoder.size**2 + 1)
        assert tuple(v.shape) == (1, 1)
