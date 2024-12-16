import os

import pytest

# fail fast to save time
pytestmark = pytest.mark.skipif(
    bool(os.environ.get("CI")), reason="need to many resources in CI"
)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from mygo.dataset import KGSDataset, MCTSDataset  # noqa: E402
from mygo.model import SmallModel, TinyModel  # noqa: E402
from mygo.tool import ModelTrainer  # noqa: E402

data_root = "data/raw"
# only download when demand to save time
download = bool(os.environ.get("download"))


def transform(data):
    device = ModelTrainer.default_device()
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, (int, tuple, list)):
        return torch.tensor(data, device=device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        raise TypeError(f"Unknown type: {type(data)}")


class TestMCTSDataset:
    @pytest.fixture
    @staticmethod
    def train_data():
        return MCTSDataset(
            data_root,
            train=True,
            download=download,
            transform=transform,
            target_transform=transform,
        )

    @pytest.fixture
    @staticmethod
    def test_data():
        return MCTSDataset(
            data_root,
            train=False,
            download=download,
            transform=transform,
            target_transform=transform,
        )

    def test_train_tiny(self, tmp_path, train_data, test_data):
        model = TinyModel(9)

        trainer = ModelTrainer(
            board_size=9,
            model=model,
            root=tmp_path,
            max_iters=10,
            eval_interval=5,
            eval_iters=2,
            train_data=train_data,
            test_data=test_data,
            plot=True,
        )

        trainer.train()

    def test_train_small(self, tmp_path, train_data, test_data):
        model = SmallModel(9)

        trainer = ModelTrainer(
            board_size=9,
            model=model,
            root=tmp_path,
            max_iters=10,
            eval_interval=5,
            eval_iters=2,
            train_data=train_data,
            test_data=test_data,
            plot=True,
        )

        trainer.train()


class TestKGSDataset:
    @pytest.fixture
    @staticmethod
    def train_data():
        return KGSDataset(
            data_root,
            train=True,
            download=download,
            game_count=2,
            transform=transform,
            target_transform=transform,
        )

    @pytest.fixture
    @staticmethod
    def test_data():
        return KGSDataset(
            data_root,
            train=False,
            download=download,
            game_count=1,
            transform=transform,
            target_transform=transform,
        )

    def test_train_tiny(self, tmp_path, train_data, test_data):
        model = TinyModel(19)

        trainer = ModelTrainer(
            board_size=19,
            model=model,
            root=tmp_path,
            max_iters=10,
            eval_interval=5,
            eval_iters=2,
            train_data=train_data,
            test_data=test_data,
            plot=True,
        )

        trainer.train()

    def test_train_small(self, tmp_path, train_data, test_data):
        model = SmallModel(19)

        trainer = ModelTrainer(
            board_size=19,
            model=model,
            root=tmp_path,
            max_iters=10,
            eval_interval=5,
            eval_iters=2,
            train_data=train_data,
            test_data=test_data,
            plot=True,
        )

        trainer.train()
