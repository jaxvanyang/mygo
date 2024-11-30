import os

import pytest

# fail fast to save time
pytestmark = pytest.mark.skipif(
    bool(os.environ.get("CI")), reason="need to many resources in CI"
)

from mygo.dataset import KGSDataset, MCTSDataset  # noqa: E402
from mygo.model import SmallModel, TinyModel  # noqa: E402
from mygo.tool import ModelTrainer  # noqa: E402

data_root = "data/raw"


class TestMCTSDataset:
    @staticmethod
    def _transform(x):
        return ModelTrainer.transform(x).argmax(1)

    @pytest.fixture
    @staticmethod
    def train_data():
        return MCTSDataset(
            data_root,
            train=True,
            download=False,
            transform=ModelTrainer.transform,
            target_transform=TestMCTSDataset._transform,
        )

    @pytest.fixture
    @staticmethod
    def test_data():
        return MCTSDataset(
            data_root,
            train=False,
            download=False,
            transform=ModelTrainer.transform,
            target_transform=TestMCTSDataset._transform,
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
        )

        trainer.train()


class TestKGSDataset:
    @pytest.fixture
    @staticmethod
    def train_data():
        return KGSDataset(
            data_root,
            train=True,
            download=False,
            game_count=2,
            transform=ModelTrainer.transform,
            target_transform=ModelTrainer.transform,
        )

    @pytest.fixture
    @staticmethod
    def test_data():
        return KGSDataset(
            data_root,
            train=False,
            download=False,
            game_count=1,
            transform=ModelTrainer.transform,
            target_transform=ModelTrainer.transform,
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
        )

        trainer.train()
