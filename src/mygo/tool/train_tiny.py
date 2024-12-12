import sys
import time
from pathlib import Path

import numpy as np
import torch

from mygo.dataset import KGSDataset
from mygo.model.tiny import TinyModel
from mygo.tool import ModelTrainer

torch.set_float32_matmul_precision("high")


def transform(data):
    if isinstance(data, torch.Tensor):
        return data.to(ModelTrainer.default_device())
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(ModelTrainer.default_device())
    else:
        raise TypeError(f"Unknown type: {type(data)}")


board_size = 19
root = Path("data")
data_root = root / "raw"
f_train_data = root / "kgs_train.pt"
f_test_data = root / "kgs_test.pt"

t0 = time.perf_counter()
if f_train_data.is_file():
    train_data = torch.load(f_train_data, weights_only=False)
else:
    train_data = KGSDataset(
        root=data_root,
        train=True,
        download=False,
        game_count=1000,
        transform=transform,
        target_transform=transform,
    )
    torch.save(train_data, f_train_data)
    print(f"Save train data at: {f_train_data}")
    sys.stdout.flush()

if f_test_data.is_file():
    test_data = torch.load(f_test_data, weights_only=False)
else:
    test_data = KGSDataset(
        root=data_root,
        train=False,
        download=False,
        game_count=100,
        transform=transform,
        target_transform=transform,
    )
    torch.save(test_data, f_test_data)
    print(f"Save test data at: {f_test_data}")
    sys.stdout.flush()

t1 = time.perf_counter()
dt = t1 - t0
print(f"Load data time: {ModelTrainer.pretty_time(dt)}")

model = TinyModel(board_size=board_size)
model_opt = torch.compile(model, mode="reduce-overhead")

trainer = ModelTrainer(
    board_size=board_size,
    model=model_opt,
    train_data=train_data,
    test_data=test_data,
    seed=42,
    plot=True,
    resume_from_checkpoint=True,
    always_save_checkpoint=False,
    max_iters=100,
    log_interval=0,
    eval_interval=500,
)

trainer.train()
