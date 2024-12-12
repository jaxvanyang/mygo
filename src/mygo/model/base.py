import numpy as np
import torch
from torch import nn


class Model(nn.Module):
    name: str

    @classmethod
    def get_model(cls, name: str):
        for subclass in cls.__subclasses__():
            if subclass.name == name:
                return subclass
        raise ValueError(f"Unknown model: {name}")

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def info(self):
        n_params = sum(p.numel() for p in self.parameters())
        return "\n".join(
            [
                f"Device: {self.device}",
                f"Parameters: {n_params:,d}",
                f"Structure:\n{self}",
            ]
        )

    def print_info(self):
        n_params = sum(p.numel() for p in self.parameters())

        print(
            "\n".join(
                [
                    f"Device: {self.device}",
                    f"Parameters: {n_params:,d}",
                    f"Structure:\n{self}",
                ]
            )
        )

    def transform(self, x):
        """Return a feedable version to the model of x."""

        device = self.device
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        elif isinstance(x, (int, tuple, list)):
            return torch.tensor(x, device=device)
        elif isinstance(x, torch.Tensor):
            return x.to(device)

        raise TypeError(f"Unknown type: {type(x)}")
