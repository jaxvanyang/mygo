from torch import nn

from .base import Model


class SmallModel(Model):
    """Small model defined in the book."""

    name = "small"

    def __init__(self, board_size: int = 19, plane_count: int = 1) -> None:
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(plane_count, 48, 7, padding="same"),
            nn.ReLU(),
            nn.Conv2d(48, 32, 5, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 32, 5, padding="same"),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(32 * board_size**2, 512),
            nn.ReLU(),
            nn.Linear(512, board_size**2),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.linear_stack(x)

        return x
