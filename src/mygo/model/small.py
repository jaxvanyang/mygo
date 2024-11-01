from torch import nn
from torch.nn import Module


class SmallModel(Module):
    """Small model defined in the book."""

    def __init__(self, board_size: int = 19, plane_count: int = 1) -> None:
        assert board_size >= 9
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
