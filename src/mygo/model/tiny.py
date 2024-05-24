from torch import nn
from torch.nn import Module


class TinyModel(Module):
    """Tiny model for experiment."""

    def __init__(self, board_size: int = 19):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 48, 3, padding="same"),
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv2d(48, 48, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(48 * (board_size // 2) ** 2, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, board_size**2),
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        x = self.linear_stack(x)

        return x
