from torch import nn

from .base import Model


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding="same")
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv_block = ConvBlock(channels, channels)
        self.conv_layer = nn.Conv2d(channels, channels, 3, padding="same")
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.conv_block(x)
        y = self.conv_layer(y)
        y = self.bn(y)
        y = x + y
        y = self.relu(y)

        return y


class PolicyHead(nn.Module):
    def __init__(self, in_channels: int, board_size: int = 19):
        super().__init__()

        n_vertices = board_size**2

        self.conv = nn.Conv2d(in_channels, 2, 1, padding="same")
        self.bn = nn.BatchNorm2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(2 * n_vertices, n_vertices + 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)

        return x


class ValueHead(nn.Module):
    def __init__(self, in_channels: int, n_hidden: int = 256, board_size: int = 19):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 1, 1, padding="same")
        self.bn = nn.BatchNorm2d(1)
        self.relu1 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(board_size**2, n_hidden)
        self.relu2 = nn.ReLU()
        self.linear2 = nn.Linear(n_hidden, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        x = self.tanh(x)

        return x


# input shape: (B, C, 19, 19)
class ZeroModel(Model):

    name = "zero"

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        board_size: int = 19,
        variant: str = "std",
    ):
        assert variant in ("std", "large")
        super().__init__()

        n_res_blocks = 19 if variant == "std" else 39

        self.conv_block = ConvBlock(in_channels, out_channels)
        self.res_blocks = nn.Sequential(
            *[ResBlock(out_channels) for _ in range(n_res_blocks)]
        )
        self.policy_head = PolicyHead(out_channels, board_size)
        self.value_head = ValueHead(out_channels, board_size=board_size)

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)
        p = self.policy_head(x)
        v = self.value_head(x)

        return p, v
