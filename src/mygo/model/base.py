from torch import nn


class Model(nn.Module):
    name: str

    @classmethod
    def get_model(cls, name: str):
        for subclass in cls.__subclasses__():
            if subclass.name == name:
                return subclass
        raise ValueError(f"Unknown model: {name}")

    def print_info(self):
        device = next(self.parameters()).device
        n_params = sum(p.numel() for p in self.parameters())

        print(
            "\n".join(
                [
                    f"Device: {device}",
                    f"Parameters: {n_params:,d}",
                    f"Structure:\n{self}",
                ]
            )
        )
