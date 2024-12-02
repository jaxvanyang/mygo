from torch import nn


class Model(nn.Module):
    name: str

    @classmethod
    def get_model(cls, name: str):
        for subclass in cls.__subclasses__():
            if subclass.name == name:
                return subclass
        raise ValueError(f"Unknown model: {name}")
