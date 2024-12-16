"""Experience utils for reinforcement learning."""

import numpy as np
from torch.utils.data import Dataset


class ExperienceBuffer:
    def __init__(
        self,
        states: list | None = None,
        actions: list | None = None,
        rewards: list | None = None,
    ):
        self.states = states or []
        self.actions = actions or []
        self.rewards = rewards or []
        assert len(self.states) == len(self.actions) == len(self.rewards)

        self._current_states = []
        self._current_actions = []

    def __len__(self):
        n_states = len(self.states)
        assert n_states == len(self.actions) == len(self.rewards)
        return n_states

    def __add__(self, other):
        out = self.__class__()
        out.states += self.states + other.states
        out.actions += self.actions + other.actions
        out.rewards += self.rewards + other.rewards

        return out

    def __repr__(self):
        return f"ExperienceBuffer({repr(self.states)}, {repr(self.actions)}, {repr(self.rewards)})"  # noqa: E501

    def begin(self):
        self._current_states.clear()
        self._current_actions.clear()

    def record(self, state, action):
        self._current_states.append(state)
        self._current_actions.append(action)

    def complete(self, reward):
        self.states += self._current_states
        self.actions += self._current_actions
        self.rewards += [reward for _ in range(len(self._current_states))]

        self.begin()


class ZeroExpDataset(Dataset):
    """Experience dataset for AlphaGo Zero model."""

    def __init__(
        self,
        exp_buffer: ExperienceBuffer,
        dtype=None,
        symmetries=False,
        transform=None,
        target_transform=None,
    ):
        self.transform = transform
        self.target_transform = target_transform

        self.features = np.array(exp_buffer.states, dtype=dtype)
        self.actions = np.array(exp_buffer.actions, dtype=dtype)
        self.values = np.array(exp_buffer.rewards, dtype=dtype).reshape(-1, 1)

        if symmetries:
            pass

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        action = self.actions[idx]
        value = self.values[idx]

        if self.transform is not None:
            feature = self.transform(feature)
        if self.target_transform is not None:
            action = self.target_transform(action)
            value = self.target_transform(value)

        return feature, (action, value)
