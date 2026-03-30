"""Minimal PyTorch DQN network for chess move selection."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - handled explicitly by callers
    torch = None
    nn = None

from .action_space import ACTION_COUNT
from .encoding import feature_size


def require_torch():
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for RL training/inference. Install the 'torch' package.")
    return torch, nn


_BaseModule = nn.Module if nn is not None else object


class BoundedDQN(_BaseModule):  # type: ignore[misc]
    """Small fully-connected Q-network over a fixed move action space."""

    def __init__(self, hidden_size: int = 512, hidden_size_2: int = 256) -> None:
        _, nn_mod = require_torch()
        super().__init__()
        self.network = nn_mod.Sequential(
            nn_mod.Linear(feature_size(), hidden_size),
            nn_mod.ReLU(),
            nn_mod.Linear(hidden_size, hidden_size_2),
            nn_mod.ReLU(),
            nn_mod.Linear(hidden_size_2, ACTION_COUNT),
        )

    def forward(self, inputs):
        return self.network(inputs)


def build_model(hidden_size: int = 512, hidden_size_2: int = 256) -> BoundedDQN:
    require_torch()
    return BoundedDQN(hidden_size=hidden_size, hidden_size_2=hidden_size_2)
