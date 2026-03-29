"""Bounded replay buffer for DQN training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List
import random


@dataclass
class Transition:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool
    legal_next_actions: list[int]


class ReplayBuffer:
    """A simple bounded replay buffer."""

    def __init__(self, capacity: int = 10000) -> None:
        self._items: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(list(self._items), batch_size)
