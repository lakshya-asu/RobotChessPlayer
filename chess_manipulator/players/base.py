"""Base protocol for chess move providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class PlayerError(RuntimeError):
    """Raised when a player adapter cannot provide a valid move."""


@dataclass(frozen=True)
class PlayerIdentity:
    """Display metadata for a player adapter."""

    name: str
    backend: str


class PlayerAdapter(Protocol):
    """Minimal FEN-in/UCI-out abstraction for turn selection."""

    @property
    def identity(self) -> PlayerIdentity:
        """Return a stable identifier for logs and summaries."""

    def select_move(self, fen: str) -> str:
        """Return a legal move in UCI notation for the supplied FEN."""
