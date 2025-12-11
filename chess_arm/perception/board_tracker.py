from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .vision_sim import PiecePose


@dataclass
class BoardTracker:
    """Tracks the mapping between FEN state and perceived piece positions."""
    fen: str
    piece_poses: Dict[str, PiecePose]

    def update_from_perception(self, observed: List[PiecePose]) -> None:
        self.piece_poses = {p.square: p for p in observed}
