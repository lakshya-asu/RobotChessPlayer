from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .vision_sim import PiecePose


@dataclass
class BoardTracker:
    """
    Tracks the mapping between a FEN state and perceived piece positions.

    FEN reconstruction from perception is not implemented yet; this class
    currently stores a FEN string and the latest observed piece poses.
    """

    fen: str
    piece_poses: Dict[str, PiecePose] = field(default_factory=dict)

    def reset(self, fen: str) -> None:
        """
        Reset tracker to a new FEN and clear stored piece poses.
        """
        self.fen = fen
        self.piece_poses.clear()

    def update_from_perception(self, observed: List[PiecePose]) -> None:
        """
        Replace stored piece poses with a new set of observations.
        """
        self.piece_poses = {p.square: p for p in observed}

    def infer_fen_from_poses(self) -> str:
        """
        Return a FEN string inferred from current piece poses.

        This is a placeholder for a full reconstruction algorithm and
        currently returns the stored FEN unchanged.
        """
        # todo: reconstruct FEN from piece_poses once piece type / color information is available
        return self.fen

    def update_fen_from_perception(self) -> None:
        """
        Update the stored FEN string using current piece poses.
        """
        self.fen = self.infer_fen_from_poses()
