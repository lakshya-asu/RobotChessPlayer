"""Fixed move-index mapping for DQN-friendly chess actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import chess

PROMOTIONS = (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PROMOTION_TO_OFFSET = {
    None: 0,
    chess.QUEEN: 1,
    chess.ROOK: 2,
    chess.BISHOP: 3,
    chess.KNIGHT: 4,
}
OFFSET_TO_PROMOTION = {offset: promotion for promotion, offset in PROMOTION_TO_OFFSET.items()}
ACTION_COUNT = 64 * 64 * len(PROMOTIONS)


@dataclass(frozen=True)
class DecodedAction:
    """Decoded move metadata for a fixed action index."""

    index: int
    move: chess.Move
    is_legal: bool


class MoveActionSpace:
    """Maps between chess moves and a bounded discrete action space."""

    size = ACTION_COUNT

    def encode(self, move: chess.Move) -> int:
        promotion = move.promotion
        if promotion not in PROMOTION_TO_OFFSET:
            raise ValueError(f"Unsupported promotion piece: {promotion}")
        return ((move.from_square * 64) + move.to_square) * len(PROMOTIONS) + PROMOTION_TO_OFFSET[promotion]

    def decode(self, index: int, board: Optional[chess.Board] = None) -> DecodedAction:
        if index < 0 or index >= self.size:
            raise ValueError(f"Action index out of range: {index}")
        pair_index, promotion_offset = divmod(index, len(PROMOTIONS))
        from_square, to_square = divmod(pair_index, 64)
        move = chess.Move(
            from_square=from_square,
            to_square=to_square,
            promotion=OFFSET_TO_PROMOTION[promotion_offset],
        )
        is_legal = board is not None and move in board.legal_moves
        return DecodedAction(index=index, move=move, is_legal=is_legal)

    def legal_indices(self, board: chess.Board) -> List[int]:
        return [self.encode(move) for move in board.legal_moves]

    def legal_mask(self, board: chess.Board) -> List[bool]:
        mask = [False] * self.size
        for index in self.legal_indices(board):
            mask[index] = True
        return mask

    def legal_moves(self, board: chess.Board) -> Iterable[chess.Move]:
        return iter(board.legal_moves)
