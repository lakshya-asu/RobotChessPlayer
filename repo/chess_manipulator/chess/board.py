"""Board-state management built around python-chess when available."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import chess
except ImportError:  # pragma: no cover - exercised in environments without python-chess
    chess = None


class BoardError(RuntimeError):
    """Raised when the chess backend is unavailable or a move is invalid."""


@dataclass
class MoveDescription:
    """Normalized details for a single UCI move."""

    uci: str
    source: str
    target: str
    promotion: str = ""
    is_capture: bool = False
    is_castling: bool = False


class ChessBoardManager:
    """Thin wrapper around python-chess for ROS-side game state."""

    def __init__(self, fen: Optional[str] = None) -> None:
        if chess is None:
            raise BoardError(
                "python-chess is required for board management. Install the 'chess' package."
            )
        self._board = chess.Board(fen) if fen else chess.Board()

    @property
    def fen(self) -> str:
        return self._board.fen()

    def set_fen(self, fen: str) -> None:
        self._board.set_fen(fen)

    def legal_move(self, uci_move: str) -> bool:
        move = chess.Move.from_uci(uci_move)
        return move in self._board.legal_moves

    def describe_move(self, uci_move: str) -> MoveDescription:
        move = chess.Move.from_uci(uci_move)
        if move not in self._board.legal_moves:
            raise BoardError(f"Illegal move: {uci_move}")
        return MoveDescription(
            uci=uci_move,
            source=chess.square_name(move.from_square),
            target=chess.square_name(move.to_square),
            promotion=chess.piece_symbol(move.promotion) if move.promotion else "",
            is_capture=self._board.is_capture(move),
            is_castling=self._board.is_castling(move),
        )

    def push_uci(self, uci_move: str) -> MoveDescription:
        description = self.describe_move(uci_move)
        self._board.push(chess.Move.from_uci(uci_move))
        return description

    def san_for_uci(self, uci_move: str) -> str:
        move = chess.Move.from_uci(uci_move)
        if move not in self._board.legal_moves:
            raise BoardError(f"Illegal move: {uci_move}")
        return self._board.san(move)

    def captured_piece_count(self) -> int:
        return 32 - len(self._board.piece_map())
