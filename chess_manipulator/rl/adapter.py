"""Runtime adapter that selects legal UCI moves from FEN strings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import chess

from .action_space import MoveActionSpace
from .encoding import encode_board
from .heuristics import best_heuristic_move
from .model import build_model, require_torch, torch


@dataclass
class AdapterResult:
    """Move-selection result for runtime consumers."""

    uci_move: str
    source: str


class RLMoveAdapter:
    """Loads an optional DQN checkpoint and always returns a legal move."""

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cpu") -> None:
        self.action_space = MoveActionSpace()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None
        self._load_checkpoint()

    def select_move(self, fen: str) -> AdapterResult:
        board = chess.Board(fen)
        if board.is_game_over():
            raise ValueError("Cannot select a move for a finished position.")
        if self.model is None:
            move = best_heuristic_move(board)
            return AdapterResult(uci_move=move.uci(), source="heuristic_fallback")

        move = self._select_model_move(board)
        if move is None:
            move = best_heuristic_move(board)
            return AdapterResult(uci_move=move.uci(), source="heuristic_fallback")
        return AdapterResult(uci_move=move.uci(), source="dqn_checkpoint")

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_path:
            return
        checkpoint = Path(self.checkpoint_path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        require_torch()
        self.model = build_model()
        payload = torch.load(checkpoint, map_location=self.device)
        state_dict = payload["model_state_dict"] if isinstance(payload, dict) else payload
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _select_model_move(self, board: chess.Board) -> Optional[chess.Move]:
        if self.model is None:
            return None
        require_torch()
        legal_indices = self.action_space.legal_indices(board)
        if not legal_indices:
            return None
        with torch.inference_mode():
            inputs = torch.tensor([encode_board(board)], dtype=torch.float32, device=self.device)
            q_values = self.model(inputs)[0]
            legal_scores = [(float(q_values[index].item()), index) for index in legal_indices]
        _, best_index = max(legal_scores, key=lambda item: (item[0], -item[1]))
        decoded = self.action_space.decode(best_index, board=board)
        return decoded.move if decoded.is_legal else None
