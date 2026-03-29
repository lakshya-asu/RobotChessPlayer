"""Bounded self-play environment for lightweight DQN training."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Optional

import chess

from .action_space import MoveActionSpace
from .encoding import encode_board
from .heuristics import best_heuristic_move, material_balance


@dataclass
class StepResult:
    state: list[float]
    reward: float
    done: bool
    info: dict


class ChessMoveEnv:
    """Single-agent environment with a bounded horizon and simple opponent policy."""

    def __init__(
        self,
        max_halfmoves: int = 80,
        opponent: str = "random",
        seed: Optional[int] = None,
    ) -> None:
        self.max_halfmoves = max_halfmoves
        self.opponent = opponent
        self.random = random.Random(seed)
        self.action_space = MoveActionSpace()
        self.board = chess.Board()
        self.agent_color = chess.WHITE
        self.steps = 0

    def reset(self, fen: Optional[str] = None) -> list[float]:
        self.board = chess.Board(fen) if fen else chess.Board()
        self.agent_color = self.board.turn
        self.steps = 0
        return encode_board(self.board)

    def step(self, action_index: int) -> StepResult:
        if self.board.is_game_over():
            return StepResult(encode_board(self.board), 0.0, True, {"terminal": True})

        decoded = self.action_space.decode(action_index, board=self.board)
        if not decoded.is_legal:
            return StepResult(
                encode_board(self.board),
                -5.0,
                True,
                {"illegal_action": True, "action_index": action_index},
            )

        reward = self._apply_move(decoded.move, self.agent_color)
        self.steps += 1
        done = self._done()

        if not done and self.board.turn != self.agent_color:
            opponent_move = self._opponent_move()
            reward -= self._apply_move(opponent_move, not self.agent_color)
            self.steps += 1
            done = self._done()

        return StepResult(encode_board(self.board), reward, done, {"fen": self.board.fen()})

    def legal_action_indices(self) -> list[int]:
        return self.action_space.legal_indices(self.board)

    def _apply_move(self, move: chess.Move, mover: chess.Color) -> float:
        before_balance = material_balance(self.board, mover)
        self.board.push(move)
        reward = material_balance(self.board, mover) - before_balance
        if self.board.is_checkmate():
            reward += 20.0
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            reward -= 0.25
        return reward

    def _opponent_move(self) -> chess.Move:
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            raise RuntimeError("Opponent requested with no legal moves available.")
        if self.opponent == "heuristic":
            return best_heuristic_move(self.board)
        return self.random.choice(legal_moves)

    def _done(self) -> bool:
        return self.board.is_game_over() or self.steps >= self.max_halfmoves
