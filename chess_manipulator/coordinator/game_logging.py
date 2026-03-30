"""Per-game trajectory logs for offline learned-player training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import chess
from chess_manipulator.rl.action_space import MoveActionSpace
from chess_manipulator.rl.encoding import encode_board


@dataclass(frozen=True)
class PlyLog:
    """One accepted half-move in a completed or partial match."""

    ply_index: int
    side_to_move: str
    player_name: str
    player_backend: str
    checkpoint_path: str
    fen_before: str
    legal_moves: list[str]
    chosen_move: str
    expected_fen: str
    confirmed_fen: str
    perception_confidence: float
    execution_success: bool
    perception_confirmed: bool


@dataclass(frozen=True)
class GameLog:
    """Serialized data product for offline RL updates and evaluation."""

    schema_version: int
    initial_fen: str
    final_fen: str
    result: str
    termination_reason: str
    white_backend: str
    black_backend: str
    white_checkpoint: str
    black_checkpoint: str
    max_plies: int
    turns: list[PlyLog] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["turns"] = [asdict(turn) for turn in self.turns]
        payload["steps"] = [_turn_to_step(turn) for turn in self.turns]
        return payload


def _turn_to_step(turn: PlyLog) -> dict[str, Any]:
    action_space = MoveActionSpace()
    board_before = chess.Board(turn.fen_before)
    board_after = chess.Board(turn.confirmed_fen)
    move = chess.Move.from_uci(turn.chosen_move)
    try:
        action_index = action_space.encode(move)
    except Exception:
        action_index = 0
    legal_action_indices = []
    for legal_move in turn.legal_moves:
        try:
            legal_action_indices.append(action_space.encode(chess.Move.from_uci(legal_move)))
        except Exception:
            continue
    reward = 0.25 if turn.execution_success and turn.perception_confirmed else -0.25
    return {
        "game_id": f"runtime-ply-{turn.ply_index}",
        "step_index": turn.ply_index - 1,
        "fen_before": turn.fen_before,
        "fen_after": turn.confirmed_fen,
        "side_to_move": turn.side_to_move,
        "uci_move": turn.chosen_move,
        "action_index": action_index,
        "reward": reward,
        "done": board_after.is_game_over(claim_draw=True),
        "legal_action_indices": legal_action_indices,
        "state": encode_board(board_before),
        "next_state": encode_board(board_after),
        "source": "runtime_match",
        "metadata": {
            "player_name": turn.player_name,
            "player_backend": turn.player_backend,
            "checkpoint_path": turn.checkpoint_path,
            "expected_fen": turn.expected_fen,
            "perception_confidence": turn.perception_confidence,
            "execution_success": turn.execution_success,
            "perception_confirmed": turn.perception_confirmed,
        },
    }


def legal_moves_for_fen(fen: str) -> list[str]:
    """Return sorted legal UCI moves for a position."""

    board = chess.Board(fen)
    return sorted(move.uci() for move in board.legal_moves)


def write_game_log(log: GameLog, output_dir: str | Path, stem: str) -> Path:
    """Write a game log JSON document and return its path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{stem}.json"
    path.write_text(json.dumps(log.to_dict(), indent=2), encoding="utf-8")
    return path
