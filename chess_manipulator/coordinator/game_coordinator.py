"""Coordinator helpers for alternating chess players and runtime orchestration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import chess
except ImportError:  # pragma: no cover - runtime dependency
    chess = None

from chess_manipulator.chess.engine import EngineConfig
from chess_manipulator.players import DQNPlayerAdapter, EnginePlayerAdapter, PlayerAdapter, PlayerError


class GameCoordinatorError(RuntimeError):
    """Raised when a coordinated game cannot proceed."""


@dataclass(frozen=True)
class TurnRecord:
    """One validated turn produced by the coordinator."""

    ply_index: int
    side_to_move: str
    player_name: str
    uci_move: str
    san_move: str
    fen_before: str
    fen_after: str


@dataclass(frozen=True)
class GameResult:
    """Summary of a coordinated game loop."""

    initial_fen: str
    final_fen: str
    result: str
    termination_reason: str
    turns: List[TurnRecord] = field(default_factory=list)


class GameCoordinator:
    """Alternate turns between two FEN-in/UCI-out players."""

    def __init__(
        self,
        white_player: PlayerAdapter,
        black_player: PlayerAdapter,
        initial_fen: Optional[str] = None,
        max_plies: int = 200,
    ) -> None:
        if chess is None:
            raise GameCoordinatorError(
                "python-chess is required for the coordinator loop. Install the 'chess' package."
            )
        self._white_player = white_player
        self._black_player = black_player
        self._initial_fen = initial_fen
        self._max_plies = max_plies

    def play(self) -> GameResult:
        board = chess.Board(self._initial_fen) if self._initial_fen else chess.Board()
        turns: List[TurnRecord] = []
        termination_reason = "max_plies_reached"

        for ply_index in range(self._max_plies):
            if board.is_game_over(claim_draw=True):
                termination_reason = "game_over"
                break

            fen_before = board.fen()
            player = self._white_player if board.turn == chess.WHITE else self._black_player
            side_to_move = "white" if board.turn == chess.WHITE else "black"

            try:
                uci_move = player.select_move(fen_before)
            except PlayerError as exc:
                raise GameCoordinatorError(
                    f"{player.identity.name} failed to produce a move for {side_to_move}: {exc}"
                ) from exc

            try:
                move = chess.Move.from_uci(uci_move)
            except ValueError as exc:
                raise GameCoordinatorError(
                    f"{player.identity.name} returned invalid UCI '{uci_move}'."
                ) from exc

            if move not in board.legal_moves:
                raise GameCoordinatorError(
                    f"{player.identity.name} returned illegal move '{uci_move}' for FEN '{fen_before}'."
                )

            san_move = board.san(move)
            board.push(move)
            turns.append(
                TurnRecord(
                    ply_index=ply_index + 1,
                    side_to_move=side_to_move,
                    player_name=player.identity.name,
                    uci_move=uci_move,
                    san_move=san_move,
                    fen_before=fen_before,
                    fen_after=board.fen(),
                )
            )

        if board.is_game_over(claim_draw=True):
            result = board.result(claim_draw=True)
        else:
            result = "*"

        return GameResult(
            initial_fen=self._initial_fen or chess.STARTING_FEN,
            final_fen=board.fen(),
            result=result,
            termination_reason=termination_reason,
            turns=turns,
        )


def build_player_adapter(
    role: str,
    backend: str,
    executable: str,
    think_time: float,
    dqn_checkpoint: str,
    device: str,
    ) -> PlayerAdapter:
    display_name = f"{role}_{backend}"
    if backend == "dqn":
        return DQNPlayerAdapter(
            policy_name=display_name,
            checkpoint_path=dqn_checkpoint,
            device=device,
        )
    return EnginePlayerAdapter(
        engine_config=EngineConfig(
            backend=backend,
            executable=executable or None,
            think_time_sec=think_time,
        ),
        display_name=display_name,
    )


def side_to_move_from_fen(fen: str) -> str:
    """Return ``white`` or ``black`` from a FEN payload."""

    if not fen:
        return "white"
    parts = fen.split()
    if len(parts) < 2:
        raise GameCoordinatorError(f"Malformed FEN: {fen}")
    return "black" if parts[1] == "b" else "white"


def expected_fen_after_move(fen: str, uci_move: str) -> str:
    """Apply a legal UCI move to a FEN string and return the next FEN."""

    if chess is None:
        raise GameCoordinatorError("python-chess is required to advance a board state.")
    board = chess.Board(fen)
    move = chess.Move.from_uci(uci_move)
    if move not in board.legal_moves:
        raise GameCoordinatorError(f"Illegal move '{uci_move}' for FEN '{fen}'.")
    board.push(move)
    return board.fen()


def perception_confirmation_valid(
    expected_fen: str,
    observed_fen: str,
    observed_confidence: float,
    confidence_threshold: float,
    observed_update_time: float,
    board_pose_update_time: float,
    min_update_time: float,
) -> bool:
    """Return whether a perception update is fresh and strong enough to trust."""

    return (
        bool(expected_fen)
        and observed_fen == expected_fen
        and observed_confidence >= confidence_threshold
        and observed_update_time >= min_update_time
        and board_pose_update_time >= min_update_time
    )


def main() -> None:
    """Run a minimal coordinator loop from the command line."""
    parser = argparse.ArgumentParser(description="Run a skeleton dual-player chess coordinator.")
    parser.add_argument("--fen", default="", help="Optional starting FEN. Defaults to the standard initial position.")
    parser.add_argument("--max-plies", type=int, default=20, help="Maximum number of half-moves to play.")
    parser.add_argument("--white-backend", default="stockfish", help="White player backend: stockfish, lc0, fairy_stockfish, sunfish, or dqn.")
    parser.add_argument("--black-backend", default="stockfish", help="Black player backend: stockfish, lc0, fairy_stockfish, sunfish, or dqn.")
    parser.add_argument("--white-executable", default="", help="Optional executable path for the white engine backend.")
    parser.add_argument("--black-executable", default="", help="Optional executable path for the black engine backend.")
    parser.add_argument("--white-dqn-checkpoint", default="", help="Optional checkpoint path for the white DQN backend.")
    parser.add_argument("--black-dqn-checkpoint", default="", help="Optional checkpoint path for the black DQN backend.")
    parser.add_argument("--device", default="cpu", help="Torch device for DQN inference.")
    parser.add_argument("--think-time", type=float, default=0.1, help="Per-move think time in seconds for engine-backed players.")
    args = parser.parse_args()

    coordinator = GameCoordinator(
        white_player=build_player_adapter(
            "white",
            args.white_backend,
            args.white_executable,
            args.think_time,
            args.white_dqn_checkpoint,
            args.device,
        ),
        black_player=build_player_adapter(
            "black",
            args.black_backend,
            args.black_executable,
            args.think_time,
            args.black_dqn_checkpoint,
            args.device,
        ),
        initial_fen=args.fen or None,
        max_plies=args.max_plies,
    )
    result = coordinator.play()

    print(f"termination_reason={result.termination_reason}")
    print(f"result={result.result}")
    print(f"plies={len(result.turns)}")
    print(f"final_fen={result.final_fen}")
    for turn in result.turns:
        print(
            f"ply={turn.ply_index} side={turn.side_to_move} "
            f"player={turn.player_name} uci={turn.uci_move} san={turn.san_move}"
        )
