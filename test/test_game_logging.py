from pathlib import Path

from chess_manipulator.coordinator import GameLog, PlyLog, legal_moves_for_fen, write_game_log
from chess_manipulator.rl.trajectory_log import read_game_logs


def test_legal_moves_for_fen_returns_starting_moves():
    moves = legal_moves_for_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

    assert "e2e4" in moves
    assert "g1f3" in moves


def test_game_log_to_dict_serializes_turns():
    log = GameLog(
        schema_version=1,
        initial_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        final_fen="8/8/8/8/8/8/8/8 b - - 0 1",
        result="*",
        termination_reason="max_plies_reached",
        white_backend="stockfish",
        black_backend="dqn",
        white_checkpoint="",
        black_checkpoint="/tmp/demo.pt",
        max_plies=1,
        turns=[
            PlyLog(
                ply_index=1,
                side_to_move="white",
                player_name="white_stockfish",
                player_backend="stockfish",
                checkpoint_path="",
                fen_before="8/8/8/8/8/8/8/8 w - - 0 1",
                legal_moves=["a2a3"],
                chosen_move="a2a3",
                expected_fen="8/8/8/8/8/P7/8/8 b - - 0 1",
                confirmed_fen="8/8/8/8/8/P7/8/8 b - - 0 1",
                perception_confidence=0.9,
                execution_success=True,
                perception_confirmed=True,
            )
        ],
    )

    payload = log.to_dict()

    assert payload["black_checkpoint"] == "/tmp/demo.pt"
    assert payload["turns"][0]["chosen_move"] == "a2a3"
    assert payload["steps"][0]["uci_move"] == "a2a3"


def test_runtime_game_log_round_trips_into_rl_log_reader(tmp_path: Path):
    log = GameLog(
        schema_version=1,
        initial_fen="8/8/8/8/8/8/P7/8 w - - 0 1",
        final_fen="8/8/8/8/8/P7/8/8 b - - 0 1",
        result="*",
        termination_reason="partial_match",
        white_backend="stockfish",
        black_backend="dqn",
        white_checkpoint="",
        black_checkpoint="/tmp/demo.pt",
        max_plies=1,
        turns=[
            PlyLog(
                ply_index=1,
                side_to_move="white",
                player_name="white_stockfish",
                player_backend="stockfish",
                checkpoint_path="",
                fen_before="8/8/8/8/8/8/P7/8 w - - 0 1",
                legal_moves=["a2a3"],
                chosen_move="a2a3",
                expected_fen="8/8/8/8/8/P7/8/8 b - - 0 1",
                confirmed_fen="8/8/8/8/8/P7/8/8 b - - 0 1",
                perception_confidence=0.95,
                execution_success=True,
                perception_confirmed=True,
            )
        ],
    )

    path = write_game_log(log, tmp_path, "runtime_match")
    loaded = read_game_logs([path])

    assert len(loaded) == 1
    assert loaded[0].step_count == 1
    assert loaded[0].transitions[0].uci_move == "a2a3"
