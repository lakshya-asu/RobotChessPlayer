from __future__ import annotations

from types import SimpleNamespace

import chess
import pytest

from chess_manipulator.rl.action_space import MoveActionSpace
from chess_manipulator.rl.checkpointing import (
    checkpoint_metadata_path,
    load_checkpoint_metadata,
)
from chess_manipulator.rl.continual import (
    evaluate_checkpoint_against_logs,
    promote_checkpoint_from_report,
)
from chess_manipulator.rl.encoding import encode_board
from chess_manipulator.rl.model import build_model, torch
from chess_manipulator.rl.train import DQNTrainer
from chess_manipulator.rl.trajectory_log import (
    LoggedTransition,
    build_game_log,
    read_game_logs,
    replay_transitions_from_logs,
    write_game_log,
)


pytest.importorskip("torch")


def _build_sample_black_game_log(game_id: str = "game-black-1") -> tuple[chess.Board, chess.Board, object]:
    space = MoveActionSpace()
    board_before = chess.Board()
    board_before.push(chess.Move.from_uci("e2e4"))
    move = chess.Move.from_uci("e7e5")
    action_index = space.encode(move)
    next_board = board_before.copy(stack=False)
    next_board.push(move)
    game_log = build_game_log(
        game_id=game_id,
        source="self_play",
        initial_fen=board_before.fen(),
        final_fen=next_board.fen(),
        result="*",
        transitions=[
            LoggedTransition(
                game_id=game_id,
                step_index=0,
                fen_before=board_before.fen(),
                fen_after=next_board.fen(),
                side_to_move="black",
                uci_move=move.uci(),
                action_index=action_index,
                reward=0.5,
                done=False,
                legal_action_indices=space.legal_indices(next_board),
                state=encode_board(board_before),
                next_state=encode_board(next_board),
                source="self_play",
                metadata={"source": "fixture"},
            )
        ],
        metadata={"agent_color": "black"},
    )
    return board_before, next_board, game_log


def test_trajectory_logs_round_trip_json_and_jsonl(tmp_path):
    _, _, json_game_log = _build_sample_black_game_log("game-black-json")
    _, _, jsonl_game_log = _build_sample_black_game_log("game-black-jsonl")
    json_path = tmp_path / "game.json"
    jsonl_path = tmp_path / "game.jsonl"

    write_game_log(json_path, json_game_log, log_format="json")
    write_game_log(jsonl_path, jsonl_game_log, log_format="jsonl")

    loaded = read_game_logs([json_path, jsonl_path])
    replay_transitions = replay_transitions_from_logs(loaded)

    assert len(loaded) == 2
    assert {game_log.game_id for game_log in loaded} == {"game-black-json", "game-black-jsonl"}
    assert all(game_log.step_count == 1 for game_log in loaded)
    assert {transition.action for transition in replay_transitions} == {
        json_game_log.transitions[0].action_index,
        jsonl_game_log.transitions[0].action_index,
    }


def test_offline_logs_seed_replay_buffer_and_fit(tmp_path):
    _, _, game_log = _build_sample_black_game_log()
    checkpoint_path = tmp_path / "trained.pt"
    trainer = DQNTrainer(
        episodes=0,
        checkpoint_path=str(checkpoint_path),
        batch_size=1,
        warmup_steps=1,
        target_sync_steps=1,
        buffer_capacity=8,
        agent_color=chess.BLACK,
    )

    loaded = trainer.seed_buffer_from_game_logs([game_log])
    optimized = trainer.fit_replay_buffer(1)

    assert loaded == 1
    assert optimized == 1
    trainer.save_checkpoint({"episodes": 0, "mean_reward": 0.0})
    assert checkpoint_path.exists()
    assert checkpoint_metadata_path(checkpoint_path).exists()


def test_checkpoint_evaluation_and_promotion_metadata_flow(tmp_path):
    _, _, game_log = _build_sample_black_game_log()
    log_path = tmp_path / "offline.json"
    write_game_log(log_path, game_log, log_format="json")

    checkpoint_path = tmp_path / "black_policy.pt"
    torch.save({"model_state_dict": build_model().state_dict(), "history": {"episodes": 0}}, checkpoint_path)

    eval_report_path = tmp_path / "evaluation.json"
    eval_args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        log=[str(log_path)],
        side_filter="black",
        device="cpu",
        report=str(eval_report_path),
    )
    report = evaluate_checkpoint_against_logs(eval_args)

    assert report["positions"] == 1
    assert report["legal_rate"] == pytest.approx(1.0)
    assert eval_report_path.exists()

    destination = tmp_path / "promoted.pt"
    promote_args = SimpleNamespace(
        checkpoint=str(checkpoint_path),
        destination=str(destination),
        report=str(eval_report_path),
        min_legal_rate=1.0,
        min_agreement_rate=0.0,
        overwrite=False,
        output=str(tmp_path / "promotion.json"),
    )
    result = promote_checkpoint_from_report(promote_args)

    assert destination.exists()
    assert checkpoint_metadata_path(destination).exists()
    metadata = load_checkpoint_metadata(destination)
    assert metadata["promotion"]["source_checkpoint"] == str(checkpoint_path)
    assert metadata["promotion"]["report"]["positions"] == 1
    assert result["promoted"] is True
