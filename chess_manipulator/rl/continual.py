"""Offline continual-learning entrypoints for the RL chess player."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import chess

from .adapter import RLMoveAdapter
from .checkpointing import append_checkpoint_event, merge_checkpoint_metadata, promote_checkpoint
from .train import DQNTrainer
from .trajectory_log import GameLog, read_game_logs


def _parse_side_filter(value: str) -> str:
    side = value.lower()
    if side not in {"white", "black", "both"}:
        raise argparse.ArgumentTypeError("side filter must be one of: white, black, both")
    return side


def _parse_agent_color(value: str) -> chess.Color:
    side = value.lower()
    if side == "white":
        return chess.WHITE
    if side == "black":
        return chess.BLACK
    raise argparse.ArgumentTypeError("agent color must be one of: white, black")


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_log_paths(paths: Sequence[str]) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in {".json", ".jsonl"}:
                    resolved.append(child)
            continue
        if path.exists():
            resolved.append(path)
            continue
        raise FileNotFoundError(f"Log path not found: {path}")
    return resolved


def _filter_game_log_by_side(game_log: GameLog, side_filter: str) -> GameLog:
    if side_filter == "both":
        return game_log
    transitions = [transition for transition in game_log.transitions if transition.side_to_move == side_filter]
    return GameLog(
        game_id=game_log.game_id,
        created_at=game_log.created_at,
        source=game_log.source,
        initial_fen=game_log.initial_fen,
        final_fen=game_log.final_fen,
        result=game_log.result,
        transitions=transitions,
        metadata=dict(game_log.metadata),
    )


def _load_game_logs(log_paths: Sequence[str], side_filter: str) -> list[GameLog]:
    resolved = _resolve_log_paths(log_paths)
    game_logs = read_game_logs(resolved)
    return [_filter_game_log_by_side(game_log, side_filter) for game_log in game_logs]


def _build_trainer(args: argparse.Namespace) -> DQNTrainer:
    checkpoint_path = args.checkpoint or str(Path("/tmp") / f"chess_manipulator_rl_{args.seed}.pt")
    return DQNTrainer(
        episodes=args.episodes,
        checkpoint_path=checkpoint_path,
        initial_checkpoint_path=args.initial_checkpoint,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        buffer_capacity=args.buffer_capacity,
        target_sync_steps=args.target_sync_steps,
        warmup_steps=args.warmup_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        max_halfmoves=args.max_halfmoves,
        opponent=args.opponent,
        agent_color=args.agent_color,
        seed=args.seed,
        device=args.device,
    )


def _summarize_logs(game_logs: Sequence[GameLog]) -> dict:
    total_transitions = sum(game_log.step_count for game_log in game_logs)
    return {
        "games": len(game_logs),
        "transitions": total_transitions,
        "game_ids": [game_log.game_id for game_log in game_logs],
    }


def collect_trajectory_logs(args: argparse.Namespace) -> dict:
    trainer = _build_trainer(args)
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    game_logs: list[GameLog] = []
    for index in range(args.episodes):
        game_id = f"collect-{index + 1}"
        log_path = None
        if output_dir is not None:
            log_path = output_dir / f"{game_id}.{args.log_format}"
        game_log = trainer.run_episode(
            source="self_play",
            game_id=game_id,
            log_path=log_path,
            log_format=args.log_format,
            training=args.train_during_collect,
            metadata={
                "mode": "collect",
                "side_filter": args.side_filter,
            },
        )
        game_logs.append(game_log)

    report = {
        "mode": "collect",
        "created_at": _utc_now_iso(),
        "log_format": args.log_format,
        "output_dir": str(output_dir) if output_dir is not None else "",
        "seed": args.seed,
        "agent_color": "white" if args.agent_color == chess.WHITE else "black",
        "train_during_collect": args.train_during_collect,
        "summary": _summarize_logs(game_logs),
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def train_from_logs_and_self_play(args: argparse.Namespace) -> dict:
    if not args.checkpoint:
        raise ValueError("A checkpoint path is required for training.")
    trainer = _build_trainer(args)
    offline_logs = _load_game_logs(args.offline_log, args.side_filter) if args.offline_log else []
    seeded = trainer.seed_buffer_from_game_logs(offline_logs)
    if args.offline_optimization_steps:
        trainer.fit_replay_buffer(args.offline_optimization_steps)

    self_play_logs: list[GameLog] = []
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(args.self_play_episodes):
        game_id = f"train-{index + 1}"
        log_path = None
        if output_dir is not None:
            log_path = output_dir / f"{game_id}.{args.log_format}"
        game_log = trainer.run_episode(
            source="self_play",
            game_id=game_id,
            log_path=log_path,
            log_format=args.log_format,
            training=True,
            metadata={
                "mode": "train",
                "side_filter": args.side_filter,
            },
        )
        self_play_logs.append(game_log)

    extra_updates = trainer.fit_replay_buffer(args.post_self_play_optimization_steps)
    history = {
        "mode": "train",
        "created_at": _utc_now_iso(),
        "offline_log_count": len(offline_logs),
        "offline_transition_count": seeded,
        "offline_optimization_steps": args.offline_optimization_steps,
        "self_play_episodes": args.self_play_episodes,
        "post_self_play_optimization_steps": args.post_self_play_optimization_steps,
        "optimization_steps": trainer.update_step,
        "post_self_play_updates": extra_updates,
        "steps": trainer.global_step,
        "seed": args.seed,
        "side_filter": args.side_filter,
        "summary": _summarize_logs(self_play_logs),
        "log_sources": [str(path) for path in _resolve_log_paths(args.offline_log)] if args.offline_log else [],
    }
    trainer.save_checkpoint(history)
    merge_checkpoint_metadata(args.checkpoint, {"continual_training": history})
    print(json.dumps(history, indent=2))
    return history


def evaluate_checkpoint_against_logs(args: argparse.Namespace) -> dict:
    if not args.checkpoint:
        raise ValueError("A checkpoint path is required for evaluation.")
    game_logs = _load_game_logs(args.log, args.side_filter)
    adapter = RLMoveAdapter(checkpoint_path=args.checkpoint, device=args.device)

    positions = 0
    legal_positions = 0
    agreement_positions = 0
    white_positions = 0
    black_positions = 0
    per_game: list[dict] = []

    for game_log in game_logs:
        game_positions = 0
        game_legal = 0
        game_agreement = 0
        for transition in game_log.transitions:
            board = chess.Board(transition.fen_before)
            if transition.side_to_move == "white":
                white_positions += 1
            else:
                black_positions += 1
            try:
                predicted = adapter.select_move(transition.fen_before)
                predicted_move = chess.Move.from_uci(predicted.uci_move)
                is_legal = predicted_move in board.legal_moves
            except Exception:
                is_legal = False
                predicted = None
            positions += 1
            game_positions += 1
            legal_positions += int(is_legal)
            game_legal += int(is_legal)
            if predicted is not None and predicted.uci_move == transition.uci_move:
                agreement_positions += 1
                game_agreement += 1
        per_game.append(
            {
                "game_id": game_log.game_id,
                "positions": game_positions,
                "legal_positions": game_legal,
                "agreement_positions": game_agreement,
            }
        )

    report = {
        "mode": "evaluate",
        "created_at": _utc_now_iso(),
        "checkpoint": args.checkpoint,
        "side_filter": args.side_filter,
        "log_count": len(game_logs),
        "positions": positions,
        "legal_positions": legal_positions,
        "agreement_positions": agreement_positions,
        "legal_rate": legal_positions / max(positions, 1),
        "agreement_rate": agreement_positions / max(positions, 1),
        "white_positions": white_positions,
        "black_positions": black_positions,
        "per_game": per_game,
        "logs": [str(path) for path in _resolve_log_paths(args.log)],
    }
    append_checkpoint_event(args.checkpoint, "evaluated", report)
    merge_checkpoint_metadata(args.checkpoint, {"evaluation": report})
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def promote_checkpoint_from_report(args: argparse.Namespace) -> dict:
    if not args.checkpoint:
        raise ValueError("A checkpoint path is required for promotion.")
    if args.report:
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    else:
        report = evaluate_checkpoint_against_logs(args)
    if report["legal_rate"] < args.min_legal_rate:
        raise RuntimeError(
            f"Checkpoint legal rate {report['legal_rate']:.3f} below threshold {args.min_legal_rate:.3f}"
        )
    if report["agreement_rate"] < args.min_agreement_rate:
        raise RuntimeError(
            f"Checkpoint agreement rate {report['agreement_rate']:.3f} below threshold {args.min_agreement_rate:.3f}"
        )

    promoted = promote_checkpoint(
        args.checkpoint,
        args.destination,
        promotion_metadata={
            "report": report,
            "thresholds": {
                "min_legal_rate": args.min_legal_rate,
                "min_agreement_rate": args.min_agreement_rate,
            },
        },
        overwrite=args.overwrite,
    )
    append_checkpoint_event(args.destination, "promoted", report)
    result = {
        "mode": "promote",
        "created_at": _utc_now_iso(),
        "source_checkpoint": args.checkpoint,
        "destination_checkpoint": args.destination,
        "promoted": True,
        "report": report,
        "metadata": promoted,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline continual-learning utilities for the RL chess player.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--episodes", type=int, default=10)
    common.add_argument("--checkpoint", default="")
    common.add_argument("--initial-checkpoint")
    common.add_argument("--batch-size", type=int, default=64)
    common.add_argument("--learning-rate", type=float, default=1e-3)
    common.add_argument("--gamma", type=float, default=0.99)
    common.add_argument("--buffer-capacity", type=int, default=10000)
    common.add_argument("--target-sync-steps", type=int, default=100)
    common.add_argument("--warmup-steps", type=int, default=250)
    common.add_argument("--epsilon-start", type=float, default=1.0)
    common.add_argument("--epsilon-end", type=float, default=0.1)
    common.add_argument("--epsilon-decay-steps", type=int, default=4000)
    common.add_argument("--max-halfmoves", type=int, default=80)
    common.add_argument("--opponent", choices=("random", "heuristic"), default="random")
    common.add_argument("--agent-color", type=_parse_agent_color, default=chess.BLACK)
    common.add_argument("--seed", type=int, default=7)
    common.add_argument("--device", default="cpu")

    collect = subparsers.add_parser("collect", parents=[common], help="Collect per-game trajectory logs.")
    collect.add_argument("--output-dir")
    collect.add_argument("--log-format", choices=("json", "jsonl"), default="json")
    collect.add_argument("--train-during-collect", action="store_true")
    collect.add_argument("--side-filter", type=_parse_side_filter, default="black")
    collect.add_argument("--report")
    collect.set_defaults(func=collect_trajectory_logs)

    train = subparsers.add_parser("train", parents=[common], help="Train from offline logs plus self-play.")
    train.add_argument("--offline-log", action="append", default=[])
    train.add_argument("--side-filter", type=_parse_side_filter, default="black")
    train.add_argument("--self-play-episodes", type=int, default=0)
    train.add_argument("--offline-optimization-steps", type=int, default=0)
    train.add_argument("--post-self-play-optimization-steps", type=int, default=0)
    train.add_argument("--output-dir")
    train.add_argument("--log-format", choices=("json", "jsonl"), default="json")
    train.set_defaults(func=train_from_logs_and_self_play)

    evaluate = subparsers.add_parser("evaluate", parents=[common], help="Evaluate a checkpoint against logs.")
    evaluate.add_argument("--log", action="append", required=True)
    evaluate.add_argument("--side-filter", type=_parse_side_filter, default="black")
    evaluate.add_argument("--report")
    evaluate.set_defaults(func=evaluate_checkpoint_against_logs)

    promote = subparsers.add_parser("promote", parents=[common], help="Promote a checkpoint if evaluation passes.")
    promote.add_argument("--log", action="append", default=[])
    promote.add_argument("--side-filter", type=_parse_side_filter, default="black")
    promote.add_argument("--report")
    promote.add_argument("--destination", required=True)
    promote.add_argument("--min-legal-rate", type=float, default=1.0)
    promote.add_argument("--min-agreement-rate", type=float, default=0.0)
    promote.add_argument("--overwrite", action="store_true")
    promote.add_argument("--output")
    promote.set_defaults(func=promote_checkpoint_from_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
